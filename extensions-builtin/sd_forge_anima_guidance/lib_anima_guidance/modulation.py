"""CLIP pooled modulation guidance for Forge Neo's Anima DiT.

The implementation follows the official Cosmos modulation-guidance path:
with ``proj`` the adapter's Linear -> SiLU -> Linear embedder of a CLIP-L
pooled vector, conditional rows receive ``proj(p) + w * (proj(p+) - proj(p-))``
(the direction is taken after projection) while unconditional rows receive
only ``proj`` of their own base pooled embedding.  Per-block learned scales
inject the projected vector into Anima's AdaLN-LoRA argument.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from backend.text_processing.parsing import parse_prompt_attention
from modules import prompt_parser
from modules.anima_artifacts import ensure_pinned_download, file_version
from modules.anima_lora_support import ANIMA_BLOCK_MAPPINGS, anima_model_block
from modules.anima_support import call_model_wrapper, inherit_wrapper_flags

logger = logging.getLogger("AnimaGuidance")

ADAPTER_URL = "https://huggingface.co/yresearch/cosmos-pooled/resolve/2b51db12d548debaf732387705518e01df943941/checkpoint_4000.pt"
ADAPTER_SHA256 = "27d4a33c817fb9ab9602b571f01f429bf34ddf96ab8b73fa4ed682f3266f84ea"
ADAPTER_MAX_BYTES = 180 * 1024 * 1024
# Rows of the pinned adapter's per-block scales: it was trained on Anima 2B.
ADAPTER_BLOCKS = 28
CLIP_URL = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors"
CLIP_SHA256 = "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"
CLIP_MAX_BYTES = 260 * 1024 * 1024

EXPECTED_ADAPTER_KEYS = (
    "scales",
    "text_embedder_clip.linear_1.weight",
    "text_embedder_clip.linear_1.bias",
    "text_embedder_clip.linear_2.weight",
    "text_embedder_clip.linear_2.bias",
)

_LOAD_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()
# One entry each (the file in use), keyed by path and file version: another
# path, or a file replaced in place (e.g. a retrained adapter), reloads and
# frees the previous weights instead of pinning every one for good.
_ADAPTER_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_CLIP_CACHE: dict[tuple, tuple[torch.nn.Module, object]] = {}
# Pooled outputs of recent prompt batches: a hires pass or a rerun with the
# same prompts skips moving CLIP-L to the GPU and back.
_POOLED_CACHE: dict[tuple, torch.Tensor] = {}
_POOLED_CACHE_SIZE = 8


def resolve_artifact(mode: str, local_path: str, automatic_path: str, *, url: str, sha256: str, maximum_bytes: int) -> str:
    if mode == "Local file":
        local_path = str(local_path or "").strip()
        if not local_path:
            raise RuntimeError("Anima modulation local artifact path is empty")
        path = os.path.abspath(os.path.expanduser(local_path))
        if not os.path.isfile(path):
            raise RuntimeError(f"Anima modulation local artifact was not found: {path}")
        return path
    if mode != "Auto-download pinned":
        raise RuntimeError(f"Unknown Anima modulation artifact mode: {mode}")
    return ensure_pinned_download(path=automatic_path, url=url, sha256=sha256, maximum_bytes=maximum_bytes)


def _load_adapter(path: str) -> dict[str, torch.Tensor]:
    path = os.path.abspath(path)
    with _LOAD_LOCK:
        cache_key = file_version(path)
        cached = _ADAPTER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise RuntimeError("Anima modulation adapter must contain a tensor dictionary")
        missing = [key for key in EXPECTED_ADAPTER_KEYS if key not in raw]
        if missing:
            raise RuntimeError(f"Anima modulation adapter is missing: {', '.join(missing)}")
        state = {}
        for key in EXPECTED_ADAPTER_KEYS:
            tensor = raw[key]
            if not torch.is_tensor(tensor):
                raise RuntimeError(f"Anima modulation adapter entry {key!r} is not a tensor")
            state[key] = tensor.detach().float().cpu().contiguous()
        _validate_adapter_tensors(state)
        _ADAPTER_CACHE.clear()
        _ADAPTER_CACHE[cache_key] = state
        return state


def check_adapter_for_dit(adapter_path: str, dit) -> None:
    """Raise if the adapter cannot patch ``dit`` (before CLIP-L is fetched)."""

    validate_for_dit(_load_adapter(adapter_path)["scales"], dit)


def _validate_adapter_tensors(state):
    scales = state["scales"]
    first_weight = state["text_embedder_clip.linear_1.weight"]
    first_bias = state["text_embedder_clip.linear_1.bias"]
    second_weight = state["text_embedder_clip.linear_2.weight"]
    second_bias = state["text_embedder_clip.linear_2.bias"]
    if scales.ndim != 2 or first_weight.ndim != 2 or second_weight.ndim != 2:
        raise RuntimeError("Anima modulation adapter has invalid matrix ranks")
    adaln_dim = int(scales.shape[1])
    if first_weight.shape[0] != adaln_dim or first_bias.shape != (adaln_dim,):
        raise RuntimeError("Anima modulation first projection does not match AdaLN width")
    if second_weight.shape != (adaln_dim, adaln_dim) or second_bias.shape != (adaln_dim,):
        raise RuntimeError("Anima modulation second projection does not match AdaLN width")


def _load_clip(path: str, config_dir: str, tokenizer_dir: str):
    path = os.path.abspath(path)
    with _LOAD_LOCK:
        cache_key = file_version(path)
        cached = _CLIP_CACHE.get(cache_key)
        if cached is not None:
            return cached
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

        config = CLIPTextConfig.from_pretrained(config_dir, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        state = load_file(path, device="cpu")
        with torch.device("meta"):
            model = CLIPTextModel(config)
        incompatible = model.load_state_dict(state, strict=False, assign=True)
        missing = [key for key in incompatible.missing_keys if not key.endswith("position_ids")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Anima modulation CLIP-L weights do not match the pinned architecture: "
                f"missing={missing}, unexpected={incompatible.unexpected_keys}"
            )
        for module in model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if not buffer.is_meta:
                    continue
                if name == "position_ids":
                    replacement = torch.arange(config.max_position_embeddings).expand((1, -1))
                    setattr(module, name, replacement)
                else:
                    raise RuntimeError(f"Anima modulation CLIP-L left unsupported meta buffer {name!r}")
        model.eval().requires_grad_(False)
        _CLIP_CACHE.clear()
        _CLIP_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer


def clip_prompt_text(prompt: str, steps: int, hires_steps: int | None = None) -> str:
    """Plain text of a Forge prompt for the pooled CLIP-L encoder.

    The reference (ComfyUI-Spectrum-KSampler) encodes through ComfyUI's
    tokenizer, which drops emphasis syntax before CLIP sees the text; Forge's
    prompt editing (``[a:b:when]``, ``[a|b]``) has no counterpart there. One
    pooled vector serves the whole pass, so the prompt of its first step is
    used, then emphasis weights are dropped (the pooled output has no
    per-token weights) and ``BREAK`` becomes a space.
    """

    schedule = prompt_parser.get_learned_conditioning_prompt_schedules([str(prompt)], max(int(steps), 1), hires_steps)
    return "".join(
        " " if text == "BREAK" and weight == -1 else text
        for text, weight in parse_prompt_attention(schedule[0][0][1])
    )


@torch.inference_mode()
def encode_clip_pooled(prompts, *, clip_path: str, config_dir: str, tokenizer_dir: str, device) -> torch.Tensor:
    device = torch.device(device)
    prompts = tuple(str(prompt) for prompt in prompts)
    cache_key = (file_version(os.path.abspath(clip_path)), config_dir, tokenizer_dir, device.type, prompts)
    cached = _POOLED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model, tokenizer = _load_clip(clip_path, config_dir, tokenizer_dir)
    compute_dtype = torch.float32 if device.type == "cpu" else torch.float16
    with _ENCODE_LOCK:
        if device.type != "cpu":
            from backend import memory_management

            memory_management.free_memory(320 * 1024 * 1024, device)
        model.to(device=device, dtype=compute_dtype)
        try:
            tokens = tokenizer(
                list(prompts),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(
                input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
                return_dict=True,
            )
            pooled = outputs.pooler_output.detach().float().cpu().contiguous()
        finally:
            # The cached model rests on the CPU in the compute dtype: fp16
            # between GPU encodes, fp32 for a CPU encoder, so a CPU cache miss
            # does not convert all of CLIP-L (~235 MiB) to fp32 and back.
            model.to(device="cpu", dtype=compute_dtype)
    if len(_POOLED_CACHE) >= _POOLED_CACHE_SIZE:
        _POOLED_CACHE.pop(next(iter(_POOLED_CACHE)))
    _POOLED_CACHE[cache_key] = pooled
    return pooled


def project_pooled(pooled: torch.Tensor, adapter: dict[str, torch.Tensor]) -> torch.Tensor:
    result = F.linear(
        pooled.float(),
        adapter["text_embedder_clip.linear_1.weight"],
        adapter["text_embedder_clip.linear_1.bias"],
    )
    result = F.silu(result)
    return F.linear(
        result,
        adapter["text_embedder_clip.linear_2.weight"],
        adapter["text_embedder_clip.linear_2.bias"],
    )


def prepare_modulation_vectors(*, adapter_path, base_positive, base_negative, direction_positive, direction_negative, weight):
    adapter = _load_adapter(adapter_path)
    projected_positive = project_pooled(base_positive, adapter)
    projected_negative = project_pooled(base_negative, adapter)
    direction = float(weight) * (
        project_pooled(direction_positive, adapter) - project_pooled(direction_negative, adapter)
    )
    return projected_positive + direction, projected_negative, adapter["scales"].clone()


def block_layout(adapter_blocks: int, model_blocks: int) -> tuple[int, ...]:
    """Model block receiving each adapter block's scale row.

    The adapter's scales are per block of the model it was trained on. On a
    depth-expanded Anima (2B = 28 blocks, 2.9B = 40, 3.8B = 52) each row goes
    to the preserved original block and the inserted copies get no
    modulation, as in sorryhyun/ComfyUI-Spectrum-KSampler
    ``mod_guidance_layouts.py`` (MIT; commit
    b1ad4f9a1a4a0c9bdf7b13dee851d8b0bdbe38a6). The lineage is Forge's
    ANIMA_BLOCK_MAPPINGS (the 28 -> 40 one matches that file's manifest).
    """

    if adapter_blocks == model_blocks:
        return tuple(range(model_blocks))
    if (adapter_blocks, model_blocks) not in ANIMA_BLOCK_MAPPINGS:
        raise RuntimeError(
            f"Anima modulation adapter for {adapter_blocks} blocks has no known layout for a {model_blocks}-block model"
        )
    return tuple(anima_model_block(index, adapter_blocks, model_blocks) for index in range(adapter_blocks))


def dit_block_count(dit) -> int:
    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise RuntimeError("Anima modulation requires Anima blocks")
    return len(blocks)


def validate_for_dit(scales: torch.Tensor, dit) -> tuple[int, ...]:
    """The model block of each scale row (see ``block_layout``)."""

    hidden_size = getattr(getattr(dit, "final_layer", None), "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Anima modulation requires final_layer.hidden_size")
    if scales.ndim != 2 or int(scales.shape[1]) != int(hidden_size) * 3:
        raise RuntimeError(
            f"Anima modulation adapter shape {tuple(scales.shape)} does not match model width {int(hidden_size) * 3}"
        )
    return block_layout(int(scales.shape[0]), dit_block_count(dit))


def _add_delta(delta, module, args, kwargs):
    """Forward pre-hook (bound to ``delta`` with ``functools.partial``)."""

    current = kwargs.get("adaln_lora_B_T_3D")
    if current is None:
        raise RuntimeError("Anima modulation could not find the block AdaLN-LoRA input")
    updated = dict(kwargs)
    updated["adaln_lora_B_T_3D"] = current + delta.to(device=current.device, dtype=current.dtype)
    return args, updated


@contextmanager
def _block_deltas(pairs):
    """Add each ``(block, delta)`` pair's delta to the block's AdaLN-LoRA input
    for the ``with`` block only; the hooks are removed on exit, also when the
    block raises.

    A forward pre-hook rather than a forward override (``modules/forward_override.py``):
    per-block compile runs every call with an override inside the blocks
    eager, while Dynamo traces a hook as a frame of its own, so modulated
    calls keep the compiled blocks. See docs/remi-architecture-notes.md.
    """

    handles = []
    try:
        for block, delta in pairs:
            handles.append(block.register_forward_pre_hook(partial(_add_delta, delta), with_kwargs=True))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def _expand_rows(value: torch.Tensor, batch: int) -> torch.Tensor:
    if value.shape[0] == batch:
        return value
    if value.shape[0] == 1:
        return value.expand(batch, -1)
    raise RuntimeError(f"Anima modulation prompt batch {value.shape[0]} does not match latent batch {batch}")


@dataclass
class ModulationPatch:
    positive: torch.Tensor
    negative: torch.Tensor
    scales: torch.Tensor
    start_layer: int
    end_layer: int

    def apply(self, model):
        """Install the patch on a clone of ``model``.

        ``start_layer`` / ``end_layer`` count the adapter's blocks (Anima
        2B's for the pinned adapter), so on a depth-expanded model they
        select the same original blocks. Returns the clamped values.
        """

        patched = model.clone()
        dit = patched.model.diffusion_model
        layout = validate_for_dit(self.scales, dit)
        total_blocks = len(layout)
        start = min(max(int(self.start_layer), 0), total_blocks - 1)
        end = total_blocks - 1 if int(self.end_layer) < 0 else min(int(self.end_layer), total_blocks - 1)
        if start > end:
            raise RuntimeError(f"Anima modulation start layer {start} is after end layer {end}")
        # (model block, adapter scale row) of each modulated block
        targets = [(layout[row], row) for row in range(start, end + 1)]

        previous = patched.model_options.get("model_function_wrapper")
        # The per-block deltas depend only on the batch layout and the input's
        # device/dtype, so they are built once, not on every model call.
        deltas: dict[tuple, tuple[torch.Tensor, ...]] = {}

        def deltas_for(branches: tuple[int, ...], chunk: int, reference: torch.Tensor) -> tuple[torch.Tensor, ...]:
            key = (branches, chunk, reference.device, reference.dtype)
            cached = deltas.get(key)
            if cached is None:
                if len(deltas) >= 8:
                    deltas.clear()
                positive, negative, scales = (t.to(reference) for t in (self.positive, self.negative, self.scales))
                pooled = torch.cat([_expand_rows(positive if branch == 0 else negative, chunk) for branch in branches], dim=0)
                cached = deltas[key] = tuple((pooled * scales[row]).unsqueeze(1) for _, row in targets)
            return cached

        def wrapper(model_function, args):
            branches = tuple(int(item) for item in args.get("cond_or_uncond", []))
            input_x = args["input"]
            if not branches or input_x.shape[0] % len(branches) != 0:
                logger.warning("Anima modulation skipped a non-divisible conditioning batch.")
                return call_model_wrapper(previous, model_function, args)
            chunk = input_x.shape[0] // len(branches)
            pairs = zip((dit.blocks[index] for index, _ in targets), deltas_for(branches, chunk, input_x))
            with _block_deltas(pairs):
                return call_model_wrapper(previous, model_function, args)

        # Per-row pooled vectors work at any latent size (Anima SPEED).
        inherit_wrapper_flags(wrapper, previous)
        patched.set_model_unet_function_wrapper(wrapper)
        return patched, start, end
