"""CLIP pooled modulation guidance for Forge Neo's Anima DiT.

The implementation follows the official Cosmos modulation-guidance path:
conditional rows receive ``CLIP(p) + w * (CLIP(p+) - CLIP(p-))`` while
unconditional rows receive only their own base pooled embedding.  Per-block
learned scales inject the projected vector into Anima's AdaLN-LoRA argument.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from lib_anima_guidance.artifacts import ensure_pinned_download

logger = logging.getLogger("AnimaGuidance")

ADAPTER_URL = "https://huggingface.co/yresearch/cosmos-pooled/resolve/2b51db12d548debaf732387705518e01df943941/checkpoint_4000.pt"
ADAPTER_SHA256 = "27d4a33c817fb9ab9602b571f01f429bf34ddf96ab8b73fa4ed682f3266f84ea"
ADAPTER_MAX_BYTES = 180 * 1024 * 1024
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
_ADAPTER_CACHE: dict[str, dict[str, torch.Tensor]] = {}
_CLIP_CACHE: dict[str, tuple[torch.nn.Module, object]] = {}


def resolve_artifact(mode: str, local_path: str, automatic_path: str, *, url: str, sha256: str, maximum_bytes: int) -> str:
    if mode == "Local file":
        path = os.path.abspath(os.path.expanduser(str(local_path).strip()))
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"Anima modulation local artifact was not found: {path}")
        return path
    if mode != "Auto-download pinned":
        raise RuntimeError(f"Unknown Anima modulation artifact mode: {mode}")
    return ensure_pinned_download(path=automatic_path, url=url, sha256=sha256, maximum_bytes=maximum_bytes)


def _load_adapter(path: str) -> dict[str, torch.Tensor]:
    path = os.path.abspath(path)
    with _LOAD_LOCK:
        cached = _ADAPTER_CACHE.get(path)
        if cached is not None:
            return cached
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as error:
            raise RuntimeError("Anima modulation requires a PyTorch build with safe weights_only loading") from error
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
        _ADAPTER_CACHE[path] = state
        return state


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
        cached = _CLIP_CACHE.get(path)
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
        _CLIP_CACHE[path] = (model, tokenizer)
        return model, tokenizer


@torch.inference_mode()
def encode_clip_pooled(prompts, *, clip_path: str, config_dir: str, tokenizer_dir: str, device) -> torch.Tensor:
    model, tokenizer = _load_clip(clip_path, config_dir, tokenizer_dir)
    device = torch.device(device)
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
            return outputs.pooler_output.detach().float().cpu().contiguous()
        finally:
            model.to(device="cpu", dtype=torch.float16)


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


def validate_for_dit(scales: torch.Tensor, dit) -> int:
    blocks = getattr(dit, "blocks", None)
    hidden_size = getattr(getattr(dit, "final_layer", None), "hidden_size", None)
    if blocks is None or hidden_size is None:
        raise RuntimeError("Anima modulation requires Anima blocks and final_layer.hidden_size")
    expected = (len(blocks), int(hidden_size) * 3)
    if tuple(scales.shape) != expected:
        raise RuntimeError(f"Anima modulation adapter shape {tuple(scales.shape)} does not match model {expected}")
    return len(blocks)


def _modulation_pre_hook(module, args, kwargs):
    local_state = getattr(module, "_forge_anima_modulation_local", None)
    delta = getattr(local_state, "delta", None)
    if delta is None:
        return args, kwargs
    current = kwargs.get("adaln_lora_B_T_3D")
    if current is None:
        raise RuntimeError("Anima modulation could not find the block AdaLN-LoRA input")
    updated = dict(kwargs)
    updated["adaln_lora_B_T_3D"] = current + delta.to(device=current.device, dtype=current.dtype)
    return args, updated


def _ensure_hooks(dit):
    for block in dit.blocks:
        if not getattr(block, "_forge_anima_modulation_hook", False):
            block._forge_anima_modulation_local = threading.local()
            block.register_forward_pre_hook(_modulation_pre_hook, with_kwargs=True)
            block._forge_anima_modulation_hook = True


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
        patched = model.clone()
        dit = patched.model.diffusion_model
        total_blocks = validate_for_dit(self.scales, dit)
        start = min(max(int(self.start_layer), 0), total_blocks - 1)
        end = total_blocks - 1 if int(self.end_layer) < 0 else min(int(self.end_layer), total_blocks - 1)
        if start > end:
            raise RuntimeError(f"Anima modulation start layer {start} is after end layer {end}")
        _ensure_hooks(dit)

        previous = patched.model_options.get("model_function_wrapper")
        while getattr(previous, "__forge_pass_wrapper_kind__", None) == "anima_modulation":
            previous = getattr(previous, "__forge_previous_wrapper__", None)

        def wrapper(model_function, args):
            branches = [int(item) for item in args.get("cond_or_uncond", [])]
            input_x = args["input"]
            if not branches or input_x.shape[0] % len(branches) != 0:
                logger.warning("Anima modulation skipped a non-divisible conditioning batch.")
                return previous(model_function, args) if previous is not None else model_function(input_x, args["timestep"], **args["c"])
            chunk = input_x.shape[0] // len(branches)
            rows = []
            for branch in branches:
                rows.append(_expand_rows(self.positive if branch == 0 else self.negative, chunk))
            pooled = torch.cat(rows, dim=0)
            scales = self.scales.to(device=input_x.device, dtype=input_x.dtype)
            try:
                for index in range(start, end + 1):
                    dit.blocks[index]._forge_anima_modulation_local.delta = (pooled.to(input_x) * scales[index]).unsqueeze(1)
                return previous(model_function, args) if previous is not None else model_function(input_x, args["timestep"], **args["c"])
            finally:
                for index in range(start, end + 1):
                    dit.blocks[index]._forge_anima_modulation_local.delta = None

        wrapper.__spectrum_cache_safe__ = True
        wrapper.__forge_pass_wrapper_kind__ = "anima_modulation"
        wrapper.__forge_previous_wrapper__ = previous
        patched.set_model_unet_function_wrapper(wrapper)
        return patched, start, end
