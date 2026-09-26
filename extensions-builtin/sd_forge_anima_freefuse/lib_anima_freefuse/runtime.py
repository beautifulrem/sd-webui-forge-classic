"""Two-pass attention collection and per-LoRA spatial routing for Anima."""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from types import MethodType

import torch
import torch.nn.functional as F

from backend.memory_management import cast_to_device
from backend.operations import main_stream_worker, weights_manual_cast
from backend.patcher.base import LowVramPatch, OnlineLoRAPatch
from backend.patcher.lora import merge_lora_to_weight
from modules_forge.packages.comfy.weight_adapter import LoRAAdapter
from modules import spectrum_force
from modules.anima_support import branch_rows, call_model_wrapper, negpip_mask_for
from modules.forward_override import forward_overrides
from .masks import fit_mask_batch, generate_masks

logger = logging.getLogger("AnimaFreeFuse")
_PATCH_LOCK = threading.RLock()
_FILENAME_ATTRIBUTE = "_forge_anima_freefuse_filename"


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    selector: str
    concept: str


def _tag_patch_value(value, filename: str) -> None:
    if not filename:
        return
    try:
        setattr(value, _FILENAME_ATTRIBUTE, str(filename))
        return
    except (AttributeError, TypeError):
        pass
    if isinstance(value, (list, tuple)):
        for child in value:
            _tag_patch_value(child, filename)


def _patch_filename(value) -> str | None:
    filename = getattr(value, _FILENAME_ATTRIBUTE, None)
    if filename:
        return str(filename)
    if isinstance(value, (list, tuple)):
        for child in value:
            filename = _patch_filename(child)
            if filename:
                return filename
    return None


def _lora_function_patches(function):
    """Return ``(key, patches)`` for Forge LoRA weight functions, else ``None``."""

    if isinstance(function, OnlineLoRAPatch):
        return function.key, function.patch
    if isinstance(function, LowVramPatch):
        return function.key, function.patches.get(function.key, [])
    return None


def _plain_lora_factors(patch, weight):
    """Return ``(down, up, scale)`` when ``patch`` is a plain LoRA on ``weight``.

    Its merged delta is ``scale * up @ down``, so the routed contribution can
    run as two rank-r matmuls instead of a dense out x in delta. Anything else
    (LoCon mid, DoRA, reshape, offsets, weight functions, strength_model,
    other adapter types) returns ``None`` and takes the merge path.
    """

    if len(patch) != 5:
        return None
    strength, adapter, strength_model, offset, function = patch
    if type(adapter) is not LoRAAdapter or strength_model != 1.0 or offset is not None or function is not None:
        return None
    up, down, alpha, mid, dora_scale, reshape = adapter.weights
    if mid is not None or dora_scale is not None or reshape is not None or weight.dim() != 2:
        return None
    up, down = up.flatten(start_dim=1), down.flatten(start_dim=1)
    if up.shape[0] != weight.shape[0] or down.shape[1] != weight.shape[1] or up.shape[1] != down.shape[0]:
        return None
    # LoRAAdapter.calculate_weight: strength * (alpha / rank) * up @ down
    return down, up, strength * (1.0 if alpha is None else alpha / down.shape[0])


def install_patch_metadata_hook() -> None:
    """Tag patch payloads without changing Forge's patch tuple ABI."""

    from backend.patcher.base import ModelPatcher
    from backend.patcher.function_chain import function_chain_contains

    current = ModelPatcher.add_patches
    if function_chain_contains(current, "_anima_freefuse_metadata"):
        return

    def add_patches_with_metadata(
        self,
        patches,
        strength_patch=1.0,
        strength_model=1.0,
        *,
        filename=None,
        online_mode=None,
    ):
        for value in patches.values():
            _tag_patch_value(value, filename)
        return current(
            self,
            patches,
            strength_patch=strength_patch,
            strength_model=strength_model,
            filename=filename,
            online_mode=online_mode,
        )

    add_patches_with_metadata._anima_freefuse_metadata = True
    add_patches_with_metadata.__wrapped__ = current
    ModelPatcher.add_patches = add_patches_with_metadata


def _normalized_name(value: str) -> str:
    return os.path.splitext(os.path.basename(str(value).strip()))[0].casefold()


def _selector_matches(selector: str, filename: str | None) -> bool:
    if not selector or not filename:
        return False
    wanted = _normalized_name(selector)
    actual = _normalized_name(filename)
    return wanted == actual


def _conditioning_indices(markers, batch: int, device) -> torch.Tensor:
    return torch.tensor(branch_rows(markers, batch) or [], device=device, dtype=torch.long)


@dataclass
class AnimaFreeFuseState:
    adapters: list[AdapterSpec]
    token_positions: dict[str, list[int]]
    background_positions: list[int]
    collect_step: int
    collect_block: int
    top_k_ratio: float
    temperature: float
    bg_scale: float
    balance_iterations: int
    feather: int
    routing_strength: float
    routing_end: float
    bias_scale: float
    positive_bias: float
    bias_blocks: set[int]
    phase: str = "idle"
    current_step: int = 0
    total_steps: int = 1
    grid_height: int = 0
    grid_width: int = 0
    similarity_samples: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    background_samples: list[torch.Tensor] = field(default_factory=list)
    masks: dict[str, torch.Tensor] = field(default_factory=dict)
    matched_files: dict[str, set[str]] = field(default_factory=dict)
    # Patcher weight keys carrying one of the adapters (set by validate_patches)
    routed_weight_keys: set[str] = field(default_factory=set)
    # Per-generation caches: patch -> adapter (patch objects are kept alive in
    # the entry, so an id is never reused), and fitted masks / attention biases
    # of the current mask set, keyed by shape, dtype and device.
    _adapter_by_patch: dict[int, tuple] = field(default_factory=dict, repr=False)
    _fitted: dict[tuple, torch.Tensor | None] = field(default_factory=dict, repr=False)

    def configure_steps(self, total_steps: int) -> None:
        self.total_steps = max(1, int(total_steps))
        self.collect_step = min(max(0, int(self.collect_step)), self.total_steps - 1)

    def begin_collect(self) -> None:
        self.phase = "collect"
        self.current_step = 0
        self.similarity_samples = {adapter.name: [] for adapter in self.adapters}
        self.background_samples = []
        self.masks = {}
        self._fitted.clear()

    def finish_collection(self) -> None:
        missing = [
            name for name, samples in self.similarity_samples.items() if not samples
        ]
        if missing or not self.background_samples:
            missing_text = ", ".join(missing or ["background"])
            raise RuntimeError(
                f"Anima FreeFuse did not collect similarity maps for {missing_text}; "
                "verify the collection block and concept phrases"
            )
        averaged = {
            name: torch.stack(samples, dim=0).mean(dim=0)
            for name, samples in self.similarity_samples.items()
        }
        background = torch.stack(self.background_samples, dim=0).mean(dim=0)
        self.masks = generate_masks(
            averaged,
            background,
            height=self.grid_height,
            width=self.grid_width,
            bg_scale=self.bg_scale,
            iterations=self.balance_iterations,
            feather=self.feather,
        )

    def begin_generate(self) -> None:
        if len(self.masks) != len(self.adapters):
            raise RuntimeError(
                "Anima FreeFuse phase 2 started without a complete mask set"
            )
        self.phase = "generate"
        self.current_step = 0
        self._fitted.clear()

    def complete(self) -> None:
        self.phase = "complete"

    def abort(self) -> None:
        self.phase = "aborted"
        self.masks = {}
        self._fitted.clear()

    def routing_active(self) -> bool:
        if self.phase != "generate" or not self.masks:
            return False
        progress = self.current_step / max(1, self.total_steps - 1)
        return progress <= min(max(float(self.routing_end), 0.0), 1.0) + 1e-8

    def adapter_for_patch(self, patch) -> str | None:
        cached = self._adapter_by_patch.get(id(patch))
        if cached is None or cached[0] is not patch:
            filename = _patch_filename(patch[1]) if len(patch) > 1 else None
            name = next((adapter.name for adapter in self.adapters if _selector_matches(adapter.selector, filename)), None)
            cached = self._adapter_by_patch[id(patch)] = (patch, name)
        return cached[1]

    def validate_patches(self, patcher) -> None:
        self.matched_files = {adapter.name: set() for adapter in self.adapters}
        self.routed_weight_keys = set()
        self._adapter_by_patch.clear()
        offline = False
        ambiguous = []
        # Offline patches live in ``patcher.patches``; online LoRAs are
        # ``OnlineLoRAPatch`` weight wrappers holding a single patch each.
        entries = [
            (key, patch, False)
            for key, patches in patcher.patches.items()
            for patch in patches
        ]
        entries.extend(
            (key, patch, True)
            for key, functions in getattr(patcher, "weight_wrapper_patches", {}).items()
            for function in functions
            if isinstance(function, OnlineLoRAPatch)
            for patch in function.patch
        )
        for key, patch, online in entries:
            matches = [
                adapter.name
                for adapter in self.adapters
                if _selector_matches(adapter.selector, _patch_filename(patch[1]))
            ]
            if len(matches) > 1:
                ambiguous.append("/".join(matches))
            elif len(matches) == 1:
                filename = _patch_filename(patch[1])
                self.matched_files[matches[0]].add(filename or "unknown")
                self.routed_weight_keys.add(key)
                offline = offline or not online
                if not math.isclose(float(patch[2]), 1.0):
                    raise ValueError(
                        "Anima FreeFuse supports standard additive LoRA patches with strength_model = 1"
                    )
        if ambiguous:
            raise ValueError(
                f"FreeFuse adapter selectors overlap: {', '.join(sorted(set(ambiguous)))}"
            )
        missing = [
            adapter.selector
            for adapter in self.adapters
            if not self.matched_files[adapter.name]
        ]
        if missing:
            raise ValueError(
                f"FreeFuse could not match loaded Anima LoRAs: {', '.join(missing)}"
            )
        if offline:
            raise ValueError(
                "Anima FreeFuse requires online LoRA patches; reload the generation after enabling the panel"
            )

    def update_grid(self, input_x: torch.Tensor, dit) -> None:
        self.grid_height = math.ceil(int(input_x.shape[-2]) / int(dit.patch_spatial))
        self.grid_width = math.ceil(int(input_x.shape[-1]) / int(dit.patch_spatial))

    @torch.no_grad()
    def collect_similarity(self, x, q, k, transformer_options) -> None:
        if self.phase != "collect" or self.current_step != self.collect_step:
            return
        spatial_tokens = self.grid_height * self.grid_width
        if spatial_tokens <= 0 or q.shape[1] < spatial_tokens:
            raise RuntimeError(
                "Anima FreeFuse could not reconcile the latent and cross-attention grids"
            )
        q = q[:, :spatial_tokens]
        hidden = x[:, :spatial_tokens]
        cond_indices = _conditioning_indices(
            transformer_options.get("cond_or_uncond", []),
            q.shape[0],
            q.device,
        )
        if cond_indices.numel() == 0:
            return
        q = q.index_select(0, cond_indices)
        k = k.index_select(0, cond_indices)
        hidden = hidden.index_select(0, cond_indices)
        text_length = k.shape[1]
        positions = {
            name: [index for index in indices if 0 <= index < text_length]
            for name, indices in self.token_positions.items()
        }
        background = [
            index for index in self.background_positions if 0 <= index < text_length
        ]
        if any(not value for value in positions.values()) or not background:
            raise RuntimeError(
                "Anima FreeFuse concept token positions exceed the active conditioning length"
            )

        # Compute only the concept columns of cross-attention, chunking image
        # queries so collection stays bounded at 512-token Anima contexts.
        score_names = [*positions, "__background__"]
        score_positions = [*positions.values(), background]
        scores = {name: [] for name in score_names}
        scale = q.shape[-1] ** -0.5
        for start in range(0, spatial_tokens, 256):
            q_chunk = q[:, start : start + 256]
            logits = torch.einsum("bshd,blhd->bhsl", q_chunk.float(), k.float()) * scale
            attention = logits.softmax(dim=-1)
            for name, indices in zip(score_names, score_positions):
                scores[name].append(attention[..., indices].mean(dim=(1, 3)))
            del logits, attention
        scores = {name: torch.cat(parts, dim=-1) for name, parts in scores.items()}

        concept_scores = {name: scores[name] for name in positions}
        count = len(concept_scores)
        hidden = hidden.float()
        selected = max(1, int(spatial_tokens * min(max(self.top_k_ratio, 0.01), 0.5)))
        temperature = max(float(self.temperature), 1e-3)

        def similarity_to_top_tokens(score: torch.Tensor) -> torch.Tensor:
            """Softmax similarity of every image token to the mean hidden
            state of the ``selected`` best-scoring tokens."""
            indices = score.topk(selected, dim=-1).indices
            core = torch.gather(hidden, 1, indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])).mean(dim=1, keepdim=True)
            similarity = torch.bmm(core, hidden.transpose(1, 2)).squeeze(1)
            return (similarity / temperature).softmax(dim=-1).detach()

        for name, own in concept_scores.items():
            competitive = own * count - sum(other for other_name, other in concept_scores.items() if other_name != name)
            self.similarity_samples[name].append(similarity_to_top_tokens(competitive))
        self.background_samples.append(similarity_to_top_tokens(scores["__background__"]))

    def mask_for_output(
        self,
        name: str,
        output: torch.Tensor,
        *,
        spatial_capable: bool,
    ) -> torch.Tensor | None:
        if not spatial_capable:
            return None
        key = ("mask", name, tuple(output.shape[:-1]), self.grid_height, self.grid_width, output.dtype, output.device)
        if key not in self._fitted:
            self._fitted[key] = self._fit_mask(name, output)
        return self._fitted[key]

    def _fit_mask(self, name: str, output: torch.Tensor) -> torch.Tensor | None:
        mask = self.masks.get(name)
        if mask is None:
            return None
        mask = fit_mask_batch(mask, int(output.shape[0]))
        height, width = self.grid_height, self.grid_width
        if output.ndim == 5 and output.shape[-3:-1] == (height, width):
            return mask.to(output).view(output.shape[0], 1, height, width, 1)
        spatial = height * width
        if output.ndim == 3 and output.shape[1] % spatial == 0:
            repeats = output.shape[1] // spatial
            return mask.to(output).reshape(output.shape[0], spatial, 1).repeat(
                1, repeats, 1
            )
        # Text projections and time-only paths have no spatial axis; official
        # FreeFuse leaves their LoRA contribution unmasked during phase 2.
        return None

    def attention_bias(
        self, q: torch.Tensor, text_length: int, transformer_options
    ) -> torch.Tensor | None:
        if not self.routing_active() or (
            self.bias_scale <= 0.0 and self.positive_bias <= 0.0
        ):
            return None
        spatial = self.grid_height * self.grid_width
        if q.shape[1] % spatial:
            return None
        markers = tuple(transformer_options.get("cond_or_uncond", []))
        key = ("bias", markers, tuple(q.shape[:2]), text_length, self.grid_height, self.grid_width, q.dtype, q.device)
        if key not in self._fitted:
            self._fitted[key] = self._build_attention_bias(q, text_length, markers)
        return self._fitted[key]

    def _build_attention_bias(self, q: torch.Tensor, text_length: int, markers) -> torch.Tensor:
        spatial = self.grid_height * self.grid_width
        repeats = q.shape[1] // spatial
        bias = torch.zeros(
            (q.shape[0], 1, q.shape[1], text_length), device=q.device, dtype=q.dtype
        )
        cond = _conditioning_indices(markers, q.shape[0], q.device)
        row_gate = torch.zeros((q.shape[0], 1, 1, 1), device=q.device, dtype=q.dtype)
        row_gate[cond] = 1.0
        for adapter in self.adapters:
            adapter_mask = fit_mask_batch(
                self.masks[adapter.name], int(q.shape[0])
            )
            mask = (
                adapter_mask
                .to(q)
                .reshape(q.shape[0], 1, spatial, 1)
                .repeat(1, 1, repeats, 1)
            )
            term = (-float(self.bias_scale) * (1.0 - mask) + float(self.positive_bias) * mask) * row_gate
            for position in self.token_positions[adapter.name]:
                if 0 <= position < text_length:
                    bias[..., position : position + 1] += term
        return bias


@dataclass(frozen=True)
class RejectedFreeFuseState:
    message: str

    def configure_steps(self, total_steps: int) -> None:
        raise RuntimeError(f"Anima FreeFuse configuration failed: {self.message}")


def apply_freefuse_rejection(model, error: Exception):
    """Ensure a caught Script callback error still aborts the selected sampler."""

    patched = model.clone()
    patched.set_transformer_option("anima_freefuse_state", RejectedFreeFuseState(str(error)))
    return patched


def _make_cross_attention_forward(
    original, module, state: AnimaFreeFuseState, block_index: int
):
    def wrapped(self, x, context=None, rope_emb=None, transformer_options={}):
        if context is None:
            return original(
                x,
                context=context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )
        # This override replaces the class forward NegPiP hooks, so apply its
        # V-only negation here: K keeps the sign-restored tokens.
        negpip_mask = negpip_mask_for(context, transformer_options)
        q, k, v = self.compute_qkv(
            x,
            context,
            rope_emb=rope_emb,
            value_context=None if negpip_mask is None else context * negpip_mask,
        )
        if block_index == state.collect_block:
            state.collect_similarity(x, q, k, transformer_options)
        bias = (
            state.attention_bias(q, k.shape[1], transformer_options)
            if block_index in state.bias_blocks
            else None
        )
        return self.compute_attention(
            q,
            k,
            v,
            transformer_options=transformer_options,
            mask=bias,
        )

    return MethodType(wrapped, module)


def _make_linear_forward(
    original,
    module,
    state: AnimaFreeFuseState,
    module_name: str,
):
    spatial_capable = not any(
        marker in module_name
        for marker in (
            ".cross_attn.k_proj",
            ".cross_attn.v_proj",
            "t_embedder",
            "adaln_modulation",
            "t_embedding_norm",
        )
    )

    def wrapped(self, x):
        weight_functions = list(getattr(self, "weight_function", []))
        # (function, key, [(patch, adapter name or None)]), resolved once per call
        resolved = []
        for function in weight_functions:
            entry = _lora_function_patches(function)
            if entry is None:
                resolved.append((function, None, None))
            else:
                resolved.append((function, entry[0], [(patch, state.adapter_for_patch(patch)) for patch in entry[1]]))
        if not any(name for _, _, patches in resolved if patches for _, name in patches):
            return original(x)

        saved_weight = self.weight_function
        saved_bias = self.bias_function
        self.weight_function = []
        self.bias_function = []
        try:
            raw_weight, raw_bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=lambda value: value,
                bias_fn=(lambda value: value) if self.bias is not None else None,
            )
        finally:
            self.weight_function = saved_weight
            self.bias_function = saved_bias

        with main_stream_worker(raw_weight, raw_bias, signal):
            # raw_weight is a private copy; it becomes the base weight as is
            # unless something else must be merged into it (the deltas below
            # still need it unpatched).
            unpatched_weight = raw_weight
            base_weight = raw_weight
            deltas = {}  # adapter name -> summed dense delta
            factors = {}  # adapter name -> [(down, up, scale)] of plain LoRAs
            for function, key, patches in resolved:
                if patches is None:
                    if base_weight is unpatched_weight:
                        base_weight = base_weight.clone()
                    base_weight = function(base_weight)
                    continue
                non_target = []
                for patch, name in patches:
                    if name is None:
                        non_target.append(patch)
                        continue
                    if state.phase != "generate":
                        continue
                    # Read from the live patch every call (strengths change
                    # per step); a plain LoRA runs low rank, the rest merges.
                    plain = _plain_lora_factors(patch, unpatched_weight)
                    if plain is not None:
                        if plain[2] != 0.0:
                            factors.setdefault(name, []).append(plain)
                        continue
                    delta = merge_lora_to_weight(
                        [patch], unpatched_weight.clone(), key
                    ).sub_(unpatched_weight)
                    if name in deltas:
                        deltas[name].add_(delta)
                    else:
                        deltas[name] = delta
                if non_target:
                    if base_weight is unpatched_weight:
                        base_weight = base_weight.clone()
                    base_weight = merge_lora_to_weight(
                        non_target, base_weight, key
                    )
            for function in saved_bias:
                raw_bias = function(raw_bias)

            output = F.linear(
                x,
                base_weight.to(dtype=x.dtype),
                None if raw_bias is None else raw_bias.to(dtype=x.dtype),
            )
            if state.phase != "generate":
                return output
            strength = min(max(float(state.routing_strength), 0.0), 1.0)
            for name in dict.fromkeys([*deltas, *factors]):
                contribution = None
                if name in deltas:
                    contribution = F.linear(x, deltas[name].to(dtype=x.dtype), None)
                for down, up, scale in factors.get(name, ()):
                    hidden = F.linear(x, cast_to_device(down, x.device, x.dtype)) * scale
                    term = F.linear(hidden, cast_to_device(up, x.device, x.dtype))
                    contribution = term if contribution is None else contribution + term
                mask = state.mask_for_output(
                    name,
                    contribution,
                    spatial_capable=spatial_capable,
                )
                if mask is not None and state.routing_active():
                    contribution = contribution * ((1.0 - strength) + mask * strength)
                output = output + contribution
            return output

    return MethodType(wrapped, module)


def apply_freefuse_patch(model, state: AnimaFreeFuseState):
    patched = model.clone()
    dit = patched.model.diffusion_model
    state.validate_patches(patched)
    previous = patched.model_options.get("model_function_wrapper")
    # Build the per-module overrides once per pass; each model call only
    # installs and removes them (fresh closures per call are costly and
    # defeat torch.compile guards).
    overrides = [
        (
            block.cross_attn,
            _make_cross_attention_forward(block.cross_attn.forward, block.cross_attn, state, index),
        )
        for index, block in enumerate(dit.blocks)
    ]
    # Only layers that carry an adapter's LoRA are routed; the rest keep
    # their normal forward and cost nothing per call.
    overrides.extend(
        (module, _make_linear_forward(module.forward, module, state, module_name))
        for module_name, module in dit.named_modules()
        if isinstance(module, torch.nn.Linear)
        and f"diffusion_model.{module_name}.weight" in state.routed_weight_keys
    )

    def wrapper(model_function, args):
        with _PATCH_LOCK:
            state.update_grid(args["input"], dit)
            with forward_overrides(overrides):
                return call_model_wrapper(previous, model_function, args)

    patched.set_model_unet_function_wrapper(wrapper)
    patched.set_transformer_option("anima_freefuse_state", state)
    # Spectrum wraps this wrapper, so it must see the request on the patcher.
    spectrum_force.request(patched, "anima_freefuse")
    return patched
