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
from einops import rearrange

from backend.attention import attention_function
from backend.operations import main_stream_worker, weights_manual_cast
from backend.patcher.base import WeightPatch
from backend.patcher.lora import merge_lora_to_weight
from .masks import generate_masks

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


def install_patch_metadata_hook() -> None:
    """Tag patch payloads without changing Forge's six-field patch tuple ABI."""

    from backend.patcher.base import ModelPatcher

    current = ModelPatcher.add_patches
    if getattr(current, "_anima_freefuse_metadata", False):
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
    add_patches_with_metadata._anima_freefuse_previous = current
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
    markers = [int(value) for value in markers or []]
    if not markers or batch % len(markers):
        return torch.arange(batch, device=device)
    chunk = batch // len(markers)
    indices = []
    for group, marker in enumerate(markers):
        if marker == 0:
            indices.extend(range(group * chunk, (group + 1) * chunk))
    return torch.tensor(indices, device=device, dtype=torch.long)


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

    def configure_steps(self, total_steps: int) -> None:
        self.total_steps = max(1, int(total_steps))
        self.collect_step = min(max(0, int(self.collect_step)), self.total_steps - 1)

    def begin_collect(self) -> None:
        self.phase = "collect"
        self.current_step = 0
        self.similarity_samples = {adapter.name: [] for adapter in self.adapters}
        self.background_samples = []
        self.masks = {}

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

    def complete(self) -> None:
        self.phase = "complete"

    def abort(self) -> None:
        self.phase = "aborted"
        self.masks = {}

    def routing_active(self) -> bool:
        if self.phase != "generate" or not self.masks:
            return False
        progress = self.current_step / max(1, self.total_steps - 1)
        return progress <= min(max(float(self.routing_end), 0.0), 1.0) + 1e-8

    def adapter_for_patch(self, patch) -> str | None:
        filename = _patch_filename(patch[1]) if len(patch) > 1 else None
        for adapter in self.adapters:
            if _selector_matches(adapter.selector, filename):
                if filename:
                    self.matched_files.setdefault(adapter.name, set()).add(filename)
                return adapter.name
        return None

    def validate_patches(self, patcher) -> None:
        self.matched_files = {adapter.name: set() for adapter in self.adapters}
        offline = []
        ambiguous = []
        for patches in patcher.patches.values():
            for patch in patches:
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
                    if len(patch) < 6 or not bool(patch[5]):
                        offline.append(matches[0])
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
        for name, own in concept_scores.items():
            competitive = own * count - sum(
                other
                for other_name, other in concept_scores.items()
                if other_name != name
            )
            selected = max(
                1, int(spatial_tokens * min(max(self.top_k_ratio, 0.01), 0.5))
            )
            indices = competitive.topk(selected, dim=-1).indices
            gathered = torch.gather(
                hidden,
                1,
                indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
            )
            core = gathered.mean(dim=1, keepdim=True)
            similarity = torch.bmm(core, hidden.transpose(1, 2)).squeeze(1)
            similarity = (similarity / max(float(self.temperature), 1e-3)).softmax(
                dim=-1
            )
            self.similarity_samples[name].append(similarity.mean(dim=0).detach())

        bg_indices = (
            scores["__background__"]
            .topk(
                max(1, int(spatial_tokens * min(max(self.top_k_ratio, 0.01), 0.5))),
                dim=-1,
            )
            .indices
        )
        bg_core = torch.gather(
            hidden,
            1,
            bg_indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
        ).mean(dim=1, keepdim=True)
        bg_similarity = torch.bmm(bg_core, hidden.transpose(1, 2)).squeeze(1)
        bg_similarity = (bg_similarity / max(float(self.temperature), 1e-3)).softmax(
            dim=-1
        )
        self.background_samples.append(bg_similarity.mean(dim=0).detach())

    def mask_for_output(
        self,
        name: str,
        output: torch.Tensor,
        *,
        spatial_capable: bool,
    ) -> torch.Tensor | None:
        if not spatial_capable:
            return None
        mask = self.masks.get(name)
        if mask is None:
            return None
        height, width = self.grid_height, self.grid_width
        if output.ndim == 5 and output.shape[-3:-1] == (height, width):
            return mask.to(output).view(1, 1, height, width, 1)
        spatial = height * width
        if output.ndim == 3 and output.shape[1] % spatial == 0:
            repeats = output.shape[1] // spatial
            return mask.to(output).reshape(1, spatial, 1).repeat(1, repeats, 1)
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
        repeats = q.shape[1] // spatial
        bias = torch.zeros(
            (q.shape[0], 1, q.shape[1], text_length), device=q.device, dtype=q.dtype
        )
        cond = _conditioning_indices(
            transformer_options.get("cond_or_uncond", []), q.shape[0], q.device
        )
        row_gate = torch.zeros((q.shape[0], 1, 1, 1), device=q.device, dtype=q.dtype)
        row_gate[cond] = 1.0
        for adapter in self.adapters:
            mask = (
                self.masks[adapter.name]
                .to(q)
                .reshape(1, 1, spatial, 1)
                .repeat(1, 1, repeats, 1)
            )
            for position in self.token_positions[adapter.name]:
                if 0 <= position < text_length:
                    bias[..., position : position + 1] += (
                        -float(self.bias_scale) * (1.0 - mask)
                        + float(self.positive_bias) * mask
                    ) * row_gate
        return bias


@dataclass(frozen=True)
class RejectedFreeFuseState:
    message: str

    def configure_steps(self, total_steps: int) -> None:
        raise RuntimeError(f"Anima FreeFuse configuration failed: {self.message}")


def apply_freefuse_rejection(model, error: Exception):
    """Ensure a caught Script callback error still aborts the selected sampler."""

    patched = model.clone()
    options = dict(patched.model_options.get("transformer_options", {}))
    options["anima_freefuse_state"] = RejectedFreeFuseState(str(error))
    patched.model_options["transformer_options"] = options
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
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        if block_index == state.collect_block:
            state.collect_similarity(x, q, k, transformer_options)
        bias = (
            state.attention_bias(q, k.shape[1], transformer_options)
            if block_index in state.bias_blocks
            else None
        )
        if bias is None:
            return self.compute_attention(
                q, k, v, transformer_options=transformer_options
            )
        q_heads = rearrange(q, "b s h d -> b h s d")
        k_heads = rearrange(k, "b s h d -> b h s d")
        v_heads = rearrange(v, "b s h d -> b h s d")
        result = attention_function(
            q_heads,
            k_heads,
            v_heads,
            self.n_heads,
            mask=bias,
            skip_reshape=True,
        )
        return self.output_dropout(self.output_proj(result))

    wrapped.__anima_freefuse_temporary__ = True
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
        routed = []
        for function in weight_functions:
            if not isinstance(function, WeightPatch):
                continue
            for patch in function.patches.get(function.key, []):
                name = state.adapter_for_patch(patch)
                if name is not None:
                    routed.append((function, patch, name))
        if not routed:
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
            unpatched_weight = raw_weight
            base_weight = raw_weight.clone()
            deltas = {}
            for function in weight_functions:
                if not isinstance(function, WeightPatch):
                    base_weight = function(base_weight)
                    continue
                non_target = []
                for patch in function.patches.get(function.key, []):
                    name = state.adapter_for_patch(patch)
                    if name is None:
                        non_target.append(patch)
                        continue
                    if state.phase == "collect":
                        continue
                    patched = merge_lora_to_weight(
                        [patch], unpatched_weight.clone(), function.key
                    )
                    delta = patched - unpatched_weight
                    if name in deltas:
                        deltas[name].add_(delta)
                    else:
                        deltas[name] = delta
                if non_target:
                    base_weight = merge_lora_to_weight(
                        non_target, base_weight, function.key
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
            for name, delta in deltas.items():
                if not torch.any(delta):
                    continue
                contribution = F.linear(x, delta.to(dtype=x.dtype), None)
                mask = state.mask_for_output(
                    name,
                    contribution,
                    spatial_capable=spatial_capable,
                )
                if mask is not None and state.routing_active():
                    contribution = contribution * ((1.0 - strength) + mask * strength)
                output = output + contribution
            return output

    wrapped.__anima_freefuse_temporary__ = True
    return MethodType(wrapped, module)


def apply_freefuse_patch(model, state: AnimaFreeFuseState):
    patched = model.clone()
    dit = patched.model.diffusion_model
    state.validate_patches(patched)
    previous = patched.model_options.get("model_function_wrapper")

    def wrapper(model_function, args):
        with _PATCH_LOCK:
            state.update_grid(args["input"], dit)
            originals = []
            try:
                for index, block in enumerate(dit.blocks):
                    cross = block.cross_attn
                    original = cross.forward
                    cross.forward = _make_cross_attention_forward(
                        original, cross, state, index
                    )
                    originals.append((cross, original))
                for module_name, module in dit.named_modules():
                    if not isinstance(module, torch.nn.Linear):
                        continue
                    original = module.forward
                    module.forward = _make_linear_forward(
                        original,
                        module,
                        state,
                        module_name,
                    )
                    originals.append((module, original))
                adjusted = dict(args)
                adjusted["c"] = dict(args["c"])
                options = dict(adjusted["c"].get("transformer_options", {}))
                options["forge_spectrum_force_actual"] = "anima_freefuse"
                options["anima_freefuse_phase"] = state.phase
                adjusted["c"]["transformer_options"] = options
                if previous is not None:
                    return previous(model_function, adjusted)
                return model_function(
                    adjusted["input"], adjusted["timestep"], **adjusted["c"]
                )
            finally:
                for module, original in reversed(originals):
                    if getattr(module.forward, "__anima_freefuse_temporary__", False):
                        module.forward = original

    wrapper.__forge_pass_wrapper_kind__ = "anima_freefuse"
    wrapper.__forge_previous_wrapper__ = previous
    patched.set_model_unet_function_wrapper(wrapper)
    options = dict(patched.model_options.get("transformer_options", {}))
    options["anima_freefuse_state"] = state
    patched.model_options["transformer_options"] = options
    return patched
