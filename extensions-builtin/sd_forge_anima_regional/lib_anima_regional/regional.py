"""Spatial cross-attention output routing for Forge Neo's Anima blocks."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import torch

_PATCH_LOCK = threading.RLock()


def parse_blocks(text: str, total: int) -> set[int]:
    result = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            result.update(range(max(0, start), min(total - 1, end) + 1))
        else:
            value = int(part)
            if 0 <= value < total:
                result.add(value)
    if not result:
        raise ValueError("Anima Regional Conditioning block selection is empty")
    return result


def rectangle_token_mask(*, latent_t, latent_h, latent_w, patch_t, patch_h, x, y, width, height, feather, device, dtype):
    token_t = math.ceil(latent_t / patch_t)
    token_h = math.ceil(latent_h / patch_h)
    token_w = math.ceil(latent_w / patch_h)
    yy = (torch.arange(token_h, device=device, dtype=torch.float32) + 0.5) / token_h
    xx = (torch.arange(token_w, device=device, dtype=torch.float32) + 0.5) / token_w
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    left = min(max(float(x), 0.0), 1.0)
    top = min(max(float(y), 0.0), 1.0)
    right = min(max(left + float(width), 0.0), 1.0)
    bottom = min(max(top + float(height), 0.0), 1.0)
    edge = max(float(feather), 1e-6)
    horizontal = torch.minimum((grid_x - left) / edge, (right - grid_x) / edge).clamp(0.0, 1.0)
    vertical = torch.minimum((grid_y - top) / edge, (bottom - grid_y) / edge).clamp(0.0, 1.0)
    mask = torch.minimum(horizontal, vertical)
    mask = mask.unsqueeze(0).expand(token_t, -1, -1).reshape(1, -1, 1)
    return mask.to(dtype=dtype)


def conditioning_rows(cond_or_uncond, total_batch: int, device, dtype):
    markers = [int(item) for item in cond_or_uncond]
    if not markers or total_batch % len(markers) != 0:
        return torch.ones((total_batch, 1, 1), device=device, dtype=dtype)
    chunk = total_batch // len(markers)
    values = []
    for marker in markers:
        values.extend([1.0 if marker == 0 else 0.0] * chunk)
    return torch.tensor(values, device=device, dtype=dtype).reshape(total_batch, 1, 1)


def fit_context(context: torch.Tensor, batch: int, like: torch.Tensor) -> torch.Tensor:
    context = context.to(device=like.device, dtype=like.dtype)
    if context.ndim == 4 and context.shape[1] == 1:
        context = context.squeeze(1)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context.shape[0] == 1:
        return context.expand(batch, -1, -1)
    if context.shape[0] == batch:
        return context
    raise RuntimeError(f"Regional prompt batch {context.shape[0]} does not match attention batch {batch}")


@dataclass
class Region:
    conditioning: torch.Tensor
    x: float
    y: float
    width: float
    height: float
    strength: float


@dataclass
class RegionalState:
    regions: list[Region]
    blocks: set[int]
    feather: float
    base_preserve: float
    start_sigma: float
    end_sigma: float
    current_masks: list[torch.Tensor] | None = None

    def active(self, sigma: float) -> bool:
        low, high = sorted((self.start_sigma, self.end_sigma))
        return low - 1e-7 <= sigma <= high + 1e-7

    def prepare_masks(self, input_x: torch.Tensor, dit):
        latent_t = int(input_x.shape[-3])
        latent_h = int(input_x.shape[-2])
        latent_w = int(input_x.shape[-1])
        self.current_masks = [
            rectangle_token_mask(
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                patch_t=int(dit.patch_temporal),
                patch_h=int(dit.patch_spatial),
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
                feather=self.feather,
                device=input_x.device,
                dtype=input_x.dtype,
            )
            for region in self.regions
        ]


def _make_cross_attention_wrapper(original_forward, state: RegionalState, block_index: int):
    def wrapped(x, context=None, rope_emb=None, transformer_options={}):
        if context is None or state.current_masks is None or block_index not in state.blocks:
            return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
        base = original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
        row_mask = conditioning_rows(
            transformer_options.get("cond_or_uncond", []),
            base.shape[0],
            base.device,
            base.dtype,
        )
        numerator = torch.zeros_like(base)
        denominator = torch.zeros((base.shape[0], base.shape[1], 1), device=base.device, dtype=base.dtype)
        for region, spatial_mask in zip(state.regions, state.current_masks):
            if spatial_mask.shape[1] != base.shape[1]:
                if base.shape[1] % spatial_mask.shape[1] != 0:
                    raise RuntimeError(
                        f"Regional token mask {spatial_mask.shape[1]} does not match Anima sequence {base.shape[1]}"
                    )
                spatial_mask = spatial_mask.repeat(1, base.shape[1] // spatial_mask.shape[1], 1)
            alpha = spatial_mask.to(base) * max(0.0, float(region.strength)) * row_mask
            if not torch.any(alpha > 0):
                continue
            regional_context = fit_context(region.conditioning, base.shape[0], context)
            regional_options = dict(transformer_options)
            # NAG's negative branch is meaningful for the base positive/negative
            # pair, not for synthetic regional contexts repeated across rows.
            regional_options["anima_nag_skip"] = "regional_context"
            regional = original_forward(x, context=regional_context, rope_emb=rope_emb, transformer_options=regional_options)
            numerator = numerator + regional * alpha
            denominator = denominator + alpha
        blend = denominator.clamp(0.0, 1.0) * (1.0 - min(max(float(state.base_preserve), 0.0), 1.0))
        regional_average = numerator / denominator.clamp_min(1e-6)
        return base * (1.0 - blend) + regional_average * blend

    wrapped.__anima_regional_temporary__ = True
    return wrapped


def apply_regional_patch(model, state: RegionalState):
    patched = model.clone()
    dit = patched.model.diffusion_model
    previous = patched.model_options.get("model_function_wrapper")

    def wrapper(model_function, args):
        sigma = float(args["timestep"].flatten()[0].item())
        if not state.active(sigma):
            return previous(model_function, args) if previous is not None else model_function(args["input"], args["timestep"], **args["c"])
        with _PATCH_LOCK:
            state.prepare_masks(args["input"], dit)
            originals = []
            try:
                for index in state.blocks:
                    module = dit.blocks[index].cross_attn
                    original = module.forward
                    module.forward = _make_cross_attention_wrapper(original, state, index)
                    originals.append((module, original))
                adjusted = dict(args)
                adjusted["c"] = dict(args["c"])
                options = dict(adjusted["c"].get("transformer_options", {}))
                options["forge_spectrum_force_actual"] = "regional_conditioning"
                adjusted["c"]["transformer_options"] = options
                return previous(model_function, adjusted) if previous is not None else model_function(adjusted["input"], adjusted["timestep"], **adjusted["c"])
            finally:
                for module, original in reversed(originals):
                    if getattr(module.forward, "__anima_regional_temporary__", False):
                        module.forward = original
                state.current_masks = None

    wrapper.__forge_pass_wrapper_kind__ = "anima_regional"
    wrapper.__forge_previous_wrapper__ = previous
    patched.set_model_unet_function_wrapper(wrapper)
    return patched
