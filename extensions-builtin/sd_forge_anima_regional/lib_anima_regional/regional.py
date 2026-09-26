"""Spatial cross-attention output routing for Forge Neo's Anima blocks."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import torch

from modules import spectrum_force
from modules.anima_support import call_model_wrapper, positive_rows
from modules.forward_override import cached_overrides, forward_overrides


_PATCH_LOCK = threading.RLock()


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


def fit_context_rows(context: torch.Tensor, batch: int, rows, like: torch.Tensor) -> torch.Tensor:
    """:func:`fit_context`, keeping only ``rows`` of the batch (all when None)."""
    context = fit_context(context, batch, like)
    return context if rows is None else context[list(rows)]


@dataclass
class Region:
    conditioning: torch.Tensor
    x: float
    y: float
    width: float
    height: float
    strength: float
    # NegPiP V-mask for this region's tokens (None without negative weights).
    negpip_mask: torch.Tensor | None = None


@dataclass
class RegionalState:
    regions: list[Region]
    blocks: set[int]
    feather: float
    base_preserve: float
    start_sigma: float
    end_sigma: float
    current_masks: list[torch.Tensor] | None = None
    # Whether each region's current mask covers any token
    region_nonempty: list[bool] = field(default_factory=list)
    # Device copies of masks, row indices and region contexts, keyed by shape,
    # device and dtype; the state lives for one sampling pass.
    cache: dict = field(default_factory=dict)

    def cached(self, key, build):
        if key not in self.cache:
            self.cache[key] = build()
        return self.cache[key]

    def end_call(self) -> None:
        self.current_masks = None

    def active(self, sigma: float) -> bool:
        return spectrum_force.in_sigma_window(sigma, self.start_sigma, self.end_sigma)

    def prepare_masks(self, input_x: torch.Tensor, dit):
        key = ("masks", tuple(input_x.shape[-3:]), input_x.device, input_x.dtype)
        self.current_masks, self.region_nonempty = self.cached(key, lambda: self._build_masks(input_x, dit))

    def _build_masks(self, input_x: torch.Tensor, dit):
        latent_t = int(input_x.shape[-3])
        latent_h = int(input_x.shape[-2])
        latent_w = int(input_x.shape[-1])
        # Built on the CPU so the emptiness check below needs no device sync
        masks = [
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
                device="cpu",
                dtype=torch.float32,
            )
            for region in self.regions
        ]
        nonempty = [bool((mask > 0).any()) and float(region.strength) > 0 for mask, region in zip(masks, self.regions)]
        return [mask.to(device=input_x.device, dtype=input_x.dtype) for mask in masks], nonempty


def _make_cross_attention_wrapper(original_forward, state: RegionalState, block_index: int):
    def wrapped(x, context=None, rope_emb=None, transformer_options={}):
        if context is None or state.current_masks is None or block_index not in state.blocks:
            return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
        base = original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
        batch = base.shape[0]
        # Regional contexts only run on the positive-prompt rows (every row
        # when the markers do not describe the batch); the others keep ``base``.
        branch = positive_rows(tuple(transformer_options.get("cond_or_uncond", [])), batch, state.cache, base.device)
        if branch is None:
            return base  # no positive-prompt row: nothing to blend
        rows = branch.rows
        region_x = branch.select(x)
        region_base = branch.select(base)
        numerator = torch.zeros_like(region_base)
        denominator = torch.zeros((region_base.shape[0], region_base.shape[1], 1), device=base.device, dtype=base.dtype)
        for number, (region, spatial_mask) in enumerate(zip(state.regions, state.current_masks)):
            if not state.region_nonempty[number]:
                continue
            if spatial_mask.shape[1] != base.shape[1]:
                if base.shape[1] % spatial_mask.shape[1] != 0:
                    raise RuntimeError(
                        f"Regional token mask {spatial_mask.shape[1]} does not match Anima sequence {base.shape[1]}"
                    )
                spatial_mask = spatial_mask.repeat(1, base.shape[1] // spatial_mask.shape[1], 1)
            alpha = spatial_mask.to(base) * float(region.strength)
            key = (number, batch, rows, context.device, context.dtype)
            regional_context = state.cached(("context", *key), lambda: fit_context_rows(region.conditioning, batch, rows, context))
            negpip_mask = (
                None
                if region.negpip_mask is None
                else state.cached(("negpip", *key), lambda: fit_context_rows(region.negpip_mask, batch, rows, context))
            )
            regional_options = branch.options(transformer_options, "regional_context", negpip_mask)
            regional = original_forward(region_x, context=regional_context, rope_emb=rope_emb, transformer_options=regional_options)
            numerator = numerator + regional * alpha
            denominator = denominator + alpha
        blend = denominator.clamp(0.0, 1.0) * (1.0 - min(max(float(state.base_preserve), 0.0), 1.0))
        regional_average = numerator / denominator.clamp_min(1e-6)
        mixed = region_base * (1.0 - blend) + regional_average * blend
        return branch.merge(base, mixed)

    return wrapped


def apply_regional_patch(model, state: RegionalState):
    patched = model.clone()
    dit = patched.model.diffusion_model
    previous = patched.model_options.get("model_function_wrapper")
    override = cached_overrides(lambda index: dit.blocks[index].cross_attn, lambda original, index: _make_cross_attention_wrapper(original, state, index))

    def wrapper(model_function, args):
        sigma = float(args["timestep"].flatten()[0].item())
        if not state.active(sigma):
            return call_model_wrapper(previous, model_function, args)
        with _PATCH_LOCK:
            state.prepare_masks(args["input"], dit)
            try:
                with forward_overrides(override(index) for index in state.blocks):
                    return call_model_wrapper(previous, model_function, args)
            finally:
                state.end_call()

    # Outside its window the wrapper is a pass-through, and the request below
    # forces actual forwards inside it, so Spectrum may cache the other steps.
    wrapper.__spectrum_cache_safe__ = previous is None or getattr(previous, "__spectrum_cache_safe__", False)
    patched.set_model_unet_function_wrapper(wrapper)
    # Spectrum wraps this wrapper, so it must see the request on the patcher.
    # Only steps inside the regional window need actual forwards.
    spectrum_force.request(patched, state.active)
    return patched
