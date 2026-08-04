"""Skimmed CFG numerical core.

Adapted from Extraltodeus/Skimmed_CFG (Apache-2.0).  Forge's CFG hook receives
both denoised predictions and their residual representation, so this module
works in denoised/x0 space and leaves the hook-specific conversion to the UI
script.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def get_skimming_mask(
    x_orig: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    cond_scale: float,
    *,
    disable_flipping_filter: bool = False,
) -> torch.Tensor:
    """Return elements whose CFG extrapolation should be limited.

    The predicates are kept equivalent to upstream Skimmed_CFG.  Inputs are
    denoised/x0 predictions, not epsilon or velocity predictions.
    """

    denoised = x_orig - (
        (x_orig - uncond)
        + cond_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign()
        == (cond * cond_scale - uncond * (cond_scale - 1.0)).sign()
    )

    if disable_flipping_filter:
        return matching_pred_signs & matching_diff_after

    deviation_influence = denoised.sign() == (denoised - x_orig).sign()
    return matching_pred_signs & matching_diff_after & deviation_influence


@torch.no_grad()
def skim_prediction(
    x_orig: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    cond_scale: float,
    skimming_scale: float,
    *,
    disable_flipping_filter: bool = False,
) -> torch.Tensor:
    """Return a skimmed copy of ``cond`` without mutating caller tensors."""

    if abs(float(cond_scale)) < 1e-8:
        return cond.clone()

    result = cond.clone()
    mask = get_skimming_mask(
        x_orig,
        result,
        uncond,
        float(cond_scale),
        disable_flipping_filter=disable_flipping_filter,
    )
    if not torch.any(mask):
        return result

    guided = x_orig - (
        (x_orig - uncond)
        + cond_scale * ((x_orig - result) - (x_orig - uncond))
    )
    fallback = x_orig - (
        (x_orig - uncond)
        + skimming_scale * ((x_orig - result) - (x_orig - uncond))
    )
    correction = guided - fallback
    result[mask] = result[mask] - correction[mask] / cond_scale
    return result


@torch.no_grad()
def apply_skim_to_predictions(
    x_orig: torch.Tensor,
    cond_denoised: torch.Tensor,
    uncond_denoised: torch.Tensor,
    cond_scale: float,
    skimming_scale: float,
    *,
    full_skim_negative: bool = False,
    disable_flipping_filter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply upstream's two-sided Skimmed CFG transform.

    The conditional prediction is skimmed first, followed by the negative
    prediction using the already-adjusted conditional result.  Cloning is
    deliberate: predictions may be reused by post-CFG hooks.
    """

    cond = cond_denoised.clone()
    uncond = uncond_denoised.clone()
    scale = float(cond_scale)
    practical = float(skimming_scale)

    if scale <= 1.0:
        return cond, uncond

    uncond = skim_prediction(
        x_orig,
        uncond,
        cond,
        scale,
        0.0 if full_skim_negative else practical,
        disable_flipping_filter=disable_flipping_filter,
    )
    cond = skim_prediction(
        x_orig,
        cond,
        uncond,
        scale - 1.0,
        practical,
        disable_flipping_filter=disable_flipping_filter,
    )
    return cond, uncond
