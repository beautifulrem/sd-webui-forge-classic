"""Skimmed CFG numerical core.

Adapted from Extraltodeus/Skimmed_CFG (Apache-2.0).  Forge's CFG hook receives
both denoised predictions and their residual representation, so this module
works in denoised/x0 space; ``make_skim_cfg_function`` adapts it to that hook.
"""

from __future__ import annotations

import torch

from lib_anima_guidance.advanced import sigma_value, standard_cfg_residual
from modules.spectrum_force import in_sigma_window


@torch.no_grad()
def get_skimming_mask(
    x_orig: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    cond_scale: float,
    *,
    disable_flipping_filter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mask, guided)``: the elements whose CFG extrapolation should
    be limited, and the CFG-guided prediction the mask was derived from.

    The predicates are kept equivalent to upstream Skimmed_CFG.  Inputs are
    denoised/x0 predictions, not epsilon or velocity predictions.
    """

    guided = x_orig - (
        (x_orig - uncond)
        + cond_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign()
        == (cond * cond_scale - uncond * (cond_scale - 1.0)).sign()
    )

    if disable_flipping_filter:
        return matching_pred_signs & matching_diff_after, guided

    deviation_influence = guided.sign() == (guided - x_orig).sign()
    return matching_pred_signs & matching_diff_after & deviation_influence, guided


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

    mask, guided = get_skimming_mask(
        x_orig,
        cond,
        uncond,
        float(cond_scale),
        disable_flipping_filter=disable_flipping_filter,
    )
    fallback = x_orig - (
        (x_orig - uncond)
        + skimming_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    # torch.where instead of boolean indexing: same values, no host sync
    return torch.where(mask, cond - (guided - fallback) / cond_scale, cond)


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

    The negative prediction is skimmed first, then the conditional one
    against the already-adjusted negative.  Both results are new tensors:
    predictions may be reused by post-CFG hooks.
    """

    scale = float(cond_scale)
    practical = float(skimming_scale)

    if scale <= 1.0:
        return cond_denoised.clone(), uncond_denoised.clone()

    uncond = skim_prediction(
        x_orig,
        uncond_denoised,
        cond_denoised,
        scale,
        0.0 if full_skim_negative else practical,
        disable_flipping_filter=disable_flipping_filter,
    )
    cond = skim_prediction(
        x_orig,
        cond_denoised,
        uncond,
        scale - 1.0,
        practical,
        disable_flipping_filter=disable_flipping_filter,
    )
    return cond, uncond


def make_skim_cfg_function(
    previous_cfg_function,
    *,
    sigma_start: float,
    sigma_end: float,
    skimming_scale: float,
    full_skim_negative: bool,
    disable_flipping_filter: bool,
    flip_sigma: float | None,
):
    """Build a Forge CFG hook while preserving an existing CFG hook.

    Forge expects ``sampler_cfg_function`` to return a residual which is
    subtracted from the sampler input.  Existing CFG functions are called with
    adjusted cond/uncond predictions, so Skimmed CFG composes before their
    combine rather than silently replacing them.
    """

    @torch.no_grad()
    def cfg_function(args):
        x_orig = args["input"]
        sigma = sigma_value(args.get("sigma", args.get("timestep")))

        adjusted_args = args
        if in_sigma_window(sigma, sigma_start, sigma_end):
            flip_filter = bool(disable_flipping_filter)
            if flip_sigma is not None and sigma > flip_sigma:
                flip_filter = not flip_filter

            cond, uncond = apply_skim_to_predictions(
                x_orig,
                args["cond_denoised"],
                args["uncond_denoised"],
                float(args["cond_scale"]),
                float(skimming_scale),
                full_skim_negative=bool(full_skim_negative),
                disable_flipping_filter=flip_filter,
            )
            adjusted_args = dict(args)
            adjusted_args["cond_denoised"] = cond
            adjusted_args["uncond_denoised"] = uncond
            adjusted_args["cond"] = x_orig - cond
            adjusted_args["uncond"] = x_orig - uncond

        if previous_cfg_function is not None:
            return previous_cfg_function(adjusted_args)
        return standard_cfg_residual(adjusted_args)

    return cfg_function
