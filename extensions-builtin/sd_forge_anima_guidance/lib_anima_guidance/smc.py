"""Sliding-Mode Control CFG for Anima flow-matching inference.

The alpha-adaptive formulation follows sorryhyun/ComfyUI-Spectrum-KSampler.
The pure state object operates in velocity space; ``make_smc_cfg_function`` is
the Forge residual-contract adapter.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from lib_anima_guidance.advanced import effective_cfg_scale, sigma_broadcast
from modules.anima_support import is_anima_auxiliary_denoiser


class SMCCFGState:
    """Stateful alpha-adaptive sliding-mode CFG controller."""

    def __init__(self, lam: float = 5.0, alpha: float = 0.2):
        self.lam = float(lam)
        self.alpha = float(alpha)
        self._e_prev: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._e_prev = None

    def combine(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
        *,
        update_state: bool = True,
    ) -> torch.Tensor:
        e = cond - uncond
        if self._e_prev is not None and self._e_prev.shape != e.shape:
            self.reset()
        e_prev = e if self._e_prev is None else self._e_prev
        surface = (e - e_prev) + self.lam * e_prev
        # Per sample: the reference's batch-wide mean (identical at batch 1)
        # would make each image's correction depend on the rest of the batch.
        gain = self.alpha * e.abs().mean(dim=tuple(range(1, e.ndim)), keepdim=True).clamp_min(1e-12)
        correction = -gain * torch.sign(surface)
        if update_state:
            self._e_prev = e.detach()
        return uncond + float(guidance_scale) * (e + correction)


def make_smc_cfg_function(state: SMCCFGState, process=None):
    """Return a Forge ``sampler_cfg_function`` running SMC in velocity space."""

    @torch.no_grad()
    def cfg_function(args):
        x_in = args["input"]
        cond = args["cond_denoised"]
        if math.isclose(float(args["cond_scale"]), 1.0):
            # Skipped at CFG 1 like the reference: there is no guidance to
            # correct, Forge skips the unconditional pass, and its
            # placeholder must not enter the error history.
            return x_in - cond

        sig = sigma_broadcast(args.get("sigma", args.get("timestep")), x_in)
        uncond = args["uncond_denoised"]
        v_cond = (x_in - cond) / sig
        v_uncond = (x_in - uncond) / sig
        v_out = state.combine(
            v_cond,
            v_uncond,
            effective_cfg_scale(args),
            update_state=not is_anima_auxiliary_denoiser(process),
        )

        # Forge computes cfg_result = x_in - returned_residual.
        return sig * v_out

    return cfg_function
