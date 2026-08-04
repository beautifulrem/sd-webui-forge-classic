"""Sliding-Mode Control CFG for Anima flow-matching inference.

The alpha-adaptive formulation follows sorryhyun/ComfyUI-Spectrum-KSampler.
The pure state object operates in velocity space; ``make_smc_cfg_function`` is
the Forge residual-contract adapter.
"""

from __future__ import annotations

from typing import Optional

import torch


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
    ) -> torch.Tensor:
        e = cond - uncond
        if self._e_prev is not None and self._e_prev.shape != e.shape:
            self.reset()
        e_prev = e if self._e_prev is None else self._e_prev
        surface = (e - e_prev) + self.lam * e_prev
        gain = self.alpha * e.abs().mean().clamp_min(1e-12)
        correction = -gain * torch.sign(surface)
        self._e_prev = e.detach()
        return uncond + float(guidance_scale) * (e + correction)


def make_smc_cfg_function(state: SMCCFGState):
    """Return a Forge ``sampler_cfg_function`` running SMC in velocity space."""

    @torch.no_grad()
    def cfg_function(args):
        x_in = args["input"]
        sigma = args.get("sigma", args.get("timestep"))
        if torch.is_tensor(sigma):
            sig = sigma.reshape(-1, *([1] * (x_in.ndim - 1))).to(
                device=x_in.device,
                dtype=x_in.dtype,
            )
            sig = sig.clamp_min(torch.finfo(x_in.dtype).tiny)
        else:
            sig = max(float(sigma), 1e-12)

        cond = args.get("cond_denoised", args["cond"])
        uncond = args.get("uncond_denoised", args["uncond"])
        v_cond = (x_in - cond) / sig
        v_uncond = (x_in - uncond) / sig
        v_out = state.combine(v_cond, v_uncond, float(args["cond_scale"]))

        # Forge computes cfg_result = x_in - returned_residual.
        return sig * v_out

    cfg_function._anima_smc_cfg = True
    cfg_function._anima_smc_state = state
    return cfg_function
