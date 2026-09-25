"""Advanced, composable guidance controllers for Anima flow inference."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from modules.anima_support import is_anima_auxiliary_denoiser
from modules.spectrum_force import in_sigma_window


def sigma_value(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().flatten()[0])
    return float(value)


def sigma_broadcast(value, reference: torch.Tensor):
    if torch.is_tensor(value):
        return value.reshape(-1, *([1] * (reference.ndim - 1))).to(
            device=reference.device, dtype=reference.dtype
        ).clamp_min(torch.finfo(reference.dtype).tiny)
    return max(float(value), 1e-12)


def standard_cfg_residual(args) -> torch.Tensor:
    cond = args["cond_denoised"]
    uncond = args["uncond_denoised"]
    denoised = uncond + (cond - uncond) * float(args["cond_scale"])
    return args["input"] - denoised


def make_guidance_range_cfg_function(
    previous_cfg_function: Callable | None,
    *,
    sigma_start: float,
    sigma_end: float,
):
    """Disable CFG outside a sigma interval while preserving the prior hook."""

    low, high = sorted((float(sigma_start), float(sigma_end)))

    @torch.no_grad()
    def cfg_function(args):
        # Same window test as the uncond skip below (and NAG / Skim / Regional)
        if in_sigma_window(sigma_value(args.get("sigma", args.get("timestep"))), low, high):
            if previous_cfg_function is not None:
                return previous_cfg_function(args)
            return standard_cfg_residual(args)
        return args["input"] - args["cond_denoised"]

    cfg_function._anima_guidance_range = (low, high)
    return cfg_function


IGNORES_UNCOND = "_anima_ignores_uncond"


def make_guidance_range_uncond_skip(cfg_function: Callable):
    """Pre-CFG hook that drops the unconditional batch outside the CFG range.

    Outside ``cfg_function``'s sigma window the result is the conditional
    prediction alone, so the negative forward pass is wasted work. It is
    only skipped while ``cfg_function`` is still the active CFG function and
    every post-CFG hook is marked as not reading ``uncond_denoised``;
    otherwise another extension would see Forge's zero placeholder.
    """

    low, high = cfg_function._anima_guidance_range

    def pre_cfg_function(model, cond, uncond, x, timestep, model_options):
        if (
            uncond is not None
            and not in_sigma_window(sigma_value(timestep), low, high)
            and model_options.get("sampler_cfg_function") is cfg_function
            and all(
                getattr(fn, IGNORES_UNCOND, False)
                for fn in model_options.get("sampler_post_cfg_function", [])
            )
        ):
            uncond = None
        return model, cond, uncond, x, timestep, model_options

    return pre_cfg_function


class MomentumGuidanceState:
    """EMA momentum in flow velocity space, isolated to one generation."""

    def __init__(self, momentum: float = 0.5, ema_decay: float = 0.6):
        self.momentum = float(momentum)
        self.ema_decay = float(ema_decay)
        self._ema: torch.Tensor | None = None

    def reset(self) -> None:
        self._ema = None

    def apply(
        self,
        denoised: torch.Tensor,
        input_x: torch.Tensor,
        sigma,
        *,
        update_state: bool = True,
    ) -> torch.Tensor:
        sig = sigma_broadcast(sigma, input_x)
        velocity = (input_x - denoised) / sig
        if self._ema is not None and (
            self._ema.shape != velocity.shape
            or self._ema.device != velocity.device
            or self._ema.dtype != velocity.dtype
        ):
            self.reset()

        if self._ema is None:
            result = velocity
            next_ema = velocity.detach()
        else:
            ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * velocity
            result = velocity + self.momentum * (velocity - ema)
            next_ema = ema.detach()

        if update_state:
            self._ema = next_ema
        return input_x - sig * result


def make_momentum_post_cfg_function(state: MomentumGuidanceState, process=None):
    @torch.no_grad()
    def post_cfg(args):
        if is_anima_auxiliary_denoiser(process):
            return args["denoised"]
        return state.apply(
            args["denoised"],
            args["input"],
            args["sigma"],
            update_state=True,
        )

    # Momentum only reads the combined prediction.
    setattr(post_cfg, IGNORES_UNCOND, True)
    return post_cfg


def _low_frequency(value: torch.Tensor) -> torch.Tensor:
    """Return a shape-preserving low-pass image without optional dependencies."""

    if value.ndim < 4 or min(value.shape[-2:]) < 2:
        return value
    height, width = value.shape[-2:]
    planes = value.reshape(-1, 1, height, width)
    low = F.avg_pool2d(planes, kernel_size=2, stride=2, ceil_mode=True)
    restored = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
    return restored.reshape(value.shape)


def fdg_combine(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
    high_frequency_scale: float,
) -> torch.Tensor:
    """Frequency-decoupled CFG with an independently bounded detail scale."""

    delta = cond - uncond
    low = _low_frequency(delta)
    high = delta - low
    return uncond + float(guidance_scale) * low + float(high_frequency_scale) * high


def make_fdg_cfg_function(high_frequency_scale: float):
    @torch.no_grad()
    def cfg_function(args):
        denoised = fdg_combine(
            args["cond_denoised"],
            args["uncond_denoised"],
            float(args["cond_scale"]),
            float(high_frequency_scale),
        )
        return args["input"] - denoised

    return cfg_function


def cfg_zero_star_scale(cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
    """Per-sample s* = <cond, uncond> / ||uncond||^2 (CFG-Zero*, eq. 11)."""

    cond_flat = cond.reshape(cond.shape[0], -1).float()
    uncond_flat = uncond.reshape(uncond.shape[0], -1).float()
    dot = (cond_flat * uncond_flat).sum(dim=1)
    # A zero unconditional velocity gives s* = 0, not NaN.
    scale = dot / uncond_flat.square().sum(dim=1).clamp_min(1e-12)
    return scale.reshape(-1, *([1] * (cond.ndim - 1))).to(cond.dtype)


def cfg_zero_star_combine(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """CFG-Zero* optimized-scale combine of two velocity predictions.

    ``uncond * s* + w * (cond - uncond * s*)``, written so that ``w = 1``
    returns ``cond`` exactly.  s* is invariant to a positive per-sample
    factor, so ``sigma * v`` residuals combine the same way as velocities.
    """

    scaled_uncond = uncond * cfg_zero_star_scale(cond, uncond)
    return cond + (float(guidance_scale) - 1.0) * (cond - scaled_uncond)


class CFGZeroStarState:
    """Counts the sampler steps of one pass for CFG-Zero*'s zero-init.

    A step is a distinct sigma: exact for single-evaluation samplers, while
    the extra sigmas of multi-evaluation samplers count as steps too.
    """

    def __init__(self, zero_init_steps: int = 1):
        self.zero_init_steps = max(int(zero_init_steps), 0)
        self._sigmas: list[float] = []

    def reset(self) -> None:
        self._sigmas = []

    def zero_init(self, sigma: float, *, update_state: bool = True) -> bool:
        if sigma in self._sigmas:
            return True
        if update_state and len(self._sigmas) < self.zero_init_steps:
            self._sigmas.append(sigma)
            return True
        return False


def make_cfg_zero_star_cfg_function(state: CFGZeroStarState, process=None):
    """Forge CFG hook for CFG-Zero* (arXiv:2503.18886).

    Forge's ``cond``/``uncond`` residuals are ``x - denoised = sigma * v`` for
    Anima's rectified flow, so the combine runs on them directly.  During the
    zero-init steps the velocity is zero: ``denoised = x`` and the solver
    leaves the latent unchanged.
    """

    @torch.no_grad()
    def cfg_function(args):
        x = args["input"]
        sigma = sigma_value(args.get("sigma", args.get("timestep")))
        if state.zero_init(sigma, update_state=not is_anima_auxiliary_denoiser(process)):
            return torch.zeros_like(x)
        return cfg_zero_star_combine(
            x - args["cond_denoised"],
            x - args["uncond_denoised"],
            float(args["cond_scale"]),
        )

    return cfg_function
