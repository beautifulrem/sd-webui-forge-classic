"""Advanced, composable guidance controllers for Anima flow inference."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

from modules.anima_support import call_model_wrapper, inherit_wrapper_flags, is_anima_auxiliary_denoiser, sigma_value
from modules.spectrum_force import in_sigma_window


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

        # Same order as the reference (pamparamm/sd-perturbed-attention
        # mg_nodes.py): extrapolate against the previous EMA, then fold this
        # step into it. The reference does this on the denoised (x0)
        # prediction; here it runs on the flow velocity, as in the paper
        # (arXiv:2602.20360), which differs once x and sigma change per step.
        previous = velocity if self._ema is None else self._ema
        result = velocity + self.momentum * (velocity - previous)

        if update_state:
            self._ema = ((1.0 - self.ema_decay) * velocity + self.ema_decay * previous).detach()
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
    Zero-init is meant for the very start of sampling from pure noise, so it
    only applies when the pass's first sigma is ``start_sigma`` (the
    schedule's maximum); hires and img2img passes start below it.
    """

    def __init__(self, zero_init_steps: int = 1, *, start_sigma: float = 1.0):
        self.zero_init_steps = max(int(zero_init_steps), 0)
        self.start_sigma = float(start_sigma)
        self.reset()

    def reset(self) -> None:
        self._sigmas: list[float] = []
        self._from_noise: bool | None = None

    def zero_init(self, sigma: float, *, update_state: bool = True) -> bool:
        if sigma in self._sigmas:
            return True
        if not update_state:
            return False
        if self._from_noise is None:
            self._from_noise = sigma >= self.start_sigma * (1.0 - 1e-5)
        if self._from_noise and len(self._sigmas) < self.zero_init_steps:
            self._sigmas.append(sigma)
            return True
        return False

    def would_zero_init(self, sigma: float) -> bool:
        """What ``zero_init(sigma)`` would return, without recording the step."""

        if sigma in self._sigmas:
            return True
        from_noise = self._from_noise
        if from_noise is None:
            from_noise = sigma >= self.start_sigma * (1.0 - 1e-5)
        return from_noise and len(self._sigmas) < self.zero_init_steps


def make_cfg_zero_star_cfg_function():
    """Forge CFG hook for CFG-Zero*'s optimized-scale combine (arXiv:2503.18886).

    Forge's ``cond``/``uncond`` residuals are ``x - denoised = sigma * v`` for
    Anima's rectified flow, so the combine runs on them directly.
    """

    @torch.no_grad()
    def cfg_function(args):
        x = args["input"]
        return cfg_zero_star_combine(
            x - args["cond_denoised"],
            x - args["uncond_denoised"],
            float(args["cond_scale"]),
        )

    return cfg_function


def make_zero_init_cfg_function(cfg_function: Callable | None, state: CFGZeroStarState, process=None):
    """CFG-Zero*'s zero-init as the outermost CFG hook.

    It must see every sigma of the pass, so it wraps the CFG range and Skim
    hooks instead of sitting inside them (the range hook does not call its
    inner function outside its window).  During the zero-init steps the
    velocity is zero: ``denoised = x`` and the solver leaves the latent
    unchanged.
    """

    @torch.no_grad()
    def zero_init_cfg_function(args):
        sigma = sigma_value(args.get("sigma", args.get("timestep")))
        if state.zero_init(sigma, update_state=not is_anima_auxiliary_denoiser(process)):
            return torch.zeros_like(args["input"])
        if cfg_function is not None:
            return cfg_function(args)
        return standard_cfg_residual(args)

    # Keeps the CFG range's unconditional-pass skip working through the wrapper.
    if hasattr(cfg_function, "_anima_guidance_range"):
        zero_init_cfg_function._anima_guidance_range = cfg_function._anima_guidance_range
    return zero_init_cfg_function


def make_zero_init_model_wrapper(previous: Callable | None, state: CFGZeroStarState, process=None):
    """Model wrapper that skips the forward passes of zero-init steps.

    The zero-init CFG hook discards the predictions of those steps, so the
    model need not run. Forge's ``model_function_wrapper`` returns the
    denoised prediction (``apply_model`` ends in ``calculate_denoised``), so
    returning the input gives ``denoised = x``: the zero velocity the CFG
    hook returns anyway. It shares the hook's step state, so probes, the
    pass-start rule and the CFG range behave the same; other steps run
    ``previous`` (or the model) unchanged.
    """

    def zero_init_model_wrapper(model_function, args):
        sigma = sigma_value(args["timestep"])
        if state.zero_init(sigma, update_state=not is_anima_auxiliary_denoiser(process)):
            return args["input"].clone()
        return call_model_wrapper(previous, model_function, args)

    return inherit_wrapper_flags(zero_init_model_wrapper, previous)


def apg_project(value: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample parallel / orthogonal parts of ``value`` w.r.t. ``direction``."""

    dims = tuple(range(1, value.ndim))
    value_f = value.float()
    unit = F.normalize(direction.float(), dim=dims)
    parallel = (value_f * unit).sum(dim=dims, keepdim=True) * unit
    return parallel.to(value.dtype), (value_f - parallel).to(value.dtype)


class APGState:
    """Adaptive Projected Guidance (Sadat et al., arXiv:2410.02416).

    Follows the paper's Algorithm 1 (also diffusers' Apache-2.0
    ``guiders/adaptive_projected_guidance.py``) on denoised predictions:
    the CFG update ``cond - uncond`` is accumulated with (reverse) momentum,
    capped to an L2 norm, split into parts parallel and orthogonal to the
    conditional prediction, and the parallel part is weighted by ``eta``.
    ``eta = 1`` with no cap and no momentum is plain CFG.
    """

    def __init__(self, eta: float = 0.0, norm_threshold: float = 0.0, momentum: float = -0.5):
        self.eta = float(eta)
        self.norm_threshold = float(norm_threshold)
        self.momentum = float(momentum)
        self._running: torch.Tensor | None = None

    def reset(self) -> None:
        self._running = None

    def combine(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
        *,
        update_state: bool = True,
    ) -> torch.Tensor:
        diff = cond - uncond
        if self.momentum != 0.0:
            running = self._running
            if running is not None and (
                running.shape != diff.shape or running.device != diff.device or running.dtype != diff.dtype
            ):
                running = None
            if running is not None:
                diff = diff + self.momentum * running
            if update_state:
                self._running = diff.detach()
        if self.norm_threshold > 0.0:
            norm = torch.linalg.vector_norm(diff.float(), dim=tuple(range(1, diff.ndim)), keepdim=True)
            diff = diff * (self.norm_threshold / norm.clamp_min(1e-12)).clamp(max=1.0).to(diff.dtype)
        parallel, orthogonal = apg_project(diff, cond)
        return cond + (float(guidance_scale) - 1.0) * (orthogonal + self.eta * parallel)


def make_apg_cfg_function(state: APGState, process=None):
    """Forge CFG hook for APG on Forge's denoised (x0) predictions.

    The paper applies APG to the denoised prediction D(z_t): projecting onto
    it is what separates the saturating component.  ``cond_denoised`` /
    ``uncond_denoised`` are exactly that for Anima's rectified flow.
    """

    @torch.no_grad()
    def cfg_function(args):
        x = args["input"]
        cond = args["cond_denoised"]
        scale = float(args["cond_scale"])
        if math.isclose(scale, 1.0):
            # Guidance adds nothing, and Forge's CFG 1 placeholder
            # unconditional prediction must not enter the momentum.
            return x - cond
        denoised = state.combine(
            cond,
            args["uncond_denoised"],
            scale,
            update_state=not is_anima_auxiliary_denoiser(process),
        )
        return x - denoised

    return cfg_function
