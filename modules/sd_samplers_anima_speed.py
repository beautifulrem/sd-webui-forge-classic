"""Spectral Progressive Diffusion (SPEED) for Anima as a Forge k-diffusion sampler.

Xiao, Chao, Yariv and Wetzstein, "Spectral Progressive Diffusion for Efficient
Image and Video Generation", arXiv:2605.18736. Training-free inference: the
early, noise-dominated steps run on a spectrally truncated half-resolution
latent; at the delta-optimal transition time (paper Eq. 9-10) the latent is
expanded in the DCT domain with the new high frequencies filled at the current
noise level, then rescaled and re-timed (Eq. 5-6) before Euler continues at
full resolution. The math follows ``utils.py`` / ``latent_image_gen.py`` of the
official code at commit ca7801c9bdffe681742e9592345bcf4885959be5
(https://github.com/howardhx/speed); the k-diffusion segmentation follows the
ComfyUI port at commit 13418a080c2a5d5c17f7025fc91990a9dc68bd91
(https://github.com/ruwwww/ComfyUI-SPEED).

Portions copyright (c) 2026 Howard Xiao and copyright (c) 2026 A. Izzuddin
Al Faruq, used under the MIT License. Permission is hereby granted, free of
charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software,
subject to inclusion of this notice. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tqdm.auto import trange

from modules.anima_feature_conflicts import (
    ARTIST_MIXER,
    FREEFUSE,
    REGIONAL,
    SPEED_SAMPLER,
    is_feature_active,
    record_conflict_resolution,
)
from modules.anima_guidance_state import reset_stateful_guidance
from modules.anima_support import require_anima_flow_denoiser
from modules.sd_samplers_anima import _denoise, _euler_step, sample_anima_flow_euler


# Paper defaults: S = 2 stages (0.5, 1.0), delta = 0.01, DCT basis.
_COARSE_SCALE = 0.5
_DELTA = 0.01
# Power-law fit of the Wan 2.1 VAE latent spectrum (paper Table 4); Anima
# encodes with the Wan VAE, so this is the measured spectrum closest to it.
_SPECTRUM_A = 219.484718
_SPECTRUM_BETA = 2.422687
# Anima is trained at 512^2-class resolutions and up; a coarse stage smaller
# than 512 x 384 px (e.g. from 768 x 768) would leave that distribution.
_MIN_COARSE_AREA = 64 * 48
_PURE_NOISE_SIGMA = 1.0 - 1e-4


def activation_time(power: float, delta: float = _DELTA) -> float:
    """Delta-optimal activation time of one frequency with signal power ``power`` (Eq. 9)."""

    return 1.0 / (1.0 + math.sqrt(delta / (power * (1.0 + power - delta))))


def transition_sigma(height: int, width: int) -> float:
    """First full-resolution sigma for the half-resolution stage (Eq. 10):
    the activation time of the highest frequency the coarse grid represents."""

    omega = _COARSE_SCALE * min(height, width) / 2.0
    return activation_time(_SPECTRUM_A * omega ** (-_SPECTRUM_BETA))


def align(sigma: float, ratio: float) -> tuple[float, float]:
    """State rescale ``kappa`` (Eq. 5) and aligned sigma (Eq. 6) after expanding
    by the linear resolution ``ratio``."""

    kappa = ratio / (1.0 + (ratio - 1.0) * sigma)
    return kappa, sigma * kappa


def _dct_matrix(n: int, like: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II matrix (scipy ``dct(type=2, norm="ortho")``)."""

    k = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(n, dtype=torch.float64)[None, :]
    matrix = torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n)) * math.sqrt(2.0 / n)
    matrix[0] /= math.sqrt(2.0)
    return matrix.to(device=like.device, dtype=torch.float32)


def dct2(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2-D DCT over the last two (spatial) dimensions, in float32."""

    return _dct_matrix(x.shape[-2], x) @ x.float() @ _dct_matrix(x.shape[-1], x).T


def idct2(coefficients: torch.Tensor) -> torch.Tensor:
    return _dct_matrix(coefficients.shape[-2], coefficients).T @ coefficients @ _dct_matrix(coefficients.shape[-1], coefficients)


def spectral_expand(x: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    """Spectral noise expansion (paper Sec. 4.1, steps i-iii).

    ``x``'s DCT spectrum is embedded in the low-frequency corner of the target
    grid; every newly exposed coefficient is ``sigma * noise``, where ``noise``
    holds unit-variance DCT coefficients at the target size. Returns float32.
    """

    height, width = x.shape[-2:]
    coefficients = sigma * noise
    coefficients[..., :height, :width] = dct2(x)
    return idct2(coefficients)


def _coarse_size(height: int, width: int) -> tuple[int, int]:
    """Half-resolution grid, snapped to Anima's 2 x 2 patches."""

    def snap(size: int) -> int:
        return max(2, int(round(_COARSE_SCALE * size)) // 2 * 2)

    return snap(height), snap(width)


@dataclass(frozen=True)
class SpeedPlan:
    size: tuple[int, int]  # coarse (height, width)
    step: int  # first full-resolution step
    sigma: float  # sigma reached at the end of the coarse stage
    aligned_sigma: float  # sigma the first full-resolution step is evaluated at
    kappa: float


def plan_speed(sigmas: list[float], height: int, width: int) -> tuple[SpeedPlan | None, str | None]:
    """The two-stage plan for a schedule, or ``(None, reason)`` when SPEED has
    nothing sound to do and the caller should run plain Flow Euler."""

    size = _coarse_size(height, width)
    if size[0] * size[1] < _MIN_COARSE_AREA:
        return None, "the half-resolution stage would be smaller than 512x384"
    total = len(sigmas) - 1
    threshold = transition_sigma(height, width)
    step = next((i for i in range(total) if sigmas[i] <= threshold), total)
    if step < 1 or step >= total or sigmas[step] <= 0.0:
        return None, "the schedule has no step to switch resolution at"
    # The rescale compensates the ortho DCT's amplitude change sqrt(hw / HW);
    # equal to the nominal 2 unless a dimension had to be rounded.
    ratio = math.sqrt(height * width / (size[0] * size[1]))
    kappa, aligned_sigma = align(sigmas[step], ratio)
    return SpeedPlan(size, step, sigmas[step], aligned_sigma, kappa), None


def fallback_reason(model, sigmas: list[float]) -> str | None:
    """Why this pass cannot run at a reduced resolution, if it cannot.

    SPEED needs a txt2img pass from pure noise and a model whose every hook
    works at any latent size. Spatial wrappers (Regional, Artist Mixer,
    FreeFuse, Spectrum, MultiDiffusion, Radial, ...), control/reference inputs
    and inpainting masks are all tied to the full-resolution grid."""

    if not sigmas or sigmas[0] < _PURE_NOISE_SIGMA:
        return "img2img / denoise < 1 does not start from pure noise"
    if getattr(model, "mask", None) is not None:
        return "inpainting mask"
    process = getattr(model, "p", None)
    for feature in (FREEFUSE, REGIONAL, ARTIST_MIXER):
        if is_feature_active(process, feature):
            return feature
    unet = getattr(getattr(getattr(process, "sd_model", None), "forge_objects", None), "unet", None)
    if getattr(unet, "controlnet_linked_list", None) is not None:
        return "ControlNet"
    options = getattr(unet, "model_options", None) or {}
    wrapper = options.get("model_function_wrapper")
    # Wrappers that work at any latent size say so (Modulation guidance)
    if wrapper is not None and not getattr(wrapper, "__resolution_independent__", False):
        return "a model wrapper (Spectrum, MultiDiffusion, Radial, ...)"
    transformer_options = options.get("transformer_options") or {}
    if transformer_options.get("patches") or transformer_options.get("patches_replace"):
        return "attention patches (ControlNet-LLLite, IP-Adapter, ...)"

    from backend.args import dynamic_args

    if getattr(dynamic_args, "ACTIVE_LLLITE_DIT", None):
        return "ControlNet-LLLite"
    if dynamic_args.ref_latents:
        return "reference latents"
    return None


def _resize(latent: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    flat = latent.reshape(1, -1, *latent.shape[-2:]).float()
    resized = torch.nn.functional.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    return resized.reshape(*latent.shape[:-2], *size).to(latent.dtype)


def _publish_full_size_latent(size: tuple[int, int]) -> None:
    """Forge returns ``state.current_latent`` for an interrupted pass; one
    stopped in the coarse stage must still decode at the requested size."""

    from modules import shared

    latent = getattr(shared.state, "current_latent", None)
    if isinstance(latent, torch.Tensor) and tuple(latent.shape[-2:]) != tuple(size):
        shared.state.current_latent = _resize(latent, size)


@torch.no_grad()
def sample_anima_speed(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Anima Flow Euler with SPEED's half-resolution first stage."""

    require_anima_flow_denoiser(model, SPEED_SAMPLER)
    extra_args = {} if extra_args is None else extra_args
    values = sigmas.detach().cpu().tolist()
    full_size = tuple(x.shape[-2:])
    process = getattr(model, "p", None)

    reason = fallback_reason(model, values)
    plan = None
    if reason is None:
        plan, reason = plan_speed(values, *full_size)
    if plan is None:
        if hasattr(process, "extra_generation_params"):
            record_conflict_resolution(process, f"{SPEED_SAMPLER} ran as Anima Flow Euler: {reason}")
        return sample_anima_flow_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

    total = len(values) - 1
    if hasattr(process, "extra_generation_params"):
        process.extra_generation_params[SPEED_SAMPLER] = f"{plan.size[1]}x{plan.size[0]} latent until step {plan.step}/{total}"

    # The initial state is the seed's own white noise, x = sigma_0 * eps.
    # Truncating its DCT gives the coarse stage unit-variance noise; the
    # discarded coefficients are independent of that stage's trajectory, so
    # they fill the new frequencies at the transition instead of fresh draws.
    coefficients = dct2(x)
    noise = coefficients / values[0]
    height, width = plan.size
    sample = idct2(coefficients[..., :height, :width]).to(x.dtype)
    run_sigmas = sigmas.clone()
    run_sigmas[plan.step] = plan.aligned_sigma

    try:
        for i in trange(total, disable=disable):
            if i == plan.step:
                expanded = spectral_expand(sample, noise, plan.sigma)
                sample = (plan.kappa * expanded).to(x.dtype)
                # SMC / APG / DCW / CFG-Zero* history is per-resolution.
                reset_stateful_guidance(model)
            denoised = _denoise(model, sample, run_sigmas, i, extra_args, callback)
            if values[i + 1] <= 0.0:
                sample = denoised
            else:
                # The coarse stage ends at the unaligned sigma it expands at.
                sample = _euler_step(sample, denoised, run_sigmas[i], sigmas[i + 1])
    except BaseException:  # also Forge's InterruptedException (Interrupt / Skip)
        if tuple(sample.shape[-2:]) != full_size:
            _publish_full_size_latent(full_size)
        raise
    return sample
