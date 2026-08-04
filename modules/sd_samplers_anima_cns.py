"""Anima-calibrated Colored Noise Sampling for Forge Neo's ER-SDE solver.

The recoloring numerics are adapted from sorryhyun/ComfyUI-Spectrum-KSampler
(MIT) and the sampler loop follows Forge Neo's existing ER-SDE implementation.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import trange

from modules import paths_internal


GAMMA_URL = (
    "https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler/releases/"
    "download/0530/cns_gamma.npz"
)
GAMMA_SHA256 = "538d5fad8253a8799600d4ff2e1a89a79e770f5a684d52f01f16e628299a7e7e"
GAMMA_MAX_BYTES = 1024 * 1024
_GAMMA_LOCK = threading.RLock()
_GAMMA_ARRAYS: Optional[dict[str, np.ndarray]] = None


def _gamma_path() -> Path:
    return Path(paths_internal.models_path) / "anima_cns" / "cns_gamma.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_gamma_file() -> Path:
    """Return a verified gamma artifact, downloading the pinned file if absent."""

    target = _gamma_path()
    with _GAMMA_LOCK:
        if target.is_file() and _sha256(target) == GAMMA_SHA256:
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".npz.download")
        request = urllib.request.Request(GAMMA_URL, headers={"User-Agent": "forge-neo-anima-cns/1"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
                received = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > GAMMA_MAX_BYTES:
                        raise RuntimeError("Anima CNS gamma download exceeded the 1 MiB safety limit")
                    output.write(chunk)
            if _sha256(temporary) != GAMMA_SHA256:
                raise RuntimeError("Anima CNS gamma checksum mismatch")
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target


def load_gamma_arrays() -> dict[str, np.ndarray]:
    global _GAMMA_ARRAYS
    if _GAMMA_ARRAYS is None:
        with _GAMMA_LOCK:
            if _GAMMA_ARRAYS is None:
                target = ensure_gamma_file()
                with np.load(target) as data:
                    _GAMMA_ARRAYS = {
                        key: np.array(data[key], copy=True)
                        for key in ("gamma", "aspects", "sigmas")
                    }
    return _GAMMA_ARRAYS


def radial_bins(height: int, width: int, count: int) -> np.ndarray:
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fy**2 + fx**2)
    maximum = radius.max()
    if maximum > 0:
        radius = radius / maximum
    edges = np.linspace(0.0, 1.0 + 1e-9, count + 1)
    return np.clip(np.digitize(radius, edges) - 1, 0, count - 1)


class CNSRecolorer:
    def __init__(
        self,
        gamma: np.ndarray,
        aspects: np.ndarray,
        sigmas: np.ndarray,
        strength: float,
    ):
        if gamma.ndim != 3:
            raise ValueError(f"CNS gamma must have shape (A,T,F), got {gamma.shape}")
        self.gamma = np.asarray(gamma, dtype=np.float64)
        self.aspects = np.asarray(aspects, dtype=np.float64).reshape(-1, 2)
        self.strength = min(max(float(strength), 0.0), 1.0)
        self.frequency_bins = self.gamma.shape[-1]
        sigma_mid = np.asarray(sigmas, dtype=np.float64)[:-1]
        self.sigma_order = np.argsort(sigma_mid)
        self.sigmas = sigma_mid[self.sigma_order]
        self.selected_gamma: Optional[np.ndarray] = None
        self.bin_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    @classmethod
    def calibrated(cls, strength: float) -> "CNSRecolorer":
        arrays = load_gamma_arrays()
        return cls(arrays["gamma"], arrays["aspects"], arrays["sigmas"], strength)

    def _select_aspect(self, height: int, width: int) -> None:
        aspect = width / max(height, 1)
        calibrated = self.aspects[:, 1] / np.maximum(self.aspects[:, 0], 1.0)
        index = int(np.argmin(np.abs(calibrated - aspect)))
        self.selected_gamma = self.gamma[index][self.sigma_order]

    def _gamma_row(self, sigma: float) -> np.ndarray:
        return np.asarray(
            [
                np.interp(float(sigma), self.sigmas, self.selected_gamma[:, index])
                for index in range(self.frequency_bins)
            ],
            dtype=np.float64,
        )

    @torch.no_grad()
    def recolor(self, white: torch.Tensor, sigma: float) -> torch.Tensor:
        if self.strength <= 0.0:
            return white
        height, width = white.shape[-2:]
        if self.selected_gamma is None:
            self._select_aspect(height, width)

        key = (height, width, str(white.device))
        bin_map = self.bin_cache.get(key)
        if bin_map is None:
            bin_map = torch.from_numpy(radial_bins(height, width, self.frequency_bins)).to(
                device=white.device,
                dtype=torch.long,
            )
            self.bin_cache[key] = bin_map

        scale_values = np.sqrt(np.clip(1.0 - self._gamma_row(float(sigma)), 0.0, 1.0))
        scale = torch.from_numpy(scale_values).to(device=white.device, dtype=torch.float32)
        spectrum = torch.fft.fft2(white.float(), dim=(-2, -1)) * scale[bin_map]
        colored = torch.fft.ifft2(spectrum, dim=(-2, -1)).real
        colored = colored / colored.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        if self.strength < 1.0:
            colored = (1.0 - self.strength) * white.float() + self.strength * colored
            colored = colored / colored.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return colored.to(dtype=white.dtype)


def _require_anima_flow(model) -> None:
    predictor = model.inner_model.predictor
    if predictor.__class__.__name__ != "PredictionDiscreteFlow":
        raise RuntimeError("Anima ER SDE CNS requires Anima/PredictionDiscreteFlow")


@torch.no_grad()
def sample_anima_er_sde_cns(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    noise_scaler=None,
    max_stage=3,
    cns_strength=1.0,
):
    """Forge ER-SDE with its per-step white-noise draw CNS-recolored."""

    from k_diffusion.sampling import (
        default_noise_sampler,
        offset_first_sigma_for_snr,
        sigma_to_half_log_snr,
    )

    _require_anima_flow(model)
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    recolorer = CNSRecolorer.calibrated(cns_strength) if cns_strength > 0.0 else None
    s_in = x.new_ones([x.shape[0]])

    def default_er_sde_noise_scaler(value):
        return value * ((value**0.3).exp() + 10.0)

    noise_scaler = default_er_sde_noise_scaler if noise_scaler is None else noise_scaler
    integration_points = 200.0
    point_indices = torch.arange(0, integration_points, dtype=torch.float32, device=x.device)

    model_sampling = model.inner_model.predictor
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()
    old_denoised = None
    old_denoised_d = None

    for index in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[index] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": index, "sigma": sigmas[index], "sigma_hat": sigmas[index], "denoised": denoised})
        stage_used = min(max_stage, index + 1)
        if sigmas[index + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[index], er_lambdas[index + 1]
            alpha_s = sigmas[index] / er_lambda_s
            alpha_t = sigmas[index + 1] / er_lambda_t
            ratio_alpha = alpha_t / alpha_s
            ratio = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)
            x = ratio_alpha * ratio * x + alpha_t * (1 - ratio) * denoised

            if stage_used >= 2:
                delta = er_lambda_t - er_lambda_s
                lambda_step = -delta / integration_points
                lambda_position = er_lambda_t + point_indices * lambda_step
                scaled_position = noise_scaler(lambda_position)
                integral = torch.sum(1 / scaled_position) * lambda_step
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[index - 1])
                x = x + alpha_t * (delta + integral * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    integral_u = torch.sum((lambda_position - er_lambda_s) / scaled_position) * lambda_step
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[index - 2]) / 2)
                    x = x + alpha_t * ((delta**2) / 2 + integral_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                noise = noise_sampler(sigmas[index], sigmas[index + 1])
                if recolorer is not None:
                    noise = recolorer.recolor(noise, float(sigmas[index]))
                variance = (er_lambda_t**2 - er_lambda_s**2 * ratio**2).sqrt().nan_to_num(nan=0.0)
                x = x + alpha_t * noise * s_noise * variance
        old_denoised = denoised
    return x
