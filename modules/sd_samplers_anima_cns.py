"""Anima-calibrated Colored Noise Sampling for Forge Neo's ER-SDE solver.

The recoloring numerics are adapted from sorryhyun/ComfyUI-Spectrum-KSampler
(MIT); the solver is Forge Neo's own ER-SDE, with its noise draws recolored.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from modules import paths_internal
from modules.anima_artifacts import ensure_pinned_download
from modules.anima_support import require_anima_flow_denoiser


GAMMA_URL = (
    "https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler/releases/"
    "download/0530/cns_gamma.npz"
)
GAMMA_SHA256 = "538d5fad8253a8799600d4ff2e1a89a79e770f5a684d52f01f16e628299a7e7e"
GAMMA_MAX_BYTES = 1024 * 1024
_GAMMA_LOCK = threading.RLock()
_GAMMA_ARRAYS: Optional[dict[str, np.ndarray]] = None
_GAMMA_RETRY_SECONDS = 600.0
_GAMMA_FAILED_AT: Optional[float] = None

logger = logging.getLogger(__name__)


def _gamma_path() -> Path:
    return Path(paths_internal.models_path) / "anima_cns" / "cns_gamma.npz"


def ensure_gamma_file() -> Path:
    """Return a verified gamma artifact, downloading the pinned file if absent."""

    return Path(
        ensure_pinned_download(
            path=str(_gamma_path()),
            url=GAMMA_URL,
            sha256=GAMMA_SHA256,
            maximum_bytes=GAMMA_MAX_BYTES,
            timeout=30,
            user_agent="forge-neo-anima-cns/1",
        )
    )


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
        # (sigmas, selected gamma) uploaded per device: the per-step lookup
        # then runs on the device, with no host sync or upload
        self.device_tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.bin_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    @classmethod
    def calibrated(cls, strength: float, arrays: dict[str, np.ndarray]) -> "CNSRecolorer":
        return cls(arrays["gamma"], arrays["aspects"], arrays["sigmas"], strength)

    @classmethod
    def try_calibrated(cls, strength: float, process=None) -> Optional["CNSRecolorer"]:
        """Return a recolorer, or ``None`` (plain ER-SDE noise) when the gamma
        artifact is unavailable, e.g. offline. Failed downloads are not retried
        for ``_GAMMA_RETRY_SECONDS`` so each generation does not stall."""

        global _GAMMA_FAILED_AT
        if _GAMMA_ARRAYS is None and _GAMMA_FAILED_AT is not None:
            if time.monotonic() - _GAMMA_FAILED_AT < _GAMMA_RETRY_SECONDS:
                reason = "gamma file unavailable (retry pending)"
                _record_cns_fallback(process, reason)
                return None
        try:
            # Only fetching/reading the artifact falls back; errors in the
            # recolorer itself are real bugs and must surface.
            arrays = load_gamma_arrays()
        except Exception as error:
            _GAMMA_FAILED_AT = time.monotonic()
            reason = f"gamma file unavailable ({type(error).__name__})"
            logger.warning("Anima CNS falls back to white noise: %s: %s", reason, error)
            _record_cns_fallback(process, reason)
            return None
        _GAMMA_FAILED_AT = None
        return cls.calibrated(strength, arrays)

    def _select_aspect(self, height: int, width: int) -> None:
        aspect = width / max(height, 1)
        calibrated = self.aspects[:, 1] / np.maximum(self.aspects[:, 0], 1.0)
        index = int(np.argmin(np.abs(calibrated - aspect)))
        self.selected_gamma = self.gamma[index][self.sigma_order]
        self.device_tables.clear()

    def _device_table(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        table = self.device_tables.get(key)
        if table is None:
            sigmas, gamma = self.sigmas, self.selected_gamma
            if len(sigmas) == 1:  # np.interp's constant: one flat segment
                sigmas, gamma = np.append(sigmas, sigmas[0] + 1.0), np.repeat(gamma, 2, axis=0)
            # float32: MPS has no float64
            table = tuple(torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=torch.float32) for value in (sigmas, gamma))
            self.device_tables[key] = table
        return table

    def _gamma_row(self, sigma, device: torch.device) -> torch.Tensor:
        """``np.interp(sigma, sigmas, selected_gamma[:, bin])`` of every bin, on
        ``device``; ``sigma`` may be a device tensor (no sync)."""

        sigmas, gamma = self._device_table(device)
        sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32).reshape(1).clamp(sigmas[0], sigmas[-1])
        upper = torch.searchsorted(sigmas, sigma, right=True).clamp(1, len(sigmas) - 1)
        lower = upper - 1
        span = sigmas[upper] - sigmas[lower]
        weight = torch.where(span > 0, (sigma - sigmas[lower]) / span, torch.zeros_like(span))
        return (gamma[lower] + weight[:, None] * (gamma[upper] - gamma[lower]))[0]

    @torch.no_grad()
    def recolor(self, white: torch.Tensor, sigma) -> torch.Tensor:
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

        scale = (1.0 - self._gamma_row(sigma, white.device)).clamp(0.0, 1.0).sqrt()
        spectrum = torch.fft.fft2(white.float(), dim=(-2, -1)) * scale[bin_map]
        colored = torch.fft.ifft2(spectrum, dim=(-2, -1)).real
        colored = colored / colored.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        if self.strength < 1.0:
            colored = (1.0 - self.strength) * white.float() + self.strength * colored
            colored = colored / colored.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return colored.to(dtype=white.dtype)


def _record_cns_fallback(process, reason: str) -> None:
    params = getattr(process, "extra_generation_params", None)
    if isinstance(params, dict):
        params["Anima CNS"] = f"white-noise fallback: {reason}"


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
    """Forge's ER-SDE with its per-step white-noise draw CNS-recolored.

    The solver is Forge's own ``sample_er_sde``; only the noise sampler it
    calls once per step (``noise_sampler(sigma, sigma_next)``) is wrapped.
    """

    from k_diffusion.sampling import default_noise_sampler, sample_er_sde

    require_anima_flow_denoiser(model, "Anima ER SDE CNS")
    recolorer = (
        CNSRecolorer.try_calibrated(cns_strength, getattr(model, "p", None))
        if cns_strength > 0.0
        else None
    )
    if recolorer is not None:
        white_noise = default_noise_sampler(x) if noise_sampler is None else noise_sampler

        def noise_sampler(sigma, sigma_next):
            return recolorer.recolor(white_noise(sigma, sigma_next), sigma)

    return sample_er_sde(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        noise_scaler=noise_scaler,
        max_stage=max_stage,
    )
