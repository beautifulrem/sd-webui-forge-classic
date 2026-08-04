"""DCW post-step SNR-t bias correction for Anima.

Numerics follow sorryhyun/ComfyUI-Spectrum-KSampler's manual DCW path.  Forge's
pre-denoiser callback supplies the same mutable sampler latent used by the
subsequent model evaluation, while a post-CFG hook captures the preceding
denoised prediction.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch


LOGGER = logging.getLogger("anima_guidance.dcw")
BANDS = ("LL", "LH", "HL", "HH")
ALL_BANDS = frozenset(BANDS)


def parse_band_mask(label: str) -> frozenset[str]:
    if label.lower() == "all":
        return ALL_BANDS
    parts = [item.strip().upper() for item in label.split("+") if item.strip()]
    invalid = [item for item in parts if item not in BANDS]
    if invalid or not parts:
        raise ValueError(f"invalid DCW band mask {label!r}; expected LL/LH/HL/HH or all")
    return frozenset(parts)


def haar_dwt_2d(value: torch.Tensor):
    a = value[..., 0::2, 0::2]
    b = value[..., 0::2, 1::2]
    c = value[..., 1::2, 0::2]
    d = value[..., 1::2, 1::2]
    scale = 0.5
    return (
        (a + b + c + d) * scale,
        (a + b - c - d) * scale,
        (a - b + c - d) * scale,
        (a - b - c + d) * scale,
    )


def haar_idwt_2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    scale = 0.5
    a = (ll + lh + hl + hh) * scale
    b = (ll + lh - hl - hh) * scale
    c = (ll - lh + hl - hh) * scale
    d = (ll - lh - hl + hh) * scale
    result = torch.empty(
        *ll.shape[:-2],
        ll.shape[-2] * 2,
        ll.shape[-1] * 2,
        device=ll.device,
        dtype=ll.dtype,
    )
    result[..., 0::2, 0::2] = a
    result[..., 0::2, 1::2] = b
    result[..., 1::2, 0::2] = c
    result[..., 1::2, 1::2] = d
    return result


def band_masked_diff(diff: torch.Tensor, bands: frozenset[str]) -> torch.Tensor:
    ll, lh, hl, hh = haar_dwt_2d(diff)
    zero = torch.zeros_like(ll)
    return haar_idwt_2d(
        ll if "LL" in bands else zero,
        lh if "LH" in bands else zero,
        hl if "HL" in bands else zero,
        hh if "HH" in bands else zero,
    )


class DCWState:
    """Per-sampling-run state for manual DCW correction."""

    def __init__(
        self,
        lam: float = -0.015,
        schedule: str = "one_minus_sigma",
        bands: frozenset[str] = frozenset({"LL"}),
    ):
        self.lam = float(lam)
        self.schedule = str(schedule)
        self.bands = frozenset(bands)
        self.last_denoised: Optional[torch.Tensor] = None
        self.current_sigma: Optional[float] = None
        self._odd_shape_warned = False

    def schedule_value(self, sigma: Optional[float]) -> float:
        if sigma is None:
            return 0.0
        if self.schedule == "one_minus_sigma":
            return 1.0 - float(sigma)
        if self.schedule == "sigma":
            return float(sigma)
        if self.schedule == "constant":
            return 1.0
        raise ValueError(f"unknown DCW schedule: {self.schedule}")

    @torch.no_grad()
    def apply_correction(self, x_in: torch.Tensor, scalar: float) -> None:
        if self.last_denoised is None or scalar == 0.0:
            return
        if self.last_denoised.shape != x_in.shape:
            self.last_denoised = None
            return
        diff = x_in - self.last_denoised
        if self.bands == ALL_BANDS:
            x_in.add_(diff, alpha=float(scalar))
            return
        height, width = x_in.shape[-2:]
        if height % 2 or width % 2:
            if not self._odd_shape_warned:
                LOGGER.warning(
                    "DCW band mask needs even latent dimensions; using broadband for %sx%s",
                    height,
                    width,
                )
                self._odd_shape_warned = True
            x_in.add_(diff, alpha=float(scalar))
            return
        masked = band_masked_diff(diff.float(), self.bands).to(dtype=x_in.dtype)
        x_in.add_(masked, alpha=float(scalar))

    @torch.no_grad()
    def before_denoiser(self, x_in: torch.Tensor, sigma: float) -> None:
        sigma = float(sigma)
        if self.current_sigma is None:
            self.current_sigma = sigma
            return
        if abs(sigma - self.current_sigma) <= 1e-8:
            return
        scalar = self.lam * self.schedule_value(self.current_sigma)
        self.apply_correction(x_in, scalar)
        self.current_sigma = sigma

    @torch.no_grad()
    def capture_denoised(self, denoised: torch.Tensor) -> torch.Tensor:
        self.last_denoised = denoised.detach().clone()
        return denoised
