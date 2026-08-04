"""Deterministic, dependency-free dimension guards for Anima generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt


@dataclass(frozen=True)
class GuardLimits:
    base_megapixels: float
    hires_megapixels: float
    max_upscale: float
    alignment: int = 16
    minimum_side: int = 64

    def validated(self) -> "GuardLimits":
        base_megapixels = max(0.1, float(self.base_megapixels))
        return GuardLimits(
            base_megapixels=base_megapixels,
            # A Hires pass cannot honor a ceiling below its base image without
            # becoming a downscale, so treat the base ceiling as the floor.
            hires_megapixels=max(base_megapixels, float(self.hires_megapixels)),
            max_upscale=max(1.0, float(self.max_upscale)),
            alignment=max(1, int(self.alignment)),
            minimum_side=max(1, int(self.minimum_side)),
        )


@dataclass(frozen=True)
class GuardResult:
    base_width: int
    base_height: int
    hires_width: int | None
    hires_height: int | None
    changes: tuple[str, ...]


def _aligned_nearest(value: float, multiple: int, minimum: int) -> int:
    units = max(1, int(floor(float(value) / multiple + 0.5)))
    return max(minimum, units * multiple)


def _aligned_floor(value: float, multiple: int, minimum: int) -> int:
    units = max(1, int(floor(float(value) / multiple)))
    return max(minimum, units * multiple)


def _fit(
    width: float,
    height: float,
    *,
    max_pixels: float,
    alignment: int,
    minimum_side: int,
    max_width: float | None = None,
    max_height: float | None = None,
) -> tuple[int, int]:
    """Preserve aspect while fitting all supplied upper bounds."""

    width = max(float(minimum_side), float(width))
    height = max(float(minimum_side), float(height))
    scale = 1.0
    if width * height > max_pixels:
        scale = min(scale, sqrt(max_pixels / (width * height)))
    if max_width is not None and width > max_width:
        scale = min(scale, max_width / width)
    if max_height is not None and height > max_height:
        scale = min(scale, max_height / height)

    if scale < 1.0:
        # Floor after a clamp: rounding upward must never cross a declared cap.
        return (
            _aligned_floor(width * scale, alignment, minimum_side),
            _aligned_floor(height * scale, alignment, minimum_side),
        )
    return (
        _aligned_nearest(width, alignment, minimum_side),
        _aligned_nearest(height, alignment, minimum_side),
    )


def guard_dimensions(
    width: int,
    height: int,
    *,
    enable_hr: bool,
    hr_scale: float,
    hr_resize_x: int,
    hr_resize_y: int,
    limits: GuardLimits,
) -> GuardResult:
    """Return safe base/Hires dimensions without mutating a processing object.

    An explicit Hires target follows Forge's one-axis inference rules. A scale
    target is derived from the guarded base. Both passes retain their aspect
    ratio while respecting pixel and per-axis upscale caps.
    """

    limits = limits.validated()
    align = limits.alignment
    minimum = max(limits.minimum_side, align)
    original_base = (int(width), int(height))
    base = _fit(
        *original_base,
        max_pixels=limits.base_megapixels * 1_000_000,
        alignment=align,
        minimum_side=minimum,
    )
    changes: list[str] = []
    if base != original_base:
        changes.append(f"base {original_base[0]}x{original_base[1]} -> {base[0]}x{base[1]}")

    if not enable_hr:
        return GuardResult(*base, None, None, tuple(changes))

    resize_x = max(0, int(hr_resize_x or 0))
    resize_y = max(0, int(hr_resize_y or 0))
    if resize_x == 0 and resize_y == 0:
        requested = (base[0] * max(1.0, float(hr_scale)), base[1] * max(1.0, float(hr_scale)))
    elif resize_y == 0:
        requested = (resize_x, resize_x * (base[1] / base[0]))
    elif resize_x == 0:
        requested = (resize_y * (base[0] / base[1]), resize_y)
    else:
        requested = (resize_x, resize_y)

    requested_int = (
        _aligned_nearest(requested[0], align, minimum),
        _aligned_nearest(requested[1], align, minimum),
    )
    hires = _fit(
        *requested,
        max_pixels=limits.hires_megapixels * 1_000_000,
        max_width=base[0] * limits.max_upscale,
        max_height=base[1] * limits.max_upscale,
        alignment=align,
        minimum_side=minimum,
    )

    # Hires fix should never accidentally become a downscale after clamping.
    hires = (max(base[0], hires[0]), max(base[1], hires[1]))
    if hires != requested_int:
        changes.append(
            f"hires {requested_int[0]}x{requested_int[1]} -> {hires[0]}x{hires[1]}"
        )
    elif (resize_x, resize_y) != (0, 0) and hires != (resize_x, resize_y):
        changes.append(
            f"hires aligned {resize_x}x{resize_y} -> {hires[0]}x{hires[1]}"
        )

    return GuardResult(*base, *hires, tuple(changes))
