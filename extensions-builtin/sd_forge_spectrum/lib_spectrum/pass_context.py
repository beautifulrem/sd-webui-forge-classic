"""Resolve the parameters that actually drive the current Forge pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingPassContext:
    steps: int
    sampler: str
    cfg: float
    is_hires: bool


def _img2img_fix_steps() -> bool:
    try:
        from modules.shared import opts
    except Exception:
        return False
    return bool(getattr(opts, "img2img_fix_steps", False))


def _img2img_actual_steps(process, requested: int, fix_steps: bool) -> int:
    """Mirror ``sd_samplers_common.setup_img2img_steps``: the sampler runs
    ``t_enc + 1`` steps of a schedule sized for ``requested`` or more steps."""

    if fix_steps:
        return requested
    denoise = float(getattr(process, "denoising_strength", 1.0) or 0.0)
    return int(min(max(denoise, 0.0), 0.999) * requested) + 1


def resolve_pass_context(process, fix_steps: bool | None = None) -> SamplingPassContext:
    is_hires = bool(getattr(process, "is_hr_pass", False))
    base_steps = max(1, int(getattr(process, "steps", 1)))
    if is_hires:
        steps = int(getattr(process, "hr_second_pass_steps", 0) or base_steps)
        sampler = getattr(process, "hr_sampler_name", None) or getattr(
            process, "sampler_name", "unknown"
        )
        cfg = float(getattr(process, "hr_cfg", getattr(process, "cfg_scale", 0.0)))
    else:
        steps = base_steps
        if getattr(process, "init_images", None) is not None:
            if fix_steps is None:
                fix_steps = _img2img_fix_steps()
            steps = _img2img_actual_steps(process, base_steps, fix_steps)
        sampler = getattr(process, "sampler_name", "unknown")
        cfg = float(getattr(process, "cfg_scale", 0.0))
    return SamplingPassContext(
        steps=max(1, steps),
        sampler=str(sampler),
        cfg=cfg,
        is_hires=is_hires,
    )
