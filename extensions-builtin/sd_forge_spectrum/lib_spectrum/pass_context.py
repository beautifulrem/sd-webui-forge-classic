"""Resolve the parameters that actually drive the current Forge pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingPassContext:
    steps: int
    sampler: str
    cfg: float
    is_hires: bool


def _setup_img2img_steps(process):
    from modules.sd_samplers_common import setup_img2img_steps

    return setup_img2img_steps(process)


def resolve_pass_context(process, img2img_steps=_setup_img2img_steps) -> SamplingPassContext:
    """``img2img_steps`` is ``sd_samplers_common.setup_img2img_steps``; the
    img2img sampler runs ``t_enc + 1`` steps of the schedule it returns."""

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
            _, t_enc = img2img_steps(process)
            steps = int(t_enc) + 1
        sampler = getattr(process, "sampler_name", "unknown")
        cfg = float(getattr(process, "cfg_scale", 0.0))
    return SamplingPassContext(
        steps=max(1, steps),
        sampler=str(sampler),
        cfg=cfg,
        is_hires=is_hires,
    )
