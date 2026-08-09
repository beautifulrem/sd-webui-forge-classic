"""Resolve the parameters that actually drive the current Forge pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingPassContext:
    steps: int
    sampler: str
    cfg: float
    is_hires: bool


def resolve_pass_context(process) -> SamplingPassContext:
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
        sampler = getattr(process, "sampler_name", "unknown")
        cfg = float(getattr(process, "cfg_scale", 0.0))
    return SamplingPassContext(
        steps=max(1, steps),
        sampler=str(sampler),
        cfg=cfg,
        is_hires=is_hires,
    )
