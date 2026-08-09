"""Small dependency-free helpers shared by built-in Anima LoRA controls."""

from __future__ import annotations


def active_template_parts(auto_timing: bool, parts):
    """Manual timing owns the single GUI window and must ignore preset parts."""

    return list(parts) if auto_timing else []


def current_sampling_position(params) -> tuple[int, int]:
    """Resolve Forge's current model-call step instead of its callback lag."""

    denoiser = getattr(params, "denoiser", None)
    callback_step = int(getattr(params, "sampling_step", 0))
    callback_steps = int(getattr(params, "total_sampling_steps", 1))
    denoiser_step = int(getattr(denoiser, "step", callback_step))
    denoiser_steps = int(getattr(denoiser, "total_steps", callback_steps))
    return max(callback_step, denoiser_step), max(
        1, callback_steps, denoiser_steps
    )
