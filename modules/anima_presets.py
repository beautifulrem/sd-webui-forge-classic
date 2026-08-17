"""Shared GUI preset registry for Remi's built-in Anima features.

The preset intentionally owns generation algorithms and their numeric tuning,
but not user-authored content such as prompts, LoRA names, or local model paths.
Extensions register their Gradio controls while the UI is being built; the
late-running preset panel can then update them in one click without importing
extension modules or coupling their runtimes together.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


SAFE_BASE_AESTHETIC_LABEL = "Official Base/Aesthetic — Safe (Recommended)"

SAFE_BASE_AESTHETIC_PRESET: dict[str, object] = {
    # Native Forge generation controls.
    "native.sampler": "ER SDE",
    "native.scheduler": "Automatic",
    "native.steps": 36,
    "native.cfg": 4.5,
    "native.distilled_cfg": 3.0,
    "native.hires": False,
    "native.refiner": False,
    # Guidance and corrective samplers. Optional algorithms remain disabled,
    # while their inactive fields are reset for reproducible later opt-in.
    "guidance.enabled": False,
    "guidance.mode": "Standard / preserve existing",
    "guidance.smc_alpha": 0.2,
    "guidance.smc_lambda": 5.0,
    "guidance.skim_enabled": False,
    "guidance.skim_scale": 2.5,
    "guidance.skim_full_negative": False,
    "guidance.skim_disable_flip": False,
    "guidance.skim_start": 0.0,
    "guidance.skim_end": 1.0,
    "guidance.skim_flip": 0.0,
    "guidance.dcw_enabled": False,
    "guidance.dcw_lambda": -0.015,
    "guidance.dcw_schedule": "one_minus_sigma",
    "guidance.dcw_bands": "LL",
    "guidance.cns_strength": 1.0,
    "guidance.unipc_solver": "bh2",
    "guidance.unipc_disabled_correctors": 0,
    "guidance.unipc_thresholding": False,
    "guidance.unipc_threshold_ratio": 0.995,
    "guidance.unipc_threshold_max": 1.0,
    "guidance.pc3_gamma": 1.0,
    "guidance.pc3_tolerance": 0.005,
    "guidance.modulation_enabled": False,
    "guidance.modulation_weight": 3.0,
    "guidance.modulation_start": 0,
    "guidance.modulation_end": -1,
    "guidance.modulation_adapter_mode": "Auto-download pinned",
    "guidance.modulation_clip_mode": "Auto-download pinned",
    "guidance.nag_enabled": False,
    "guidance.nag_scale": 2.0,
    "guidance.nag_tau": 2.5,
    "guidance.nag_alpha": 0.5,
    "guidance.nag_start": 0.0,
    "guidance.nag_end": 0.5,
    "guidance.momentum_enabled": False,
    "guidance.momentum_strength": 0.5,
    "guidance.momentum_ema": 0.6,
    "guidance.fdg_detail": 2.0,
    "guidance.range_enabled": False,
    "guidance.range_start": 0.0,
    "guidance.range_end": 1.0,
    # Scheduler and cache acceleration.
    "dynamic_shift.enabled": False,
    "dynamic_shift.start": 5.0,
    "dynamic_shift.end": 1.5,
    "dynamic_shift.curve": "Cosine",
    "dynamic_shift.hires_mode": "Same as first pass",
    "dynamic_shift.hires_start": 3.0,
    "dynamic_shift.hires_end": 1.5,
    "spectrum.enabled": False,
    "spectrum.weight": 0.25,
    "spectrum.degree": 4,
    "spectrum.regularization": 0.1,
    "spectrum.window": 2,
    "spectrum.window_growth": 0.0,
    "spectrum.warmup": 6,
    "spectrum.stop": 0.9,
    "spectrum.tail_actual": 3,
    "spectrum.history": 10,
    "spectrum.schedule": "Window",
    "spectrum.refresh_ratio": 0.0,
    "spectrum.sea_beta": 2.0,
    "spectrum.policy": "Conservative",
    "spectrum.verbose": False,
    "compile.preset": "Automatic",
    # Spatial and post-processing stacks.
    "regional.enabled": False,
    "regional.blocks": "0-39",
    "regional.start": 0.0,
    "regional.end": 0.65,
    "regional.feather": 0.03,
    "regional.base_preserve": 0.15,
    "regional.region1_active": True,
    "regional.region1_x": 0.0,
    "regional.region1_y": 0.0,
    "regional.region1_width": 0.5,
    "regional.region1_height": 1.0,
    "regional.region1_strength": 1.0,
    "regional.region2_active": True,
    "regional.region2_x": 0.5,
    "regional.region2_y": 0.0,
    "regional.region2_width": 0.5,
    "regional.region2_height": 1.0,
    "regional.region2_strength": 1.0,
    "regional.region3_active": False,
    "regional.region3_x": 0.25,
    "regional.region3_y": 0.25,
    "regional.region3_width": 0.5,
    "regional.region3_height": 0.5,
    "regional.region3_strength": 1.0,
    "artist.enabled": False,
    "artist.global_strength": 0.7,
    "artist.optimization": "Balance",
    "artist.combine": "Output average",
    "artist.fusion": "Interpolate",
    "artist.cache": True,
    "artist.apply_uncond": False,
    "freefuse.enabled": False,
    "freefuse.collect_step": 3,
    "freefuse.collect_block": 18,
    "freefuse.top_k": 0.10,
    "freefuse.temperature": 300,
    "freefuse.background_scale": 0.95,
    "freefuse.balance_iterations": 15,
    "freefuse.feather": 1,
    "freefuse.routing_strength": 1.0,
    "freefuse.routing_end": 1.0,
    "freefuse.bias_scale": 6.0,
    "freefuse.positive_bias": 1.0,
    "freefuse.bias_blocks": "0-39",
    "pid.enabled": False,
    "pid.degrade_sigma": 0.0,
    "pid.color_correction": True,
    "hires_guard.enabled": False,
    "hires_guard.policy": "Clamp unsafe values",
    "hires_guard.preset": "Balanced",
    "hires_guard.align": True,
    "hires_guard.base_mp": 1.2,
    "hires_guard.hires_mp": 2.1,
    "hires_guard.max_upscale": 1.75,
    # Auxiliary opt-in helpers.
    "resolution.random_standard": False,
    "resolution.random_highres": False,
    "prompt_rescale.enabled": False,
    "prompt_rescale.source_steps": 35,
    "prompt_rescale.fractions": False,
    "lora_stage.enabled": False,
    "lora_layer.lora_enabled": False,
    "lora_layer.lokr_enabled": False,
}

# Only these controls are ever locked by Remi's mutual-exclusion GUI. The
# preset unlocks them after moving the whole set back to a compatible state;
# all other components keep their extension-defined interactivity.
PRESET_CONFLICT_CONTROLS = frozenset(
    {
        "native.sampler",
        "native.scheduler",
        "native.hires",
        "native.refiner",
        "guidance.mode",
        "guidance.momentum_enabled",
        "guidance.dcw_enabled",
        "dynamic_shift.enabled",
        "regional.enabled",
        "artist.enabled",
        "freefuse.enabled",
        "pid.enabled",
    }
)


_DEFAULT_VALUE = object()
_CONTROLS: dict[str, OrderedDict[str, tuple[Any, object]]] = {}


def reset_preset_controls(tab: str) -> None:
    """Start a fresh registry for a newly built txt2img/img2img tab."""

    _CONTROLS[tab] = OrderedDict()


def register_preset_control(
    tab: str,
    name: str,
    component: Any,
    value: object = _DEFAULT_VALUE,
) -> None:
    """Register one preset-owned component, replacing stale UI references."""

    if name not in SAFE_BASE_AESTHETIC_PRESET:
        raise KeyError(f"Unknown Anima preset field: {name}")
    if value is _DEFAULT_VALUE:
        value = SAFE_BASE_AESTHETIC_PRESET[name]
    _CONTROLS.setdefault(tab, OrderedDict())[name] = (component, value)


def preset_controls(tab: str) -> tuple[tuple[str, ...], tuple[Any, ...], tuple[object, ...]]:
    """Return a stable component/value snapshot suitable for a Gradio event."""

    controls = _CONTROLS.get(tab, OrderedDict())
    names = tuple(controls)
    return (
        names,
        tuple(component for component, _value in controls.values()),
        tuple(value for _component, value in controls.values()),
    )
