"""
Dynamic Shift Scheduler for sd-webui-forge-classic (neo)

Ramps the flow-matching "shift" value across sampling steps instead of using
one constant value, via p.sampler_noise_scheduler_override. High shift early
biases steps toward the high-noise / composition regime; low shift late gives
more steps to the detail regime (or the reverse, if you prefer).

Only activates for flow-matching models (sd_model.use_shift == True, e.g.
Anima, Lumina, Qwen-Image...). While enabled, the Schedule type dropdown and
the built-in Shift slider are bypassed for the sigma schedule of the affected
passes.
"""

import logging
import sys
from pathlib import Path

from backend.logging import setup_logger
from modules import script_callbacks, scripts
from modules.anima_feature_conflicts import FLOW_SAMPLERS, resolve_schedule_conflicts
from remi import ui

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib_dynshift.schedule import CURVES, compute_sigmas  # noqa: E402

NAME = "Dynamic Shift"
logger = logging.getLogger("DynamicShift")
setup_logger(logger)

# infotext keys
K_ENABLED = "DynShift enabled"
K_START = "DynShift start"
K_END = "DynShift end"
K_CURVE = "DynShift curve"
K_HR_MODE = "DynShift hires mode"
K_HR_START = "DynShift hires start"
K_HR_END = "DynShift hires end"

HR_MODES = ["Same as first pass", "Custom", "Static (built-in shift)"]

# The Sampler script's params (owner "sampler") the panel reads
SAMPLER_PARAM = "sampler.sampler_name"
STEPS_PARAM = "sampler.steps"

NOTE_FLOW_SAMPLER = "**Compatibility:** this sampler locks its scheduler, so Dynamic Shift is unavailable."
NOTE_SCHEDULE = "**Compatibility:** Dynamic Shift controls the schedule; Schedule Type stays Automatic while it is on."


def _shift(name, default, label, infotext, preset, *, hidden=False):
    return ui.Slider(name, 0.1, 12.0, default, 0.05, label=label, infotext=infotext, preset=preset, hidden=hidden)


def _curve(value):
    # An unknown curve (old preset, infotext, API value) runs as "Linear",
    # as process() always treated it; the form then shows what will run
    return value if value in CURVES else "Linear"


class DynamicShiftScript(scripts.Script):
    section = "sampler"
    sorting_priority = 5

    def title(self):
        return NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        # Sampler / scheduler compatibility is resolved by the core conflict
        # locks (the enable switch is the "dynamic_shift.enabled" role); the
        # notes explain the lock the user sees.
        flow_sampler = ui.param(SAMPLER_PARAM, in_=sorted(FLOW_SAMPLERS))
        # img2img has no Hires. fix: its hires params keep their defaults, unrendered
        hidden = is_img2img
        return ui.Section(
            NAME,
            enable=ui.Toggle(
                "enabled",
                False,
                label=NAME,
                paste=lambda d: str(d.get(K_ENABLED, False)) == "True",
                preset="dynamic_shift.enabled",
            ),
            children=[
                ui.Markdown(NOTE_FLOW_SAMPLER, visible_if=flow_sampler),
                ui.Markdown(NOTE_SCHEDULE, visible_if=ui.all_of(ui.param("enabled"), ui.not_(flow_sampler))),
                ui.Row(
                    _shift("shift_start", 5.0, "Shift @ first step", K_START, "dynamic_shift.start"),
                    _shift("shift_end", 1.5, "Shift @ last step", K_END, "dynamic_shift.end"),
                ),
                ui.Custom(
                    "curve",
                    "shift-curve",
                    "Cosine",
                    props={"choices": CURVES, "start": "shift_start", "end": "shift_end", "steps": STEPS_PARAM},
                    label="Curve",
                    infotext=K_CURVE,
                    preset="dynamic_shift.curve",
                    validate=_curve,
                ),
                ui.Choice("hr_mode", HR_MODES, HR_MODES[0], label="Hires fix", infotext=K_HR_MODE, preset="dynamic_shift.hires_mode", hidden=hidden),
                ui.Row(
                    _shift("hr_start", 3.0, "Hires shift @ first step", K_HR_START, "dynamic_shift.hires_start", hidden=hidden),
                    _shift("hr_end", 1.5, "Hires shift @ last step", K_HR_END, "dynamic_shift.hires_end", hidden=hidden),
                    visible_if=None if hidden else ui.param("hr_mode", eq="Custom"),
                ),
            ],
        )

    def process(self, p, enabled, shift_start, shift_end, curve, hr_mode, hr_start, hr_end):
        # XYZ grid (or API) overrides, set as attributes on p
        shift_start = getattr(p, "dynshift_start", shift_start)
        shift_end = getattr(p, "dynshift_end", shift_end)
        curve = getattr(p, "dynshift_curve", curve)
        if any(hasattr(p, a) for a in ("dynshift_start", "dynshift_end", "dynshift_curve")):
            enabled = True

        if not enabled:
            return

        if not getattr(p.sd_model, "use_shift", False):
            logger.warning("Current model is not a flow/shift model; Dynamic Shift skipped")
            return

        conflict_state = resolve_schedule_conflicts(
            sampler=str(getattr(p, "sampler_name", "")),
            scheduler=str(getattr(p, "scheduler", "Automatic")),
            dynamic_shift=True,
        )
        if not conflict_state.dynamic_shift:
            reason = "; ".join(conflict_state.messages) or "sampler locks its scheduler"
            p.extra_generation_params[K_ENABLED] = False
            p.extra_generation_params["DynShift disabled"] = reason
            logger.warning("%s; Dynamic Shift skipped", reason)
            return
        # Check ownership before touching the scheduler so a skipped run keeps the user's choice.
        if p.sampler_noise_scheduler_override is not None:
            logger.warning("Another script already overrides the noise scheduler; Dynamic Shift skipped")
            return

        if conflict_state.scheduler != str(getattr(p, "scheduler", "Automatic")):
            p.extra_generation_params["DynShift replaced scheduler"] = getattr(
                p, "scheduler", "Automatic"
            )
            p.scheduler = conflict_state.scheduler

        if curve not in CURVES:
            curve = "Linear"

        def override(steps):
            import torch

            if getattr(p, "is_hr_pass", False):
                if hr_mode == "Custom":
                    s0, s1 = hr_start, hr_end
                elif hr_mode == "Static (built-in shift)":
                    # constant-shift schedule using the model's current (hires)
                    # shift value, i.e. what FlowMatchEulerDiscrete would do
                    s = float(getattr(p.sd_model.forge_objects.unet.model.predictor, "shift", 1.0))
                    s0 = s1 = s
                else:
                    s0, s1 = shift_start, shift_end
            else:
                s0, s1 = shift_start, shift_end

            sigmas = compute_sigmas(steps, s0, s1, curve)
            logger.info("%d steps | shift %g -> %g (%s)", steps, s0, s1, curve)
            return torch.tensor(sigmas, dtype=torch.float32)

        p.sampler_noise_scheduler_override = override

        p.extra_generation_params[K_ENABLED] = True
        p.extra_generation_params[K_START] = shift_start
        p.extra_generation_params[K_END] = shift_end
        p.extra_generation_params[K_CURVE] = curve
        if getattr(p, "enable_hr", False):
            p.extra_generation_params[K_HR_MODE] = hr_mode
            if hr_mode == "Custom":
                p.extra_generation_params[K_HR_START] = hr_start
                p.extra_generation_params[K_HR_END] = hr_end


# ---------------------------------------------------------------------------
# XYZ grid integration


def _set_attr(field):
    def apply(p, x, xs):
        setattr(p, field, x)

    return apply


def _register_xyz():
    xyz = None
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid", "xyz_grid"):
            xyz = data.module
            break
    if xyz is None:
        return

    existing = [x.label for x in xyz.axis_options]
    for label, opt in [
        ("[DynShift] Start", xyz.AxisOption("[DynShift] Start", float, _set_attr("dynshift_start"))),
        ("[DynShift] End", xyz.AxisOption("[DynShift] End", float, _set_attr("dynshift_end"))),
        ("[DynShift] Curve", xyz.AxisOption("[DynShift] Curve", str, _set_attr("dynshift_curve"), choices=lambda: CURVES)),
    ]:
        if label not in existing:
            xyz.axis_options.append(opt)


script_callbacks.on_before_ui(_register_xyz)
