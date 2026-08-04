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

import sys
from pathlib import Path

import gradio as gr

from modules import script_callbacks, scripts
from modules.ui_components import InputAccordion

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib_dynshift.schedule import CURVES, compute_sigmas, preview_svg  # noqa: E402

NAME = "Dynamic Shift"

# infotext keys
K_ENABLED = "DynShift enabled"
K_START = "DynShift start"
K_END = "DynShift end"
K_CURVE = "DynShift curve"
K_HR_MODE = "DynShift hires mode"
K_HR_START = "DynShift hires start"
K_HR_END = "DynShift hires end"

HR_MODES = ["Same as first pass", "Custom", "Static (built-in shift)"]


class DynamicShiftScript(scripts.Script):
    section = "sampler"
    create_group = False
    sorting_priority = 5

    def title(self):
        return NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # InputAccordion makes the native accordion header itself the enable
        # control, matching Forge's Hires Fix / Refiner UI pattern.
        with InputAccordion(False, label=NAME) as enabled:
            with gr.Row():
                curve = gr.Dropdown(label="Curve", choices=CURVES, value="Cosine")
            with gr.Row():
                shift_start = gr.Slider(
                    label="Shift @ first step", minimum=0.1, maximum=12.0, step=0.05, value=5.0,
                )
                shift_end = gr.Slider(
                    label="Shift @ last step", minimum=0.1, maximum=12.0, step=0.05, value=1.5,
                )
            preview = gr.HTML(value=preview_svg(30, 5.0, 1.5, "Cosine"))

            if not is_img2img:
                with gr.Row():
                    hr_mode = gr.Dropdown(label="Hires fix", choices=HR_MODES, value=HR_MODES[0])
                with gr.Row():
                    hr_start = gr.Slider(
                        label="Hires shift @ first step", minimum=0.1, maximum=12.0, step=0.05, value=3.0, visible=False,
                    )
                    hr_end = gr.Slider(
                        label="Hires shift @ last step", minimum=0.1, maximum=12.0, step=0.05, value=1.5, visible=False,
                    )

                def _toggle_hr(mode):
                    vis = mode == "Custom"
                    return gr.update(visible=vis), gr.update(visible=vis)

                hr_mode.change(_toggle_hr, inputs=[hr_mode], outputs=[hr_start, hr_end], show_progress=False)
            else:
                hr_mode = gr.Dropdown(value=HR_MODES[0], visible=False, choices=HR_MODES)
                hr_start = gr.Slider(value=3.0, visible=False)
                hr_end = gr.Slider(value=1.5, visible=False)

            def _update_preview(s0, s1, c):
                return preview_svg(30, s0, s1, c)

            for comp in (shift_start, shift_end, curve):
                comp.change(_update_preview, inputs=[shift_start, shift_end, curve], outputs=[preview], show_progress=False)

        self.infotext_fields = [
            (enabled, lambda d: str(d.get(K_ENABLED, False)) == "True"),
            (shift_start, K_START),
            (shift_end, K_END),
            (curve, K_CURVE),
            (hr_mode, K_HR_MODE),
            (hr_start, K_HR_START),
            (hr_end, K_HR_END),
        ]
        self.paste_field_names = [K_ENABLED, K_START, K_END, K_CURVE, K_HR_MODE, K_HR_START, K_HR_END]

        return [enabled, shift_start, shift_end, curve, hr_mode, hr_start, hr_end]

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
            print(f"[{NAME}] current model is not a flow/shift model; skipping.")
            return

        if p.sampler_noise_scheduler_override is not None:
            print(f"[{NAME}] another script already overrides the noise scheduler; skipping.")
            return

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
            print(f"[{NAME}] {steps} steps | shift {s0:g} -> {s1:g} ({curve})")
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
