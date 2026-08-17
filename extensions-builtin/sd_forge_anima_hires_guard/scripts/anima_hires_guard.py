"""Anima-only Hires safety limits exposed as native Forge Neo controls."""

from __future__ import annotations

import logging

import gradio as gr

from lib_anima_hires import GuardLimits, guard_dimensions
from modules import scripts
from modules.anima_support import is_anima_engine
from modules.anima_presets import register_preset_control
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaHiresGuard")

PRESETS = {
    "Low VRAM": GuardLimits(0.90, 1.50, 1.50),
    "Balanced": GuardLimits(1.20, 2.10, 1.75),
    "Detail": GuardLimits(1.40, 2.36, 2.00),
}


def _is_anima(process) -> bool:
    return is_anima_engine(getattr(process, "sd_model", None))


class AnimaHiresGuardScript(scripts.Script):
    # Run after the bundled resolution randomizer (priority 0), so the guard
    # validates the dimensions that will actually enter processing.
    sorting_priority = 2100

    def title(self):
        return "Anima Hires Guard"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def ui(self, is_img2img):
        if is_img2img:
            return []

        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown(
                "Anima only. Validates the base pass and Hires target after "
                "resolution randomization. **Report only** never changes dimensions."
            )
            with gr.Row():
                policy = gr.Dropdown(
                    ["Clamp unsafe values", "Report only"],
                    value="Clamp unsafe values",
                    label="Policy",
                )
                preset = gr.Dropdown(
                    ["Balanced", "Low VRAM", "Detail", "Custom"],
                    value="Balanced",
                    label="Safety preset",
                    info="Custom uses the three sliders below.",
                )
                align = gr.Checkbox(
                    True,
                    label="Align to 16 px",
                    info="Matches Anima's 8x VAE reduction and 2x transformer patches.",
                )
            with gr.Row():
                base_mp = gr.Slider(
                    0.4,
                    3.0,
                    value=1.2,
                    step=0.05,
                    label="Base maximum (MP)",
                )
                hires_mp = gr.Slider(
                    0.8,
                    6.0,
                    value=2.1,
                    step=0.05,
                    label="Hires maximum (MP)",
                )
                max_upscale = gr.Slider(
                    1.0,
                    3.0,
                    value=1.75,
                    step=0.05,
                    label="Maximum upscale per axis",
                )

        self.infotext_fields = [
            (policy, "Anima Hires policy"),
            (preset, "Anima Hires preset"),
            (align, "Anima Hires align"),
            (base_mp, "Anima base MP"),
            (hires_mp, "Anima Hires MP"),
            (max_upscale, "Anima Hires max upscale"),
        ]
        for name, component in {
            "hires_guard.enabled": enabled,
            "hires_guard.policy": policy,
            "hires_guard.preset": preset,
            "hires_guard.align": align,
            "hires_guard.base_mp": base_mp,
            "hires_guard.hires_mp": hires_mp,
            "hires_guard.max_upscale": max_upscale,
        }.items():
            register_preset_control(self.tabname, name, component)
        return [enabled, policy, preset, align, base_mp, hires_mp, max_upscale]

    def after_model_load(
        self,
        process,
        enabled,
        policy,
        preset,
        align,
        base_mp,
        hires_mp,
        max_upscale,
        **kwargs,
    ):
        if not enabled or not _is_anima(process):
            return

        limits = PRESETS.get(str(preset))
        if limits is None:
            limits = GuardLimits(base_mp, hires_mp, max_upscale)
        if not align:
            limits = GuardLimits(
                limits.base_megapixels,
                limits.hires_megapixels,
                limits.max_upscale,
                alignment=1,
            )
        limits = limits.validated()

        result = guard_dimensions(
            process.width,
            process.height,
            enable_hr=bool(getattr(process, "enable_hr", False)),
            hr_scale=float(getattr(process, "hr_scale", 1.0)),
            hr_resize_x=int(getattr(process, "hr_resize_x", 0) or 0),
            hr_resize_y=int(getattr(process, "hr_resize_y", 0) or 0),
            limits=limits,
        )
        process.extra_generation_params["Anima Hires guard"] = str(policy)
        process.extra_generation_params["Anima Hires preset"] = str(preset)
        process.extra_generation_params["Anima Hires limits"] = (
            f"base {limits.base_megapixels:g} MP, hires {limits.hires_megapixels:g} MP, "
            f"upscale {limits.max_upscale:g}x, align {limits.alignment}"
        )
        if not result.changes:
            return

        summary = "; ".join(result.changes)
        process.extra_generation_params["Anima Hires adjustments"] = summary
        if str(policy) == "Report only":
            logger.warning("Anima Hires Guard report: %s", summary)
            return

        process.width = result.base_width
        process.height = result.base_height
        if result.hires_width is not None and result.hires_height is not None:
            # Use an explicit target so both axes retain the exact guarded,
            # aligned values; a scalar hr_scale cannot express rounding.
            process.hr_resize_x = result.hires_width
            process.hr_resize_y = result.hires_height
        logger.info("Anima Hires Guard applied: %s", summary)
