"""Anima-only Hires safety limits exposed as native Forge Neo controls."""

from __future__ import annotations

import logging
import math
import sys

from backend.logging import setup_logger
from lib_anima_hires import BALANCED, GuardLimits, guard_dimensions
from modules import scripts, shared
from modules.anima_support import is_anima_engine, require_finite, table_params
from remi import ui

logger = logging.getLogger("AnimaHiresGuard")
setup_logger(logger)

PRESETS = {
    "Low VRAM": GuardLimits(0.90, 1.50, 1.50),
    "Balanced": BALANCED,
    "Detail": GuardLimits(1.40, 2.36, 2.00),
}

# Marks a "Report only" note in the infotext, which then lists changes the
# guard did not make
REPORT_ONLY_PREFIX = "report only (not applied): "


def _is_anima(process) -> bool:
    return is_anima_engine(getattr(process, "sd_model", None))


def _forge_resolution_step() -> int:
    """The grid Forge rounds every Hires target to (processing's ``sRound``).

    ``modules.ui`` reads ``opts.res_step`` once at import because the setting
    needs a restart, so its copy is the one in effect when it is loaded.
    """

    ui = sys.modules.get("modules.ui")
    step = getattr(ui, "_STEP", None) or getattr(shared.opts, "res_step", 64)
    return max(1, int(step))


# (preset control name, infotext key) of each script argument, in order
FIELDS = (
    ("enabled", "Anima Hires guard enabled"),
    ("policy", "Anima Hires policy"),
    ("preset", "Anima Hires preset"),
    ("align", "Anima Hires align"),
    ("base_mp", "Anima base MP"),
    ("hires_mp", "Anima Hires MP"),
    ("max_upscale", "Anima Hires max upscale"),
)


class AnimaHiresGuardScript(scripts.Script):
    # Panel position only. The guard checks the final dimensions because it
    # runs in after_model_load, after the resolution randomizer's before_process.
    sorting_priority = 2100

    def title(self):
        return "Anima Hires Guard"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def params(self, is_img2img):
        # txt2img only (show): the guard sizes the Hires. fix target
        custom = ui.param("preset", eq="Custom")
        root = ui.Section(
            self.title(),
            enable=ui.Toggle("enabled", False, label=self.title()),
            children=[
                ui.Markdown(
                    "Anima only. Validates the base pass and Hires target after "
                    "resolution randomization. **Report only** never changes dimensions."
                ),
                ui.Row(
                    ui.Choice("policy", ["Clamp unsafe values", "Report only"], "Clamp unsafe values", label="Policy"),
                    ui.Choice("preset", ["Balanced", "Low VRAM", "Detail", "Custom"], "Balanced", label="Safety preset", info="Custom uses the three sliders below."),
                    ui.Toggle(
                        "align",
                        True,
                        label="Align to 16 px",
                        info="Matches Anima's 8x VAE reduction and 2x transformer patches. Hires targets also follow Forge's Resolution Step setting.",
                    ),
                ),
                # Read only with the Custom preset: the named presets bring their own limits
                ui.Row(
                    ui.Slider("base_mp", 0.4, 3.0, 1.2, 0.05, label="Base maximum (MP)"),
                    ui.Slider("hires_mp", 0.8, 6.0, 2.1, 0.05, label="Hires maximum (MP)"),
                    ui.Slider("max_upscale", 1.0, 3.0, 1.75, 0.05, label="Maximum upscale per axis"),
                    enabled_if=custom,
                ),
            ],
        )
        self.param_order = table_params(((name, key, f"hires_guard.{name}") for name, key in FIELDS), root)
        return root

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
        # Before anything is guarded or recorded: max(0.1, nan) would shrink
        # the job to the 0.1 MP floor and record it as an ordinary clamp
        require_finite(self.title(), base_mp=base_mp, hires_mp=hires_mp, max_upscale=max_upscale)
        if not enabled or not _is_anima(process):
            return
        if getattr(process, "enable_hr", False) and getattr(shared.opts, "use_old_hires_fix_width_height", False):
            # Width/height are then the final hires size and processing
            # derives the first pass (and hr_resize_x/y) from them later.
            reason = "skipped: the legacy hires width/height setting is on"
            process.extra_generation_params["Anima Hires adjustments"] = reason
            logger.warning("Anima Hires Guard %s", reason)
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
        # txt2img's Upscale button: width/height are the finished image's size
        # (modules/txt2img.py) and there is no base pass to shrink.
        upscaling_image = bool(getattr(process, "txt2img_upscale", False)) or getattr(process, "firstpass_image", None) is not None
        # Forge rounds the Hires target to its Resolution Step after this hook
        # (processing.calculate_target_resolution): guard on a grid that
        # rounding keeps, or a checked 1440 px axis renders as 1472 px.
        hires_alignment = math.lcm(limits.alignment, _forge_resolution_step())

        result = guard_dimensions(
            process.width,
            process.height,
            enable_hr=bool(getattr(process, "enable_hr", False)),
            hr_scale=float(getattr(process, "hr_scale", 1.0)),
            hr_resize_x=int(getattr(process, "hr_resize_x", 0) or 0),
            hr_resize_y=int(getattr(process, "hr_resize_y", 0) or 0),
            limits=limits,
            fixed_base=upscaling_image,
            hires_alignment=hires_alignment,
        )
        # The params' ``infotext`` keys (``FIELDS``), so pasted parameters restore the panel
        values = (True, str(policy), str(preset), bool(align), float(base_mp), float(hires_mp), float(max_upscale))
        process.extra_generation_params.update(zip((key for _, key in FIELDS), values))
        process.extra_generation_params["Anima Hires limits"] = (
            f"base {limits.base_megapixels:g} MP, hires {limits.hires_megapixels:g} MP, "
            f"upscale {limits.max_upscale:g}x, align {limits.alignment}, hires align {hires_alignment}"
        )
        report_only = str(policy) == "Report only"
        if not report_only and result.hires_width is not None and result.hires_height is not None:
            # Always an explicit target, so both axes keep the exact guarded,
            # grid-aligned values: a scalar hr_scale cannot express rounding.
            process.hr_resize_x = result.hires_width
            process.hr_resize_y = result.hires_height
        if not result.changes:
            return

        summary = "; ".join(result.changes)
        if report_only:
            # The image keeps the unsafe size: say so, or the infotext reads as a clamp
            process.extra_generation_params["Anima Hires adjustments"] = f"{REPORT_ONLY_PREFIX}{summary}"
            logger.warning("Anima Hires Guard report: %s", summary)
            return

        process.extra_generation_params["Anima Hires adjustments"] = summary
        process.width = result.base_width
        process.height = result.base_height
        logger.info("Anima Hires Guard applied: %s", summary)
