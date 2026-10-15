"""One-click, GUI-visible baseline for Remi's built-in Anima stack."""

from __future__ import annotations

import logging

from backend.logging import setup_logger
from modules import scripts
from modules.anima_presets import PRESET_DESCRIPTIONS, PRESETS, SAFE_BASE_AESTHETIC_LABEL, turbo_settings_advice
from modules.anima_support import is_anima_engine
from remi import ui

logger = logging.getLogger("anima_presets")
setup_logger(logger)


def owner_relative(patch: dict, owner: str) -> dict:
    """A patch relative to the core owner (core params bare, others
    ``owner.param``) made relative to ``owner``: core params become
    ``core.param`` and ``owner``'s own params bare."""

    relative = {}
    for key, entry in patch.items():
        if key.startswith("__"):
            relative[key] = entry
        elif "." not in key:
            relative[f"core.{key}"] = entry
        else:
            target, name = key.split(".", 1)
            relative[name if target == owner else key] = entry
    return relative


class AnimaRemiPresetScript(scripts.Script):
    # Default section, below the panels the preset configures.
    sorting_priority = 100000

    def title(self):
        return "Anima Remi Presets"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        # The preset owns no processing setting: its one param picks what Apply writes.
        return ui.Section(
            self.title(),
            ui.Choice("preset", list(PRESETS), SAFE_BASE_AESTHETIC_LABEL, label="Preset"),
            *(ui.Markdown(PRESET_DESCRIPTIONS[label], visible_if=ui.param("preset", eq=label)) for label in PRESETS if PRESET_DESCRIPTIONS.get(label)),
            ui.Markdown(
                "Optional guidance, DAVE, Cross-Attention Boost, cache, spatial, "
                "Hires, Refiner and post-processing stacks start disabled. Applying a preset "
                "preserves prompt text, model paths and LoRA/artist names."
            ),
            ui.Button("Apply preset", "apply", "primary"),
        )

    @ui.action("apply")
    def apply(self, ctx: ui.ActionContext, values: dict) -> dict:
        """Every control bound to an Anima preset field, native and extension
        ones alike (the layouts' ``preset=`` bindings), set to the chosen
        preset, with the conflict locks resolved again over the current form
        (``ctx.core``/``ctx.scripts``) and the preset's values. Unknown labels
        apply the default preset."""

        from remi.core.presets import anima_preset_patch

        tab = ctx.tab if ctx.tab in ("txt2img", "img2img") else self.tabname or "txt2img"
        owner = ctx.owner or self.script_id
        current = {**ctx.scripts, "core": ctx.core, owner: values}
        return owner_relative(anima_preset_patch(tab, str(values.get("preset") or ""), values=current), owner)

    def process(self, p, *args):
        if not is_anima_engine(getattr(p, "sd_model", None)):
            return
        checkpoint = getattr(getattr(p.sd_model, "sd_checkpoint_info", None), "name", "")
        advice = turbo_settings_advice(checkpoint, float(p.cfg_scale), int(p.steps))
        if advice is not None:
            logger.warning(advice)
