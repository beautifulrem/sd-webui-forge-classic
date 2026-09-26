"""Forge Neo GUI integration for the Anima front-loaded cross-attention boost."""

from __future__ import annotations

import logging

import gradio as gr

from backend.logging import setup_logger
from lib_anima_xattn_boost.boost import DEFAULT_BAND, DEFAULT_GAIN, DEFAULT_RENORM, BoostState, apply_xattn_boost_patch
from modules import scripts
from modules.anima_presets import register_preset_control
from modules.anima_support import is_anima_engine
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaXattnBoost")
setup_logger(logger)

# Infotext keys, in the order of the controls returned by ``ui``
INFOTEXT_KEYS = ("Anima xattn boost enabled", "Anima xattn boost strength", "Anima xattn boost band", "Anima xattn boost renorm")
# Anima preset fields, in the same order
PRESET_FIELDS = ("xattn_boost.enabled", "xattn_boost.strength", "xattn_boost.band", "xattn_boost.renorm")


def generation_params(strength, band, renorm) -> dict:
    """Infotext for one boosted pass; keys match ``INFOTEXT_KEYS``."""

    return dict(zip(INFOTEXT_KEYS, (True, float(strength), float(band), float(renorm))))


class AnimaXattnBoostScript(scripts.Script):
    sorting_priority = 2029

    def title(self):
        return "Anima Cross-Attention Boost"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown(
                "Improves adherence to weak tags and multi-subject bindings by amplifying the positive prompt's "
                "cross-attention during the first high-noise steps (sigma at or above the band). No extra model "
                "forwards. It amplifies every tag, framing and count tags included: expect some composition and "
                "identity drift."
            )
            with gr.Row():
                strength = gr.Slider(1.0, 3.0, value=DEFAULT_GAIN, step=0.05, label="Strength", info="Cross-attention residual gain; 1 is off. 2.0 is the only strength the reference ships.")
                band = gr.Slider(0.5, 1.0, value=DEFAULT_BAND, step=0.01, label="Band (sigma)", info="Boost model calls whose sigma is at or above this; 0.85 is the only tested band.")
                renorm = gr.Slider(0.0, 1.0, value=DEFAULT_RENORM, step=0.05, label="Norm matching", info="Pulls each boosted image back toward its unboosted mean token norm (0 = raw gain).")
        controls = [enable, strength, band, renorm]
        self.infotext_fields = [PasteField(control, key) for control, key in zip(controls, INFOTEXT_KEYS)]
        self.paste_field_names = list(INFOTEXT_KEYS)
        for name, control in zip(PRESET_FIELDS, controls):
            register_preset_control(self.tabname, name, control)
        return controls

    def process_before_every_sampling(self, p, enable, strength, band, renorm, **kwargs):
        if not enable or float(strength) == 1.0 or not is_anima_engine(getattr(p, "sd_model", None)):
            return
        unet = p.sd_model.forge_objects.unet
        state = BoostState(gain=float(strength), band=float(band), renorm=float(renorm))
        # A fresh patch per pass on the pass's own patcher: nothing outlives it,
        # and forwards are only overridden inside a boosted model call.
        p.sd_model.forge_objects.unet = apply_xattn_boost_patch(unet, state)
        p.extra_generation_params.update(generation_params(strength, band, renorm))
        logger.debug("Anima xattn boost: gain %.3f, band %.3f, renorm %.3f", float(strength), float(band), float(renorm))
