"""Forge Neo GUI integration for the Anima front-loaded cross-attention boost."""

from __future__ import annotations

import logging

from backend.logging import setup_logger
from lib_anima_xattn_boost.boost import DEFAULT_BAND, DEFAULT_GAIN, DEFAULT_RENORM, BoostState, apply_xattn_boost_patch
from modules import scripts
from modules.anima_support import AnimaPatchPanel, clamp, is_anima_engine, require_finite, to_float
from remi import ui

logger = logging.getLogger("AnimaXattnBoost")
setup_logger(logger)

# Infotext keys, in script argument order
INFOTEXT_KEYS = ("Anima xattn boost enabled", "Anima xattn boost strength", "Anima xattn boost band", "Anima xattn boost renorm")
# Anima preset fields, in the same order
PRESET_FIELDS = ("xattn_boost.enabled", "xattn_boost.strength", "xattn_boost.band", "xattn_boost.renorm")


def generation_params(strength, band, renorm) -> dict:
    """Infotext for one boosted pass; keys match ``INFOTEXT_KEYS``."""

    return dict(zip(INFOTEXT_KEYS, (True, float(strength), float(band), float(renorm))))


class AnimaXattnBoostScript(AnimaPatchPanel, scripts.Script):
    INFOTEXT_KEYS = INFOTEXT_KEYS
    PRESET_FIELDS = PRESET_FIELDS
    sorting_priority = 2029

    def title(self):
        return "Anima Cross-Attention Boost"

    def params(self, is_img2img):
        return self.panel_params(
            ui.Section(
                self.title(),
                enable=ui.Toggle("enabled", False, label=self.title()),
                children=[
                    ui.Markdown(
                        "Improves adherence to weak tags and multi-subject bindings by amplifying the positive prompt's "
                        "cross-attention during the first high-noise steps (sigma at or above the band). No extra model "
                        "forwards. It amplifies every tag, framing and count tags included: expect some composition and "
                        "identity drift."
                    ),
                    ui.Row(
                        ui.Slider("strength", 1.0, 3.0, DEFAULT_GAIN, 0.05, label="Strength", info="Cross-attention residual gain; 1 is off. 2.0 is the only strength the reference ships."),
                        ui.Slider("band", 0.5, 1.0, DEFAULT_BAND, 0.01, label="Band (sigma)", info="Boost model calls whose sigma is at or above this; 0.85 is the only tested band."),
                        ui.Slider("renorm", 0.0, 1.0, DEFAULT_RENORM, 0.05, label="Norm matching", info="Pulls each boosted image back toward its unboosted mean token norm (0 = raw gain)."),
                    ),
                ],
            )
        )

    def process_before_every_sampling(self, p, enable, strength, band, renorm, **kwargs):
        # Before anything is installed: a NaN gain would blacken the image
        require_finite(self.title(), strength=strength, band=band, renorm=renorm)
        # The sliders' ranges; what is applied is what the infotext records
        strength = clamp(to_float(strength, DEFAULT_GAIN), 1.0, 3.0)
        band = clamp(to_float(band, DEFAULT_BAND), 0.5, 1.0)
        renorm = clamp(to_float(renorm, DEFAULT_RENORM), 0.0, 1.0)
        if not enable or strength == 1.0 or not is_anima_engine(getattr(p, "sd_model", None)):
            return
        state = BoostState(gain=strength, band=band, renorm=renorm)
        # Forwards are only overridden inside a boosted model call.
        self.patch_pass(p, lambda unet: apply_xattn_boost_patch(unet, state), generation_params(strength, band, renorm))
        logger.debug("Anima xattn boost: gain %.3f, band %.3f, renorm %.3f", strength, band, renorm)
