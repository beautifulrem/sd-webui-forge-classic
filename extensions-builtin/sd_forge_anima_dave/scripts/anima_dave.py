"""Forge Neo GUI integration for Anima DAVE (DC attenuation for diversity)."""

from __future__ import annotations

import logging

import gradio as gr

from backend.logging import setup_logger
from lib_anima_dave.dave import DEFAULT_BLOCKS, DEFAULT_STRENGTH, DEFAULT_TAU, DaveState, apply_dave_patch, model_blocks
from modules import scripts
from modules.anima_support import AnimaPatchPanel, clamp, is_anima_engine, require_finite, to_float
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaDAVE")
setup_logger(logger)

# Infotext keys, in the order of the controls returned by ``ui``
INFOTEXT_KEYS = ("Anima DAVE enabled", "Anima DAVE strength", "Anima DAVE tau", "Anima DAVE blocks")
# Anima preset fields, in the same order
PRESET_FIELDS = ("dave.enabled", "dave.strength", "dave.tau", "dave.blocks")


def generation_params(strength, tau, blocks) -> dict:
    """Infotext for one DAVE pass; keys match ``INFOTEXT_KEYS``."""

    return dict(zip(INFOTEXT_KEYS, (True, float(strength), float(tau), str(blocks))))


class AnimaDaveScript(AnimaPatchPanel, scripts.Script):
    INFOTEXT_KEYS = INFOTEXT_KEYS
    PRESET_FIELDS = PRESET_FIELDS
    sorting_priority = 2028

    def title(self):
        return "Anima DAVE Diversity"

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown(
                "Adds same-prompt layout diversity across seeds by removing part of the spatial mean (DC) "
                "of early-mid Anima block outputs during the first steps (DAVE, ICML 2026). "
                "Window width hurts text and hands more than strength: keep tau at or below 0.10."
            )
            with gr.Row():
                strength = gr.Slider(0.0, 1.0, value=DEFAULT_STRENGTH, step=0.01, label="Strength", info="Fraction of the DC removed per block; diversity often peaks low (0.05-0.2).")
                tau = gr.Slider(0.01, 1.0, value=DEFAULT_TAU, step=0.01, label="Tau", info="Leading fraction of the schedule DAVE runs on.")
                blocks = gr.Textbox(
                    DEFAULT_BLOCKS,
                    label="Blocks (2B layout)",
                    info="28-block indices; mapped onto the original blocks of 40/52-block models.",
                )
        controls = self.register_paste_fields([enable, strength, tau, blocks])
        return controls

    def process_before_every_sampling(self, p, enable, strength, tau, blocks, **kwargs):
        # Before anything is installed: a NaN tau aborts the job mid-sampling
        require_finite("Anima DAVE", strength=strength, tau=tau)
        # The sliders' ranges; what is applied is what the infotext records
        strength = clamp(to_float(strength, DEFAULT_STRENGTH), 0.0, 1.0)
        tau = clamp(to_float(tau, DEFAULT_TAU), 0.01, 1.0)
        if not enable or strength <= 0.0 or not is_anima_engine(getattr(p, "sd_model", None)):
            return
        total_blocks = len(p.sd_model.forge_objects.unet.model.diffusion_model.blocks)
        selected = model_blocks(str(blocks), total_blocks)
        state = DaveState(blocks=selected, strength=strength, tau=tau)
        # Block forwards are only overridden inside a windowed model call.
        self.patch_pass(p, lambda unet: apply_dave_patch(unet, state), generation_params(strength, tau, blocks))
        logger.debug("Anima DAVE: strength %.3f, tau %.3f, model blocks %s", strength, tau, selected)
