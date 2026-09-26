"""Forge Neo GUI integration for Anima DAVE (DC attenuation for diversity)."""

from __future__ import annotations

import logging

import gradio as gr

from backend.logging import setup_logger
from lib_anima_dave.dave import DEFAULT_BLOCKS, DEFAULT_STRENGTH, DEFAULT_TAU, DaveState, apply_dave_patch, model_blocks
from modules import scripts
from modules.anima_support import is_anima_engine
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaDAVE")
setup_logger(logger)

# Infotext keys, in the order of the controls returned by ``ui``
INFOTEXT_KEYS = ("Anima DAVE enabled", "Anima DAVE strength", "Anima DAVE tau", "Anima DAVE blocks")


def generation_params(strength, tau, blocks) -> dict:
    """Infotext for one DAVE pass; keys match ``INFOTEXT_KEYS``."""

    return dict(zip(INFOTEXT_KEYS, (True, float(strength), float(tau), str(blocks))))


class AnimaDaveScript(scripts.Script):
    sorting_priority = 2028

    def title(self):
        return "Anima DAVE Diversity"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

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
        controls = [enable, strength, tau, blocks]
        self.infotext_fields = [PasteField(control, key) for control, key in zip(controls, INFOTEXT_KEYS)]
        self.paste_field_names = list(INFOTEXT_KEYS)
        return controls

    def process_before_every_sampling(self, p, enable, strength, tau, blocks, **kwargs):
        if not enable or float(strength) <= 0.0 or not is_anima_engine(getattr(p, "sd_model", None)):
            return
        unet = p.sd_model.forge_objects.unet
        total_blocks = len(unet.model.diffusion_model.blocks)
        selected = model_blocks(str(blocks), total_blocks)
        state = DaveState(blocks=selected, strength=float(strength), tau=float(tau))
        # A fresh patch per pass on the pass's own patcher: nothing outlives it,
        # and block forwards are only overridden inside a windowed model call.
        p.sd_model.forge_objects.unet = apply_dave_patch(unet, state)
        p.extra_generation_params.update(generation_params(strength, tau, blocks))
        logger.debug("Anima DAVE: strength %.3f, tau %.3f, model blocks %s", float(strength), float(tau), selected)
