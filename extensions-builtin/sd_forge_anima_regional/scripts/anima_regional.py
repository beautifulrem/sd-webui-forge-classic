"""Forge Neo GUI integration for Anima regional prompt conditioning."""

from __future__ import annotations

import logging

import gradio as gr
import torch

from backend.logging import setup_logger
from lib_anima_regional.regional import Region, RegionalState, apply_regional_patch, parse_blocks
from modules import prompt_parser, scripts
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaRegional")
setup_logger(logger)


def _extract_conditioning(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _extract_conditioning(value[0])
    if isinstance(value, dict):
        for key in ("crossattn", "cross_attn", "c_crossattn"):
            if torch.is_tensor(value.get(key)):
                return value[key]
    raise RuntimeError("Could not extract Anima regional conditioning tensor")


def _encode(p, text: str):
    value = prompt_parser.SdConditioning(
        [str(text)],
        width=p.width,
        height=p.height,
        distilled_cfg_scale=getattr(p, "distilled_cfg_scale", None),
    )
    result = p.sd_model.get_learned_conditioning(value)
    return _extract_conditioning(result).detach().float().cpu().contiguous()


class AnimaRegionalScript(scripts.Script):
    sorting_priority = 2027

    def title(self):
        return "Anima Regional Conditioning"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown("Routes independently encoded prompts into feathered rectangular regions at selected Anima cross-attention blocks.")
            with gr.Row():
                blocks = gr.Textbox("0-27", label="Anima blocks")
                start = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Start progress")
                end = gr.Slider(0.0, 1.0, value=0.65, step=0.01, label="End progress")
                feather = gr.Slider(0.0, 0.25, value=0.03, step=0.005, label="Edge feather")
                base_preserve = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Base preserve")
            region_controls = []
            defaults = [
                (True, "", 0.0, 0.0, 0.5, 1.0, 1.0),
                (True, "", 0.5, 0.0, 0.5, 1.0, 1.0),
                (False, "", 0.25, 0.25, 0.5, 0.5, 1.0),
            ]
            for index, values in enumerate(defaults, start=1):
                with gr.Accordion(f"Region {index}", open=index <= 2):
                    with gr.Row():
                        active = gr.Checkbox(values[0], label="Active")
                        prompt = gr.Textbox(values[1], label="Regional prompt")
                        strength = gr.Slider(0.0, 2.0, value=values[6], step=0.05, label="Strength")
                    with gr.Row():
                        x = gr.Slider(0.0, 1.0, value=values[2], step=0.01, label="X")
                        y = gr.Slider(0.0, 1.0, value=values[3], step=0.01, label="Y")
                        width = gr.Slider(0.0, 1.0, value=values[4], step=0.01, label="Width")
                        height = gr.Slider(0.0, 1.0, value=values[5], step=0.01, label="Height")
                    region_controls.extend([active, prompt, x, y, width, height, strength])
        controls = [enable, blocks, start, end, feather, base_preserve, *region_controls]
        keys = [
            "Anima regional enabled",
            "Anima regional blocks",
            "Anima regional start",
            "Anima regional end",
            "Anima regional feather",
            "Anima regional base preserve",
            *[f"Anima region {index} {name}" for index in range(1, 4) for name in ("active", "prompt", "x", "y", "width", "height", "strength")],
        ]
        self.infotext_fields = [PasteField(control, key) for control, key in zip(controls, keys)]
        self.paste_field_names = keys
        return controls

    def process_before_every_sampling(self, p, enable, blocks, start, end, feather, base_preserve, *region_values, **kwargs):
        if not enable or type(getattr(p, "sd_model", None)).__name__ != "Anima":
            return
        unet = p.sd_model.forge_objects.unet
        total_blocks = len(unet.model.diffusion_model.blocks)
        selected_blocks = parse_blocks(str(blocks), total_blocks)
        regions = []
        for offset in range(0, len(region_values), 7):
            active, text, x, y, width, height, strength = region_values[offset : offset + 7]
            if not active or not str(text).strip() or float(strength) <= 0.0 or float(width) <= 0.0 or float(height) <= 0.0:
                continue
            regions.append(Region(_encode(p, str(text)), float(x), float(y), float(width), float(height), float(strength)))
        if not regions:
            logger.warning("Anima Regional Conditioning was enabled but no active non-empty region exists.")
            return
        predictor = unet.model.predictor
        state = RegionalState(
            regions=regions,
            blocks=selected_blocks,
            feather=float(feather),
            base_preserve=float(base_preserve),
            start_sigma=float(predictor.percent_to_sigma(float(start))),
            end_sigma=float(predictor.percent_to_sigma(float(end))),
        )
        p.sd_model.forge_objects.unet = apply_regional_patch(unet, state)
        p.extra_generation_params["Anima regional conditioning"] = True
        p.extra_generation_params["Anima regional regions"] = len(regions)
        p.extra_generation_params["Anima regional blocks"] = str(blocks)
        p.extra_generation_params["Anima regional range"] = f"{float(start):.2f}-{float(end):.2f}"
        p.extra_generation_params["Anima regional feather"] = float(feather)
        p.extra_generation_params["Anima regional base preserve"] = float(base_preserve)
