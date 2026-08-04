"""Native Anima guidance and correction controls for Forge Neo."""

from __future__ import annotations

import math

import gradio as gr
import torch

from lib_anima_guidance.skim import apply_skim_to_predictions
from modules import scripts
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion


def _is_anima(p) -> bool:
    return type(getattr(p, "sd_model", None)).__name__ == "Anima"


def _active_for_sigma(sigma: float, sigma_start: float, sigma_end: float) -> bool:
    high = max(float(sigma_start), float(sigma_end))
    low = min(float(sigma_start), float(sigma_end))
    eps = max(1e-8, high * 1e-7)
    return low - eps <= float(sigma) <= high + eps


def _make_skim_cfg_function(
    previous_cfg_function,
    *,
    sigma_start: float,
    sigma_end: float,
    skimming_scale: float,
    full_skim_negative: bool,
    disable_flipping_filter: bool,
    flip_sigma: float | None,
):
    """Build a Forge CFG hook while preserving an existing CFG hook.

    Forge expects ``sampler_cfg_function`` to return a residual which is
    subtracted from the sampler input.  Existing CFG functions are called with
    adjusted cond/uncond predictions, so Skimmed CFG composes before their
    combine rather than silently replacing them.
    """

    @torch.no_grad()
    def cfg_function(args):
        x_orig = args["input"]
        sigma_tensor = args.get("sigma", args.get("timestep"))
        sigma = float(sigma_tensor.flatten()[0]) if torch.is_tensor(sigma_tensor) else float(sigma_tensor)

        adjusted_args = args
        if _active_for_sigma(sigma, sigma_start, sigma_end):
            flip_filter = bool(disable_flipping_filter)
            if flip_sigma is not None and sigma > flip_sigma:
                flip_filter = not flip_filter

            cond, uncond = apply_skim_to_predictions(
                x_orig,
                args["cond_denoised"],
                args["uncond_denoised"],
                float(args["cond_scale"]),
                float(skimming_scale),
                full_skim_negative=bool(full_skim_negative),
                disable_flipping_filter=flip_filter,
            )
            adjusted_args = dict(args)
            adjusted_args["cond_denoised"] = cond
            adjusted_args["uncond_denoised"] = uncond
            adjusted_args["cond"] = x_orig - cond
            adjusted_args["uncond"] = x_orig - uncond

        if previous_cfg_function is not None:
            return previous_cfg_function(adjusted_args)

        cond = adjusted_args["cond_denoised"]
        uncond = adjusted_args["uncond_denoised"]
        denoised = uncond + (cond - uncond) * float(adjusted_args["cond_scale"])
        return x_orig - denoised

    cfg_function._anima_skimmed_cfg = True
    cfg_function._anima_wrapped_cfg = previous_cfg_function
    return cfg_function


class AnimaGuidanceScript(scripts.Script):
    sorting_priority = 2030

    def title(self):
        return "Anima Guidance & Corrections"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown(
                "Only applies to Anima. Guidance patches are installed on a cloned "
                "model for the current generation and recorded in PNG metadata."
            )
            skim_enable = gr.Checkbox(False, label="Skimmed CFG anti-burn")
            with gr.Row():
                skim_scale = gr.Slider(
                    0.0,
                    10.0,
                    value=2.5,
                    step=0.1,
                    label="Skimming fallback CFG",
                    info="Only flagged values are limited to this effective CFG; 1.5–3 is a useful Anima range.",
                )
                full_negative = gr.Checkbox(
                    False,
                    label="Full skim negative",
                    info="Uses a zero fallback for flagged negative-prompt values.",
                )
                disable_flip_filter = gr.Checkbox(
                    False,
                    label="Disable flipping filter",
                    info="More aggressive; keeps values normally rejected by the sign-flip safety filter.",
                )
            with gr.Row():
                start_percent = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Skim start")
                end_percent = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Skim end")
                flip_percent = gr.Slider(
                    0.0,
                    1.0,
                    value=0.0,
                    step=0.01,
                    label="Filter flip point",
                    info="0 disables timed inversion of the flipping filter.",
                )

        controls = [
            enable,
            skim_enable,
            skim_scale,
            full_negative,
            disable_flip_filter,
            start_percent,
            end_percent,
            flip_percent,
        ]
        keys = [
            "Anima guidance enabled",
            "Anima Skimmed CFG",
            "Anima skim CFG",
            "Anima skim full negative",
            "Anima skim disable flip filter",
            "Anima skim start",
            "Anima skim end",
            "Anima skim flip",
        ]
        self.infotext_fields = [PasteField(component, key) for component, key in zip(controls, keys)]
        self.paste_field_names = keys
        return controls

    def process_before_every_sampling(
        self,
        p,
        enable: bool,
        skim_enable: bool,
        skim_scale: float,
        full_negative: bool,
        disable_flip_filter: bool,
        start_percent: float,
        end_percent: float,
        flip_percent: float,
        *args,
        **kwargs,
    ):
        if not enable or not skim_enable or not _is_anima(p):
            return

        unet = p.sd_model.forge_objects.unet.clone()
        predictor = unet.model.predictor
        sigma_start = float(predictor.percent_to_sigma(float(start_percent)))
        sigma_end = float(predictor.percent_to_sigma(float(end_percent)))
        flip_sigma = None
        if 0.0 < float(flip_percent) < 1.0:
            flip_sigma = float(predictor.percent_to_sigma(float(flip_percent)))

        previous = unet.model_options.get("sampler_cfg_function")
        unet.set_model_sampler_cfg_function(
            _make_skim_cfg_function(
                previous,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                skimming_scale=float(skim_scale),
                full_skim_negative=bool(full_negative),
                disable_flipping_filter=bool(disable_flip_filter),
                flip_sigma=flip_sigma,
            ),
            disable_cfg1_optimization=True,
        )
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params["Anima Skimmed CFG"] = True
        p.extra_generation_params["Anima skim CFG"] = float(skim_scale)
        if full_negative:
            p.extra_generation_params["Anima skim full negative"] = True
        if disable_flip_filter:
            p.extra_generation_params["Anima skim disable flip filter"] = True
        if not math.isclose(float(start_percent), 0.0) or not math.isclose(float(end_percent), 1.0):
            p.extra_generation_params["Anima skim range"] = f"{float(start_percent):.2f}-{float(end_percent):.2f}"
        if flip_sigma is not None:
            p.extra_generation_params["Anima skim flip"] = float(flip_percent)
