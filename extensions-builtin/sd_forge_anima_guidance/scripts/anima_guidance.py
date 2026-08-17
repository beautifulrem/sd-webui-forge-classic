"""Native Anima guidance and correction controls for Forge Neo."""

from __future__ import annotations

import math
import os
import logging

import gradio as gr
import torch

from backend import memory_management
from backend.logging import setup_logger
from lib_anima_guidance.modulation import (
    ADAPTER_MAX_BYTES,
    ADAPTER_SHA256,
    ADAPTER_URL,
    CLIP_MAX_BYTES,
    CLIP_SHA256,
    CLIP_URL,
    ModulationPatch,
    encode_clip_pooled,
    prepare_modulation_vectors,
    resolve_artifact,
)
from lib_anima_guidance.prompts import effective_prompt_batch
from lib_anima_guidance.skim import apply_skim_to_predictions
from lib_anima_guidance.smc import SMCCFGState, make_smc_cfg_function
from lib_anima_guidance.dcw import DCWState, parse_band_mask
from lib_anima_guidance.advanced import (
    MomentumGuidanceState,
    make_fdg_cfg_function,
    make_guidance_range_cfg_function,
    make_momentum_post_cfg_function,
)
from lib_anima_guidance.nag import NAGAttentionModifier
from backend.nn.anima_attention import ANIMA_ATTENTION_MODIFIERS
from modules import paths, script_callbacks, scripts
from modules.anima_feature_conflicts import (
    record_conflict_resolution,
    resolve_guidance_conflicts,
)
from modules.anima_presets import register_preset_control
from modules.anima_support import is_anima_auxiliary_denoiser, is_anima_engine
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaGuidance")
setup_logger(logger)


def _is_anima(p) -> bool:
    return is_anima_engine(getattr(p, "sd_model", None))


def _dcw_before_denoiser(params) -> None:
    process = getattr(params.denoiser, "p", None)
    if is_anima_auxiliary_denoiser(process):
        return
    state = getattr(process, "_anima_dcw_state", None)
    if state is None:
        return
    sigma = params.sigma
    sigma_value = float(sigma.flatten()[0]) if torch.is_tensor(sigma) else float(sigma)
    state.before_denoiser(params.x, sigma_value)


def _capture_dcw_denoised(process, state: DCWState, denoised):
    """Do not let PC3 endpoint probes replace the main-step DCW history."""

    if is_anima_auxiliary_denoiser(process):
        return denoised
    return state.capture_denoised(denoised)


script_callbacks.on_cfg_denoiser(_dcw_before_denoiser, name="anima_dcw_pre_step")


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

    def __init__(self):
        self._sampler_component = None

    def title(self):
        return "Anima Guidance & Corrections"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        if getattr(component, "elem_id", None) == f"{self.tabname}_sampling":
            self._sampler_component = component

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown(
                "Only applies to Anima. Guidance patches are installed on a cloned "
                "model for the current generation and recorded in PNG metadata."
            )
            conflict_status = gr.Markdown("")
            guidance_mode = gr.Dropdown(
                ["Standard / preserve existing", "SMC-CFG", "FDG (experimental)"],
                value="Standard / preserve existing",
                label="CFG guidance mode",
                info="SMC-CFG and FDG are mutually exclusive combine modes. Skimmed CFG is applied before the selected combine.",
            )
            with gr.Row():
                smc_alpha = gr.Slider(
                    0.0,
                    1.0,
                    value=0.2,
                    step=0.01,
                    label="SMC adaptive alpha",
                    info="Switching gain = alpha x mean absolute conditional residual.",
                )
                smc_lambda = gr.Slider(
                    0.0,
                    10.0,
                    value=5.0,
                    step=0.1,
                    label="SMC surface lambda",
                )
            with gr.Accordion("Advanced Anima guidance", open=False):
                gr.Markdown(
                    "NAG modifies cross-attention; Momentum modifies the final flow velocity. "
                    "NAG guides the base prompt path while synthetic Regional/Artist branches stay isolated. "
                    "Momentum is limited to single-evaluation Euler samplers and is disabled with SMC-CFG or FDG."
                )
                nag_enable = gr.Checkbox(False, label="Enable NAG attention guidance")
                with gr.Row():
                    nag_scale = gr.Slider(0.0, 8.0, value=2.0, step=0.05, label="NAG scale")
                    nag_tau = gr.Slider(0.1, 10.0, value=2.5, step=0.1, label="NAG normalization tau")
                    nag_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="NAG blend alpha")
                with gr.Row():
                    nag_start = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="NAG start")
                    nag_end = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="NAG end")
                momentum_enable = gr.Checkbox(False, label="Enable Momentum Guidance")
                with gr.Row():
                    momentum_strength = gr.Slider(0.0, 1.5, value=0.5, step=0.01, label="Momentum strength")
                    momentum_ema = gr.Slider(0.0, 0.99, value=0.6, step=0.01, label="Momentum EMA decay")
                fdg_high_scale = gr.Slider(
                    0.0,
                    10.0,
                    value=2.0,
                    step=0.1,
                    label="FDG detail guidance",
                    info="Used only by FDG. FDG takes priority over DCW and is disabled with the CNS sampler.",
                )
                guidance_range_enable = gr.Checkbox(False, label="Limit CFG to a sampling range")
                with gr.Row():
                    guidance_range_start = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="CFG range start")
                    guidance_range_end = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="CFG range end")
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
            with gr.Accordion("DCW latent bias correction", open=False):
                dcw_enable = gr.Checkbox(False, label="Enable DCW")
                with gr.Row():
                    dcw_lambda = gr.Slider(
                        -0.1,
                        0.1,
                        value=-0.015,
                        step=0.001,
                        label="DCW lambda",
                        info="Anima LL-band starting point: -0.015.",
                    )
                    dcw_schedule = gr.Dropdown(
                        ["one_minus_sigma", "sigma", "constant"],
                        value="one_minus_sigma",
                        label="DCW schedule",
                    )
                    dcw_bands = gr.Dropdown(
                        ["LL", "LH", "HL", "HH", "LH+HL+HH", "all"],
                        value="LL",
                        label="DCW frequency bands",
                        info="LL is the Anima-tuned default; all reproduces broadband correction.",
                    )
            with gr.Accordion("CNS colored-noise sampler", open=False):
                cns_strength = gr.Slider(
                    0.0,
                    1.0,
                    value=1.0,
                    step=0.05,
                    label="Anima ER SDE CNS strength",
                    info="Used only by the Anima ER SDE CNS sampler. 0 is stock ER-SDE noise; 1 is full calibrated recoloring.",
                )
            with gr.Accordion("Flow corrective sampler tuning", open=False):
                gr.Markdown(
                    "These controls are wired only to `Anima Flow UniPC2` and "
                    "`Anima Flow PC3`; upstream defaults are preserved."
                )
                with gr.Row():
                    flow_unipc_solver_type = gr.Dropdown(
                        ["bh2", "bh1"],
                        value="bh2",
                        label="UniPC solver type",
                    )
                    flow_unipc_disable_corrector_first = gr.Slider(
                        0,
                        10,
                        value=0,
                        step=1,
                        label="UniPC disabled early correctors",
                    )
                    flow_unipc_thresholding = gr.Checkbox(
                        False,
                        label="UniPC dynamic thresholding",
                    )
                with gr.Row():
                    flow_unipc_dynamic_thresholding_ratio = gr.Slider(
                        0.5,
                        1.0,
                        value=0.995,
                        step=0.001,
                        label="UniPC threshold percentile",
                    )
                    flow_unipc_sample_max_value = gr.Slider(
                        1.0,
                        10.0,
                        value=1.0,
                        step=0.1,
                        label="UniPC threshold maximum",
                    )
                with gr.Row():
                    flow_pc3_gamma = gr.Slider(
                        0.0,
                        1.0,
                        value=1.0,
                        step=0.05,
                        label="PC3 maximum correction gamma",
                    )
                    flow_pc3_tolerance = gr.Slider(
                        0.0001,
                        0.05,
                        value=0.005,
                        step=0.0005,
                        label="PC3 correction tolerance",
                    )
            with gr.Accordion("CLIP modulation guidance", open=False):
                gr.Markdown(
                    "Adds the official-style CLIP pooled modulation path to Anima. "
                    "The first automatic use downloads a pinned 163 MiB adapter and 235 MiB CLIP-L encoder."
                )
                modulation_enable = gr.Checkbox(False, label="Enable CLIP modulation guidance")
                with gr.Row():
                    modulation_weight = gr.Slider(
                        -20.0,
                        20.0,
                        value=3.0,
                        step=0.05,
                        label="Direction weight",
                        info="CLIP(base) + weight x (CLIP(positive direction) - CLIP(negative direction)).",
                    )
                    modulation_start = gr.Slider(0, 63, value=0, step=1, label="Start block")
                    modulation_end = gr.Slider(
                        -1,
                        63,
                        value=-1,
                        step=1,
                        label="End block",
                        info="-1 means the final Anima block.",
                    )
                modulation_base = gr.Textbox(
                    "",
                    label="Base CLIP prompt override",
                    info="Empty uses each image's normal positive prompt. The normal negative prompt is used for unconditional rows.",
                )
                with gr.Row():
                    modulation_positive = gr.Textbox(
                        "high quality, detailed, accurate anatomy",
                        label="Positive modulation direction",
                    )
                    modulation_negative = gr.Textbox(
                        "low quality, artifacts, bad anatomy",
                        label="Negative modulation direction",
                    )
                with gr.Row():
                    adapter_mode = gr.Dropdown(
                        ["Auto-download pinned", "Local file"],
                        value="Auto-download pinned",
                        label="Modulation adapter",
                    )
                    adapter_path = gr.Textbox("", label="Local adapter .pt path")
                with gr.Row():
                    clip_mode = gr.Dropdown(
                        ["Auto-download pinned", "Local file"],
                        value="Auto-download pinned",
                        label="CLIP-L encoder",
                    )
                    clip_path = gr.Textbox("", label="Local CLIP-L .safetensors path")

        if self._sampler_component is not None:
            conflict_inputs = [
                self._sampler_component,
                guidance_mode,
                momentum_enable,
                dcw_enable,
            ]
            conflict_outputs = [
                guidance_mode,
                momentum_enable,
                dcw_enable,
                conflict_status,
            ]

            def update_conflicts(selected, sampler, mode, momentum, dcw):
                state = resolve_guidance_conflicts(
                    sampler=str(sampler),
                    mode=str(mode),
                    momentum=bool(momentum),
                    dcw=bool(dcw),
                    selected=selected,
                )
                if state.messages:
                    status = "**Compatibility:** " + "; ".join(state.messages)
                elif state.mode == "FDG (experimental)":
                    status = "**Compatibility:** FDG owns CFG combine; Momentum and DCW are locked."
                elif state.momentum:
                    status = "**Compatibility:** Momentum requires Standard CFG guidance."
                elif state.dcw:
                    status = "**Compatibility:** DCW is active, so FDG is unavailable."
                else:
                    status = ""
                return (
                    gr.update(value=state.mode, choices=list(state.mode_choices)),
                    gr.update(
                        value=state.momentum,
                        interactive=state.momentum_interactive,
                    ),
                    gr.update(value=state.dcw, interactive=state.dcw_interactive),
                    gr.update(value=status),
                )

            for selected, component in (
                ("sampler", self._sampler_component),
                ("mode", guidance_mode),
                ("momentum", momentum_enable),
                ("dcw", dcw_enable),
            ):
                component.change(
                    fn=lambda sampler, mode, momentum, dcw, selected=selected: update_conflicts(
                        selected, sampler, mode, momentum, dcw
                    ),
                    inputs=conflict_inputs,
                    outputs=conflict_outputs,
                    queue=False,
                    show_progress=False,
                )

        controls = [
            enable,
            guidance_mode,
            smc_alpha,
            smc_lambda,
            skim_enable,
            skim_scale,
            full_negative,
            disable_flip_filter,
            start_percent,
            end_percent,
            flip_percent,
            dcw_enable,
            dcw_lambda,
            dcw_schedule,
            dcw_bands,
            cns_strength,
            flow_unipc_solver_type,
            flow_unipc_disable_corrector_first,
            flow_unipc_thresholding,
            flow_unipc_dynamic_thresholding_ratio,
            flow_unipc_sample_max_value,
            flow_pc3_gamma,
            flow_pc3_tolerance,
            modulation_enable,
            modulation_weight,
            modulation_start,
            modulation_end,
            modulation_base,
            modulation_positive,
            modulation_negative,
            adapter_mode,
            adapter_path,
            clip_mode,
            clip_path,
            nag_enable,
            nag_scale,
            nag_tau,
            nag_alpha,
            nag_start,
            nag_end,
            momentum_enable,
            momentum_strength,
            momentum_ema,
            fdg_high_scale,
            guidance_range_enable,
            guidance_range_start,
            guidance_range_end,
        ]
        preset_controls = {
            "guidance.enabled": enable,
            "guidance.mode": guidance_mode,
            "guidance.smc_alpha": smc_alpha,
            "guidance.smc_lambda": smc_lambda,
            "guidance.skim_enabled": skim_enable,
            "guidance.skim_scale": skim_scale,
            "guidance.skim_full_negative": full_negative,
            "guidance.skim_disable_flip": disable_flip_filter,
            "guidance.skim_start": start_percent,
            "guidance.skim_end": end_percent,
            "guidance.skim_flip": flip_percent,
            "guidance.dcw_enabled": dcw_enable,
            "guidance.dcw_lambda": dcw_lambda,
            "guidance.dcw_schedule": dcw_schedule,
            "guidance.dcw_bands": dcw_bands,
            "guidance.cns_strength": cns_strength,
            "guidance.unipc_solver": flow_unipc_solver_type,
            "guidance.unipc_disabled_correctors": flow_unipc_disable_corrector_first,
            "guidance.unipc_thresholding": flow_unipc_thresholding,
            "guidance.unipc_threshold_ratio": flow_unipc_dynamic_thresholding_ratio,
            "guidance.unipc_threshold_max": flow_unipc_sample_max_value,
            "guidance.pc3_gamma": flow_pc3_gamma,
            "guidance.pc3_tolerance": flow_pc3_tolerance,
            "guidance.modulation_enabled": modulation_enable,
            "guidance.modulation_weight": modulation_weight,
            "guidance.modulation_start": modulation_start,
            "guidance.modulation_end": modulation_end,
            "guidance.modulation_adapter_mode": adapter_mode,
            "guidance.modulation_clip_mode": clip_mode,
            "guidance.nag_enabled": nag_enable,
            "guidance.nag_scale": nag_scale,
            "guidance.nag_tau": nag_tau,
            "guidance.nag_alpha": nag_alpha,
            "guidance.nag_start": nag_start,
            "guidance.nag_end": nag_end,
            "guidance.momentum_enabled": momentum_enable,
            "guidance.momentum_strength": momentum_strength,
            "guidance.momentum_ema": momentum_ema,
            "guidance.fdg_detail": fdg_high_scale,
            "guidance.range_enabled": guidance_range_enable,
            "guidance.range_start": guidance_range_start,
            "guidance.range_end": guidance_range_end,
        }
        for name, component in preset_controls.items():
            register_preset_control(self.tabname, name, component)
        keys = [
            "Anima guidance enabled",
            "Anima CFG guidance mode",
            "Anima SMC alpha",
            "Anima SMC lambda",
            "Anima Skimmed CFG",
            "Anima skim CFG",
            "Anima skim full negative",
            "Anima skim disable flip filter",
            "Anima skim start",
            "Anima skim end",
            "Anima skim flip",
            "Anima DCW",
            "Anima DCW lambda",
            "Anima DCW schedule",
            "Anima DCW bands",
            "Anima CNS strength",
            "Anima Flow UniPC solver type",
            "Anima Flow UniPC disabled correctors",
            "Anima Flow UniPC thresholding",
            "Anima Flow UniPC threshold percentile",
            "Anima Flow UniPC threshold maximum",
            "Anima Flow PC3 gamma",
            "Anima Flow PC3 tolerance",
            "Anima modulation guidance",
            "Anima modulation weight",
            "Anima modulation start block",
            "Anima modulation end block",
            "Anima modulation base prompt",
            "Anima modulation positive direction",
            "Anima modulation negative direction",
            "Anima modulation adapter mode",
            "Anima modulation adapter path",
            "Anima modulation CLIP mode",
            "Anima modulation CLIP path",
            "Anima NAG",
            "Anima NAG scale",
            "Anima NAG tau",
            "Anima NAG alpha",
            "Anima NAG start",
            "Anima NAG end",
            "Anima Momentum Guidance",
            "Anima Momentum strength",
            "Anima Momentum EMA",
            "Anima FDG detail guidance",
            "Anima CFG active range",
            "Anima CFG range start",
            "Anima CFG range end",
        ]
        self.infotext_fields = [PasteField(component, key) for component, key in zip(controls, keys)]
        self.paste_field_names = keys
        return controls

    def process_before_every_sampling(
        self,
        p,
        enable: bool,
        guidance_mode: str,
        smc_alpha: float,
        smc_lambda: float,
        skim_enable: bool,
        skim_scale: float,
        full_negative: bool,
        disable_flip_filter: bool,
        start_percent: float,
        end_percent: float,
        flip_percent: float,
        dcw_enable: bool,
        dcw_lambda: float,
        dcw_schedule: str,
        dcw_bands: str,
        cns_strength: float,
        flow_unipc_solver_type: str,
        flow_unipc_disable_corrector_first: int,
        flow_unipc_thresholding: bool,
        flow_unipc_dynamic_thresholding_ratio: float,
        flow_unipc_sample_max_value: float,
        flow_pc3_gamma: float,
        flow_pc3_tolerance: float,
        modulation_enable: bool,
        modulation_weight: float,
        modulation_start: int,
        modulation_end: int,
        modulation_base: str,
        modulation_positive: str,
        modulation_negative: str,
        adapter_mode: str,
        adapter_path: str,
        clip_mode: str,
        clip_path: str,
        nag_enable: bool = False,
        nag_scale: float = 2.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.5,
        nag_start: float = 0.0,
        nag_end: float = 0.5,
        momentum_enable: bool = False,
        momentum_strength: float = 0.5,
        momentum_ema: float = 0.6,
        fdg_high_scale: float = 2.0,
        guidance_range_enable: bool = False,
        guidance_range_start: float = 0.0,
        guidance_range_end: float = 1.0,
        *args,
        **kwargs,
    ):
        active_sampler = (
            (getattr(p, "hr_sampler_name", None) or p.sampler_name)
            if getattr(p, "is_hr_pass", False)
            else p.sampler_name
        )
        conflict_state = resolve_guidance_conflicts(
            sampler=str(active_sampler),
            mode=str(guidance_mode),
            momentum=bool(momentum_enable),
            dcw=bool(dcw_enable),
        )
        guidance_mode = conflict_state.mode
        momentum_enable = conflict_state.momentum
        dcw_enable = conflict_state.dcw
        smc_enabled = guidance_mode == "SMC-CFG" and float(smc_alpha) > 0.0
        fdg_enabled = guidance_mode == "FDG (experimental)"
        dcw_enabled = bool(dcw_enable) and not math.isclose(float(dcw_lambda), 0.0)
        cns_selected = active_sampler == "Anima ER SDE CNS"
        if not enable or not _is_anima(p):
            return

        if conflict_state.messages:
            resolution = "; ".join(conflict_state.messages)
            record_conflict_resolution(p, *conflict_state.messages)
            logger.warning("Anima compatibility resolver: %s", resolution)

        if cns_selected:
            p.cns_strength = min(max(float(cns_strength), 0.0), 1.0)
            p.extra_generation_params["Anima CNS strength"] = p.cns_strength

        if active_sampler == "Anima Flow UniPC2":
            p.flow_unipc_solver_type = str(flow_unipc_solver_type)
            p.flow_unipc_disable_corrector_first = max(
                0, int(flow_unipc_disable_corrector_first)
            )
            p.flow_unipc_thresholding = bool(flow_unipc_thresholding)
            p.flow_unipc_dynamic_thresholding_ratio = min(
                max(float(flow_unipc_dynamic_thresholding_ratio), 0.5), 1.0
            )
            p.flow_unipc_sample_max_value = max(
                1.0, float(flow_unipc_sample_max_value)
            )
            p.extra_generation_params["Anima Flow UniPC solver type"] = (
                p.flow_unipc_solver_type
            )
            p.extra_generation_params["Anima Flow UniPC disabled correctors"] = (
                p.flow_unipc_disable_corrector_first
            )
            if p.flow_unipc_thresholding:
                p.extra_generation_params["Anima Flow UniPC thresholding"] = (
                    f"p={p.flow_unipc_dynamic_thresholding_ratio:g}, "
                    f"max={p.flow_unipc_sample_max_value:g}"
                )

        if active_sampler == "Anima Flow PC3":
            p.flow_pc3_gamma = min(max(float(flow_pc3_gamma), 0.0), 1.0)
            p.flow_pc3_tolerance = min(
                max(float(flow_pc3_tolerance), 0.0001), 0.05
            )
            p.extra_generation_params["Anima Flow PC3 gamma"] = p.flow_pc3_gamma
            p.extra_generation_params["Anima Flow PC3 tolerance"] = (
                p.flow_pc3_tolerance
            )

        modulation_enabled = bool(modulation_enable)
        advanced_enabled = bool(nag_enable) or bool(momentum_enable) or fdg_enabled or bool(guidance_range_enable)
        if not skim_enable and not smc_enabled and not dcw_enabled and not modulation_enabled and not advanced_enabled:
            return

        unet = p.sd_model.forge_objects.unet
        if modulation_enabled:
            try:
                automatic_adapter = os.path.join(paths.models_path, "Anima", "modulation_guidance", "checkpoint_4000.pt")
                automatic_clip = os.path.join(paths.models_path, "CLIP", "Anima-Mod-Guidance", "clip_l.safetensors")
                resolved_adapter = resolve_artifact(
                    str(adapter_mode),
                    str(adapter_path),
                    automatic_adapter,
                    url=ADAPTER_URL,
                    sha256=ADAPTER_SHA256,
                    maximum_bytes=ADAPTER_MAX_BYTES,
                )
                resolved_clip = resolve_artifact(
                    str(clip_mode),
                    str(clip_path),
                    automatic_clip,
                    url=CLIP_URL,
                    sha256=CLIP_SHA256,
                    maximum_bytes=CLIP_MAX_BYTES,
                )
                positive_prompts, negative_prompts = effective_prompt_batch(p)
                base_prompts = [str(modulation_base)] * len(positive_prompts) if str(modulation_base).strip() else positive_prompts
                all_clip_prompts = [*base_prompts, *negative_prompts, str(modulation_positive), str(modulation_negative)]
                pooled = encode_clip_pooled(
                    all_clip_prompts,
                    clip_path=resolved_clip,
                    config_dir=os.path.join(paths.script_path, "backend", "huggingface", "black-forest-labs", "FLUX.1-schnell", "text_encoder"),
                    tokenizer_dir=os.path.join(paths.script_path, "backend", "huggingface", "black-forest-labs", "FLUX.1-schnell", "tokenizer"),
                    device=memory_management.text_encoder_device(),
                )
                batch = len(positive_prompts)
                projected_positive, projected_negative, scales = prepare_modulation_vectors(
                    adapter_path=resolved_adapter,
                    base_positive=pooled[:batch],
                    base_negative=pooled[batch : 2 * batch],
                    direction_positive=pooled[-2:-1],
                    direction_negative=pooled[-1:],
                    weight=float(modulation_weight),
                )
                unet, effective_start, effective_end = ModulationPatch(
                    positive=projected_positive,
                    negative=projected_negative,
                    scales=scales,
                    start_layer=int(modulation_start),
                    end_layer=int(modulation_end),
                ).apply(unet)
                p.extra_generation_params["Anima modulation guidance"] = True
                p.extra_generation_params["Anima modulation weight"] = float(modulation_weight)
                p.extra_generation_params["Anima modulation blocks"] = f"{effective_start}-{effective_end}"
                p.extra_generation_params["Anima modulation positive"] = str(modulation_positive)
                p.extra_generation_params["Anima modulation negative"] = str(modulation_negative)
                if str(modulation_base).strip():
                    p.extra_generation_params["Anima modulation base override"] = str(modulation_base)
            except Exception as error:
                modulation_enabled = False
                logger.exception("Anima modulation guidance was disabled for this generation: %s", error)
                p.extra_generation_params["Anima modulation guidance"] = f"disabled: {type(error).__name__}"

        if not modulation_enabled:
            unet = unet.clone()
        predictor = unet.model.predictor
        sigma_start = float(predictor.percent_to_sigma(float(start_percent)))
        sigma_end = float(predictor.percent_to_sigma(float(end_percent)))
        flip_sigma = None
        if 0.0 < float(flip_percent) < 1.0:
            flip_sigma = float(predictor.percent_to_sigma(float(flip_percent)))

        if nag_enable:
            nag_sigma_start = float(predictor.percent_to_sigma(float(nag_start)))
            nag_sigma_end = float(predictor.percent_to_sigma(float(nag_end)))
            def report_unbatched_nag():
                message = "inactive: positive/negative branches could not be GPU-batched"
                p.extra_generation_params["Anima NAG"] = message
                logger.warning(
                    "Anima NAG is inactive for this generation because Forge split the positive and negative branches; reduce batch/resolution or free VRAM"
                )

            unet.append_transformer_option(
                ANIMA_ATTENTION_MODIFIERS,
                NAGAttentionModifier(
                    scale=float(nag_scale),
                    tau=float(nag_tau),
                    alpha=float(nag_alpha),
                    sigma_start=nag_sigma_start,
                    sigma_end=nag_sigma_end,
                    on_unbatched=report_unbatched_nag,
                ),
            )
            unet.set_transformer_option("forge_spectrum_force_actual", "anima_nag")
            unet.disable_model_cfg1_optimization()
            p.extra_generation_params["Anima NAG"] = True
            p.extra_generation_params["Anima NAG scale"] = float(nag_scale)
            p.extra_generation_params["Anima NAG tau"] = float(nag_tau)
            p.extra_generation_params["Anima NAG alpha"] = float(nag_alpha)
            p.extra_generation_params["Anima NAG range"] = f"{float(nag_start):.2f}-{float(nag_end):.2f}"

        previous = unet.model_options.get("sampler_cfg_function")
        active_cfg_function = previous
        if smc_enabled:
            active_cfg_function = make_smc_cfg_function(
                SMCCFGState(lam=float(smc_lambda), alpha=float(smc_alpha)),
                process=p,
            )
            p.extra_generation_params["Anima CFG guidance mode"] = "SMC-CFG"
            p.extra_generation_params["Anima SMC alpha"] = float(smc_alpha)
            p.extra_generation_params["Anima SMC lambda"] = float(smc_lambda)
            if previous is not None:
                p.extra_generation_params["Anima CFG replaced existing"] = getattr(
                    previous, "__name__", type(previous).__name__
                )
        elif fdg_enabled:
            active_cfg_function = make_fdg_cfg_function(float(fdg_high_scale))
            p.extra_generation_params["Anima CFG guidance mode"] = "FDG (experimental)"
            p.extra_generation_params["Anima FDG detail guidance"] = float(fdg_high_scale)
            if previous is not None:
                p.extra_generation_params["Anima CFG replaced existing"] = getattr(
                    previous, "__name__", type(previous).__name__
                )

        if skim_enable:
            active_cfg_function = _make_skim_cfg_function(
                active_cfg_function,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                skimming_scale=float(skim_scale),
                full_skim_negative=bool(full_negative),
                disable_flipping_filter=bool(disable_flip_filter),
                flip_sigma=flip_sigma,
            )

        if guidance_range_enable:
            range_sigma_start = float(predictor.percent_to_sigma(float(guidance_range_start)))
            range_sigma_end = float(predictor.percent_to_sigma(float(guidance_range_end)))
            active_cfg_function = make_guidance_range_cfg_function(
                active_cfg_function,
                sigma_start=range_sigma_start,
                sigma_end=range_sigma_end,
            )
            p.extra_generation_params["Anima CFG active range"] = (
                f"{float(guidance_range_start):.2f}-{float(guidance_range_end):.2f}"
            )

        if smc_enabled or fdg_enabled or skim_enable or guidance_range_enable:
            unet.set_model_sampler_cfg_function(active_cfg_function, disable_cfg1_optimization=True)

        momentum_samplers = {"Euler", "Anima Flow Euler", "Anima FreeFuse Euler"}
        momentum_enabled = (
            bool(momentum_enable)
            and not (smc_enabled or fdg_enabled)
            and active_sampler in momentum_samplers
        )
        if momentum_enable and not momentum_enabled:
            reason = (
                f"incompatible CFG mode {guidance_mode}"
                if smc_enabled or fdg_enabled
                else f"sampler {active_sampler} performs unsupported intermediate evaluations"
            )
            p.extra_generation_params["Anima Momentum Guidance"] = f"disabled: {reason}"
            logger.warning("Anima Momentum Guidance was disabled: %s", reason)
        if momentum_enabled:
            momentum_state = MomentumGuidanceState(
                momentum=float(momentum_strength),
                ema_decay=float(momentum_ema),
            )
            unet.set_model_sampler_post_cfg_function(
                make_momentum_post_cfg_function(momentum_state, process=p),
                disable_cfg1_optimization=True,
            )
            unet.set_transformer_option("forge_spectrum_force_actual", "anima_momentum")
            p.extra_generation_params["Anima Momentum Guidance"] = True
            p.extra_generation_params["Anima Momentum strength"] = float(momentum_strength)
            p.extra_generation_params["Anima Momentum EMA"] = float(momentum_ema)

        if dcw_enabled:
            dcw_state = DCWState(
                lam=float(dcw_lambda),
                schedule=str(dcw_schedule),
                bands=parse_band_mask(str(dcw_bands)),
            )
            p._anima_dcw_state = dcw_state
            unet.set_model_sampler_post_cfg_function(
                lambda hook_args: _capture_dcw_denoised(
                    p, dcw_state, hook_args["denoised"]
                ),
                disable_cfg1_optimization=True,
            )
            p.extra_generation_params["Anima DCW"] = True
            p.extra_generation_params["Anima DCW lambda"] = float(dcw_lambda)
            p.extra_generation_params["Anima DCW schedule"] = str(dcw_schedule)
            p.extra_generation_params["Anima DCW bands"] = str(dcw_bands)

        p.sd_model.forge_objects.unet = unet

        if skim_enable:
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
