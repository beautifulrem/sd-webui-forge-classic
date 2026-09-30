"""Native Anima guidance and correction controls for Forge Neo."""

from __future__ import annotations

import math
import os
import logging
from types import SimpleNamespace

import gradio as gr

from backend import memory_management
from backend.logging import setup_logger
from lib_anima_guidance.modulation import (
    ADAPTER_BLOCKS,
    ADAPTER_MAX_BYTES,
    ADAPTER_SHA256,
    ADAPTER_URL,
    CLIP_MAX_BYTES,
    CLIP_SHA256,
    CLIP_URL,
    ModulationPatch,
    block_layout,
    check_adapter_for_dit,
    clip_prompt_text,
    dit_block_count,
    encode_clip_pooled,
    prepare_modulation_vectors,
    resolve_artifact,
)
from lib_anima_guidance.skim import make_skim_cfg_function
from lib_anima_guidance.smc import SMCCFGState, make_smc_cfg_function
from lib_anima_guidance.dcw import DCWState, parse_band_mask
from lib_anima_guidance.advanced import (
    IGNORES_UNCOND,
    APGState,
    CFGZeroStarState,
    MomentumGuidanceState,
    make_apg_cfg_function,
    make_cfg_zero_star_cfg_function,
    make_fdg_cfg_function,
    make_guidance_range_cfg_function,
    make_guidance_range_uncond_skip,
    make_momentum_post_cfg_function,
    make_tcfg_cfg_function,
    make_zero_init_cfg_function,
    make_zero_init_model_wrapper,
    sigma_value,
)
from lib_anima_guidance.nag import NAGAttentionModifier
from backend.nn.anima_attention import ANIMA_ATTENTION_MODIFIERS
from modules import paths, script_callbacks, scripts, sd_samplers_anima_cns, shared, spectrum_force
from modules.anima_guidance_state import clear_stateful_guidance, register_stateful_guidance
from modules.anima_feature_conflicts import (
    APG_GUIDANCE,
    CFG_ZERO_STAR_GUIDANCE,
    CNS_SAMPLER,
    DCW_INPAINT_MESSAGE,
    FDG_GUIDANCE,
    GUIDANCE_MODES,
    SMC_GUIDANCE,
    STANDARD_GUIDANCE,
    TCFG_GUIDANCE,
    record_conflict_resolution,
    resolve_guidance_conflicts,
)
from modules.anima_presets import register_preset_control
from modules.anima_support import effective_prompt_batch, is_anima_auxiliary_denoiser, is_anima_engine, process_ref, schedule_steps, script_arguments
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaGuidance")
setup_logger(logger)


_REQUIRED = object()

# One row per script argument, in the order ui() returns the components (the
# API's positional order): (name of the control and parameter, infotext key
# or None, preset name or None, default for an argument list stored before the
# control was added, see script_arguments, or _REQUIRED). The local
# modulation file paths have no key: they are never written to infotext, and
# a pasted PNG must not pick which local file is loaded. Keys use only word
# characters, spaces, "-" and "/": parse_generation_parameters reads no other.
SCRIPT_ARGS = (
    ("enable", "Anima guidance enabled", "guidance.enabled", _REQUIRED),
    ("guidance_mode", "Anima CFG guidance mode", "guidance.mode", _REQUIRED),
    ("smc_alpha", "Anima SMC alpha", "guidance.smc_alpha", _REQUIRED),
    ("smc_lambda", "Anima SMC lambda", "guidance.smc_lambda", _REQUIRED),
    ("skim_enable", "Anima Skimmed CFG", "guidance.skim_enabled", _REQUIRED),
    ("skim_scale", "Anima skim CFG", "guidance.skim_scale", _REQUIRED),
    ("full_negative", "Anima skim full negative", "guidance.skim_full_negative", _REQUIRED),
    ("disable_flip_filter", "Anima skim disable flip filter", "guidance.skim_disable_flip", _REQUIRED),
    ("start_percent", "Anima skim start", "guidance.skim_start", _REQUIRED),
    ("end_percent", "Anima skim end", "guidance.skim_end", _REQUIRED),
    ("flip_percent", "Anima skim flip", "guidance.skim_flip", _REQUIRED),
    ("dcw_enable", "Anima DCW", "guidance.dcw_enabled", _REQUIRED),
    ("dcw_lambda", "Anima DCW lambda", "guidance.dcw_lambda", _REQUIRED),
    ("dcw_schedule", "Anima DCW schedule", "guidance.dcw_schedule", _REQUIRED),
    ("dcw_bands", "Anima DCW bands", "guidance.dcw_bands", _REQUIRED),
    ("cns_strength", "Anima CNS strength", "guidance.cns_strength", _REQUIRED),
    ("flow_unipc_solver_type", "Anima Flow UniPC solver type", "guidance.unipc_solver", _REQUIRED),
    ("flow_unipc_disable_corrector_first", "Anima Flow UniPC disabled correctors", "guidance.unipc_disabled_correctors", _REQUIRED),
    ("flow_unipc_thresholding", "Anima Flow UniPC thresholding", "guidance.unipc_thresholding", _REQUIRED),
    ("flow_unipc_dynamic_thresholding_ratio", "Anima Flow UniPC threshold percentile", "guidance.unipc_threshold_ratio", _REQUIRED),
    ("flow_unipc_sample_max_value", "Anima Flow UniPC threshold maximum", "guidance.unipc_threshold_max", _REQUIRED),
    ("flow_pc3_gamma", "Anima Flow PC3 gamma", "guidance.pc3_gamma", _REQUIRED),
    ("flow_pc3_tolerance", "Anima Flow PC3 tolerance", "guidance.pc3_tolerance", _REQUIRED),
    ("modulation_enable", "Anima modulation guidance", "guidance.modulation_enabled", _REQUIRED),
    ("modulation_weight", "Anima modulation weight", "guidance.modulation_weight", _REQUIRED),
    ("modulation_start", "Anima modulation start block", "guidance.modulation_start", _REQUIRED),
    ("modulation_end", "Anima modulation end block", "guidance.modulation_end", _REQUIRED),
    ("modulation_base", "Anima modulation base prompt", None, _REQUIRED),
    ("modulation_positive", "Anima modulation positive direction", None, _REQUIRED),
    ("modulation_negative", "Anima modulation negative direction", None, _REQUIRED),
    ("adapter_mode", "Anima modulation adapter mode", "guidance.modulation_adapter_mode", _REQUIRED),
    ("adapter_path", None, None, _REQUIRED),
    ("clip_mode", "Anima modulation CLIP mode", "guidance.modulation_clip_mode", _REQUIRED),
    ("clip_path", None, None, _REQUIRED),
    ("nag_enable", "Anima NAG", "guidance.nag_enabled", False),
    ("nag_scale", "Anima NAG scale", "guidance.nag_scale", 2.0),
    ("nag_tau", "Anima NAG tau", "guidance.nag_tau", 2.5),
    ("nag_alpha", "Anima NAG alpha", "guidance.nag_alpha", 0.5),
    ("nag_start", "Anima NAG start", "guidance.nag_start", 0.0),
    ("nag_end", "Anima NAG end", "guidance.nag_end", 0.5),
    ("momentum_enable", "Anima Momentum Guidance", "guidance.momentum_enabled", False),
    ("momentum_strength", "Anima Momentum strength", "guidance.momentum_strength", 0.5),
    ("momentum_ema", "Anima Momentum EMA", "guidance.momentum_ema", 0.6),
    ("fdg_high_scale", "Anima FDG detail guidance", "guidance.fdg_detail", 12.0),
    ("guidance_range_enable", "Anima CFG active range", "guidance.range_enabled", False),
    ("guidance_range_start", "Anima CFG range start", "guidance.range_start", 0.0),
    ("guidance_range_end", "Anima CFG range end", "guidance.range_end", 1.0),
    ("cfg_zero_init_steps", "Anima CFG-Zero zero-init steps", "guidance.cfg_zero_init_steps", 1),
    ("apg_eta", "Anima APG eta", "guidance.apg_eta", 0.0),
    ("apg_norm_threshold", "Anima APG norm threshold", "guidance.apg_norm_threshold", 0.0),
    ("apg_momentum", "Anima APG momentum", "guidance.apg_momentum", -0.5),
)
SCRIPT_ARG_NAMES = tuple(row[0] for row in SCRIPT_ARGS)
SCRIPT_ARG_DEFAULTS = {name: default for name, _, _, default in SCRIPT_ARGS if default is not _REQUIRED}
INFOTEXT_KEYS = {name: key for name, key, _, _ in SCRIPT_ARGS if key is not None}

# The CFG range, NAG and Skim windows go through Forge's percent_to_sigma,
# unlike the step fractions of other panels.
MODEL_TIME_INFO = (
    "Fraction of model (sigma) time, not of the step count: with Anima's shift 3, "
    "0.5 is sigma 0.75, so the steps a window covers depend on the scheduler."
)


def _record(p, name: str, value, *, unless_recorded: bool = False) -> None:
    """Write ``value`` to infotext under the key SCRIPT_ARGS gives control ``name``.

    The key is the one ``infotext_fields`` pastes back into that control.
    ``unless_recorded`` keeps a value an earlier pass of the job wrote.
    """

    if unless_recorded:
        p.extra_generation_params.setdefault(INFOTEXT_KEYS[name], value)
    else:
        p.extra_generation_params[INFOTEXT_KEYS[name]] = value


def _record_switched_off(p, args: SimpleNamespace, dcw_enabled: bool) -> None:
    """Record the Standard mode and each sub-feature this pass leaves off.

    Forge's UI paste leaves a control whose key is missing as it is, so
    without these a pasted PNG keeps whatever the panel had switched on.
    Only the switch is written, not the settings of a feature that is off.
    A hires pass whose sampler drops a feature keeps the first pass's value.
    """

    if args.guidance_mode == STANDARD_GUIDANCE:
        _record(p, "guidance_mode", STANDARD_GUIDANCE, unless_recorded=True)
    if not args.skim_enable:
        _record(p, "skim_enable", False, unless_recorded=True)
    if not dcw_enabled:
        _record(p, "dcw_enable", False, unless_recorded=True)
    if not args.modulation_enable:
        _record(p, "modulation_enable", False, unless_recorded=True)
    if not args.nag_enable:
        _record(p, "nag_enable", False, unless_recorded=True)
    if not args.momentum_enable:
        _record(p, "momentum_enable", False, unless_recorded=True)
    if not args.guidance_range_enable:
        _record(p, "guidance_range_enable", False, unless_recorded=True)


def _is_anima(p) -> bool:
    return is_anima_engine(getattr(p, "sd_model", None))


def _dcw_before_denoiser(params) -> None:
    process = getattr(params.denoiser, "p", None)
    if is_anima_auxiliary_denoiser(process):
        return
    state = getattr(process, "_anima_dcw_state", None)
    if state is None:
        return
    if getattr(params.denoiser, "mask", None) is not None:
        # params.x is a blended copy here, not the sampler's latent: the
        # in-place correction would be lost. Installing DCW already checks
        # p.mask; this catches a mask set after that.
        process._anima_dcw_state = None
        params_dict = process.extra_generation_params
        for key in [key for key in params_dict if key.startswith("Anima DCW")]:
            del params_dict[key]
        _record(process, "dcw_enable", False)
        record_conflict_resolution(process, DCW_INPAINT_MESSAGE)
        logger.warning("Anima DCW was disabled: %s", DCW_INPAINT_MESSAGE)
        return
    state.before_denoiser(params.x, sigma_value(params.sigma))


def _capture_dcw_denoised(process, state: DCWState, denoised):
    """Do not let PC3 endpoint probes replace the main-step DCW history."""

    if is_anima_auxiliary_denoiser(process):
        return denoised
    return state.capture_denoised(denoised)


script_callbacks.on_cfg_denoiser(_dcw_before_denoiser, name="anima_dcw_pre_step")


def _stash_nag_negative(params) -> None:
    process = getattr(params.denoiser, "p", None)
    modifier = getattr(process, "_anima_nag_modifier", None)
    if modifier is not None and not is_anima_auxiliary_denoiser(process):
        modifier.set_negative_context(params.text_uncond)


script_callbacks.on_cfg_denoiser(_stash_nag_negative, name="anima_nag_negative_context")


def _cns_strength(args: SimpleNamespace) -> float:
    return min(max(float(args.cns_strength), 0.0), 1.0)


def _prefetch_cns_gamma(p, args: SimpleNamespace) -> None:
    """Fetch the CNS gamma table before sampling when a pass will use it.

    The sampler only reads the downloaded file: a first download inside it
    would stall up to its 30 s timeout with the model on the GPU. Without
    Guidance enabled the sampler runs at its default strength 1.
    """

    samplers = {p.sampler_name}
    if getattr(p, "enable_hr", False):
        samplers.add(getattr(p, "hr_sampler_name", None) or p.sampler_name)
    if CNS_SAMPLER not in samplers or not _is_anima(p):
        return
    strength = _cns_strength(args) if args.enable else float(getattr(p, "cns_strength", 1.0))
    if strength > 0.0:
        sd_samplers_anima_cns.prefetch_gamma(p)


def _install_sampler_tuning(p, args: SimpleNamespace, active_sampler: str) -> None:
    """Settings of one sampler, applied and recorded only while it is active."""

    if active_sampler == CNS_SAMPLER:
        p.cns_strength = _cns_strength(args)
        _record(p, "cns_strength", p.cns_strength)

    if active_sampler == "Anima Flow UniPC2":
        p.flow_unipc_solver_type = str(args.flow_unipc_solver_type)
        p.flow_unipc_disable_corrector_first = max(
            0, int(args.flow_unipc_disable_corrector_first)
        )
        p.flow_unipc_thresholding = bool(args.flow_unipc_thresholding)
        p.flow_unipc_dynamic_thresholding_ratio = min(
            max(float(args.flow_unipc_dynamic_thresholding_ratio), 0.5), 1.0
        )
        p.flow_unipc_sample_max_value = max(
            1.0, float(args.flow_unipc_sample_max_value)
        )
        _record(p, "flow_unipc_solver_type", p.flow_unipc_solver_type)
        _record(p, "flow_unipc_disable_corrector_first", p.flow_unipc_disable_corrector_first)
        _record(p, "flow_unipc_thresholding", p.flow_unipc_thresholding)
        if p.flow_unipc_thresholding:
            _record(p, "flow_unipc_dynamic_thresholding_ratio", p.flow_unipc_dynamic_thresholding_ratio)
            _record(p, "flow_unipc_sample_max_value", p.flow_unipc_sample_max_value)

    if active_sampler == "Anima Flow PC3":
        p.flow_pc3_gamma = min(max(float(args.flow_pc3_gamma), 0.0), 1.0)
        p.flow_pc3_tolerance = min(
            max(float(args.flow_pc3_tolerance), 0.0001), 0.05
        )
        _record(p, "flow_pc3_gamma", p.flow_pc3_gamma)
        _record(p, "flow_pc3_tolerance", p.flow_pc3_tolerance)


def _install_modulation(p, args: SimpleNamespace, unet):
    """A clone of ``unet`` with CLIP modulation guidance, or None if that failed.

    A failure is logged and recorded, and the generation continues without it.
    """

    try:
        dit = unet.model.diffusion_model
        if str(args.adapter_mode) == "Auto-download pinned":
            # Fail before downloading ~400 MB for a model it cannot patch.
            block_layout(ADAPTER_BLOCKS, dit_block_count(dit))
        automatic_adapter = os.path.join(paths.models_path, "Anima", "modulation_guidance", "checkpoint_4000.pt")
        automatic_clip = os.path.join(paths.models_path, "CLIP", "Anima-Mod-Guidance", "clip_l.safetensors")
        # The local paths are typed in the web UI: only files in the models
        # folder, and none under --hide-ui-dir-config (see resolve_local_artifact).
        local_file_policy = dict(
            models_root=paths.models_path,
            local_files_allowed=not shared.cmd_opts.hide_ui_dir_config,
        )
        resolved_adapter = resolve_artifact(
            str(args.adapter_mode),
            str(args.adapter_path),
            automatic_adapter,
            url=ADAPTER_URL,
            sha256=ADAPTER_SHA256,
            maximum_bytes=ADAPTER_MAX_BYTES,
            **local_file_policy,
        )
        check_adapter_for_dit(resolved_adapter, dit)
        resolved_clip = resolve_artifact(
            str(args.clip_mode),
            str(args.clip_path),
            automatic_clip,
            url=CLIP_URL,
            sha256=CLIP_SHA256,
            maximum_bytes=CLIP_MAX_BYTES,
            **local_file_policy,
        )
        base = str(args.modulation_base)
        positive_prompts, negative_prompts = effective_prompt_batch(p)
        base_prompts = [base] * len(positive_prompts) if base.strip() else positive_prompts
        all_clip_prompts = [*base_prompts, *negative_prompts, str(args.modulation_positive), str(args.modulation_negative)]
        steps, hires_steps = schedule_steps(p)
        pooled = encode_clip_pooled(
            [clip_prompt_text(prompt, steps, hires_steps) for prompt in all_clip_prompts],
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
            weight=float(args.modulation_weight),
        )
        patched, effective_start, effective_end = ModulationPatch(
            positive=projected_positive,
            negative=projected_negative,
            scales=scales,
            start_layer=int(args.modulation_start),
            end_layer=int(args.modulation_end),
        ).apply(unet)
        _record(p, "modulation_enable", True)
        _record(p, "modulation_weight", float(args.modulation_weight))
        _record(p, "modulation_start", int(effective_start))
        _record(p, "modulation_end", int(effective_end))
        _record(p, "modulation_positive", str(args.modulation_positive))
        _record(p, "modulation_negative", str(args.modulation_negative))
        _record(p, "adapter_mode", str(args.adapter_mode))
        _record(p, "clip_mode", str(args.clip_mode))
        # Empty too: a pasted PNG must not keep the panel's override.
        _record(p, "modulation_base", base)
        return patched
    except Exception as error:
        logger.exception("Anima modulation guidance was disabled for this generation: %s", error)
        _record(p, "modulation_enable", f"disabled: {type(error).__name__}")
        return None


def _install_nag(p, args: SimpleNamespace, unet, predictor) -> None:
    sigma_start = float(predictor.percent_to_sigma(float(args.nag_start)))
    sigma_end = float(predictor.percent_to_sigma(float(args.nag_end)))

    def report_unbatched_nag(p=process_ref(p)):
        _record(p, "nag_enable", "inactive: no negative context for a positive-only call")
        logger.warning(
            "Anima NAG was skipped: a model call had no negative-prompt context to guide against "
            "(no unconditional rows and no stashed negative context, or an unknown batch layout)"
        )

    nag_modifier = NAGAttentionModifier(
        scale=float(args.nag_scale),
        tau=float(args.nag_tau),
        alpha=float(args.nag_alpha),
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        on_unbatched=report_unbatched_nag,
    )
    unet.append_transformer_option(ANIMA_ATTENTION_MODIFIERS, nag_modifier)
    # Positive-only calls (CFG 1, or branches Forge split for VRAM)
    # get the negative context from the CFG callback instead: no
    # unconditional forward pass is needed just for NAG.
    p._anima_nag_modifier = nag_modifier
    # NAG only changes the output inside its own sigma window.
    spectrum_force.request(unet, spectrum_force.sigma_window(sigma_start, sigma_end))
    _record(p, "nag_enable", True)
    _record(p, "nag_scale", float(args.nag_scale))
    _record(p, "nag_tau", float(args.nag_tau))
    _record(p, "nag_alpha", float(args.nag_alpha))
    _record(p, "nag_start", float(args.nag_start))
    _record(p, "nag_end", float(args.nag_end))


def _cfg_mode_function(p, args: SimpleNamespace, previous):
    """The combine hook of the selected CFG mode; ``previous`` for Standard."""

    mode = args.guidance_mode
    if mode == SMC_GUIDANCE:
        cfg_function = make_smc_cfg_function(
            register_stateful_guidance(p, SMCCFGState(lam=float(args.smc_lambda), alpha=float(args.smc_alpha))),
            process=process_ref(p),
        )
        _record(p, "guidance_mode", SMC_GUIDANCE)
        _record(p, "smc_alpha", float(args.smc_alpha))
        _record(p, "smc_lambda", float(args.smc_lambda))
        return cfg_function
    if mode == FDG_GUIDANCE:
        cfg_function = make_fdg_cfg_function(float(args.fdg_high_scale))
        _record(p, "guidance_mode", FDG_GUIDANCE)
        _record(p, "fdg_high_scale", float(args.fdg_high_scale))
        return cfg_function
    if mode == CFG_ZERO_STAR_GUIDANCE:
        cfg_function = make_cfg_zero_star_cfg_function()
        _record(p, "guidance_mode", CFG_ZERO_STAR_GUIDANCE)
        _record(p, "cfg_zero_init_steps", max(0, int(args.cfg_zero_init_steps)))
        return cfg_function
    if mode == APG_GUIDANCE:
        cfg_function = make_apg_cfg_function(
            register_stateful_guidance(
                p,
                APGState(
                    eta=float(args.apg_eta),
                    norm_threshold=float(args.apg_norm_threshold),
                    momentum=float(args.apg_momentum),
                ),
            ),
            process=process_ref(p),
        )
        _record(p, "guidance_mode", APG_GUIDANCE)
        _record(p, "apg_eta", float(args.apg_eta))
        _record(p, "apg_norm_threshold", float(args.apg_norm_threshold))
        _record(p, "apg_momentum", float(args.apg_momentum))
        return cfg_function
    if mode == TCFG_GUIDANCE:
        cfg_function = make_tcfg_cfg_function()
        _record(p, "guidance_mode", TCFG_GUIDANCE)
        return cfg_function
    return previous


def _install_cfg_hooks(p, args: SimpleNamespace, unet, predictor, cfg_mode_enabled: bool) -> None:
    """The CFG combine mode, wrapped by Skimmed CFG, the CFG range and zero-init."""

    mode = args.guidance_mode
    fdg_enabled = mode == FDG_GUIDANCE
    previous = unet.model_options.get("sampler_cfg_function")
    active_cfg_function = _cfg_mode_function(p, args, previous)
    if previous is not None and active_cfg_function is not previous:
        p.extra_generation_params["Anima CFG replaced existing"] = getattr(
            previous, "__name__", type(previous).__name__
        )

    if args.skim_enable:
        flip_sigma = None
        if 0.0 < float(args.flip_percent) < 1.0:
            flip_sigma = float(predictor.percent_to_sigma(float(args.flip_percent)))
        active_cfg_function = make_skim_cfg_function(
            active_cfg_function,
            sigma_start=float(predictor.percent_to_sigma(float(args.start_percent))),
            sigma_end=float(predictor.percent_to_sigma(float(args.end_percent))),
            skimming_scale=float(args.skim_scale),
            full_skim_negative=bool(args.full_negative),
            disable_flipping_filter=bool(args.disable_flip_filter),
            flip_sigma=flip_sigma,
        )

    if args.guidance_range_enable:
        active_cfg_function = make_guidance_range_cfg_function(
            active_cfg_function,
            sigma_start=float(predictor.percent_to_sigma(float(args.guidance_range_start))),
            sigma_end=float(predictor.percent_to_sigma(float(args.guidance_range_end))),
        )
        _record(p, "guidance_range_enable", True)
        _record(p, "guidance_range_start", float(args.guidance_range_start))
        _record(p, "guidance_range_end", float(args.guidance_range_end))

    zero_init_steps = max(0, int(args.cfg_zero_init_steps)) if mode == CFG_ZERO_STAR_GUIDANCE else 0
    if zero_init_steps > 0:
        # Outermost, so zero-init sees every sigma of the pass (the CFG
        # range does not call its inner hook outside its window). It
        # zeroes only a pass that starts from pure noise.
        zero_init_state = register_stateful_guidance(
            p,
            CFGZeroStarState(
                zero_init_steps,
                start_sigma=float(predictor.percent_to_sigma(0.0)),
            ),
        )
        active_cfg_function = make_zero_init_cfg_function(
            active_cfg_function,
            zero_init_state,
            process=process_ref(p),
        )
        # The same steps skip the model: their predictions are discarded.
        unet.set_model_unet_function_wrapper(
            make_zero_init_model_wrapper(
                unet.model_options.get("model_function_wrapper"),
                zero_init_state,
                process=process_ref(p),
            )
        )
        # Spectrum must neither forecast nor learn from those steps.
        spectrum_force.request(unet, zero_init_state.would_zero_init)

    if cfg_mode_enabled or args.skim_enable or args.guidance_range_enable:
        # At CFG 1 Skimmed CFG, SMC-CFG, CFG-Zero*, APG, TCFG and the CFG range all
        # reduce to the conditional prediction, AND-prompt weights or not
        # (CFG-Zero*'s zero-init reads neither), so only FDG (whose detail band still guides there) keeps
        # Forge from skipping that pass. A wrapped hook from another
        # extension set its own flag already.
        unet.set_model_sampler_cfg_function(
            active_cfg_function,
            disable_cfg1_optimization=bool(fdg_enabled),
        )
    if args.guidance_range_enable:
        # Outside the window only the conditional prediction is used, so
        # the unconditional forward pass is skipped on those steps.
        unet.set_model_sampler_pre_cfg_function(
            make_guidance_range_uncond_skip(active_cfg_function)
        )


def _install_momentum(p, args: SimpleNamespace, unet) -> None:
    momentum_state = register_stateful_guidance(
        p,
        MomentumGuidanceState(
            momentum=float(args.momentum_strength),
            ema_decay=float(args.momentum_ema),
        ),
    )
    # Momentum reads only the combined prediction, so it needs no
    # unconditional pass at CFG 1 (or outside the CFG range).
    unet.set_model_sampler_post_cfg_function(
        make_momentum_post_cfg_function(momentum_state, process=process_ref(p)),
    )
    spectrum_force.request(unet, "anima_momentum")
    _record(p, "momentum_enable", True)
    _record(p, "momentum_strength", float(args.momentum_strength))
    _record(p, "momentum_ema", float(args.momentum_ema))


def _install_dcw(p, args: SimpleNamespace, unet) -> None:
    dcw_state = register_stateful_guidance(
        p,
        DCWState(
            lam=float(args.dcw_lambda),
            schedule=str(args.dcw_schedule),
            bands=parse_band_mask(str(args.dcw_bands)),
        ),
    )
    p._anima_dcw_state = dcw_state

    # DCW only reads the final (post-CFG) prediction, so at CFG 1 it
    # does not need the unconditional pass Forge otherwise skips.
    def dcw_post_cfg(hook_args, p=process_ref(p), dcw_state=dcw_state):
        return _capture_dcw_denoised(p, dcw_state, hook_args["denoised"])

    setattr(dcw_post_cfg, IGNORES_UNCOND, True)
    unet.set_model_sampler_post_cfg_function(dcw_post_cfg)
    _record(p, "dcw_enable", True)
    _record(p, "dcw_lambda", float(args.dcw_lambda))
    _record(p, "dcw_schedule", str(args.dcw_schedule))
    _record(p, "dcw_bands", str(args.dcw_bands))


def _record_skim(p, args: SimpleNamespace) -> None:
    # Every setting, defaults too: a pasted PNG must not keep the panel's.
    _record(p, "skim_enable", True)
    _record(p, "skim_scale", float(args.skim_scale))
    _record(p, "full_negative", bool(args.full_negative))
    _record(p, "disable_flip_filter", bool(args.disable_flip_filter))
    _record(p, "start_percent", float(args.start_percent))
    _record(p, "end_percent", float(args.end_percent))
    _record(p, "flip_percent", float(args.flip_percent))


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
                list(GUIDANCE_MODES),
                value=STANDARD_GUIDANCE,
                label="CFG guidance mode",
                info="SMC-CFG, CFG-Zero*, APG, TCFG and FDG are mutually exclusive combine modes. Skimmed CFG is applied before the selected combine. TCFG has no settings: it drops the part of the negative prediction that does not lie along the direction it shares with the positive one.",
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
                    "Momentum is limited to single-evaluation Euler samplers and requires the Standard CFG mode."
                )
                nag_enable = gr.Checkbox(False, label="Enable NAG attention guidance")
                with gr.Row():
                    nag_scale = gr.Slider(0.0, 8.0, value=2.0, step=0.05, label="NAG scale")
                    nag_tau = gr.Slider(0.1, 10.0, value=2.5, step=0.1, label="NAG normalization tau")
                    nag_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="NAG blend alpha")
                with gr.Row():
                    nag_start = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="NAG start", info=MODEL_TIME_INFO)
                    nag_end = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="NAG end", info=MODEL_TIME_INFO)
                momentum_enable = gr.Checkbox(False, label="Enable Momentum Guidance")
                with gr.Row():
                    momentum_strength = gr.Slider(0.0, 1.5, value=0.5, step=0.01, label="Momentum strength")
                    momentum_ema = gr.Slider(0.0, 0.99, value=0.6, step=0.01, label="Momentum EMA decay")
                fdg_high_scale = gr.Slider(
                    0.0,
                    30.0,
                    value=12.0,
                    step=0.1,
                    label="FDG detail guidance",
                    info="Used only by FDG: the guidance scale of the high-frequency band (the low band uses CFG). Keep it above CFG; below CFG it weakens detail guidance. FDG and DCW exclude each other: in this panel the one turned on last wins and locks the other; in API calls with both on, FDG wins. FDG is disabled with the CNS sampler.",
                )
                cfg_zero_init_steps = gr.Slider(
                    0,
                    10,
                    value=1,
                    step=1,
                    label="CFG-Zero* zero-init steps",
                    info="Used only by CFG-Zero*. The first N sampler steps output zero velocity (the latent is left unchanged) when the pass starts from pure noise; hires and img2img passes are not zeroed. 0 keeps only the optimized scale.",
                )
                with gr.Row():
                    apg_eta = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0,
                        step=0.01,
                        label="APG eta",
                        info="Used only by APG. Weight of the guidance part parallel to the prediction (the oversaturating part); 1 keeps it like CFG.",
                    )
                    apg_norm_threshold = gr.Slider(
                        0.0,
                        100.0,
                        value=0.0,
                        step=0.5,
                        label="APG norm threshold",
                        info="Used only by APG. Caps the L2 norm of the guidance update over the whole latent (grows with resolution); 0 disables.",
                    )
                    apg_momentum = gr.Slider(
                        -0.95,
                        0.95,
                        value=-0.5,
                        step=0.01,
                        label="APG momentum",
                        info="Used only by APG. Negative values are the paper's reverse momentum; 0 disables.",
                    )
                guidance_range_enable = gr.Checkbox(False, label="Limit CFG to a sampling range")
                with gr.Row():
                    guidance_range_start = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.01, label="CFG range start", info=MODEL_TIME_INFO
                    )
                    guidance_range_end = gr.Slider(
                        0.0, 1.0, value=1.0, step=0.01, label="CFG range end", info=MODEL_TIME_INFO
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
                start_percent = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Skim start", info=MODEL_TIME_INFO)
                end_percent = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Skim end", info=MODEL_TIME_INFO)
                flip_percent = gr.Slider(
                    0.0,
                    1.0,
                    value=0.0,
                    step=0.01,
                    label="Filter flip point",
                    info="0 disables timed inversion of the flipping filter. A model (sigma) time fraction, like Skim start/end.",
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
                        info="proj(base) + weight x (proj(positive direction) - proj(negative direction)), where proj is the adapter's projection of the CLIP-L pooled vector.",
                    )
                    modulation_start = gr.Slider(0, 63, value=0, step=1, label="Start block")
                    modulation_end = gr.Slider(
                        -1,
                        63,
                        value=-1,
                        step=1,
                        label="End block",
                        info="-1 means the final block. Start/end count the adapter's blocks (the pinned one has Anima 2B's 28); on 2.9B/3.8B they select the same original blocks.",
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
                    adapter_path = gr.Textbox(
                        "",
                        label="Local adapter .pt path",
                        info="Inside the models folder (absolute, or relative to it).",
                        **shared.hide_dirs,
                    )
                with gr.Row():
                    clip_mode = gr.Dropdown(
                        ["Auto-download pinned", "Local file"],
                        value="Auto-download pinned",
                        label="CLIP-L encoder",
                    )
                    clip_path = gr.Textbox(
                        "",
                        label="Local CLIP-L .safetensors path",
                        info="Inside the models folder (absolute, or relative to it).",
                        **shared.hide_dirs,
                    )

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
                elif state.mode == FDG_GUIDANCE:
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

        # Each control is the local variable named in SCRIPT_ARGS.
        controls = locals()
        fields = [(controls[name], key, preset_name) for name, key, preset_name, _ in SCRIPT_ARGS]
        for component, _, preset_name in fields:
            if preset_name is not None:
                register_preset_control(self.tabname, preset_name, component)
        self.infotext_fields = [PasteField(component, key) for component, key, _ in fields if key is not None]
        self.paste_field_names = [key for _, key, _ in fields if key is not None]
        return [component for component, _, _ in fields]

    @staticmethod
    def _arguments(values) -> SimpleNamespace:
        # Positional script args are named by SCRIPT_ARGS, the table ui()
        # returns its components from, so their order is written only once.
        # Forge's own keywords (x, noise, c, uc) are not script args.
        return script_arguments("Anima guidance", SCRIPT_ARG_NAMES, values, defaults=SCRIPT_ARG_DEFAULTS)

    def process(self, p, *values):
        # Once per job, before any pass samples.
        _prefetch_cns_gamma(p, self._arguments(values))

    def process_before_every_sampling(self, p, *values, **kwargs):
        self._process_before_every_sampling(p, self._arguments(values))

    def _process_before_every_sampling(self, p, args: SimpleNamespace):
        # The global pre-denoiser hook applies whatever DCW state p holds: a
        # pass without DCW (e.g. hires after it was dropped for the sampler)
        # must not keep correcting towards the previous pass's history.
        p._anima_dcw_state = None
        p._anima_nag_modifier = None
        clear_stateful_guidance(p)
        active_sampler = (
            (getattr(p, "hr_sampler_name", None) or p.sampler_name)
            if getattr(p, "is_hr_pass", False)
            else p.sampler_name
        )
        # SMC with alpha <= 0 installs nothing, so it must not evict Momentum.
        if str(args.guidance_mode) == SMC_GUIDANCE and float(args.smc_alpha) <= 0.0:
            args.guidance_mode = STANDARD_GUIDANCE
        conflict_state = resolve_guidance_conflicts(
            sampler=str(active_sampler),
            mode=str(args.guidance_mode),
            momentum=bool(args.momentum_enable),
            dcw=bool(args.dcw_enable),
            inpaint_mask=getattr(p, "mask", None) is not None,
        )
        args.guidance_mode = conflict_state.mode
        args.momentum_enable = conflict_state.momentum
        args.dcw_enable = conflict_state.dcw
        # Every mode but Standard installs a combine hook: SMC with alpha <= 0
        # became Standard above, and the resolver only resets modes to it.
        cfg_mode_enabled = args.guidance_mode != STANDARD_GUIDANCE
        dcw_enabled = bool(args.dcw_enable) and not math.isclose(float(args.dcw_lambda), 0.0)
        if not args.enable or not _is_anima(p):
            return

        # Every _record() key is the control's ``infotext_fields`` key, so
        # pasted parameters restore the panel.
        _record(p, "enable", True)

        if conflict_state.messages:
            resolution = "; ".join(conflict_state.messages)
            record_conflict_resolution(p, *conflict_state.messages)
            logger.warning("Anima compatibility resolver: %s", resolution)

        _record_switched_off(p, args, dcw_enabled)
        _install_sampler_tuning(p, args, active_sampler)

        if not (
            args.skim_enable
            or cfg_mode_enabled
            or dcw_enabled
            or args.modulation_enable
            or args.nag_enable
            or args.momentum_enable
            or args.guidance_range_enable
        ):
            return

        unet = p.sd_model.forge_objects.unet
        patched = _install_modulation(p, args, unet) if args.modulation_enable else None
        unet = unet.clone() if patched is None else patched
        predictor = unet.model.predictor
        if args.nag_enable:
            _install_nag(p, args, unet, predictor)
        _install_cfg_hooks(p, args, unet, predictor, cfg_mode_enabled)
        # resolve_guidance_conflicts() already dropped Momentum for unsupported
        # samplers and every mode but Standard CFG (or reset the mode to
        # Standard when Momentum was the last switch), and recorded why.
        if args.momentum_enable:
            _install_momentum(p, args, unet)
        if dcw_enabled:
            _install_dcw(p, args, unet)
        p.sd_model.forge_objects.unet = unet

        if args.skim_enable:
            _record_skim(p, args)
