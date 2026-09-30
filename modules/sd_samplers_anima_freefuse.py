"""Two-pass Euler driver used by the native Anima FreeFuse integration."""

from __future__ import annotations

import logging

import torch
from tqdm.auto import trange

from modules.anima_guidance_state import reset_stateful_guidance
from modules.sd_samplers_anima import denoise_step, euler_step
from k_diffusion.sampling import sample_euler

logger = logging.getLogger("AnimaFreeFuseSampler")


def _runtime_state(model):
    process = getattr(model, "p", None)
    sd_model = getattr(process, "sd_model", None)
    forge_objects = getattr(sd_model, "forge_objects", None)
    unet = getattr(forge_objects, "unet", None)
    options = getattr(unet, "model_options", {})
    return options.get("transformer_options", {}).get("anima_freefuse_state")


@torch.no_grad()
def sample_anima_freefuse_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Collect masks in a partial pass, then restart Euler from the same noise."""

    runtime = _runtime_state(model)
    if runtime is None:
        logger.warning(
            "Anima FreeFuse Euler selected without an active FreeFuse panel; using ordinary Euler."
        )
        return sample_euler(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
        )

    # On any error, FreeFuse's Script.cleanup (run by process_images' finally)
    # restores the online-LoRA mode it switched on for this generation.
    if float(s_churn) != 0.0:
        raise ValueError(
            "Anima FreeFuse Euler requires S-churn = 0 so both passes share the same deterministic path"
        )

    extra_args = {} if extra_args is None else extra_args
    total_steps = len(sigmas) - 1
    runtime.configure_steps(total_steps)
    collect_steps = min(total_steps, runtime.collect_step + 1)

    # Both passes start from ``x``. The probe gets its own copy: denoiser
    # callbacks (on_cfg_denoiser, DCW) may change the model input in place.
    # CFGDenoiser raises on Interrupt / Skip at the start of each model call.
    runtime.begin_collect()
    probe = x.clone()
    try:
        for index in range(collect_steps):
            runtime.current_step = index
            denoised = denoise_step(model, probe, sigmas, index, extra_args, None)
            probe = euler_step(probe, denoised, sigmas[index], sigmas[index + 1])
        runtime.finish_collection()
    except BaseException:  # also Forge's InterruptedException (Interrupt / Skip)
        runtime.abort()
        raise

    # The probe pass must be invisible to the real pass. CFGDenoiser's prompt
    # schedule step needs nothing: denoise_step sets it before every call.
    # Stateful guidance carries state across calls and is reset.
    reset_stateful_guidance(model)
    runtime.begin_generate()

    result = x
    try:
        for index in trange(total_steps, disable=disable):
            runtime.current_step = index
            denoised = denoise_step(model, result, sigmas, index, extra_args, callback)
            result = euler_step(result, denoised, sigmas[index], sigmas[index + 1])
    except BaseException:
        runtime.abort()
        raise
    runtime.complete()
    return result
