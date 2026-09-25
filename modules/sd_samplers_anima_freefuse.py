"""Two-pass Euler driver used by the native Anima FreeFuse integration."""

from __future__ import annotations

import logging

import torch
from tqdm.auto import trange

from modules.shared import state as webui_state
from modules.anima_guidance_state import reset_stateful_guidance
from k_diffusion.sampling import sample_euler, to_d

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
    s_in = x.new_ones([x.shape[0]])
    initial_x = x.detach().clone()
    original_model_step = int(getattr(model, "step", 0))

    runtime.begin_collect()
    probe = initial_x.detach().clone()
    try:
        for index in range(collect_steps):
            if webui_state.interrupted or webui_state.skipped:
                from modules.sd_samplers_common import InterruptedException

                raise InterruptedException
            runtime.current_step = index
            sigma = sigmas[index]
            denoised = model(probe, sigma * s_in, **extra_args)
            probe = probe + to_d(probe, sigma, denoised) * (sigmas[index + 1] - sigma)
        runtime.finish_collection()
    except BaseException:  # also Forge's InterruptedException (Interrupt / Skip)
        runtime.abort()
        raise

    # CFGDenoiser owns prompt schedule state; the probe pass must be invisible
    # to the real pass. Stateful guidance receives the same reset treatment.
    model.step = original_model_step
    reset_stateful_guidance(model)
    runtime.begin_generate()

    result = initial_x
    try:
        for index in trange(total_steps, disable=disable):
            runtime.current_step = index
            sigma = sigmas[index]
            denoised = model(result, sigma * s_in, **extra_args)
            if callback is not None:
                callback(
                    {
                        "x": result,
                        "i": index,
                        "sigma": sigma,
                        "sigma_hat": sigma,
                        "denoised": denoised,
                    }
                )
            result = result + to_d(result, sigma, denoised) * (
                sigmas[index + 1] - sigma
            )
    except BaseException:
        runtime.abort()
        raise
    runtime.complete()
    return result
