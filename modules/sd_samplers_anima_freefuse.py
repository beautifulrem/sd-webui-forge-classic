"""Two-pass Euler driver used by the native Anima FreeFuse integration."""

from __future__ import annotations

import logging

import torch
from tqdm.auto import trange

from modules.shared import state as webui_state
from modules_forge.packages.k_diffusion.sampling import to_d

logger = logging.getLogger("AnimaFreeFuseSampler")


def _runtime_state(model):
    process = getattr(model, "p", None)
    sd_model = getattr(process, "sd_model", None)
    forge_objects = getattr(sd_model, "forge_objects", None)
    unet = getattr(forge_objects, "unet", None)
    options = getattr(unet, "model_options", {})
    return options.get("transformer_options", {}).get("anima_freefuse_state")


def _reset_stateful_guidance(model) -> None:
    """Discard probe-pass history before the real pass starts."""

    process = getattr(model, "p", None)
    dcw = getattr(process, "_anima_dcw_state", None)
    if dcw is not None:
        dcw.last_denoised = None
        dcw.current_sigma = None

    sd_model = getattr(process, "sd_model", None)
    unet = getattr(getattr(sd_model, "forge_objects", None), "unet", None)
    cfg_function = getattr(unet, "model_options", {}).get("sampler_cfg_function")
    while cfg_function is not None:
        smc = getattr(cfg_function, "_anima_smc_state", None)
        if smc is not None and hasattr(smc, "reset"):
            smc.reset()
        cfg_function = getattr(cfg_function, "_anima_wrapped_cfg", None)


def _restore_online_after_error(model) -> None:
    process = getattr(model, "p", None)
    if process is None or not getattr(process, "_anima_freefuse_enabled", False):
        return
    if getattr(process, "_anima_freefuse_borrowed_online", False):
        return
    from backend.args import dynamic_args

    dynamic_args.online_lora = bool(
        getattr(process, "_anima_freefuse_original_online", False)
    )


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
        from k_diffusion.sampling import sample_euler

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

    if float(s_churn) != 0.0:
        _restore_online_after_error(model)
        raise ValueError(
            "Anima FreeFuse Euler requires S-churn = 0 so both passes share the same deterministic path"
        )

    extra_args = {} if extra_args is None else extra_args
    total_steps = len(sigmas) - 1
    try:
        runtime.configure_steps(total_steps)
    except Exception:
        _restore_online_after_error(model)
        raise
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
    except Exception:
        runtime.abort()
        _restore_online_after_error(model)
        raise

    # CFGDenoiser owns prompt schedule state; the probe pass must be invisible
    # to the real pass. Stateful guidance receives the same reset treatment.
    model.step = original_model_step
    _reset_stateful_guidance(model)
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
    except Exception:
        runtime.abort()
        _restore_online_after_error(model)
        raise
    runtime.complete()
    return result
