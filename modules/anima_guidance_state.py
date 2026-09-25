"""Lifecycle helpers shared by Anima samplers and guidance extensions."""

from __future__ import annotations


def reset_stateful_guidance(model) -> None:
    """Discard auxiliary/probe-pass history before a real sampling pass."""

    process = getattr(model, "p", None)
    dcw = getattr(process, "_anima_dcw_state", None)
    if dcw is not None:
        dcw.last_denoised = None
        dcw.current_sigma = None

    sd_model = getattr(process, "sd_model", None)
    unet = getattr(getattr(sd_model, "forge_objects", None), "unet", None)
    model_options = getattr(unet, "model_options", {})
    cfg_function = model_options.get("sampler_cfg_function")
    while cfg_function is not None:
        smc = getattr(cfg_function, "_anima_smc_state", None)
        if smc is not None and hasattr(smc, "reset"):
            smc.reset()
        cfg_function = getattr(cfg_function, "_anima_wrapped_cfg", None)

    for post_function in model_options.get("sampler_post_cfg_function", ()):
        momentum = getattr(post_function, "_anima_momentum_state", None)
        if momentum is not None and hasattr(momentum, "reset"):
            momentum.reset()
