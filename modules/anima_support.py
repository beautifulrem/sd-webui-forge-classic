"""Shared capability checks for Anima-only integrations."""

from __future__ import annotations

from contextlib import contextmanager


_AUXILIARY_DENOISER_ATTRIBUTE = "_anima_auxiliary_denoiser"


def is_anima_engine(model) -> bool:
    return bool(getattr(model, "is_anima_engine", False))


def is_anima_auxiliary_denoiser(process) -> bool:
    """Return whether the current denoiser call is a solver-only probe."""

    return bool(getattr(process, _AUXILIARY_DENOISER_ATTRIBUTE, False))


def effective_prompt_batch(process) -> tuple[list[str], list[str]]:
    """Return the positive and negative prompts used by the current pass."""

    if bool(getattr(process, "is_hr_pass", False)):
        positive = getattr(process, "hr_prompts", None) or getattr(
            process, "prompts", None
        )
        negative = getattr(process, "hr_negative_prompts", None) or getattr(
            process, "negative_prompts", None
        )
    else:
        positive = getattr(process, "prompts", None)
        negative = getattr(process, "negative_prompts", None)

    positive = list(positive or [getattr(process, "prompt", "")])
    negative = list(negative or [getattr(process, "negative_prompt", "")])
    if len(positive) == 1 and len(negative) > 1:
        positive *= len(negative)
    if len(negative) == 1 and len(positive) > 1:
        negative *= len(positive)
    if len(positive) != len(negative):
        raise RuntimeError(
            "Anima positive/negative prompt batch sizes differ for the current pass"
        )
    return [str(value) for value in positive], [str(value) for value in negative]


@contextmanager
def anima_auxiliary_denoiser(model):
    """Mark a solver-only model call without leaking process-global state."""

    process = getattr(model, "p", None)
    if process is None:
        yield
        return

    missing = object()
    previous = getattr(process, _AUXILIARY_DENOISER_ATTRIBUTE, missing)
    setattr(process, _AUXILIARY_DENOISER_ATTRIBUTE, True)
    try:
        yield
    finally:
        if previous is missing:
            delattr(process, _AUXILIARY_DENOISER_ATTRIBUTE)
        else:
            setattr(process, _AUXILIARY_DENOISER_ATTRIBUTE, previous)


def require_anima_flow_denoiser(model, feature: str) -> None:
    """Reject similarly shaped RF models before applying Anima-specific math."""

    process = getattr(model, "p", None)
    engine = getattr(process, "sd_model", None)
    if not is_anima_engine(engine):
        raise RuntimeError(f"{feature} requires an active Anima checkpoint")

    predictor = getattr(getattr(model, "inner_model", None), "predictor", None)
    from backend.modules.k_prediction import PredictionDiscreteFlow

    if not isinstance(predictor, PredictionDiscreteFlow):
        raise RuntimeError(
            f"{feature} requires Anima's discrete rectified-flow predictor"
        )




NEGPIP_MASK_KEY = "negpip_mask"


def split_conditioning(conds):
    """Split ``get_learned_conditioning`` output into ``(crossattn, mask)``.

    While NegPiP is active the output is ``{"crossattn", "c_negpip_mask"}``:
    negative-weight tokens are sign-restored in ``crossattn`` and NegPiP's
    attention hook negates only V with the mask. Callers that build their own
    contexts (Regional, Artist Mixer) keep the mask and pass it to the forward
    under ``NEGPIP_MASK_KEY`` so those tokens get the same semantics as in the
    main prompt. Other outputs are returned unchanged with ``mask=None``.
    """

    if not isinstance(conds, dict):
        return conds, None
    return conds.get("crossattn", conds.get("c_crossattn")), conds.get("c_negpip_mask")


def negpip_mask_for(context, transformer_options):
    """NegPiP V-mask from ``transformer_options`` if it matches ``context``.

    The mask describes one specific context's tokens; any other context (a
    different length, or a batch it cannot tile onto) must not use it.
    """

    import torch

    mask = (transformer_options or {}).get(NEGPIP_MASK_KEY)
    if not torch.is_tensor(mask) or not torch.is_tensor(context):
        return None
    if mask.ndim != context.ndim or mask.shape[1] != context.shape[1]:
        return None
    if mask.shape[0] != context.shape[0]:
        if context.shape[0] % mask.shape[0]:
            return None
        mask = mask.repeat(context.shape[0] // mask.shape[0], *([1] * (mask.ndim - 1)))
    return mask.to(context)
