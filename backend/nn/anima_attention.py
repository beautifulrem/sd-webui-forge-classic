"""Composable attention modifiers for Anima's native attention path."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch


ANIMA_ATTENTION_MODIFIERS = "anima_attention_modifiers"
# Set for cross-attention while modifiers are active: ``project_kv(context)``
# projects another text context with this layer's K/V weights (e.g. NAG's
# negative prompt when CFG 1 skips the unconditional branch).
ANIMA_PROJECT_KV = "anima_project_kv"


def active_attention_modifiers(transformer_options: dict[str, Any]) -> tuple:
    """The modifiers that act on this model call.

    A modifier may expose ``active(transformer_options) -> bool`` (e.g. NAG
    outside its sigma window); one without it is always active.
    """

    return tuple(
        modifier
        for modifier in transformer_options.get(ANIMA_ATTENTION_MODIFIERS) or ()
        if getattr(modifier, "active", None) is None or modifier.active(transformer_options)
    )


def mask_honouring_attention(base_attention: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """``base_attention``, or PyTorch SDPA when it would not apply a mask.

    SageAttention's CUDA kernels ignore ``attn_mask`` (FreeFuse's concept bias
    would silently do nothing) and FlashAttention rejects it (an error log and
    an SDPA fallback on every biased call), so masked calls on those backends
    go to ``attention_pytorch`` directly; unmasked calls keep the fast one."""

    from backend import attention as backends

    if base_attention is backends.attention_sage or base_attention is backends.attention_flash:
        return backends.attention_pytorch
    return base_attention


def run_anima_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    base_attention: Callable[..., torch.Tensor],
    transformer_options: dict[str, Any] | None,
    is_self_attention: bool,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Anima attention through the ordered per-generation modifier chain.

    Modifiers receive a ``next_attention`` callable and the unflattened
    ``[batch, tokens, heads, head_dim]`` tensors, and chain in list order
    without replacing one another.  NAG is the only modifier today; the
    cross-attention wrappers (Regional, Artist Mixer, FreeFuse, ...) replace the
    attention module's ``forward`` through ``modules.forward_override``
    instead, which composes them separately. A call that carries a mask runs
    on a backend that applies it (:func:`mask_honouring_attention`).
    """

    options = transformer_options or {}

    def base(q_value, k_value, v_value, *, mask=None):
        q_heads = q_value.movedim(-2, 1)
        k_heads = k_value.movedim(-2, 1)
        v_heads = v_value.movedim(-2, 1)
        attention = base_attention if mask is None else mask_honouring_attention(base_attention)
        return attention(
            q_heads,
            k_heads,
            v_heads,
            heads,
            mask=mask,
            skip_reshape=True,
            transformer_options=options,
        )

    modifiers: Sequence[Callable[..., torch.Tensor]] = options.get(ANIMA_ATTENTION_MODIFIERS) or ()
    call = base
    for modifier in reversed(tuple(modifiers)):
        call = partial(modifier, call, transformer_options=options, is_self_attention=is_self_attention)
    return call(q, k, v, mask=mask)
