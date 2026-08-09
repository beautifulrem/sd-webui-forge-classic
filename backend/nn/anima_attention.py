"""Composable attention modifiers for Anima's native attention path."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


ANIMA_ATTENTION_MODIFIERS = "anima_attention_modifiers"


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
    ``[batch, tokens, heads, head_dim]`` tensors.  Keeping this seam inside the
    native Anima attention module lets temporary cross-attention wrappers such
    as Regional, Artist Mixer, and FreeFuse compose without replacing one
    another.
    """

    options = transformer_options or {}

    def base(q_value, k_value, v_value, *, mask=None):
        q_heads = q_value.movedim(-2, 1)
        k_heads = k_value.movedim(-2, 1)
        v_heads = v_value.movedim(-2, 1)
        return base_attention(
            q_heads,
            k_heads,
            v_heads,
            heads,
            mask=mask,
            skip_reshape=True,
            transformer_options=options,
        )

    modifiers: Sequence[Callable[..., torch.Tensor]] = tuple(
        options.get(ANIMA_ATTENTION_MODIFIERS, ())
    )
    call = base
    for modifier in reversed(modifiers):
        next_attention = call

        def call(
            q_value,
            k_value,
            v_value,
            *,
            mask=None,
            _modifier=modifier,
            _next=next_attention,
        ):
            return _modifier(
                _next,
                q_value,
                k_value,
                v_value,
                mask=mask,
                transformer_options=options,
                is_self_attention=is_self_attention,
            )

    return call(q, k, v, mask=mask)
