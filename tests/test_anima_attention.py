from __future__ import annotations

import torch

from backend.nn.anima_attention import ANIMA_ATTENTION_MODIFIERS, run_anima_attention


def test_anima_attention_modifier_chain_is_ordered_and_mask_aware():
    seen = []

    def base(q, k, v, heads, *, mask, **kwargs):
        seen.append(("base", mask))
        return q.movedim(1, 2).flatten(2)

    def outer(next_attention, q, k, v, *, mask, **kwargs):
        seen.append(("outer", mask))
        return next_attention(q, k, v, mask=mask) + 1

    def inner(next_attention, q, k, v, *, mask, **kwargs):
        seen.append(("inner", mask))
        return next_attention(q, k, v, mask=mask) * 2

    q = torch.ones(1, 2, 1, 3)
    mask = torch.ones(1, 1, 2, 2)
    result = run_anima_attention(
        q,
        q,
        q,
        heads=1,
        base_attention=base,
        transformer_options={ANIMA_ATTENTION_MODIFIERS: [outer, inner]},
        is_self_attention=False,
        mask=mask,
    )
    assert [entry[0] for entry in seen] == ["outer", "inner", "base"]
    assert all(entry[1] is mask for entry in seen)
    assert torch.equal(result, torch.full((1, 2, 3), 3.0))
