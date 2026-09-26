"""``compute_kv`` (the K/V projection NAG uses for a stashed negative context)
must honour NegPiP's V-only negation exactly like ``compute_qkv``."""

import torch

from backend import operations
from backend.nn.anima import SelfCrossAttention


def test_compute_kv_value_context_matches_compute_qkv():
    generator = torch.Generator().manual_seed(7)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        attention = SelfCrossAttention(32, context_dim=16, n_heads=2, head_dim=8)
    attention.load_state_dict({k: torch.randn(v.shape, generator=generator) for k, v in attention.state_dict().items()})
    x = torch.randn(2, 5, 32, generator=generator)
    context = torch.randn(2, 4, 16, generator=generator)
    mask = torch.tensor([1.0, -1.0, 1.0, 1.0]).reshape(1, 4, 1).expand(2, 4, 1)

    with torch.inference_mode():
        _, k, v = attention.compute_qkv(x, context, value_context=context * mask)
        k2, v2 = attention.compute_kv(context, value_context=context * mask)
        _, v_plain = attention.compute_kv(context)

    assert torch.equal(k, k2) and torch.equal(v, v2)
    assert not torch.equal(v2, v_plain)


def test_compute_qkv_cross_attention_matches_the_plain_projections():
    """compute_qkv takes its cross-attention K/V from compute_kv; the result must
    stay the plain q/k/v projections + norms (V from ``value_context``)."""
    generator = torch.Generator().manual_seed(11)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        attention = SelfCrossAttention(32, context_dim=16, n_heads=2, head_dim=8)
    attention.load_state_dict({k: torch.randn(v.shape, generator=generator) for k, v in attention.state_dict().items()})
    x = torch.randn(2, 5, 32, generator=generator)
    context = torch.randn(2, 4, 16, generator=generator)
    value_context = -context

    with torch.inference_mode():
        heads = (2, 8)
        expected_q = attention.q_norm(attention.q_proj(x).unflatten(-1, heads))
        expected_k = attention.k_norm(attention.k_proj(context).unflatten(-1, heads))
        for vc in (None, value_context):
            q, k, v = attention.compute_qkv(x, context, value_context=vc)
            expected_v = attention.v_proj(context if vc is None else vc).unflatten(-1, heads)
            assert torch.equal(q, expected_q) and torch.equal(k, expected_k) and torch.equal(v, expected_v)
