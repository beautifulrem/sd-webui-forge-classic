from __future__ import annotations

import pytest
import torch

from backend import attention as backends
from backend.nn.anima_attention import run_anima_attention


def _mask_dropping_backend(calls):
    """Stands in for SageAttention's CUDA kernels: runs SDPA without the mask."""

    def attention(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
        calls.append(mask)
        return backends.attention_pytorch(q, k, v, heads, mask=None, skip_reshape=skip_reshape)

    return attention


def _inputs():
    generator = torch.Generator().manual_seed(0)
    # [batch, tokens, heads, head_dim]
    q = torch.randn(1, 4, 2, 8, generator=generator)
    k = torch.randn(1, 5, 2, 8, generator=generator)
    v = torch.randn(1, 5, 2, 8, generator=generator)
    bias = torch.zeros(1, 1, 4, 5)
    bias[..., 2:] = -1e4  # a FreeFuse-style additive bias: hide keys 2..4
    return q, k, v, bias


def _run(q, k, v, base, mask=None):
    return run_anima_attention(q, k, v, heads=2, base_attention=base, transformer_options={}, is_self_attention=False, mask=mask)


@pytest.mark.parametrize("backend_name", ["attention_sage", "attention_flash"])
def test_masked_calls_bypass_a_backend_that_drops_the_mask(monkeypatch, backend_name):
    calls = []
    dropping = _mask_dropping_backend(calls)
    monkeypatch.setattr(backends, backend_name, dropping)
    q, k, v, bias = _inputs()

    masked = _run(q, k, v, dropping, mask=bias)
    assert calls == []  # went to SDPA, not the active backend
    expected = backends.attention_pytorch(q.movedim(-2, 1), k.movedim(-2, 1), v.movedim(-2, 1), 2, mask=bias, skip_reshape=True)
    assert torch.allclose(masked, expected)
    only_visible = backends.attention_pytorch(q.movedim(-2, 1), k[:, :2].movedim(-2, 1), v[:, :2].movedim(-2, 1), 2, skip_reshape=True)
    assert torch.allclose(masked, only_visible, atol=1e-5)  # the bias took effect

    plain = _run(q, k, v, dropping)
    assert calls == [None]  # unmasked calls keep the active backend
    assert not torch.allclose(plain, masked, atol=1e-3)


def test_backends_that_apply_masks_are_kept_for_masked_calls():
    calls = []

    def honouring(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
        calls.append(mask)
        return backends.attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape)

    q, k, v, bias = _inputs()
    _run(q, k, v, honouring, mask=bias)
    assert len(calls) == 1 and calls[0] is bias
