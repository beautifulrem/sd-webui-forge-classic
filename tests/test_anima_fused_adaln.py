import pytest
import torch
from torch import nn

from backend.nn import anima


def _reference(x, norm, y, z):
    return torch.addcmul(z, norm(x), 1 + y)


def _inputs(x_dtype=torch.float32, mod_dtype=torch.bfloat16):
    torch.manual_seed(0)
    x = torch.randn(2, 1, 3, 4, 16, dtype=x_dtype)
    y = torch.randn(2, 1, 1, 1, 16, dtype=mod_dtype)
    z = torch.randn(2, 1, 1, 1, 16, dtype=mod_dtype)
    return x, nn.LayerNorm(16, elementwise_affine=False, eps=1e-6), y, z


@pytest.fixture
def fused_calls(monkeypatch):
    calls = []
    real = anima.ck.adaln

    def spy(x, scale, shift, eps=1e-6):
        calls.append((x.dtype, scale.dtype, shift.dtype, eps))
        return real(x, scale, shift, eps)

    monkeypatch.setattr(anima.ck, "adaln", spy)
    monkeypatch.setattr(anima, "_has_fused_adaln", lambda x: True)
    return calls


def test_cpu_keeps_the_native_path_bit_for_bit():
    x, norm, y, z = _inputs()

    assert not anima._has_fused_adaln(x)
    assert torch.equal(anima._fn(x, norm, y, z), _reference(x, norm, y, z))


@pytest.mark.parametrize("x_dtype, mod_dtype", [(torch.float32, torch.bfloat16), (torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16)])
def test_fused_adaln_matches_the_native_formula(fused_calls, x_dtype, mod_dtype):
    x, norm, y, z = _inputs(x_dtype, mod_dtype)

    out = anima._fn(x, norm, y, z)
    ref = _reference(x, norm, y, z)

    assert fused_calls == [(x_dtype, x_dtype, x_dtype, 1e-6)]
    assert out.dtype == ref.dtype and out.shape == ref.shape
    tol = 1e-5 if x_dtype == torch.float32 else 3e-2
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)


def test_fused_adaln_is_skipped_when_it_would_downcast(fused_calls):
    x, norm, y, z = _inputs(torch.bfloat16, torch.float32)

    out = anima._fn(x, norm, y, z)

    assert fused_calls == []
    assert out.dtype == torch.float32


def test_fused_adaln_is_skipped_for_affine_norms(fused_calls):
    x, _, y, z = _inputs()
    norm = nn.LayerNorm(16, eps=1e-6)

    anima._fn(x, norm, y, z)

    assert fused_calls == []


def test_fused_adaln_is_skipped_while_compiling(fused_calls, monkeypatch):
    x, norm, y, z = _inputs()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    anima._fn(x, norm, y, z)

    assert fused_calls == []
