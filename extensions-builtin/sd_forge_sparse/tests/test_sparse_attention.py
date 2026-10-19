"""The attention patch itself, on the CPU: Sol-Attn is a GPU kernel (NVIDIA
sm_80+ or ROCm with matrix cores), so a stand-in with the kernel's calling
convention computes exact attention and records its calls. The patch is
installed as the enabled panel installs it and reached through the
backend's own attention entry point (``backend.attention.attention_pytorch``
and its ``optimized_attention_override`` hook)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import remi_port_harness as harness
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sparse.py"
HEADS, DIM_HEAD, TOKENS = 2, 128, 64


@pytest.fixture(scope="module")
def sparse():
    from remi.testing import webui_state

    with webui_state():
        yield harness.load_script(SCRIPT, "sparse_attention_under_test")


@pytest.fixture
def kernel(sparse, monkeypatch):
    """The Sol-Attn stand-in: exact attention over ``(b, n, heads, d)``;
    every call recorded. The CPU passes for a supported GPU."""
    calls = []

    def sol_attn(q, k, v, *, tau, scale=None, token_aug=0):
        calls.append({"shape": tuple(q.shape), "dtype": q.dtype, "tau": tau, "scale": scale, "token_aug": token_aug})
        out = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2).float() for t in (q, k, v)), scale=scale)
        return out.transpose(1, 2).to(q.dtype)

    monkeypatch.setattr(sparse.ck, "sol_attn", sol_attn)
    monkeypatch.setattr(sparse.SparseAttnPatch, "ineligible", staticmethod(lambda q, k, v, dim_head: None))
    monkeypatch.setattr(sparse, "unavailable_reason", lambda device: None)
    return calls


class _Predictor:
    def percent_to_sigma(self, percent):
        return 1.0 - percent


def _enabled_unet(sparse, **values):
    """The unet after the enabled panel's process_before_every_sampling."""

    class Unet:
        model = SimpleNamespace(predictor=_Predictor())

        def __init__(self):
            self.model_options = {"transformer_options": {}}

        def clone(self):
            return self

    script = sparse.SparseAttentionForForge()
    p = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=Unet())), extra_generation_params={}, comment=lambda text: None)
    args = harness.script_layout(script).values_to_args({"enable": True, "min_tokens": 0, **values})
    script.process_before_every_sampling(p, *args)
    return p.sd_model.forge_objects.unet


def _qkv(skip_reshape=False, tokens=TOKENS, seed=0):
    generator = torch.Generator().manual_seed(seed)
    if skip_reshape:
        return [torch.randn(1, HEADS, tokens, DIM_HEAD, generator=generator) for _ in range(3)]
    return [torch.randn(1, tokens, HEADS * DIM_HEAD, generator=generator) for _ in range(3)]


def _attention(unet, q, k, v, *, sigma=0.5, **kwargs):
    from backend.attention import attention_pytorch

    options = {**unet.model_options["transformer_options"], "sigmas": torch.tensor([sigma])}
    return attention_pytorch(q, k, v, HEADS, transformer_options=options, **kwargs)


@pytest.mark.parametrize("layout", [{}, {"skip_reshape": True}, {"skip_reshape": True, "skip_output_reshape": True}], ids=["tokens", "heads", "heads-out"])
def test_eligible_attention_runs_through_the_sparse_kernel(sparse, kernel, layout):
    from backend.attention import attention_pytorch

    unet = _enabled_unet(sparse, tau=1.5, extra_tokens=64)
    q, k, v = _qkv(layout.get("skip_reshape", False))

    out = _attention(unet, q, k, v, **layout)

    assert kernel == [{"shape": (1, TOKENS, HEADS, DIM_HEAD), "dtype": torch.bfloat16, "tau": 1.5, "scale": None, "token_aug": 64}]
    # the kernel's (b, n, heads, d) result in the caller's layout and dtype: the dense result
    dense = attention_pytorch(q, k, v, HEADS, **layout)
    assert out.shape == dense.shape and out.dtype == q.dtype
    torch.testing.assert_close(out, dense, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("case", "values", "call"),
    [
        ("mask", {}, {"mask": torch.zeros(TOKENS, TOKENS)}),
        ("sigma above the range", {"percent": [0.25, 0.75]}, {"sigma": 0.9}),
        ("sigma below the range", {"percent": [0.25, 0.75]}, {"sigma": 0.1}),
        ("below the token threshold", {"min_tokens": TOKENS * 2}, {}),
        ("dense block", {"dense_blocks": "0-2, 7"}, {"transformer_options_block": 7}),
    ],
)
def test_ineligible_calls_stay_dense(sparse, kernel, case, values, call):
    from backend.attention import attention_pytorch

    unet = _enabled_unet(sparse, **values)
    q, k, v = _qkv()
    call = dict(call)
    if (block := call.pop("transformer_options_block", None)) is not None:
        unet.model_options["transformer_options"]["block_index"] = block
    out = _attention(unet, q, k, v, **call)
    assert kernel == [], case
    mask = call.get("mask")
    torch.testing.assert_close(out, attention_pytorch(q, k, v, HEADS, mask=mask))


def test_a_previous_override_still_runs_for_dense_calls(sparse, kernel):
    """Sparse attention wraps an override another feature installed first."""
    seen = []

    def previous(func, *args, **kwargs):
        seen.append(args[3])
        return func(*args, **kwargs)

    chained = sparse.make_attention_override(sparse.SparseAttnPatch(1.25, 1.0, 0.0, TOKENS * 2, 0, set(), False), previous)
    q, k, v = _qkv()
    from backend.attention import attention_pytorch

    out = attention_pytorch(q, k, v, HEADS, transformer_options={"optimized_attention_override": chained})
    assert seen == [HEADS] and kernel == []
    torch.testing.assert_close(out, attention_pytorch(q, k, v, HEADS))


def _meta(device="cuda", dtype=torch.bfloat16, shape=(1, TOKENS, HEADS, DIM_HEAD)):
    return SimpleNamespace(device=SimpleNamespace(type=device), dtype=dtype, shape=shape)


@pytest.mark.parametrize(
    ("q", "k", "dim_head", "available", "reason"),
    [
        (_meta(), _meta(), DIM_HEAD, True, None),
        (_meta(device="mps"), _meta(), DIM_HEAD, True, "CUDA device is required"),
        (_meta(), _meta(), DIM_HEAD, False, "sol_attn is not available for this GPU"),
        (_meta(dtype=torch.float64), _meta(dtype=torch.float64), DIM_HEAD, True, 'dtype "torch.float64" is not supported'),
        (_meta(shape=(1, TOKENS, HEADS, 64)), _meta(shape=(1, TOKENS, HEADS, 64)), 64, True, "head_dim (64) != 128"),
        (_meta(), _meta(shape=(1, 77, HEADS, DIM_HEAD)), DIM_HEAD, True, "cross-attention"),
        (_meta(), _meta(dtype=torch.float16), DIM_HEAD, True, "mixed dtypes: torch.bfloat16/torch.float16/torch.bfloat16"),
    ],
)
def test_the_kernel_rules(sparse, monkeypatch, q, k, dim_head, available, reason):
    monkeypatch.setattr(sparse.ck, "sol_attn_is_available", lambda device: available)
    assert sparse.SparseAttnPatch.ineligible(q, k, q, dim_head) == reason


def test_the_panel_states_the_gpu_requirement(sparse):
    schema = harness.script_layout(sparse.SparseAttentionForForge()).schema()
    note = schema["children"][0]
    assert note["type"] == "markdown" and "GPU only" in note["text"] and "MPS" in note["text"]
