"""The attention patch itself, on the CPU. The fused Sol-Attn kernels need a
GPU (NVIDIA sm_80+ or ROCm with matrix cores), so for that path a stand-in
with the kernel's calling convention computes exact attention and records its
calls; elsewhere the patch runs Comfy Kitchen's reference implementation,
which is tested here for real. The patch is installed as the enabled panel
installs it and reached through the backend's own attention entry point
(``backend.attention.attention_pytorch`` and its ``optimized_attention_override``
hook)."""

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
    monkeypatch.setattr(sparse.SparseAttnPatch, "ineligible", staticmethod(lambda q, k, v, dim_head, fused=True: None))
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
    ("q", "k", "dim_head", "available", "reason", "reference_reason"),
    [
        (_meta(), _meta(), DIM_HEAD, True, None, None),
        (_meta(device="mps"), _meta(), DIM_HEAD, True, "CUDA device is required", None),
        (_meta(), _meta(), DIM_HEAD, False, "sol_attn is not available for this GPU", None),
        (_meta(dtype=torch.float64), _meta(dtype=torch.float64), DIM_HEAD, True, 'dtype "torch.float64" is not supported', 'dtype "torch.float64" is not supported'),
        (_meta(shape=(1, TOKENS, HEADS, 64)), _meta(shape=(1, TOKENS, HEADS, 64)), 64, True, "head_dim (64) != 128", "head_dim (64) != 128"),
        (_meta(), _meta(shape=(1, 77, HEADS, DIM_HEAD)), DIM_HEAD, True, "cross-attention", "cross-attention"),
        (_meta(), _meta(dtype=torch.float16), DIM_HEAD, True, "mixed dtypes: torch.bfloat16/torch.float16/torch.bfloat16", "mixed dtypes: torch.bfloat16/torch.float16/torch.bfloat16"),
    ],
)
def test_the_kernel_rules(sparse, monkeypatch, q, k, dim_head, available, reason, reference_reason):
    """The fused kernels need a capable GPU; the reference takes the same
    calls on any device (so both make the same image)."""
    monkeypatch.setattr(sparse.ck, "sol_attn_is_available", lambda device: available)
    assert sparse.SparseAttnPatch.ineligible(q, k, q, dim_head) == reason
    assert sparse.SparseAttnPatch.ineligible(q, k, q, dim_head, False) == reference_reason


def test_the_panel_states_where_it_is_faster(sparse):
    schema = harness.script_layout(sparse.SparseAttentionForForge()).schema()
    note = schema["children"][0]
    assert note["type"] == "markdown" and "GPUs only" in note["text"] and "MPS" in note["text"] and "reference implementation" in note["text"]


# region The reference implementation (no fused kernel: MPS, the CPU, older GPUs)


def _structured_qkv(tokens=256, heads=HEADS, batch=1, skip_reshape=False, seed=0):
    """Queries and keys that share a low-rank structure, so attention is
    peaked (as in a real model) and the routing keeps some key blocks exact
    and pools the others."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(batch, tokens, heads, 8, generator=generator)
    projection = torch.randn(8, DIM_HEAD, generator=generator) * 1.5
    q = base @ projection + 0.3 * torch.randn(batch, tokens, heads, DIM_HEAD, generator=generator)
    k = base @ projection + 0.3 * torch.randn(batch, tokens, heads, DIM_HEAD, generator=generator)
    v = torch.randn(batch, tokens, heads, DIM_HEAD, generator=generator)
    if skip_reshape:
        return [t.transpose(1, 2).contiguous() for t in (q, k, v)]
    return [t.reshape(batch, tokens, heads * DIM_HEAD) for t in (q, k, v)]


def _comfy_kitchen_reference(q, k, v, tau):
    """Comfy Kitchen's own reference over all heads at once, ``(B, T, H, D)``."""
    import comfy_kitchen

    return comfy_kitchen.registry.get_implementation("sol_attn", backend="eager")(q, k, v, tau=tau)


def test_the_reference_is_comfy_kitchens_in_head_chunks(sparse, monkeypatch):
    """Chunking the heads (to bound memory) gives Comfy Kitchen's result for
    the whole tensor: every (batch, head) is independent."""
    q, k, v = (t.view(2, 256, HEADS * 2, DIM_HEAD) for t in _structured_qkv(batch=2, heads=HEADS * 2))
    monkeypatch.setattr(sparse, "REFERENCE_SCORE_BYTES", 256 * 256 * 4)  # one head at a time
    torch.testing.assert_close(sparse.reference_sol_attn(q, k, v, tau=1.25), _comfy_kitchen_reference(q, k, v, 1.25))
    monkeypatch.setattr(sparse, "REFERENCE_SCORE_BYTES", 2**30)  # all at once
    torch.testing.assert_close(sparse.reference_sol_attn(q, k, v, tau=1.25), _comfy_kitchen_reference(q, k, v, 1.25))


def test_the_reference_is_reproducible(sparse, monkeypatch):
    """The same inputs give the same output on every run, so a pasted
    infotext regenerates the image: no order-dependent accumulation
    (``scatter_add_`` sums in a varying order on MPS and CUDA, and a flipped
    routing decision changed the real model's image), and on a ragged last
    block too."""

    def unordered(*args, **kwargs):
        raise AssertionError("scatter_add_ adds in a varying order on GPUs")

    q, k, v = (t.view(1, 250, HEADS, DIM_HEAD) for t in _structured_qkv(tokens=250))
    with monkeypatch.context() as patched:
        patched.setattr(torch.Tensor, "scatter_add_", unordered)
        first = sparse.reference_sol_attn(q, k, v, tau=1.25)
        assert torch.equal(first, sparse.reference_sol_attn(q, k, v, tau=1.25))
    torch.testing.assert_close(first, _comfy_kitchen_reference(q, k, v, 1.25))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS (Apple Silicon)")
def test_the_reference_is_reproducible_on_mps(sparse):
    """Where it runs instead of the kernels: bit-identical repeats on MPS."""

    generator = torch.Generator().manual_seed(0)  # unstructured: many block scores near the routing threshold
    q, k, v = (torch.randn(1, 1024, 4, DIM_HEAD, generator=generator).to("mps", torch.bfloat16) for _ in range(3))
    outputs = [sparse.reference_sol_attn(q, k, v, tau=1.25) for _ in range(6)]
    assert all(torch.equal(out, outputs[0]) for out in outputs[1:])


@pytest.mark.parametrize("layout", [{}, {"skip_reshape": True}, {"skip_reshape": True, "skip_output_reshape": True}], ids=["tokens", "heads", "heads-out"])
def test_without_the_kernels_the_patch_runs_the_reference(sparse, monkeypatch, layout):
    """On the CPU (as on MPS): the enabled panel patches attention, eligible
    calls give Comfy Kitchen's sparse result (not the dense one), in the
    caller's layout and dtype, and the job says the reference ran."""
    from backend.attention import attention_pytorch
    from modules import devices

    monkeypatch.setattr(devices, "device", torch.device("cpu"))
    assert sparse.unavailable_reason(devices.device) == "CUDA device is required"
    unet = _enabled_unet(sparse, tau=1.25)
    q, k, v = _structured_qkv(skip_reshape=layout.get("skip_reshape", False))

    out = _attention(unet, q, k, v, **layout)

    dense = attention_pytorch(q, k, v, HEADS, **layout)
    assert out.shape == dense.shape and out.dtype == q.dtype
    as_bthd = (lambda t: t.transpose(1, 2)) if layout.get("skip_reshape") else (lambda t: t.view(1, 256, HEADS, DIM_HEAD))
    expected = _comfy_kitchen_reference(*(as_bthd(t) for t in (q, k, v)), 1.25)
    expected = expected.transpose(1, 2) if layout.get("skip_output_reshape") else expected.reshape(1, 256, HEADS * DIM_HEAD)
    torch.testing.assert_close(out, expected)
    # sparse: some key blocks were pooled, so the result is not dense attention
    assert (out - dense).abs().max() > 1e-2


def test_the_reference_falls_back_to_dense_when_it_cannot_run(sparse, monkeypatch):
    """Out of memory (or too many tokens for the reference): that call stays
    dense, with a warning, instead of failing the job."""
    from backend.attention import attention_pytorch
    from modules import devices

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("MPS backend out of memory")

    monkeypatch.setattr(devices, "device", torch.device("cpu"))
    monkeypatch.setattr(sparse, "reference_sol_attn", fail)
    warnings = []
    monkeypatch.setattr(sparse.logger, "warning", warnings.append)
    unet = _enabled_unet(sparse)
    q, k, v = _structured_qkv()
    out = _attention(unet, q, k, v)
    _attention(unet, q, k, v)
    torch.testing.assert_close(out, attention_pytorch(q, k, v, HEADS))
    assert sum("reference sparse attention failed" in message for message in warnings) == 1  # once, not per call


# endregion
