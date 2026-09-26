import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import SpectrumNode, _call_samples


def _branch_signatures(conditioning, branches):
    return _call_samples(torch.ones(1), conditioning, branches)[1]


def _reference(conditioning, branches):
    """The previous per-branch implementation: one host transfer per branch."""

    def signature(context):
        flattened = context.detach().reshape(-1)
        if flattened.numel() == 0:
            sample = ()
        else:
            stride = max(1, flattened.numel() // 16)
            values = flattened[::stride][:16].float().cpu().tolist()
            sample = tuple(round(float(value), 6) for value in values)
        return tuple(context.shape), str(context.dtype), sample

    if not isinstance(conditioning, dict) or not branches:
        return None
    context = conditioning.get("c_crossattn")
    if not torch.is_tensor(context) or context.shape[0] % len(branches):
        return None
    return {key: signature(chunk) for key, chunk in zip(branches, context.chunk(len(branches), dim=0))}


CASES = [
    ((2, 77, 64), torch.float32, [1, 0]),
    ((2, 5, 3), torch.float16, [0, 1]),  # fewer than 16 values per branch
    ((1, 512, 1024), torch.bfloat16, [0]),
    ((4, 7, 9), torch.float32, [0, 1]),  # two rows per branch
    ((2, 17, 1), torch.float32, [1, 0]),  # stride does not divide the branch
    ((2, 0, 8), torch.float32, [0, 1]),  # empty context
]


@pytest.mark.parametrize("shape,dtype,branches", CASES)
def test_signatures_match_the_per_branch_reference(shape, dtype, branches):
    context = torch.randn(shape, generator=torch.Generator().manual_seed(0)).to(dtype)
    conditioning = {"c_crossattn": context}
    assert _branch_signatures(conditioning, branches) == _reference(conditioning, branches)


def test_non_contiguous_context_matches_the_reference():
    context = torch.randn(64, 2, 77).permute(1, 2, 0)
    assert not context.is_contiguous()
    conditioning = {"c_crossattn": context}
    assert _branch_signatures(conditioning, [0, 1]) == _reference(conditioning, [0, 1])


def test_unsplittable_inputs_return_none():
    assert _branch_signatures(None, [0]) is None
    assert _branch_signatures({"c_crossattn": torch.zeros(2, 3)}, []) is None
    assert _branch_signatures({"c_crossattn": torch.zeros(3, 3)}, [0, 1]) is None
    assert _branch_signatures({}, [0, 1]) is None


def test_all_branches_share_one_host_transfer(monkeypatch):
    calls = []
    original = torch.Tensor.tolist

    def counting_tolist(self):
        calls.append(tuple(self.shape))
        return original(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    signatures = _branch_signatures({"c_crossattn": torch.randn(2, 77, 64)}, [1, 0])
    assert set(signatures) == {0, 1}
    assert len(calls) == 1


def test_each_model_call_makes_one_host_transfer(monkeypatch):
    """The sigma rides along with the signature gather: one copy per call."""

    class FakePatcher:
        def __init__(self):
            self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
            self.model_options = {}
            self.wrapper = None

        def clone(self):
            return self

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper

    patcher = SpectrumNode.patch(
        FakePatcher(), steps=10, weight=1.0, degree=1, lam=1e-6, window_size=2,
        flex_window=0.0, warmup_steps=5, stop_caching_step=1.0, tail_actual_steps=1,
        history_size=3, schedule="Window", refresh_ratio=0.0, sea_beta=2.0,
        compat_policy="Conservative", verbose=False, sea_cache_dir="unused",
        sea_cache_context={}, process=SimpleNamespace(),
    )
    transfers = []
    for name in ("tolist", "item", "__float__"):
        original = getattr(torch.Tensor, name)

        def counting(self, *args, _name=name, _original=original):
            transfers.append(_name)
            return _original(self, *args)

        monkeypatch.setattr(torch.Tensor, name, counting)

    context = torch.randn(2, 77, 64)
    for step in range(3):  # warmup: every call is an actual forward
        before = len(transfers)
        patcher.wrapper(
            lambda x, timestep, **_kwargs: x,
            {
                "input": torch.zeros(2, 4),
                "timestep": torch.tensor([1.0 - step * 0.1]),
                "c": {"c_crossattn": context},
                "cond_or_uncond": [0, 1],
            },
        )
        assert transfers[before:] == ["tolist"]


def test_call_sigma_is_read_exactly():
    sigma = torch.tensor([0.123456789], dtype=torch.float64)
    value, signatures = _call_samples(sigma, {"c_crossattn": torch.randn(2, 3, 4)}, [0, 1])
    assert value == float(sigma[0])
    assert set(signatures) == {0, 1}
    assert _call_samples(torch.tensor([0.5, 0.5]), None, [0])[0] == 0.5
    assert _call_samples(0.25, {"c_crossattn": torch.randn(1, 3)}, [0])[0] == 0.25
