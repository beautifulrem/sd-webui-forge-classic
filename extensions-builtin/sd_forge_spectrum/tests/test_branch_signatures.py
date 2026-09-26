import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import _branch_signatures


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
