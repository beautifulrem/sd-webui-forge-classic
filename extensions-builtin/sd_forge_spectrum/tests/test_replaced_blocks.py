"""A branch whose blocks are replaced over part of the pass: Skip Layer
Guidance (``modules/processing_scripts/slg.py``, upstream be6fa5d4) runs a
Moss-Image step as two calls between its start and end percents, cond alone
and then uncond alone with ``patches_replace`` skipping DiT blocks, and one
CFG batch outside them. The uncond features on the two sides of the switch
come from different forwards, so its history must not span it."""

import pathlib
import sys
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import SpectrumNode, _replaced_blocks

STEPS = 16
DEGREE = 1
COND, UNCOND = 0, 1
SLG_STEPS = range(8)  # start 0.0, end 0.5


def _skip(args, _extra):
    return args


SLG_OPTIONS = {"patches_replace": {"dit": {("single_block", 22): _skip, ("single_block", 24): _skip}}}


class _FakePatcher:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
        self.model_options = {}
        self.wrapper = None

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, wrapper):
        self.wrapper = wrapper


def _patch(policy):
    process = SimpleNamespace()
    patcher = SpectrumNode.patch(
        _FakePatcher(), steps=STEPS, weight=1.0, degree=DEGREE, lam=1e-6, window_size=2,
        flex_window=0.0, warmup_steps=2, stop_caching_step=1.0, tail_actual_steps=1,
        history_size=DEGREE + 2, schedule="Window", refresh_ratio=0.0, sea_beta=2.0,
        compat_policy=policy, verbose=False, sea_cache_dir="unused",
        sea_cache_context={}, process=process,
    )
    return patcher


CONTEXTS = {COND: torch.ones(1, 4, 8), UNCOND: torch.zeros(1, 4, 8)}


def _truth(branch, step):
    # Skipping layers shifts the uncond output while SLG is active.
    return 1.0 * step if branch == COND else 2.0 * step + (40.0 if step in SLG_STEPS else 0.0)


def _call(patcher, step, branches, transformer_options, actual):
    def model_function(x, timestep, **_kwargs):
        for branch in branches:
            actual[branch].append(step)
        return torch.cat([torch.full_like(x[:1], _truth(branch, step)) for branch in branches])

    context = torch.cat([CONTEXTS[branch] for branch in branches])
    c = {"c_crossattn": context, "transformer_options": {**transformer_options, "cond_or_uncond": list(branches)}}
    sigma = torch.tensor([1.0 - step * 0.06])
    return patcher.wrapper(model_function, {"input": torch.zeros(len(branches), 1), "timestep": sigma, "c": c, "cond_or_uncond": list(branches)})


def _run(policy):
    patcher = _patch(policy)
    actual = {COND: [], UNCOND: []}
    errors = []
    for step in range(STEPS):
        if step in SLG_STEPS:
            outputs = {COND: _call(patcher, step, [COND], {}, actual), UNCOND: _call(patcher, step, [UNCOND], SLG_OPTIONS, actual)}
        else:
            batch = _call(patcher, step, [UNCOND, COND], {}, actual)
            outputs = {UNCOND: batch[:1], COND: batch[1:]}
        errors.extend((step, branch, abs(float(output[0, 0]) - _truth(branch, step))) for branch, output in outputs.items())
    return actual, errors


def test_replaced_blocks_are_listed_in_order():
    assert _replaced_blocks({}) == () and _replaced_blocks(None) == () and _replaced_blocks({"patches_replace": {"dit": {}}}) == ()
    assert _replaced_blocks(SLG_OPTIONS) == (("dit", "('single_block', 22)"), ("dit", "('single_block', 24)"))


def test_skip_layer_guidance_window_does_not_leak_into_the_forecasts():
    for policy in ("Conservative", "Legacy / fastest"):
        actual, errors = _run(policy)
        assert all(error < 1e-3 for *_where, error in errors), (policy, [item for item in errors if item[2] >= 1e-3])
        # Both branches still cache, inside and outside the SLG window.
        for branch in (COND, UNCOND):
            assert [step for step in SLG_STEPS if step not in actual[branch]], (policy, branch, actual)
            assert [step for step in range(SLG_STEPS.stop, STEPS) if step not in actual[branch]], (policy, branch, actual)
        # The uncond branch rebuilds its history after the switch.
        assert [step for step in actual[UNCOND] if step >= SLG_STEPS.stop][: DEGREE + 2] == list(range(SLG_STEPS.stop, SLG_STEPS.stop + DEGREE + 2)), (policy, actual)
