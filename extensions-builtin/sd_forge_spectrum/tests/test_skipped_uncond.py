"""The unconditional branch missing from some steps: "Ignore Negative Prompt
during Early Steps" (``skip_early_cond``: no uncond for the first steps) and
"Skip Negative Prompt during Later Steps" (``s_min_uncond``: no uncond on every
other step below a sigma, or on all of them with ``s_min_uncond_all``).
``CFGDenoiser`` sets the CFG scale to 1 there and ``sampling_function`` drops
the uncond, so the model call holds the cond chunk alone.

Upstream 34022261 removed Spectrum's refusal to run with these options; Remi's
per-branch history copes: a branch missing from a step loses its forecaster
(no forecast across the gap), the cond branch keeps caching."""

import pathlib
import sys
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import SpectrumNode

STEPS = 16
DEGREE = 1
COND, UNCOND = 0, 1


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
    return patcher, process._spectrum_states[0]


CONTEXTS = {COND: torch.ones(1, 4, 8), UNCOND: torch.zeros(1, 4, 8)}


def _sigma(step):
    return torch.tensor([1.0 - step * 0.06])


def _run(uncond_steps, policy="Conservative"):
    """Sample STEPS steps, with the uncond branch only on ``uncond_steps``.

    The cond output follows one linear trend; the uncond output a linear trend
    that jumps by 50 after every step it is missing from, so a forecast built
    from features on both sides of a gap is far off."""

    patcher, state = _patch(policy)
    actual = {COND: [], UNCOND: []}
    errors = []
    forecaster_keys = []
    for step in range(STEPS):
        gaps = sum(1 for earlier in range(step) if earlier not in uncond_steps)
        truth = {COND: 1.0 * step, UNCOND: 2.0 * step + 50.0 * gaps}
        branches = [UNCOND, COND] if step in uncond_steps else [COND]

        def model_function(x, timestep, step=step, branches=branches, truth=truth, **_kwargs):
            for branch in branches:
                actual[branch].append(step)
            return torch.cat([torch.full_like(x[:1], truth[branch]) for branch in branches])

        context = torch.cat([CONTEXTS[branch] for branch in branches])
        args = {"input": torch.zeros(len(branches), 1), "timestep": _sigma(step), "c": {"c_crossattn": context}, "cond_or_uncond": branches}
        output = patcher.wrapper(model_function, args)
        for row, branch in enumerate(branches):
            errors.append((step, branch, abs(float(output[row, 0]) - truth[branch])))
        forecaster_keys.append({key: list(forecaster.times) for key, forecaster in state.forecasters.items()})
    return actual, errors, forecaster_keys


def _cached(actual, branch, present):
    return [step for step in present if step not in actual[branch]]


def test_early_steps_without_uncond_cache_the_cond_branch():
    early = 6  # skip_early_cond: no uncond for steps 0..5
    uncond_steps = set(range(early, STEPS))
    for policy in ("Conservative", "Legacy / fastest"):
        actual, errors, keys = _run(uncond_steps, policy)
        assert all(error < 1e-3 for *_where, error in errors), (policy, errors)
        # The cond branch caches on its own before the uncond one appears.
        assert _cached(actual, COND, range(early)), (policy, actual)
        # The uncond branch starts its history when it appears: actual until
        # its forecaster is ready (degree + 2 features), then it caches too.
        assert actual[UNCOND][: DEGREE + 2] == list(range(early, early + DEGREE + 2)), (policy, actual)
        assert _cached(actual, UNCOND, uncond_steps), (policy, actual)
        assert sorted(keys[early - 1]) == [COND] and keys[early][UNCOND] == [float(early)]


def test_later_steps_without_uncond_never_forecast_across_the_gap():
    # s_min_uncond: uncond missing on every other step from step 8 on.
    uncond_steps = set(range(8)) | {step for step in range(8, STEPS) if step % 2 == 0}
    actual, errors, keys = _run(uncond_steps)
    assert all(error < 1e-3 for *_where, error in errors), errors
    # After the first gap the uncond branch never has degree + 2 features in a
    # row again: every one of its later steps is actual, none is forecast
    # from history on the other side of a gap.
    assert [step for step in range(9, STEPS) if step in uncond_steps] == [step for step in actual[UNCOND] if step >= 9]
    # The uncond history is dropped at the step after the one it missed and
    # starts again from the step it returns on.
    assert keys[10][UNCOND] == [10.0] and keys[12][UNCOND] == [12.0]
    # The cond branch keeps its history and still caches in that stretch.
    assert _cached(actual, COND, range(8, STEPS - 1)), actual


def test_uncond_dropped_for_all_later_steps():
    # s_min_uncond with s_min_uncond_all: no uncond from step 9 on.
    uncond_steps = set(range(9))
    actual, errors, keys = _run(uncond_steps)
    assert all(error < 1e-3 for *_where, error in errors), errors
    assert all(UNCOND not in step_keys for step_keys in keys[10:])
    assert _cached(actual, COND, range(9, STEPS - 1)), actual
