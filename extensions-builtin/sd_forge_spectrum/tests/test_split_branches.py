"""Forge splits cond and uncond into separate model calls when they do not fit
one batch: two wrapper calls per sigma, uncond-only then cond-only
(``calc_cond_uncond_batch`` reverses its batch order to [1, 0])."""

import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import SpectrumNode

STEPS = 10
UNCOND_FIRST = (1, 0)


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
        _FakePatcher(), steps=STEPS, weight=1.0, degree=1, lam=1e-6, window_size=2,
        flex_window=0.0, warmup_steps=3, stop_caching_step=1.0, tail_actual_steps=1,
        history_size=3, schedule="Window", refresh_ratio=0.0, sea_beta=2.0,
        compat_policy=policy, verbose=False, sea_cache_dir="unused",
        sea_cache_context={}, process=process,
    )
    return patcher, process._spectrum_states[0]


def _trend(branch, step):
    return (1.0 + branch) * step  # a linear trend per branch


CONTEXTS = {0: torch.ones(1, 4, 8), 1: torch.zeros(1, 4, 8)}


def _sigma(step):
    return torch.tensor([1.0 - step * 0.09])


def _run_split(policy, calls_per_step=1):
    patcher, state = _patch(policy)
    actual = {0: [], 1: []}
    outputs = {0: [], 1: []}
    forecaster_keys = []

    for step in range(STEPS):
        # calls_per_step > 1 repeats the pair at the same sigma (a second
        # model evaluation of the step), which must not be a new branch step.
        for repeat in range(calls_per_step):
            for branch in UNCOND_FIRST:

                def model_function(x, timestep, branch=branch, step=step, repeat=repeat, **_kwargs):
                    actual[branch].append(step if repeat == 0 else (step, repeat))
                    return torch.full_like(x, _trend(branch, step))

                args = {"input": torch.zeros(1, 1), "timestep": _sigma(step), "c": {"c_crossattn": CONTEXTS[branch]}, "cond_or_uncond": [branch]}
                output = patcher.wrapper(model_function, args)
                if repeat == 0:
                    outputs[branch].append(output)
        forecaster_keys.append(sorted(state.forecasters))
    return actual, outputs, forecaster_keys


def _batched_actual_steps(policy):
    patcher, _state = _patch(policy)
    actual = []
    for step in range(STEPS):

        def model_function(x, timestep, step=step, **_kwargs):
            actual.append(step)
            return torch.cat([torch.full_like(x[:1], _trend(branch, step)) for branch in UNCOND_FIRST])

        context = torch.cat([CONTEXTS[branch] for branch in UNCOND_FIRST])
        patcher.wrapper(model_function, {"input": torch.zeros(2, 1), "timestep": _sigma(step), "c": {"c_crossattn": context}, "cond_or_uncond": list(UNCOND_FIRST)})
    return actual


class SplitBranchTests(unittest.TestCase):
    def test_split_calls_forecast_both_branches_on_the_batched_schedule(self):
        for policy in ("Conservative", "Legacy / fastest"):
            with self.subTest(policy=policy):
                actual, outputs, forecaster_keys = _run_split(policy)

                # Each branch's first call at a sigma feeds its own forecaster,
                # so both are forecast, on the steps one CFG batch would be.
                self.assertEqual(actual[0], actual[1])
                self.assertEqual(actual[0], _batched_actual_steps(policy))
                self.assertEqual(actual[0][:3], [0, 1, 2])  # warmup
                self.assertIn(STEPS - 1, actual[0])  # tail
                self.assertTrue([step for step in range(STEPS) if step not in actual[0]], actual[0])
                self.assertTrue(all(keys == [0, 1] for keys in forecaster_keys[:STEPS - 1]), forecaster_keys)
                # (The tail step, STEPS - 1, forecasts nothing and frees the history.)
                self.assertEqual(forecaster_keys[STEPS - 1], [])
                for branch in UNCOND_FIRST:
                    for step in range(STEPS):
                        # Forecasts continue each branch's own trend.
                        torch.testing.assert_close(outputs[branch][step], torch.full((1, 1), _trend(branch, step)), rtol=0, atol=1e-3)

    def test_repeated_split_calls_at_a_sigma_run_actual_without_feeding_history(self):
        # Conservative: a branch's later calls at the same sigma are repeats
        # (e.g. a second model evaluation), actual and not recorded again.
        actual, outputs, _keys = _run_split("Conservative", calls_per_step=2)
        single, _outputs, _keys = _run_split("Conservative")

        for branch in UNCOND_FIRST:
            self.assertEqual([step for step in actual[branch] if isinstance(step, int)], single[branch])
            self.assertEqual([step for step in actual[branch] if not isinstance(step, int)], [(step, 1) for step in range(STEPS)])
            for step in range(STEPS):
                torch.testing.assert_close(outputs[branch][step], torch.full((1, 1), _trend(branch, step)), rtol=0, atol=1e-3)

    def test_legacy_forecasts_each_and_sub_prompt_on_its_own(self):
        # AND sub-prompts share cond_or_uncond 0 in one batch: each chunk has
        # its own trend and must not be merged with the other's on cached steps.
        branches = [0, 0, 1]
        slopes = torch.tensor([1.0, 5.0, 2.0]).reshape(3, 1)
        context = torch.stack([torch.full((4, 8), float(row)) for row in range(3)])
        patcher, _state = _patch("Legacy / fastest")
        actual = []
        for step in range(STEPS):

            def model_function(x, timestep, step=step, **_kwargs):
                actual.append(step)
                return slopes * step

            output = patcher.wrapper(model_function, {"input": torch.zeros(3, 1), "timestep": _sigma(step), "c": {"c_crossattn": context}, "cond_or_uncond": branches})
            torch.testing.assert_close(output, slopes * step, rtol=0, atol=1e-3)
        self.assertTrue([step for step in range(STEPS) if step not in actual], actual)


if __name__ == "__main__":
    unittest.main()
