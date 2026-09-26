"""Forge splits cond and uncond into separate model calls when they do not fit
one batch: two wrapper calls per sigma, cond-only then uncond-only."""

import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import SpectrumNode

STEPS = 10


class _FakePatcher:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
        self.model_options = {}
        self.wrapper = None

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, wrapper):
        self.wrapper = wrapper


def _run_split(policy):
    process = SimpleNamespace()
    patcher = SpectrumNode.patch(
        _FakePatcher(), steps=STEPS, weight=1.0, degree=1, lam=1e-6, window_size=2,
        flex_window=0.0, warmup_steps=3, stop_caching_step=1.0, tail_actual_steps=1,
        history_size=3, schedule="Window", refresh_ratio=0.0, sea_beta=2.0,
        compat_policy=policy, verbose=False, sea_cache_dir="unused",
        sea_cache_context={}, process=process,
    )
    state = process._spectrum_states[0]
    actual = {0: [], 1: []}
    outputs = {0: [], 1: []}
    forecaster_keys = []
    contexts = {0: torch.ones(1, 4, 8), 1: torch.zeros(1, 4, 8)}

    for step in range(STEPS):
        sigma = torch.tensor([1.0 - step * 0.09])
        for branch in (0, 1):

            def model_function(x, timestep, branch=branch, step=step, **_kwargs):
                actual[branch].append(step)
                return torch.full_like(x, (1.0 + branch) * step)  # a linear trend per branch

            args = {"input": torch.zeros(1, 1), "timestep": sigma, "c": {"c_crossattn": contexts[branch]}, "cond_or_uncond": [branch]}
            outputs[branch].append(patcher.wrapper(model_function, args))
        forecaster_keys.append(sorted(state.forecasters))
    return actual, outputs, forecaster_keys


class SplitBranchTests(unittest.TestCase):
    def test_split_calls_forecast_only_the_conditional_branch(self):
        for policy in ("Conservative", "Legacy / fastest"):
            with self.subTest(policy=policy):
                actual, outputs, forecaster_keys = _run_split(policy)

                # The uncond call is never a new step: Conservative forces it
                # actual, and Legacy has no forecaster for branch 1 (dropped at
                # each new cond-only step, never fed by a repeated-sigma call).
                self.assertEqual(actual[1], list(range(STEPS)))
                self.assertTrue(all(keys == [0] for keys in forecaster_keys[3:]), forecaster_keys)

                cached = [step for step in range(STEPS) if step not in actual[0]]
                self.assertTrue(cached, actual[0])
                self.assertEqual(actual[0][:3], [0, 1, 2])  # warmup
                self.assertIn(STEPS - 1, actual[0])  # tail
                for step in range(STEPS):
                    # Forecast cond outputs continue the cond trend; uncond is the model's own.
                    torch.testing.assert_close(outputs[0][step], torch.full((1, 1), float(step)), rtol=0, atol=1e-3)
                    torch.testing.assert_close(outputs[1][step], torch.full((1, 1), 2.0 * step))


if __name__ == "__main__":
    unittest.main()
