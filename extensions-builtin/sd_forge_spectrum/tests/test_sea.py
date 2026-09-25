import math
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import SpectrumState
from lib_spectrum.sea import (
    SeaCalibration,
    count_refreshes,
    flow_time,
    l1rel,
    sea_filter,
    sea_gain,
    solve_delta_for_refresh_ratio,
    window_refresh_fraction,
)


class SeaTests(unittest.TestCase):
    def test_filter_preserves_shape_and_finite_values(self):
        value = torch.randn(1, 4, 8, 10)
        filtered = sea_filter(value, sigma=0.5, beta=2.0)
        self.assertEqual(filtered.shape, value.shape)
        self.assertTrue(torch.isfinite(filtered).all())

    def test_relative_distance_uses_most_changed_batch_row(self):
        previous = torch.ones(2, 1, 1, 2)
        current = previous.clone()
        current[0] += 0.1
        current[1] += 2.0

        self.assertAlmostEqual(l1rel(current, previous), 2.0)

    def test_delta_solver_tracks_requested_refresh_count(self):
        distances = [0.1, 0.2, 0.05, 0.4, 0.1, 0.15]
        delta = solve_delta_for_refresh_ratio(distances, 0.5)
        self.assertTrue(math.isfinite(delta) and delta > 0)
        self.assertLessEqual(count_refreshes(distances, delta), round(len(distances) * 0.5))

    def test_variance_exploding_sigmas_map_into_the_flow_range(self):
        self.assertEqual(flow_time(0.25, flow=True), 0.25)
        for sigma in (0.03, 1.0, 14.6):
            t = flow_time(sigma, flow=False)
            self.assertTrue(0.0 < t < 1.0)
            self.assertAlmostEqual((1.0 - t) / t, 1.0 / sigma)  # same SNR
            gain = sea_gain(8, 8, t, 2.0, "cpu")
            self.assertTrue(torch.isfinite(gain).all() and (gain > 0).all())

    def test_window_fraction_is_bounded(self):
        fraction = window_refresh_fraction(30, 6, 3, 2.0, 0.25)
        self.assertTrue(0.0 <= fraction <= 1.0)


SEA = "SEA (auto-calibrated)"


def _calibration(cache_dir, steps=10):
    return SeaCalibration(
        cache_dir=cache_dir,
        context={"model": "Test"},
        steps=steps,
        warmup_steps=2,
        tail_actual_steps=2,
        window_size=2.0,
        flex_window=0.0,
        refresh_ratio=0.0,
    )


def _state(calibration, steps=10):
    return SpectrumState(
        steps=steps,
        window_size=2.0,
        flex_window=0.0,
        warmup_steps=2,
        tail_actual_steps=2,
        schedule=SEA,
        sea_beta=2.0,
        sea_calibration=calibration,
        verbose=False,
    )


def _run(state, steps=10, batch=lambda step: 1):
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(1, 4, 1, 8, 8, generator=generator)
    for step in range(steps):
        sigma = 1.0 - step / steps
        noise = torch.randn(image.shape, generator=generator)
        latent = (1.0 - sigma) * image + sigma * noise
        # Branch copies of x are identical, as Forge batches them.
        state.advance(sigma, latent.repeat(batch(step), 1, 1, 1, 1))


class SeaStateTests(unittest.TestCase):
    def test_calibration_is_saved_once_and_loaded_by_the_next_generation(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            state = _state(_calibration(cache_dir))
            self.assertIsNone(state.sea_delta)
            _run(state)
            # Distances from the cacheable steps [warmup, stop_at) only.
            self.assertEqual(len(state.sea_distances), state.stop_at - state.warmup_steps)
            self.assertTrue(state.sea_saved)
            self.assertTrue(state.sea_delta > 0)
            self.assertTrue(pathlib.Path(state.sea_calibration.path).is_file())

            again = _state(_calibration(cache_dir))
            self.assertEqual(again.sea_delta, state.sea_delta)
            _run(again)
            self.assertEqual(again.sea_distances, [])  # calibrated: nothing re-collected
            self.assertIsNone(_state(_calibration(cache_dir, steps=12)).sea_delta)  # other config

    def test_refresh_decision_follows_the_accumulated_distance(self):
        state = _state(None)
        state.sea_delta = 0.5
        state.forecasters[0] = SimpleNamespace(ready=True)
        state.step = state.warmup_steps
        state.sea_accumulated = 0.4
        self.assertEqual(state.decide([0]), "cached")
        state.sea_accumulated = 0.5
        self.assertEqual(state.decide([0]), "actual")
        state.actual_completed()
        self.assertEqual(state.sea_accumulated, 0.0)
        self.assertEqual(state.decide([0]), "cached")
        state.step = state.stop_at
        self.assertEqual(state.decide([0]), "actual")  # tail steps are always actual

    def test_uncond_dropping_out_of_the_batch_keeps_measuring(self):
        # Anima's CFG active range: cond+uncond (2 rows), then cond only.
        dropped = _state(None)
        _run(dropped, batch=lambda step: 2 if step < 5 else 1)
        single = _state(None)
        _run(single)
        self.assertEqual(dropped.sea_distances, single.sea_distances)
        self.assertAlmostEqual(dropped.sea_accumulated, single.sea_accumulated)


if __name__ == "__main__":
    unittest.main()
