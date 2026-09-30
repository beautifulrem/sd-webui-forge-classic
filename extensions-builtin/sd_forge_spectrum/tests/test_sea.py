import math
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum import forecaster, sea
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

    def test_gain_prior_is_built_once_per_latent_size(self):
        sea._power_spectrum.cache_clear()
        with mock.patch.object(sea, "radial_frequency", wraps=sea.radial_frequency) as built:
            gains = [sea_gain(8, 10, t, 2.0, "cpu") for t in (0.9, 0.5, 0.1)]
            self.assertEqual(built.call_count, 1)  # only sigma changed
            sea_gain(8, 12, 0.5, 2.0, "cpu")
            sea_gain(8, 10, 0.5, 3.0, "cpu")
            self.assertEqual(built.call_count, 3)  # new size, new beta
        for t, gain in zip((0.9, 0.5, 0.1), gains):
            spectrum = sea.radial_frequency(8, 10, "cpu").clamp_min(1.0 / 10).pow(-2.0)
            expected = (1 - t) * spectrum / ((1 - t) ** 2 * spectrum + t * t + 1e-8)
            self.assertTrue(torch.equal(gain, expected / expected.mean()))

    def test_window_fraction_is_bounded(self):
        fraction = window_refresh_fraction(30, 6, 3, 2.0, 0.25)
        self.assertTrue(0.0 <= fraction <= 1.0)


SEA = "SEA (auto-calibrated)"


def _calibration(cache_dir, steps=10, sea_beta=2.0):
    return SeaCalibration(
        cache_dir=cache_dir,
        context={"model": "Test"},
        steps=steps,
        warmup_steps=2,
        tail_actual_steps=2,
        window_size=2.0,
        flex_window=0.0,
        refresh_ratio=0.0,
        sea_beta=sea_beta,
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


def _run(state, steps=10, batch=lambda step: 1, copies=lambda step: 1):
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(1, 4, 1, 8, 8, generator=generator)
    for step in range(steps):
        sigma = 1.0 - step / steps
        noise = torch.randn(image.shape, generator=generator)
        latent = (1.0 - sigma) * image + sigma * noise
        # Branch copies of x are identical, as Forge batches them.
        state.advance(sigma, latent.repeat(batch(step), 1, 1, 1, 1), copies(step))


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

    @unittest.skipIf(os.name != "posix", "POSIX permission bits")
    def test_saved_calibration_is_readable_like_a_plain_new_file(self):
        # Not mkstemp's 0600, and not a hard-coded 0644 under umask 077.
        from modules import anima_user_data

        for umask, expected in ((0o022, 0o644), (0o077, 0o600)):
            with self.subTest(umask=oct(umask)), tempfile.TemporaryDirectory() as cache_dir, mock.patch.object(anima_user_data, "_UMASK", umask):
                calibration = _calibration(cache_dir)
                calibration.solve_and_save([0.1, 0.2, 0.3])
                self.assertEqual(stat.S_IMODE(os.stat(calibration.path).st_mode), expected)
                self.assertEqual(os.listdir(cache_dir), [os.path.basename(calibration.path)])  # no temporary left

    def test_spectral_beta_is_part_of_the_calibration_key(self):
        # The saved delta is solved from beta-filtered distances.
        self.assertNotEqual(_calibration("unused", sea_beta=2.0).key, _calibration("unused", sea_beta=3.0).key)

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

    def test_only_one_branch_copy_is_filtered(self):
        # cond + uncond (+ an AND sub-prompt) batch the same x: filter one copy.
        filtered_rows = []

        def recording_filter(value, *args):
            filtered_rows.append(value.shape[0])
            return sea_filter(value, *args)

        batched = _state(None)
        with mock.patch.object(forecaster, "sea_filter", recording_filter):
            _run(batched, batch=lambda step: 3 if step < 5 else 1, copies=lambda step: 3 if step < 5 else 1)
        self.assertEqual(filtered_rows, [1] * 10)
        single = _state(None)
        _run(single)
        self.assertEqual(batched.sea_distances, single.sea_distances)
        self.assertEqual(batched.sea_accumulated, single.sea_accumulated)

    def test_the_sampling_wrapper_filters_one_copy_of_a_cfg_batch(self):
        class Patcher:
            model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
            model_options = {}
            wrapper = None

            def clone(self):
                return self

            def set_model_unet_function_wrapper(self, wrapper):
                self.wrapper = wrapper

        with tempfile.TemporaryDirectory() as cache_dir:
            patcher = forecaster.SpectrumNode.patch(
                Patcher(), steps=10, weight=1.0, degree=1, lam=1e-6, window_size=2,
                flex_window=0.0, warmup_steps=2, stop_caching_step=1.0, tail_actual_steps=2,
                history_size=3, schedule=SEA, refresh_ratio=0.0, sea_beta=2.0,
                compat_policy="Legacy / fastest", verbose=False, sea_cache_dir=cache_dir,
                sea_cache_context={}, process=SimpleNamespace(),
            )
            filtered_rows = []

            def recording_filter(value, *args):
                filtered_rows.append(value.shape[0])
                return sea_filter(value, *args)

            x = torch.randn(1, 4, 1, 8, 8)
            args = {"input": torch.cat([x, x]), "timestep": torch.tensor([0.9]), "c": {"c_crossattn": torch.randn(2, 4, 8)}, "cond_or_uncond": [1, 0]}
            with mock.patch.object(forecaster, "sea_filter", recording_filter):
                patcher.wrapper(lambda value, timestep, **_kwargs: value, args)
        self.assertEqual(filtered_rows, [1])


if __name__ == "__main__":
    unittest.main()
