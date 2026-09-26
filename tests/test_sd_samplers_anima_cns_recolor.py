from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from modules import sd_samplers_anima_cns
from modules.sd_samplers_anima_cns import CNSRecolorer, radial_bins


class CNSTests(unittest.TestCase):
    def _recolorer(self, strength):
        gamma = np.full((1, 2, 8), 0.25, dtype=np.float64)
        aspects = np.asarray([[1024, 1024]], dtype=np.float64)
        sigmas = np.asarray([1.0, 0.5, 0.0], dtype=np.float64)
        return CNSRecolorer(gamma, aspects, sigmas, strength)

    def test_radial_bins_are_bounded(self):
        bins = radial_bins(8, 12, 16)
        self.assertEqual(bins.shape, (8, 12))
        self.assertGreaterEqual(int(bins.min()), 0)
        self.assertLess(int(bins.max()), 16)

    def test_zero_strength_is_exact_passthrough(self):
        white = torch.randn(2, 4, 8, 8)
        result = self._recolorer(0.0).recolor(white, 0.5)
        self.assertIs(result, white)

    def test_recolored_noise_is_finite_and_rms_normalized(self):
        torch.manual_seed(8)
        white = torch.randn(2, 4, 16, 16)
        result = self._recolorer(1.0).recolor(white, 0.5)
        self.assertTrue(torch.isfinite(result).all())
        std = result.float().std(dim=(-2, -1))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-4, rtol=1e-4))

    def test_device_gamma_lookup_matches_numpy_interp(self):
        rng = np.random.default_rng(3)
        gamma = rng.uniform(-0.2, 1.1, size=(3, 12, 16))
        aspects = np.asarray([[1024, 1024], [832, 1216], [1216, 832]], dtype=np.float64)
        sigmas = np.concatenate([np.sort(rng.uniform(0.0, 1.0, 12))[::-1], [0.0]])
        recolorer = CNSRecolorer(gamma, aspects, sigmas, 1.0)
        recolorer._select_aspect(104, 152)
        midpoints = recolorer.sigmas
        probes = [-0.5, 0.0, midpoints[0], midpoints[5], midpoints[-1], 1.0, 2.0, *rng.uniform(0.0, 1.0, 20)]
        for sigma in probes:
            expected = np.sqrt(np.clip(1.0 - np.asarray([np.interp(sigma, midpoints, recolorer.selected_gamma[:, index]) for index in range(16)]), 0.0, 1.0))
            for value in (sigma, torch.tensor(sigma, dtype=torch.float32), torch.tensor(sigma, dtype=torch.float64)):
                actual = (1.0 - recolorer._gamma_row(value, torch.device("cpu"))).clamp(0.0, 1.0).sqrt()
                np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6, err_msg=f"sigma={sigma}")

    def test_single_calibrated_sigma_is_constant_like_numpy(self):
        recolorer = CNSRecolorer(np.linspace(0.0, 0.5, 8).reshape(1, 1, 8), np.asarray([[1, 1]]), np.asarray([0.7, 0.0]), 1.0)
        recolorer._select_aspect(8, 8)
        for sigma in (0.1, 0.7, 3.0):
            np.testing.assert_allclose(recolorer._gamma_row(sigma, torch.device("cpu")).numpy(), np.linspace(0.0, 0.5, 8), atol=1e-7)

    def test_unavailable_gamma_falls_back_without_retrying_every_run(self):
        process = SimpleNamespace(extra_generation_params={})
        failing = mock.Mock(side_effect=OSError("offline"))
        with mock.patch.object(sd_samplers_anima_cns, "_GAMMA_ARRAYS", None), mock.patch.object(
            sd_samplers_anima_cns, "_GAMMA_FAILED_AT", None
        ), mock.patch.object(sd_samplers_anima_cns, "load_gamma_arrays", failing):
            self.assertIsNone(CNSRecolorer.try_calibrated(1.0, process))
            self.assertIsNone(CNSRecolorer.try_calibrated(1.0, process))
        self.assertEqual(failing.call_count, 1)
        self.assertIn("white-noise fallback", process.extra_generation_params["Anima CNS"])


if __name__ == "__main__":
    unittest.main()
