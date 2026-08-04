from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

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


if __name__ == "__main__":
    unittest.main()
