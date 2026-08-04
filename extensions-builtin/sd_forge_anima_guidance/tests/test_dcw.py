from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.dcw import ALL_BANDS, DCWState, band_masked_diff, haar_dwt_2d, haar_idwt_2d, parse_band_mask


class DCWTests(unittest.TestCase):
    def test_haar_round_trip(self):
        torch.manual_seed(9)
        value = torch.randn(2, 4, 8, 10)
        rebuilt = haar_idwt_2d(*haar_dwt_2d(value))
        self.assertTrue(torch.allclose(value, rebuilt, atol=1e-6, rtol=1e-6))

    def test_all_band_mask_is_round_trip(self):
        value = torch.randn(1, 4, 8, 8)
        self.assertTrue(torch.allclose(value, band_masked_diff(value, ALL_BANDS), atol=1e-6, rtol=1e-6))

    def test_invalid_band_rejected(self):
        with self.assertRaises(ValueError):
            parse_band_mask("LL+BAD")

    def test_correction_starts_on_second_sigma(self):
        state = DCWState(lam=-0.015, bands=frozenset({"LL"}))
        x = torch.ones(1, 4, 8, 8)
        state.capture_denoised(torch.zeros_like(x))
        state.before_denoiser(x, 0.9)
        self.assertTrue(torch.equal(x, torch.ones_like(x)))
        state.before_denoiser(x, 0.8)
        self.assertFalse(torch.equal(x, torch.ones_like(x)))


if __name__ == "__main__":
    unittest.main()
