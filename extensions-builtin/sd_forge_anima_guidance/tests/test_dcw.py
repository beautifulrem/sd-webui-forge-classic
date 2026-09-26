from __future__ import annotations

import logging
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
        # x += lam * (1 - previous sigma) * LL(x - denoised); a constant diff
        # of 1 is all low band, so every element moves by the full scalar.
        expected = torch.full_like(x, 1 + (-0.015) * (1 - 0.9))
        self.assertTrue(torch.allclose(x, expected, rtol=0.0, atol=1e-7))
        state.before_denoiser(x, 0.8)  # same sigma (a second evaluation): no-op
        self.assertTrue(torch.allclose(x, expected, rtol=0.0, atol=1e-7))

    def test_default_ll_mode_leaves_the_detail_bands_untouched(self):
        # A diff with energy in every band: only its LL band may move x.
        torch.manual_seed(21)
        state = DCWState(lam=-0.015, bands=parse_band_mask("LL"))
        x = torch.randn(2, 4, 8, 6)
        denoised = torch.randn_like(x)
        state.capture_denoised(denoised)
        state.before_denoiser(x, 0.9)
        before = x.clone()
        state.before_denoiser(x, 0.8)

        scalar = -0.015 * (1 - 0.9)
        change = haar_dwt_2d(x - before)
        diff = haar_dwt_2d(before - denoised)
        self.assertTrue(all(band.abs().max() > 0.1 for band in diff))
        self.assertTrue(torch.allclose(change[0], scalar * diff[0], rtol=0.0, atol=1e-6))
        for band in change[1:]:  # LH, HL, HH
            self.assertTrue(torch.allclose(band, torch.zeros_like(band), rtol=0.0, atol=1e-6))

    def test_warnings_use_the_panels_visible_logger(self):
        # Forge's root logger is at ERROR: a plain module logger's warnings
        # never show. The panel's "AnimaGuidance" logger has its own handler.
        from lib_anima_guidance import dcw

        self.assertIs(dcw.LOGGER, logging.getLogger("AnimaGuidance"))


if __name__ == "__main__":
    unittest.main()
