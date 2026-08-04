from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.skim import apply_skim_to_predictions, skim_prediction


def _inputs():
    torch.manual_seed(1234)
    x = torch.randn(2, 4, 8, 8)
    cond = torch.randn_like(x)
    uncond = torch.randn_like(x)
    return x, cond, uncond


class SkimmedCFGTests(unittest.TestCase):
    def test_inputs_are_not_mutated(self):
        x, cond, uncond = _inputs()
        cond_before = cond.clone()
        uncond_before = uncond.clone()
        apply_skim_to_predictions(x, cond, uncond, 4.0, 2.5)
        self.assertTrue(torch.equal(cond, cond_before))
        self.assertTrue(torch.equal(uncond, uncond_before))

    def test_equal_fallback_scale_is_identity_for_single_side(self):
        x, cond, uncond = _inputs()
        result = skim_prediction(x, cond, uncond, 4.0, 4.0)
        self.assertTrue(torch.allclose(result, cond))

    def test_cfg_one_is_safe_identity(self):
        x, cond, uncond = _inputs()
        result_cond, result_uncond = apply_skim_to_predictions(x, cond, uncond, 1.0, 2.5)
        self.assertTrue(torch.equal(result_cond, cond))
        self.assertTrue(torch.equal(result_uncond, uncond))

    def test_full_negative_and_aggressive_filter_remain_finite(self):
        x, cond, uncond = _inputs()
        result_cond, result_uncond = apply_skim_to_predictions(
            x,
            cond,
            uncond,
            7.0,
            2.0,
            full_skim_negative=True,
            disable_flipping_filter=True,
        )
        self.assertTrue(torch.isfinite(result_cond).all())
        self.assertTrue(torch.isfinite(result_uncond).all())


if __name__ == "__main__":
    unittest.main()
