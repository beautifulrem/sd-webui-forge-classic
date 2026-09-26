from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.advanced import (
    fdg_combine,
    make_fdg_cfg_function,
    make_guidance_range_cfg_function,
    standard_cfg_residual,
)
from lib_anima_guidance.skim import apply_skim_to_predictions, make_skim_cfg_function, skim_prediction


def _inputs():
    torch.manual_seed(1234)
    x = torch.randn(2, 4, 8, 8)
    cond = torch.randn_like(x)
    uncond = torch.randn_like(x)
    return x, cond, uncond


class SkimmedCFGTests(unittest.TestCase):
    def test_cfg_one_leaves_the_conditional_prediction_alone(self):
        # Forge skips the negative pass at CFG 1 and passes zeros instead.
        x, cond, _ = _inputs()
        new_cond, _ = apply_skim_to_predictions(x, cond, torch.zeros_like(cond), 1.0, 2.0)
        self.assertTrue(torch.equal(new_cond, cond))

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

    def test_skim_prediction_matches_the_upstream_formula_exactly(self):
        # Upstream Skimmed_CFG, written out: mask from the guided prediction,
        # then the masked elements are pulled towards the fallback scale.
        x, cond, uncond = _inputs()
        scale, fallback_scale = 5.0, 2.0
        guided = x - ((x - uncond) + scale * ((x - cond) - (x - uncond)))
        fallback = x - ((x - uncond) + fallback_scale * ((x - cond) - (x - uncond)))
        mask = ((cond - uncond).sign() == cond.sign()) & (
            cond.sign() == (cond * scale - uncond * (scale - 1.0)).sign()
        )
        for disable_flip in (True, False):
            expected_mask = mask if disable_flip else mask & (guided.sign() == (guided - x).sign())
            expected = torch.where(expected_mask, cond - (guided - fallback) / scale, cond)
            actual = skim_prediction(x, cond, uncond, scale, fallback_scale, disable_flipping_filter=disable_flip)
            self.assertTrue(torch.equal(actual, expected), disable_flip)


class SkimCFGFunctionChainTests(unittest.TestCase):
    def _args(self, sigma):
        x, cond, uncond = _inputs()
        return {
            "input": x,
            "sigma": torch.tensor([sigma]),
            "cond_denoised": cond,
            "uncond_denoised": uncond,
            "cond": x - cond,
            "uncond": x - uncond,
            "cond_scale": 5.0,
        }

    def _skim(self, previous, **overrides):
        options = dict(
            sigma_start=0.8, sigma_end=0.2, skimming_scale=2.0,
            full_skim_negative=False, disable_flipping_filter=False, flip_sigma=None,
        )
        options.update(overrides)
        return make_skim_cfg_function(previous, **options)

    def test_skim_feeds_adjusted_predictions_into_the_next_combine(self):
        args = self._args(0.5)
        cfg = self._skim(make_fdg_cfg_function(1.0))
        x = args["input"]
        cond, uncond = apply_skim_to_predictions(x, args["cond_denoised"], args["uncond_denoised"], 5.0, 2.0)
        # The FDG stage combines the skimmed predictions, not the raw ones.
        expected = x - fdg_combine(cond, uncond, 5.0, 1.0)
        self.assertTrue(torch.allclose(cfg(args), expected, atol=1e-5))
        self.assertFalse(torch.allclose(cfg(args), x - fdg_combine(args["cond_denoised"], args["uncond_denoised"], 5.0, 1.0)))

    def test_chain_inside_a_cfg_range_and_outside_the_skim_window(self):
        seen = []

        def previous(args):
            seen.append(args)
            return standard_cfg_residual(args)

        cfg = make_guidance_range_cfg_function(self._skim(previous), sigma_start=1.0, sigma_end=0.0)
        outside = self._args(0.9)
        self.assertTrue(torch.allclose(cfg(outside), standard_cfg_residual(outside)))
        self.assertIs(seen[-1], outside)  # skim is a pass-through outside its window
        inside = self._args(0.5)
        cfg(inside)
        self.assertIsNot(seen[-1], inside)
        self.assertTrue(torch.equal(seen[-1]["cond"], inside["input"] - seen[-1]["cond_denoised"]))


if __name__ == "__main__":
    unittest.main()
