from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.smc import SMCCFGState, make_smc_cfg_function


class SMCCFGTests(unittest.TestCase):
    def test_alpha_zero_matches_standard_cfg(self):
        torch.manual_seed(44)
        cond = torch.randn(2, 4, 8, 8)
        uncond = torch.randn_like(cond)
        scale = 4.0
        actual = SMCCFGState(alpha=0.0).combine(cond, uncond, scale)
        expected = uncond + scale * (cond - uncond)
        self.assertTrue(torch.allclose(actual, expected))

    def test_state_resets_safely_after_shape_change(self):
        state = SMCCFGState(alpha=0.2)
        first = state.combine(torch.ones(1, 4, 8, 8), torch.zeros(1, 4, 8, 8), 4.0)
        second = state.combine(torch.ones(1, 4, 16, 16), torch.zeros(1, 4, 16, 16), 4.0)
        self.assertEqual(first.shape, (1, 4, 8, 8))
        self.assertEqual(second.shape, (1, 4, 16, 16))
        self.assertTrue(torch.isfinite(second).all())

    def test_previous_error_changes_later_control_surface(self):
        state = SMCCFGState(alpha=0.2, lam=5.0)
        uncond = torch.zeros(1, 1, 2, 2)
        first = state.combine(torch.ones_like(uncond), uncond, 4.0)
        second = state.combine(-torch.ones_like(uncond), uncond, 4.0)
        fresh = SMCCFGState(alpha=0.2, lam=5.0).combine(-torch.ones_like(uncond), uncond, 4.0)
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(torch.equal(second, fresh))

    def test_each_image_correction_is_independent_of_the_batch(self):
        torch.manual_seed(45)
        cond, uncond = torch.randn(2, 4, 1, 8, 8), torch.randn(2, 4, 1, 8, 8)
        # A second image with a much larger residual must not change the first.
        cond[1] = cond[1] * 50.0
        batched = SMCCFGState(alpha=0.2).combine(cond, uncond, 4.0)
        alone = SMCCFGState(alpha=0.2).combine(cond[:1], uncond[:1], 4.0)
        self.assertTrue(torch.allclose(batched[:1], alone, atol=1e-5))
        # First step: surface = lam * e, so the gain is alpha * mean|e| per image.
        e = cond[:1] - uncond[:1]
        expected = uncond[:1] + 4.0 * (e - 0.2 * e.abs().mean() * torch.sign(e))
        self.assertTrue(torch.allclose(alone, expected, atol=1e-5))

    def test_auxiliary_combine_does_not_advance_history(self):
        state = SMCCFGState(alpha=0.2, lam=5.0)
        uncond = torch.zeros(1, 1, 2, 2)
        state.combine(torch.ones_like(uncond), uncond, 4.0)
        previous = state._e_prev.clone()

        state.combine(
            -torch.ones_like(uncond),
            uncond,
            4.0,
            update_state=False,
        )

        self.assertTrue(torch.equal(state._e_prev, previous))

    def test_forge_adapter_combines_the_denoised_predictions(self):
        torch.manual_seed(46)
        x = torch.randn(1, 4, 1, 8, 8)
        cond_denoised, uncond_denoised = torch.randn_like(x), torch.randn_like(x)
        args = {
            "input": x,
            "sigma": torch.tensor([0.7]),
            "cond_denoised": cond_denoised,
            "uncond_denoised": uncond_denoised,
            # Forge's cond / uncond are residuals, not predictions.
            "cond": x - cond_denoised,
            "uncond": x - uncond_denoised,
            "cond_scale": 4.0,
        }
        residual = make_smc_cfg_function(SMCCFGState(alpha=0.0))(args)
        expected = x - (uncond_denoised + 4.0 * (cond_denoised - uncond_denoised))
        self.assertTrue(torch.allclose(residual, expected, atol=1e-5))

    def test_forge_adapter_skips_at_cfg_1_without_advancing_history(self):
        torch.manual_seed(47)
        x = torch.randn(1, 4, 1, 8, 8)
        cond_denoised = torch.randn_like(x)
        state = SMCCFGState(alpha=0.2)
        args = {
            "input": x,
            "sigma": torch.tensor([0.7]),
            "cond_denoised": cond_denoised,
            # Forge's CFG 1 placeholder: the unconditional pass was skipped.
            "uncond_denoised": torch.zeros_like(x),
            "cond_scale": 1.0,
        }
        residual = make_smc_cfg_function(state)(args)
        self.assertTrue(torch.equal(residual, x - cond_denoised))
        self.assertIsNone(state._e_prev)


if __name__ == "__main__":
    unittest.main()
