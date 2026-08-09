from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.advanced import (
    MomentumGuidanceState,
    fdg_combine,
    make_guidance_range_cfg_function,
)


class MomentumGuidanceTests(unittest.TestCase):
    def test_first_step_is_identity_and_second_adds_velocity_momentum(self):
        state = MomentumGuidanceState(momentum=0.5, ema_decay=0.5)
        x = torch.zeros(1, 1, 2, 2)
        first = -torch.ones_like(x)
        second = -torch.full_like(x, 3.0)
        self.assertTrue(torch.equal(state.apply(first, x, 1.0), first))
        actual = state.apply(second, x, 1.0)
        self.assertTrue(torch.allclose(actual, torch.full_like(x, -3.5)))

    def test_probe_can_avoid_advancing_state(self):
        state = MomentumGuidanceState()
        x = torch.zeros(1, 1, 2, 2)
        state.apply(-torch.ones_like(x), x, 1.0)
        previous = state._ema.clone()
        state.apply(-torch.full_like(x, 2.0), x, 1.0, update_state=False)
        self.assertTrue(torch.equal(state._ema, previous))

    def test_shape_change_resets_safely(self):
        state = MomentumGuidanceState()
        state.apply(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), 1.0)
        result = state.apply(torch.ones(1, 1, 3, 3), torch.zeros(1, 1, 3, 3), 1.0)
        self.assertEqual(result.shape, (1, 1, 3, 3))
        self.assertTrue(torch.isfinite(result).all())


class FrequencyDecoupledGuidanceTests(unittest.TestCase):
    def test_equal_frequency_scales_match_standard_cfg(self):
        torch.manual_seed(12)
        cond = torch.randn(1, 4, 7, 9)
        uncond = torch.randn_like(cond)
        scale = 4.0
        actual = fdg_combine(cond, uncond, scale, scale)
        expected = uncond + scale * (cond - uncond)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_detail_scale_changes_nonconstant_signal(self):
        cond = torch.zeros(1, 1, 4, 4)
        cond[..., ::2, ::2] = 1.0
        uncond = torch.zeros_like(cond)
        self.assertFalse(
            torch.equal(fdg_combine(cond, uncond, 4.0, 1.0), fdg_combine(cond, uncond, 4.0, 4.0))
        )

    def test_anima_five_dimensional_latent_preserves_shape(self):
        cond = torch.randn(2, 4, 1, 7, 9)
        uncond = torch.randn_like(cond)
        result = fdg_combine(cond, uncond, 4.0, 2.0)
        self.assertEqual(result.shape, cond.shape)
        self.assertTrue(torch.isfinite(result).all())


class GuidanceRangeTests(unittest.TestCase):
    def test_outside_uses_conditional_and_inside_delegates(self):
        x = torch.full((1, 1, 1, 1), 10.0)
        cond = torch.full_like(x, 3.0)
        uncond = torch.full_like(x, 1.0)
        previous = lambda args: torch.full_like(x, 5.0)
        hook = make_guidance_range_cfg_function(previous, sigma_start=0.2, sigma_end=0.8)
        base = {"input": x, "cond_denoised": cond, "uncond_denoised": uncond, "cond_scale": 4.0}
        self.assertTrue(torch.equal(hook({**base, "sigma": torch.tensor([0.5])}), torch.full_like(x, 5.0)))
        self.assertTrue(torch.equal(hook({**base, "sigma": torch.tensor([0.9])}), x - cond))


if __name__ == "__main__":
    unittest.main()
