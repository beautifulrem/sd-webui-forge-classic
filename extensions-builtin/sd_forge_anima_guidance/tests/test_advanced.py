from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.advanced import (
    CFGZeroStarState,
    MomentumGuidanceState,
    cfg_zero_star_combine,
    cfg_zero_star_scale,
    fdg_combine,
    make_cfg_zero_star_cfg_function,
    IGNORES_UNCOND,
    make_guidance_range_cfg_function,
    make_guidance_range_uncond_skip,
    make_momentum_post_cfg_function,
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


class CFGZeroStarTests(unittest.TestCase):
    def test_scale_one_reproduces_plain_cfg(self):
        # cond = uncond + a component orthogonal to uncond, so s* = 1.
        uncond = torch.tensor([[1.0, 2.0, 0.0, 0.0]]).reshape(1, 1, 2, 2)
        cond = uncond + torch.tensor([[0.0, 0.0, 3.0, -1.0]]).reshape(1, 1, 2, 2)
        self.assertTrue(torch.allclose(cfg_zero_star_scale(cond, uncond), torch.ones(1, 1, 1, 1)))
        actual = cfg_zero_star_combine(cond, uncond, 4.5)
        self.assertTrue(torch.allclose(actual, uncond + 4.5 * (cond - uncond)))

    def test_orthogonal_predictions_give_zero_scale(self):
        cond = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).reshape(2, 1, 1, 2)
        uncond = torch.tensor([[0.0, 3.0], [5.0, 0.0]]).reshape(2, 1, 1, 2)
        self.assertTrue(torch.equal(cfg_zero_star_scale(cond, uncond), torch.zeros(2, 1, 1, 1)))
        # s* = 0 drops the unconditional term: w * cond.
        self.assertTrue(torch.allclose(cfg_zero_star_combine(cond, uncond, 4.0), 4.0 * cond))

    def test_scale_is_per_sample_and_cfg_one_is_the_conditional_prediction(self):
        torch.manual_seed(3)
        cond, uncond = torch.randn(2, 4, 1, 3, 5), torch.randn(2, 4, 1, 3, 5)
        scale = cfg_zero_star_scale(cond, uncond)
        for index in range(2):
            expected = (cond[index] * uncond[index]).sum() / uncond[index].square().sum()
            self.assertTrue(torch.allclose(scale[index].flatten(), expected.reshape(1)))
        self.assertTrue(torch.equal(cfg_zero_star_combine(cond, uncond, 1.0), cond))
        # Forge's CFG 1 placeholder (zero uncond) stays finite.
        self.assertTrue(torch.equal(cfg_zero_star_combine(cond, torch.zeros_like(cond), 1.0), cond))

    def test_cfg_function_works_on_velocity_residuals(self):
        torch.manual_seed(4)
        x = torch.randn(1, 4, 8, 8)
        sigma = 0.7
        v_cond, v_uncond = torch.randn_like(x), torch.randn_like(x)
        args = {
            "input": x,
            "sigma": torch.tensor([sigma]),
            "cond_denoised": x - sigma * v_cond,
            "uncond_denoised": x - sigma * v_uncond,
            "cond_scale": 4.0,
        }
        residual = make_cfg_zero_star_cfg_function(CFGZeroStarState(0))(args)
        # Forge's residual is x - denoised = sigma * v for rectified flow.
        expected = sigma * cfg_zero_star_combine(v_cond, v_uncond, 4.0)
        self.assertTrue(torch.allclose(residual, expected, atol=1e-5))

    def test_zero_init_zeroes_the_first_steps_only(self):
        state = CFGZeroStarState(2)
        cfg = make_cfg_zero_star_cfg_function(state)
        x = torch.ones(1, 1, 2, 2)
        base = {"input": x, "cond_denoised": x * 0.5, "uncond_denoised": x * 0.25, "cond_scale": 3.0}
        residuals = [cfg({**base, "sigma": torch.tensor([sigma])}) for sigma in (1.0, 1.0, 0.8, 0.6)]
        # Two steps (the repeated 1.0 is another evaluation of step 0) output
        # zero velocity, i.e. denoised = x and an unchanged latent.
        for residual in residuals[:3]:
            self.assertTrue(torch.equal(residual, torch.zeros_like(x)))
        self.assertFalse(torch.equal(residuals[3], torch.zeros_like(x)))
        state.reset()
        self.assertTrue(torch.equal(cfg({**base, "sigma": torch.tensor([0.6])}), torch.zeros_like(x)))

    def test_auxiliary_probe_does_not_consume_zero_init_steps(self):
        from types import SimpleNamespace

        process = SimpleNamespace()
        state = CFGZeroStarState(1)
        cfg = make_cfg_zero_star_cfg_function(state, process=process)
        x = torch.ones(1, 1, 2, 2)
        base = {"input": x, "cond_denoised": x * 0.5, "uncond_denoised": x * 0.25, "cond_scale": 3.0}
        from modules import anima_support

        setattr(process, anima_support._AUXILIARY_DENOISER_ATTRIBUTE, True)
        self.assertFalse(torch.equal(cfg({**base, "sigma": torch.tensor([0.9])}), torch.zeros_like(x)))
        setattr(process, anima_support._AUXILIARY_DENOISER_ATTRIBUTE, False)
        self.assertTrue(torch.equal(cfg({**base, "sigma": torch.tensor([1.0])}), torch.zeros_like(x)))


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


    def test_cfg_one_does_not_need_the_unconditional_prediction(self):
        # Forge skips the negative pass at CFG 1 and passes zeros instead.
        cfg = make_guidance_range_cfg_function(None, sigma_start=0.2, sigma_end=0.8)
        x, cond = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        args = {"input": x, "cond_denoised": cond, "uncond_denoised": torch.zeros_like(cond), "cond_scale": 1.0}
        for sigma in (0.5, 0.9):
            self.assertTrue(torch.allclose(cfg({**args, "sigma": torch.tensor([sigma])}), x - cond))

class GuidanceRangeUncondSkipTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_guidance_range_cfg_function(None, sigma_start=0.2, sigma_end=0.8)
        self.skip = make_guidance_range_uncond_skip(self.cfg)
        self.uncond = [{"model_conds": {}}]

    def uncond_after(self, sigma, **options):
        options.setdefault("sampler_cfg_function", self.cfg)
        return self.skip(None, [], self.uncond, None, torch.tensor([sigma]), options)[2]

    def test_uncond_only_runs_inside_the_window(self):
        self.assertIsNone(self.uncond_after(0.9))
        self.assertIsNone(self.uncond_after(0.1))
        self.assertIs(self.uncond_after(0.5), self.uncond)

    def test_kept_when_another_hook_may_read_uncond(self):
        self.assertIs(self.uncond_after(0.9, sampler_cfg_function=lambda args: 0), self.uncond)
        self.assertIs(self.uncond_after(0.9, sampler_post_cfg_function=[lambda args: 0]), self.uncond)

        def safe(args):
            return args["denoised"]

        setattr(safe, IGNORES_UNCOND, True)
        self.assertIsNone(self.uncond_after(0.9, sampler_post_cfg_function=[safe]))

    def test_momentum_does_not_keep_the_unconditional_pass(self):
        momentum = make_momentum_post_cfg_function(MomentumGuidanceState())
        self.assertIsNone(self.uncond_after(0.9, sampler_post_cfg_function=[momentum]))


if __name__ == "__main__":
    unittest.main()
