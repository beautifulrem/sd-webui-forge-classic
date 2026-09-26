from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.advanced import (
    APGState,
    CFGZeroStarState,
    MomentumGuidanceState,
    apg_project,
    cfg_zero_star_combine,
    cfg_zero_star_scale,
    fdg_combine,
    make_apg_cfg_function,
    make_cfg_zero_star_cfg_function,
    IGNORES_UNCOND,
    make_guidance_range_cfg_function,
    make_guidance_range_uncond_skip,
    make_momentum_post_cfg_function,
    make_zero_init_cfg_function,
    make_zero_init_model_wrapper,
)


class MomentumGuidanceTests(unittest.TestCase):
    def test_first_step_is_identity_and_second_adds_velocity_momentum(self):
        state = MomentumGuidanceState(momentum=0.5, ema_decay=0.5)
        x = torch.zeros(1, 1, 2, 2)
        first = -torch.ones_like(x)
        second = -torch.full_like(x, 3.0)
        self.assertTrue(torch.equal(state.apply(first, x, 1.0), first))
        actual = state.apply(second, x, 1.0)
        # Reference mg_nodes.py: v + momentum * (v - m_prev) with m_prev = 1,
        # then m = (1 - ema) * v + ema * m_prev.
        self.assertTrue(torch.allclose(actual, torch.full_like(x, -(3.0 + 0.5 * (3.0 - 1.0)))))
        self.assertTrue(torch.allclose(state._ema, torch.full_like(x, 0.5 * 3.0 + 0.5 * 1.0)))

    def test_momentum_runs_on_velocity_not_on_denoised(self):
        # With x = 0 and sigma = 1 the velocity is just -denoised, so the case
        # above cannot tell the two spaces apart; here x and sigma change.
        state = MomentumGuidanceState(momentum=0.5, ema_decay=0.5)
        x1, sigma1, d1 = torch.full((1, 1, 2, 2), 2.0), 0.8, torch.full((1, 1, 2, 2), 0.5)
        x2, sigma2, d2 = torch.full((1, 1, 2, 2), 1.5), torch.tensor([0.5]), torch.full((1, 1, 2, 2), 1.0)
        self.assertTrue(torch.allclose(state.apply(d1, x1, sigma1), d1))
        actual = state.apply(d2, x2, sigma2)
        # v1 = (2 - 0.5) / 0.8 = 1.875, v2 = (1.5 - 1) / 0.5 = 1
        # v = v2 + 0.5 * (v2 - v1) = 0.5625, denoised = 1.5 - 0.5 * 0.5625
        self.assertTrue(torch.allclose(actual, torch.full_like(x2, 1.21875)))
        self.assertTrue(torch.allclose(state._ema, torch.full_like(x2, 0.5 * 1.0 + 0.5 * 1.875)))
        # Momentum on the denoised prediction (the reference) would give
        # d2 + 0.5 * (d2 - d1) = 1.25.
        self.assertFalse(torch.allclose(actual, d2 + 0.5 * (d2 - d1)))

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

    def test_default_detail_scale_is_above_cfg(self):
        # The paper / reference guide the high band harder than the low band
        # (which uses CFG); a detail scale below CFG would weaken details.
        from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET as preset

        self.assertGreater(preset["guidance.fdg_detail"], preset["native.cfg"])

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
        residual = make_cfg_zero_star_cfg_function()(args)
        # Forge's residual is x - denoised = sigma * v for rectified flow.
        expected = sigma * cfg_zero_star_combine(v_cond, v_uncond, 4.0)
        self.assertTrue(torch.allclose(residual, expected, atol=1e-5))

    def test_zero_init_zeroes_the_first_steps_only(self):
        state = CFGZeroStarState(2)
        cfg = make_zero_init_cfg_function(make_cfg_zero_star_cfg_function(), state)
        x = torch.ones(1, 1, 2, 2)
        base = {"input": x, "cond_denoised": x * 0.5, "uncond_denoised": x * 0.25, "cond_scale": 3.0}
        residuals = [cfg({**base, "sigma": torch.tensor([sigma])}) for sigma in (1.0, 1.0, 0.8, 0.6)]
        # Two steps (the repeated 1.0 is another evaluation of step 0) output
        # zero velocity, i.e. denoised = x and an unchanged latent.
        for residual in residuals[:3]:
            self.assertTrue(torch.equal(residual, torch.zeros_like(x)))
        self.assertFalse(torch.equal(residuals[3], torch.zeros_like(x)))
        state.reset()
        self.assertTrue(torch.equal(cfg({**base, "sigma": torch.tensor([1.0])}), torch.zeros_like(x)))

    def test_pass_starting_mid_schedule_is_not_zeroed(self):
        # Hires / img2img passes start below the schedule's maximum sigma:
        # zero-init is only for sampling from pure noise.
        state = CFGZeroStarState(1, start_sigma=1.0)
        cfg = make_zero_init_cfg_function(make_cfg_zero_star_cfg_function(), state)
        x = torch.ones(1, 1, 2, 2)
        base = {"input": x, "cond_denoised": x * 0.5, "uncond_denoised": x * 0.25, "cond_scale": 3.0}
        for sigma in (0.6, 0.4):
            self.assertFalse(torch.equal(cfg({**base, "sigma": torch.tensor([sigma])}), torch.zeros_like(x)))

    def test_zero_init_sees_steps_outside_the_cfg_range(self):
        # The CFG range skips its inner hook outside its window, so zero-init
        # wraps it: the first step (sigma 1.0, outside [0.2, 0.8]) is the one
        # zeroed, not the first in-window step.
        state = CFGZeroStarState(1)
        ranged = make_guidance_range_cfg_function(make_cfg_zero_star_cfg_function(), sigma_start=0.2, sigma_end=0.8)
        cfg = make_zero_init_cfg_function(ranged, state)
        x = torch.ones(1, 1, 2, 2)
        base = {"input": x, "cond_denoised": x * 0.5, "uncond_denoised": x * 0.25, "cond_scale": 3.0}
        self.assertTrue(torch.equal(cfg({**base, "sigma": torch.tensor([1.0])}), torch.zeros_like(x)))
        self.assertTrue(torch.equal(cfg({**base, "sigma": torch.tensor([0.9])}), x - base["cond_denoised"]))
        inside = cfg({**base, "sigma": torch.tensor([0.7])})
        self.assertTrue(torch.allclose(inside, cfg_zero_star_combine(x * 0.5, x * 0.75, 3.0)))
        # The range's unconditional-pass skip still recognises the hook.
        skip = make_guidance_range_uncond_skip(cfg)
        uncond = [{"model_conds": {}}]
        self.assertIsNone(skip(None, [], uncond, None, torch.tensor([0.9]), {"sampler_cfg_function": cfg})[2])

    def test_auxiliary_probe_does_not_consume_zero_init_steps(self):
        from types import SimpleNamespace

        process = SimpleNamespace()
        state = CFGZeroStarState(1)
        cfg = make_zero_init_cfg_function(make_cfg_zero_star_cfg_function(), state, process=process)
        x = torch.ones(1, 1, 2, 2)
        base = {"input": x, "cond_denoised": x * 0.5, "uncond_denoised": x * 0.25, "cond_scale": 3.0}
        from modules import anima_support

        setattr(process, anima_support._AUXILIARY_DENOISER_ATTRIBUTE, True)
        self.assertFalse(torch.equal(cfg({**base, "sigma": torch.tensor([0.9])}), torch.zeros_like(x)))
        setattr(process, anima_support._AUXILIARY_DENOISER_ATTRIBUTE, False)
        self.assertTrue(torch.equal(cfg({**base, "sigma": torch.tensor([1.0])}), torch.zeros_like(x)))


    def test_zero_init_steps_do_not_call_the_model(self):
        from types import SimpleNamespace

        from modules import anima_support

        process = SimpleNamespace()
        state = CFGZeroStarState(1)
        calls = []

        def previous(model_function, args):  # an earlier wrapper is kept in the chain
            calls.append(("previous", float(args["timestep"][0])))
            return model_function(args["input"], args["timestep"], **args["c"])

        def model_function(x, t, **c):
            calls.append(("model", float(t[0])))
            return x * 0.5

        wrapper = make_zero_init_model_wrapper(previous, state, process=process)
        cfg = make_zero_init_cfg_function(make_cfg_zero_star_cfg_function(), state, process=process)
        x = torch.ones(2, 1, 2, 2)

        def run(sigma):
            args = {"input": x, "timestep": torch.tensor([sigma, sigma]), "c": {}, "cond_or_uncond": [0, 1]}
            return wrapper(model_function, args)

        # A PC3 probe at a new sigma runs the model and consumes no step.
        setattr(process, anima_support._AUXILIARY_DENOISER_ATTRIBUTE, True)
        self.assertTrue(torch.equal(run(0.9), x * 0.5))
        setattr(process, anima_support._AUXILIARY_DENOISER_ATTRIBUTE, False)
        calls.clear()

        # Step 0 (and a split second batch of it): no forward, denoised = x.
        for _ in range(2):
            denoised = run(1.0)
            self.assertTrue(torch.equal(denoised, x))
            self.assertIsNot(denoised, x)
        self.assertEqual(calls, [])
        cond = {"input": x[:1], "sigma": torch.tensor([1.0]), "cond_denoised": x[:1], "uncond_denoised": x[:1], "cond_scale": 3.0}
        self.assertTrue(torch.equal(cfg(cond), torch.zeros_like(x[:1])))
        # Step 1 runs the chain.
        self.assertTrue(torch.equal(run(0.8), x * 0.5))
        self.assertEqual([name for name, _ in calls], ["previous", "model"])

    def test_zero_init_model_wrapper_follows_the_pass_start_rule(self):
        state = CFGZeroStarState(1, start_sigma=1.0)
        wrapper = make_zero_init_model_wrapper(None, state)
        x = torch.ones(1, 1, 2, 2)
        args = {"input": x, "timestep": torch.tensor([0.6]), "c": {}, "cond_or_uncond": [0]}
        # A hires / img2img pass starts mid-schedule: nothing is skipped.
        self.assertTrue(torch.equal(wrapper(lambda x, t, **c: x * 0.5, args), x * 0.5))
        self.assertFalse(state.would_zero_init(0.6))
        fresh = CFGZeroStarState(1, start_sigma=1.0)
        self.assertTrue(fresh.would_zero_init(1.0))
        self.assertEqual(fresh._sigmas, [])  # a peek records nothing


class AdaptiveProjectedGuidanceTests(unittest.TestCase):
    def test_eta_one_without_cap_or_momentum_is_plain_cfg(self):
        torch.manual_seed(5)
        cond, uncond = torch.randn(2, 4, 1, 6, 5), torch.randn(2, 4, 1, 6, 5)
        state = APGState(eta=1.0, norm_threshold=0.0, momentum=0.0)
        for _ in range(2):
            actual = state.combine(cond, uncond, 4.5)
            self.assertTrue(torch.allclose(actual, uncond + 4.5 * (cond - uncond), atol=1e-5))

    def test_parallel_and_orthogonal_parts_are_orthogonal_per_sample(self):
        torch.manual_seed(6)
        value, direction = torch.randn(3, 4, 1, 5, 7), torch.randn(3, 4, 1, 5, 7)
        parallel, orthogonal = apg_project(value, direction)
        self.assertTrue(torch.allclose(parallel + orthogonal, value, atol=1e-6))
        dot = (orthogonal * direction).flatten(1).sum(1)
        self.assertTrue(torch.allclose(dot, torch.zeros(3), atol=1e-4))
        # Each parallel part is a multiple of its own sample's direction.
        cosine = torch.nn.functional.cosine_similarity(parallel.flatten(1), direction.flatten(1), dim=1)
        self.assertTrue(torch.allclose(cosine.abs(), torch.ones(3), atol=1e-5))

    def test_eta_zero_removes_the_parallel_update(self):
        cond = torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1)
        uncond = torch.tensor([1.0, -1.0]).reshape(1, 2, 1, 1)
        # diff = (1, 1): parallel (1, 0), orthogonal (0, 1) to cond.
        actual = APGState(eta=0.0, momentum=0.0).combine(cond, uncond, 3.0)
        self.assertTrue(torch.allclose(actual, cond + 2.0 * torch.tensor([0.0, 1.0]).reshape(1, 2, 1, 1)))

    def test_norm_threshold_caps_each_sample_update(self):
        cond = torch.zeros(2, 1, 1, 2)
        cond[..., 0] = 1.0
        uncond = cond.clone()
        uncond[0, ..., 1] = -30.0  # update norm 30, capped to 3
        uncond[1, ..., 1] = -1.0  # update norm 1, left alone
        actual = APGState(eta=1.0, norm_threshold=3.0, momentum=0.0).combine(cond, uncond, 2.0)
        update = actual - cond
        self.assertTrue(torch.allclose(update.flatten(1).norm(dim=1), torch.tensor([3.0, 1.0])))

    def test_momentum_accumulates_and_resets(self):
        cond = torch.zeros(1, 2, 1, 1)
        cond[:, 0] = 1.0
        uncond = cond.clone()
        uncond[:, 1] = -1.0  # diff (0, 1), orthogonal to cond
        state = APGState(eta=1.0, momentum=-0.5)
        first = state.combine(cond, uncond, 2.0)
        second = state.combine(cond, uncond, 2.0)
        self.assertTrue(torch.allclose(first - cond, torch.tensor([0.0, 1.0]).reshape(1, 2, 1, 1)))
        # running = diff + momentum * running = 1 - 0.5 = 0.5
        self.assertTrue(torch.allclose(second - cond, torch.tensor([0.0, 0.5]).reshape(1, 2, 1, 1)))
        state.reset()
        self.assertTrue(torch.equal(state.combine(cond, uncond, 2.0), first))

    def test_cfg_function_works_on_denoised_predictions_and_skips_cfg_one(self):
        from types import SimpleNamespace

        from modules import anima_support
        from modules.anima_guidance_state import clear_stateful_guidance, register_stateful_guidance, reset_stateful_guidance

        torch.manual_seed(7)
        x = torch.randn(1, 4, 1, 4, 4)
        cond, uncond = torch.randn_like(x), torch.randn_like(x)
        process = SimpleNamespace()
        clear_stateful_guidance(process)
        state = register_stateful_guidance(process, APGState(eta=0.0, norm_threshold=0.0, momentum=-0.5))
        cfg = make_apg_cfg_function(state, process=process)
        args = {"input": x, "sigma": torch.tensor([0.7]), "cond_denoised": cond, "uncond_denoised": uncond}
        expected = x - APGState(eta=0.0, momentum=-0.5).combine(cond, uncond, 4.0)
        self.assertTrue(torch.allclose(cfg({**args, "cond_scale": 4.0}), expected, atol=1e-6))
        self.assertIsNotNone(state._running)
        reset_stateful_guidance(SimpleNamespace(p=process))
        self.assertIsNone(state._running)
        # CFG 1: the conditional prediction, and the zero placeholder
        # unconditional prediction does not enter the momentum.
        placeholder = {**args, "uncond_denoised": torch.zeros_like(x), "cond_scale": 1.0}
        self.assertTrue(torch.equal(cfg(placeholder), x - cond))
        self.assertIsNone(state._running)
        # Auxiliary probes do not advance the momentum.
        setattr(process, anima_support._AUXILIARY_DENOISER_ATTRIBUTE, True)
        cfg({**args, "cond_scale": 4.0})
        self.assertIsNone(state._running)


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
