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
    _low_frequency,
    apg_project,
    cfg_zero_star_combine,
    cfg_zero_star_scale,
    fdg_combine,
    make_apg_cfg_function,
    make_cfg_zero_star_cfg_function,
    IGNORES_UNCOND,
    make_fdg_cfg_function,
    make_guidance_range_cfg_function,
    make_guidance_range_uncond_skip,
    make_momentum_post_cfg_function,
    make_tcfg_cfg_function,
    make_zero_init_cfg_function,
    make_zero_init_model_wrapper,
    tcfg_combine,
    tcfg_damp_uncond,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _run_forge_combine(x, cond_value, uncond_value, *, cond_scale=4.0, strengths=(0.7, 0.8), sigma=0.5, **options):
    """Forge's real ``sampling_function_inner`` with fixed model predictions."""

    from backend.sampling.sampling_function import sampling_function_inner

    class Model:
        apply_model = None  # the wrapper below replaces it

        def memory_required(self, shape):
            return 0

    def model_wrapper(apply_model, args):
        return torch.cat([cond_value if kind == 0 else uncond_value for kind in args["cond_or_uncond"]])

    cond = [{"model_conds": {}, "strength": strength} for strength in strengths]
    uncond = [{"model_conds": {}}]
    return sampling_function_inner(
        Model(), x, torch.tensor([sigma]), uncond, cond, cond_scale,
        model_options={"model_function_wrapper": model_wrapper, **options},
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

    def test_low_band_matches_the_frozen_kornia_pyramid(self):
        # Generated once with kornia 0.6.12: pyrup(pyrdown(value)), the low
        # band of the reference node's two-level Laplacian pyramid.
        fixture = torch.load(FIXTURES / "fdg_pyramid_low_band.pt")
        self.assertTrue(torch.allclose(_low_frequency(fixture["value"]), fixture["low"], atol=1e-6))

    def test_low_band_matches_kornia_when_installed(self):
        try:
            import kornia
        except ImportError:
            self.skipTest("kornia is not installed")
        torch.manual_seed(13)
        for shape in ((1, 4, 16, 16), (2, 16, 104, 152), (1, 3, 6, 10)):
            value = torch.randn(shape)
            expected = kornia.geometry.pyrup(kornia.geometry.pyrdown(value))
            self.assertTrue(torch.allclose(_low_frequency(value), expected, atol=1e-6), shape)
        # Anima's (B, C, T, H, W) latents filter each plane the same way.
        value = torch.randn(2, 4, 1, 8, 12)
        expected = kornia.geometry.pyrup(kornia.geometry.pyrdown(value.reshape(2, 4, 8, 12)))
        self.assertTrue(torch.allclose(_low_frequency(value), expected.reshape(value.shape), atol=1e-6))

    def test_odd_sides_keep_their_size(self):
        value = torch.randn(1, 4, 7, 9)
        low = _low_frequency(value)
        self.assertEqual(low.shape, value.shape)
        self.assertTrue(torch.isfinite(low).all())

    def test_constant_delta_is_all_low_band(self):
        # A constant survives the reflect-padded blur and the bilinear
        # resampling unchanged, so it has no detail band: the detail scale
        # is inert.
        uncond = torch.randn(1, 4, 4, 6)
        cond = uncond + 0.5
        expected = uncond + 4.0 * 0.5
        for detail in (0.0, 1.0, 9.0):
            self.assertTrue(torch.allclose(fdg_combine(cond, uncond, 4.0, detail), expected, atol=1e-6))

    def test_zero_mean_checkerboard_delta_is_all_detail_band(self):
        # The binomial taps 1 - 4 + 6 - 4 + 1 cancel on the checkerboard (the
        # reflect padding keeps it alternating), so its low band is zero and
        # the whole delta is guided by the detail scale.
        uncond = torch.randn(1, 4, 4, 6)
        delta = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(2, 3)
        cond = uncond + delta
        for detail in (0.0, 1.0, 9.0):
            actual = fdg_combine(cond, uncond, 4.0, detail)
            self.assertTrue(torch.allclose(actual, uncond + detail * delta, atol=1e-6))

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


def _tcfg_svd_reference(cond, uncond):
    """TCFG's damping as the paper states it: a full SVD of [uncond; cond] per sample, in float64."""

    damped = []
    for c, u in zip(cond.double(), uncond.double()):
        _, _, vh = torch.linalg.svd(torch.stack([u.flatten(), c.flatten()]), full_matrices=False)
        v1 = vh[0]
        damped.append(((u.flatten() @ v1) * v1).reshape(u.shape))
    return torch.stack(damped)


class TCFGTests(unittest.TestCase):
    def test_damping_matches_a_full_svd_projection(self):
        torch.manual_seed(20)
        base = torch.randn(3, 4, 1, 6, 5)
        cases = {
            "independent": (torch.randn_like(base), torch.randn_like(base)),
            # Typical CFG: the two predictions differ by a small part.
            "close": (base, base + 0.1 * torch.randn_like(base)),
            "uncond larger": (base, 3.0 * base + torch.randn_like(base)),
        }
        for name, (cond, uncond) in cases.items():
            with self.subTest(name):
                actual = tcfg_damp_uncond(cond, uncond)
                self.assertEqual(actual.dtype, torch.float32)
                expected = _tcfg_svd_reference(cond, uncond)
                self.assertTrue(torch.allclose(actual.double(), expected, atol=1e-5, rtol=1e-4))

    def test_the_damped_part_is_what_the_svd_calls_tangential(self):
        # uncond' lies along v1 and the dropped part is orthogonal to it.
        torch.manual_seed(21)
        cond, uncond = torch.randn(2, 16, 8), torch.randn(2, 16, 8)
        damped = tcfg_damp_uncond(cond, uncond)
        dropped = uncond - damped
        self.assertTrue(torch.allclose((dropped * damped).sum(dim=(1, 2)), torch.zeros(2), atol=1e-4))
        # Parallel predictions have no tangential part: plain CFG.
        self.assertTrue(torch.allclose(tcfg_damp_uncond(cond, -2.0 * cond), -2.0 * cond, atol=1e-5))
        self.assertTrue(torch.allclose(tcfg_combine(cond, 0.5 * cond, 4.0), 0.5 * cond + 4.0 * (cond - 0.5 * cond), atol=1e-5))

    def test_residuals_or_velocities_give_the_same_combine(self):
        # Forge's residual is sigma * v; the SVD direction ignores that factor.
        torch.manual_seed(22)
        cond, uncond = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        self.assertTrue(torch.allclose(tcfg_combine(0.7 * cond, 0.7 * uncond, 5.0), 0.7 * tcfg_combine(cond, uncond, 5.0), atol=1e-5))

    def test_degenerate_predictions_stay_finite(self):
        zero, value = torch.zeros(1, 4, 2, 2), torch.randn(1, 4, 2, 2)
        self.assertTrue(torch.equal(tcfg_damp_uncond(zero, zero), zero))
        # A zero unconditional prediction projects to zero.
        self.assertTrue(torch.allclose(tcfg_damp_uncond(value, zero), zero, atol=1e-6))
        # Equal singular values: every direction is singular, uncond is kept.
        cond = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 2)
        uncond = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 2)
        self.assertTrue(torch.allclose(tcfg_damp_uncond(cond, uncond), uncond))

    def test_each_image_is_damped_on_its_own(self):
        torch.manual_seed(23)
        cond, uncond = torch.randn(3, 4, 1, 4, 4), torch.randn(3, 4, 1, 4, 4)
        batched = tcfg_combine(cond, uncond, 4.5)
        for index in range(3):
            single = tcfg_combine(cond[index : index + 1], uncond[index : index + 1], 4.5)
            self.assertTrue(torch.allclose(batched[index : index + 1], single, atol=1e-6))
        # Changing one image leaves the others alone.
        other = uncond.clone()
        other[1] = 10.0 * torch.randn_like(other[1])
        changed = tcfg_combine(cond, other, 4.5)
        self.assertTrue(torch.equal(changed[0], batched[0]))
        self.assertTrue(torch.equal(changed[2], batched[2]))

    def test_cfg_function_runs_on_velocity_residuals(self):
        torch.manual_seed(24)
        x = torch.randn(2, 4, 8, 8)
        sigma = 0.6
        v_cond, v_uncond = torch.randn_like(x), torch.randn_like(x)
        args = {
            "input": x,
            "sigma": torch.tensor([sigma] * 2),
            "cond_denoised": x - sigma * v_cond,
            "uncond_denoised": x - sigma * v_uncond,
            "cond_scale": 4.0,
        }
        residual = make_tcfg_cfg_function()(args)
        damped = _tcfg_svd_reference(v_cond, v_uncond).float()
        expected = sigma * (damped + 4.0 * (v_cond - damped))
        self.assertTrue(torch.allclose(residual, expected, atol=1e-4))

    def test_half_inputs_are_computed_in_float32(self):
        torch.manual_seed(25)
        x = torch.randn(1, 4, 8, 8)
        cond, uncond = x - 0.5 * torch.randn_like(x), x - 0.5 * torch.randn_like(x)
        args = {"input": x, "cond_denoised": cond, "uncond_denoised": uncond, "cond_scale": 5.0}
        half = {key: value.half() if torch.is_tensor(value) else value for key, value in args.items()}
        full = make_tcfg_cfg_function()(args)
        actual = make_tcfg_cfg_function()(half)
        self.assertEqual(actual.dtype, torch.float16)
        expected = make_tcfg_cfg_function()({key: value.float() if torch.is_tensor(value) else value for key, value in half.items()})
        self.assertTrue(torch.allclose(actual.float(), expected, atol=1e-2))
        self.assertTrue(torch.allclose(actual.float(), full, atol=5e-2))

    def test_cfg_one_is_the_conditional_prediction(self):
        torch.manual_seed(26)
        x = torch.randn(1, 4, 8, 8)
        cond, uncond = torch.randn_like(x), torch.randn_like(x)
        cfg = make_tcfg_cfg_function()
        placeholder = {"input": x, "cond_denoised": cond, "uncond_denoised": torch.zeros_like(x), "cond_scale": 1.0, "edit_strength": 1.5}
        self.assertTrue(torch.equal(cfg(placeholder), x - cond))
        # A real unconditional pass at CFG 1 and weight 1 is plain CFG 1 too.
        real = {"input": x, "cond_denoised": cond, "uncond_denoised": uncond, "cond_scale": 1.0,
                "model_options": {"disable_cfg1_optimization": True}}
        self.assertTrue(torch.allclose(cfg(real), x - cond, atol=1e-6))


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

    def test_full_window_matches_forge_default_combine_with_and_prompt_weights(self):
        # AND sub-prompt weights summing to 1.5: Forge's default combine
        # multiplies the guidance by that sum (edit_strength). Every combine
        # mode at its identity settings must do the same.
        from lib_anima_guidance.skim import make_skim_cfg_function
        from lib_anima_guidance.smc import SMCCFGState, make_smc_cfg_function

        torch.manual_seed(5)
        x = torch.randn(1, 4, 8, 8)
        cond_value, uncond_value = torch.randn_like(x), torch.randn_like(x)
        expected = _run_forge_combine(x, cond_value, uncond_value)
        self.assertTrue(torch.allclose(expected, uncond_value + (cond_value - uncond_value) * 4.0 * 1.5))

        ranged = make_guidance_range_cfg_function(None, sigma_start=1.0, sigma_end=0.0)
        identity_hooks = {
            "range": ranged,
            "fdg equal scales": make_fdg_cfg_function(4.0),
            "apg eta 1": make_apg_cfg_function(APGState(eta=1.0, norm_threshold=0.0, momentum=0.0)),
            "smc alpha 0": make_smc_cfg_function(SMCCFGState(lam=5.0, alpha=0.0)),
            "skim outside its window": make_skim_cfg_function(
                None, sigma_start=0.4, sigma_end=0.0, skimming_scale=2.0,
                full_skim_negative=False, disable_flipping_filter=False, flip_sigma=None,
            ),
        }
        for name, hook in identity_hooks.items():
            with self.subTest(name):
                actual = _run_forge_combine(
                    x, cond_value, uncond_value,
                    sampler_cfg_function=hook,
                    sampler_pre_cfg_function=[make_guidance_range_uncond_skip(ranged)] if hook is ranged else [],
                )
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

        # CFG-Zero* is plain CFG when s* = 1: the conditional residual is the
        # unconditional one plus a part orthogonal to it.
        uncond_residual = torch.randn_like(x)
        extra = torch.randn_like(x)
        extra -= (extra * uncond_residual).sum() / uncond_residual.square().sum() * uncond_residual
        cond_star, uncond_star = x - (uncond_residual + extra), x - uncond_residual
        actual = _run_forge_combine(x, cond_star, uncond_star, sampler_cfg_function=make_cfg_zero_star_cfg_function())
        self.assertTrue(torch.allclose(actual, _run_forge_combine(x, cond_star, uncond_star), atol=1e-5))

        # TCFG is plain CFG when the residuals are parallel (no tangential part).
        cond_par, uncond_par = x - 0.5 * uncond_residual, x - uncond_residual
        actual = _run_forge_combine(x, cond_par, uncond_par, sampler_cfg_function=make_tcfg_cfg_function())
        self.assertTrue(torch.allclose(actual, _run_forge_combine(x, cond_par, uncond_par), atol=1e-5))

    def test_and_prompt_weight_acts_like_a_larger_cfg_scale(self):
        # Outside the identity settings too, weights summing to 1.5 at CFG 4
        # guide like one prompt at CFG 6 (Skimmed CFG's mask and fallback
        # included), as in Forge's default combine.
        from lib_anima_guidance.skim import make_skim_cfg_function
        from lib_anima_guidance.smc import SMCCFGState, make_smc_cfg_function

        torch.manual_seed(9)
        x = torch.randn(1, 4, 8, 8)
        cond_value, uncond_value = torch.randn_like(x), torch.randn_like(x)
        hooks = {
            "cfg-zero*": make_cfg_zero_star_cfg_function,
            "tcfg": make_tcfg_cfg_function,
            "apg": lambda: make_apg_cfg_function(APGState(eta=0.0, norm_threshold=5.0, momentum=-0.5)),
            "smc": lambda: make_smc_cfg_function(SMCCFGState(lam=5.0, alpha=0.2)),
            "skim": lambda: make_skim_cfg_function(
                None, sigma_start=1.0, sigma_end=0.0, skimming_scale=2.0,
                full_skim_negative=False, disable_flipping_filter=False, flip_sigma=None,
            ),
        }
        for name, make_hook in hooks.items():
            with self.subTest(name):
                weighted = _run_forge_combine(x, cond_value, uncond_value, sampler_cfg_function=make_hook())
                single = _run_forge_combine(
                    x, cond_value, uncond_value, cond_scale=6.0, strengths=(1.0,), sampler_cfg_function=make_hook(),
                )
                self.assertTrue(torch.allclose(weighted, single, atol=1e-5))
        # FDG scales both bands by the weight.
        weighted = _run_forge_combine(x, cond_value, uncond_value, sampler_cfg_function=make_fdg_cfg_function(8.0))
        single = _run_forge_combine(
            x, cond_value, uncond_value, cond_scale=6.0, strengths=(1.0,), sampler_cfg_function=make_fdg_cfg_function(12.0),
        )
        self.assertTrue(torch.allclose(weighted, single, atol=1e-5))

    def test_cfg_one_ignores_the_placeholder_even_with_and_prompt_weights(self):
        # At CFG 1 Forge skips the negative pass; the combine modes return
        # the conditional prediction instead of guiding against zeros.
        from lib_anima_guidance.skim import make_skim_cfg_function
        from lib_anima_guidance.smc import SMCCFGState, make_smc_cfg_function

        torch.manual_seed(10)
        x = torch.randn(1, 4, 8, 8)
        cond = torch.randn_like(x)
        args = {
            "input": x, "sigma": torch.tensor([0.5]), "cond_denoised": cond,
            "uncond_denoised": torch.zeros_like(x), "cond_scale": 1.0, "edit_strength": 1.5,
        }
        skim = make_skim_cfg_function(
            make_cfg_zero_star_cfg_function(), sigma_start=1.0, sigma_end=0.0, skimming_scale=2.0,
            full_skim_negative=False, disable_flipping_filter=False, flip_sigma=None,
        )
        for hook in (
            make_cfg_zero_star_cfg_function(),
            make_tcfg_cfg_function(),
            make_apg_cfg_function(APGState()),
            make_smc_cfg_function(SMCCFGState()),
            skim,
        ):
            self.assertTrue(torch.equal(hook(args), x - cond))


    def test_cfg_one_has_no_jump_at_the_window_edge_with_and_prompt_weights(self):
        # Forge's own combine gives edit_strength * cond against its CFG 1
        # zero placeholder; the CFG range (with its uncond skip) and Skimmed
        # CFG alone give the conditional prediction inside and outside their
        # windows instead, like SMC-CFG, APG and CFG-Zero*.
        from lib_anima_guidance.skim import make_skim_cfg_function

        torch.manual_seed(11)
        x = torch.randn(1, 4, 8, 8)
        cond_value, uncond_value = torch.randn_like(x), torch.randn_like(x)
        ranged = make_guidance_range_cfg_function(None, sigma_start=0.8, sigma_end=0.2)
        hooks = {
            "range": {"sampler_cfg_function": ranged,
                      "sampler_pre_cfg_function": [make_guidance_range_uncond_skip(ranged)]},
            "skim": {"sampler_cfg_function": make_skim_cfg_function(
                None, sigma_start=0.8, sigma_end=0.2, skimming_scale=2.0,
                full_skim_negative=False, disable_flipping_filter=False, flip_sigma=None,
            )},
        }
        for name, options in hooks.items():
            for sigma in (0.5, 0.9):
                with self.subTest(name, sigma=sigma):
                    actual = _run_forge_combine(
                        x, cond_value, uncond_value, cond_scale=1.0, sigma=sigma, **options,
                    )
                    self.assertTrue(torch.allclose(actual, cond_value, atol=1e-6))
            with self.subTest(name, weights="sum to 1"):
                # Where the weights sum to 1 this is Forge's default combine.
                actual = _run_forge_combine(
                    x, cond_value, uncond_value, cond_scale=1.0, strengths=(0.4, 0.6), **options,
                )
                expected = _run_forge_combine(x, cond_value, uncond_value, cond_scale=1.0, strengths=(0.4, 0.6))
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_cfg_one_with_a_real_unconditional_pass_keeps_forge_default_combine(self):
        # With disable_cfg1_optimization the negative pass runs, so inside
        # the window the range combines as Forge does.
        torch.manual_seed(12)
        x = torch.randn(1, 4, 8, 8)
        cond_value, uncond_value = torch.randn_like(x), torch.randn_like(x)
        ranged = make_guidance_range_cfg_function(None, sigma_start=0.8, sigma_end=0.2)
        actual = _run_forge_combine(
            x, cond_value, uncond_value, cond_scale=1.0,
            sampler_cfg_function=ranged, disable_cfg1_optimization=True,
        )
        expected = _run_forge_combine(x, cond_value, uncond_value, cond_scale=1.0, disable_cfg1_optimization=True)
        self.assertTrue(torch.allclose(expected, uncond_value + (cond_value - uncond_value) * 1.5, atol=1e-6))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

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
