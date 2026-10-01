import sys
from types import SimpleNamespace

import pytest
import torch

import modules
import modules.sd_samplers_anima as anima_samplers
import modules.sd_samplers_anima_speed as speed
from modules.anima_feature_conflicts import CONFLICT_METADATA_KEY, REGIONAL, SPEED_SAMPLER, set_feature_active
from modules.anima_guidance_state import register_stateful_guidance


MU, STD = 0.7, 0.4


def _anima_sigmas(steps, start=1.0, dtype=torch.float32):
    t = torch.linspace(start, 0.0, steps + 1, dtype=dtype)
    return 3 * t / (1 + 2 * t)  # Anima's shift 3


def _gaussian_model(x, sigma, **_extra_args):
    """Closed-form RF denoiser for i.i.d. N(MU, STD^2) data."""

    t = sigma.view(-1, *([1] * (x.ndim - 1)))
    return MU + (1 - t) * STD**2 / ((1 - t) ** 2 * STD**2 + t**2) * (x - (1 - t) * MU)


class _RecordingModel:
    def __init__(self, process=None):
        self.p = process if process is not None else SimpleNamespace(extra_generation_params={})
        self.step = 0
        self.total_steps = 0
        self.shapes = []

    def __call__(self, x, sigma, **extra_args):
        self.shapes.append(tuple(x.shape[-2:]))
        return _gaussian_model(x, sigma, **extra_args)


@pytest.fixture(autouse=True)
def _no_engine_check(monkeypatch):
    monkeypatch.setattr(speed, "require_anima_flow_denoiser", lambda _model, _feature: None)
    monkeypatch.setattr(anima_samplers, "require_anima_flow_denoiser", lambda _model, _feature: None)


def _noise(*shape, seed=0, dtype=torch.float32):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_dct_is_orthonormal_and_matches_scipy():
    x = _noise(2, 3, 10, 14, dtype=torch.float64)
    assert torch.allclose(speed.idct2(speed.dct2(x)), x.float(), atol=1e-5)
    fft = pytest.importorskip("scipy.fft")
    expected = fft.dctn(x.numpy(), type=2, norm="ortho", axes=(-2, -1))
    assert torch.allclose(speed.dct2(x), torch.from_numpy(expected).float(), atol=1e-4)


def test_spectral_expansion_keeps_the_low_band_and_fills_the_rest_with_scaled_noise():
    low = _noise(2, 4, 8, 12, seed=1)
    noise = _noise(2, 4, 16, 24, seed=2)

    expanded = speed.spectral_expand(low, noise.clone(), 0.6)

    coefficients = speed.dct2(expanded)
    assert expanded.shape[-2:] == (16, 24)
    assert torch.allclose(coefficients[..., :8, :12], speed.dct2(low), atol=1e-4)
    high = torch.ones(16, 24, dtype=torch.bool)
    high[:8, :12] = False
    assert torch.allclose(coefficients[..., high], 0.6 * noise[..., high], atol=1e-4)


def test_injected_noise_matches_the_aligned_sigma():
    """After the Eq. 5 rescale the new coefficients carry noise at exactly the
    aligned sigma of Eq. 6, the level the model is queried at."""

    noise = speed.dct2(_noise(4, 16, 64, 64, seed=3))  # unit-variance coefficients
    low = torch.zeros(4, 16, 32, 32)
    sigma = 0.7
    kappa, aligned = speed.align(sigma, 2.0)

    coefficients = speed.dct2(kappa * speed.spectral_expand(low, noise, sigma))

    high = torch.ones(64, 64, dtype=torch.bool)
    high[:32, :32] = False
    assert aligned == pytest.approx(2 * sigma / (1 + sigma))
    assert float(coefficients[..., high].std()) == pytest.approx(aligned, rel=0.02)


def test_truncating_then_expanding_pure_noise_returns_the_seed_noise():
    noise = _noise(1, 16, 32, 48, seed=4)
    coefficients = speed.dct2(noise)
    coarse = speed.idct2(coefficients[..., :16, :24])
    kappa, aligned = speed.align(1.0, 2.0)

    assert (kappa, aligned) == (1.0, 1.0)
    assert float(coarse.std()) == pytest.approx(1.0, rel=0.05)
    assert torch.allclose(speed.spectral_expand(coarse, coefficients, 1.0), noise, atol=1e-4)


def test_stage_schedule_is_monotone_and_ends_at_full_resolution():
    model = _RecordingModel()
    sigmas = _anima_sigmas(30)
    x = _noise(1, 16, 128, 96)

    result = speed.sample_anima_speed(model, x, sigmas, disable=True)

    plan, _ = speed.plan_speed(sigmas.tolist(), 128, 96)
    assert result.shape == x.shape
    assert len(model.shapes) == 30
    assert model.shapes[0] == (64, 48) and model.shapes[-1] == (128, 96)
    areas = [h * w for h, w in model.shapes]
    assert areas == sorted(areas)
    assert areas.count(64 * 48) == plan.step and 0 < plan.step < 30
    assert sigmas[plan.step] <= speed.transition_sigma(128, 96) < sigmas[plan.step - 1]
    assert SPEED_SAMPLER in model.p.extra_generation_params


@pytest.mark.parametrize(
    "size, sigma, step",
    [
        ((128, 128), 0.694, 18),  # 1024 x 1024
        ((152, 104), 0.748, 16),  # 832 x 1216
    ],
)
def test_default_switch_is_the_delta_optimal_sigma_of_the_wan_vae_spectrum(size, sigma, step):
    # Paper Table 4 Wan 2.1 VAE fit; anima_lora's Anima beta = 2.26 is not
    # adopted (no amplitude published with it), see the research doc.
    assert (speed._SPECTRUM_A, speed._SPECTRUM_BETA) == (219.484718, 2.422687)
    assert speed.transition_sigma(*size) == pytest.approx(sigma, abs=1e-3)

    plan, _ = speed.plan_speed(_anima_sigmas(30).tolist(), *size)

    assert plan.step == step


def _first_step_at_or_below(sigmas, threshold):
    return next(i for i, sigma in enumerate(sigmas) if sigma <= threshold)


def test_switch_sigma_override_moves_the_switch_step():
    sigmas = _anima_sigmas(30).tolist()
    automatic, _ = speed.plan_speed(sigmas, 128, 128)

    later, _ = speed.plan_speed(sigmas, 128, 128, 0.5)
    earlier, _ = speed.plan_speed(sigmas, 128, 128, 0.8)

    assert later.step == _first_step_at_or_below(sigmas, 0.5) > automatic.step
    assert earlier.step == _first_step_at_or_below(sigmas, 0.8) < automatic.step
    assert later.sigma <= 0.5 < sigmas[later.step - 1]
    assert later.aligned_sigma == pytest.approx(speed.align(later.sigma, 2.0)[1])


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), (0, None), (0.0, None), (-0.3, None), ("", None), ("0.6", 0.6), (0.2, 0.5), (0.5, 0.5), (0.9, 0.9), (0.99, 0.9), (5, 0.9),
     # Non-finite values are the automatic default, not a bound
     (float("inf"), None), ("inf", None), (float("-inf"), None), (float("nan"), None), ("NaN", None)],
)
def test_switch_sigma_override_is_bounded(value, expected):
    assert speed.resolve_switch_sigma(value) == expected


def test_out_of_range_overrides_plan_like_their_bound():
    sigmas = _anima_sigmas(30).tolist()

    assert speed.plan_speed(sigmas, 128, 128, 0.1) == speed.plan_speed(sigmas, 128, 128, 0.5)
    assert speed.plan_speed(sigmas, 128, 128, 1.0) == speed.plan_speed(sigmas, 128, 128, 0.9)
    assert speed.plan_speed(sigmas, 128, 128, 0.0) == speed.plan_speed(sigmas, 128, 128)


def _fake_shared(monkeypatch, switch_sigma):
    fake_shared = SimpleNamespace(state=SimpleNamespace(current_latent=None), opts=SimpleNamespace(**{speed.SWITCH_SIGMA_OPTION: switch_sigma}))
    monkeypatch.setitem(sys.modules, "modules.shared", fake_shared)
    monkeypatch.setattr(modules, "shared", fake_shared, raising=False)


def test_the_switch_sigma_setting_drives_the_run_and_is_recorded(monkeypatch):
    _fake_shared(monkeypatch, 0.55)
    model = _RecordingModel()
    sigmas = _anima_sigmas(30)

    speed.sample_anima_speed(model, _noise(1, 4, 128, 128), sigmas, disable=True)

    step = _first_step_at_or_below(sigmas.tolist(), 0.55)
    assert model.shapes.count((64, 64)) == step
    assert model.p.extra_generation_params[speed.SWITCH_SIGMA_INFOTEXT] == 0.55
    assert f"until step {step}/30" in model.p.extra_generation_params[SPEED_SAMPLER]


def test_the_automatic_switch_sigma_writes_no_infotext(monkeypatch):
    _fake_shared(monkeypatch, 0.0)
    model = _RecordingModel()

    speed.sample_anima_speed(model, _noise(1, 4, 128, 128), _anima_sigmas(30), disable=True)

    assert speed.SWITCH_SIGMA_INFOTEXT not in model.p.extra_generation_params
    assert SPEED_SAMPLER in model.p.extra_generation_params


def test_the_switch_sigma_setting_is_pasteable_from_its_infotext_key():
    from modules.shared_options import options_templates

    option = options_templates[speed.SWITCH_SIGMA_OPTION]

    assert option.infotext == speed.SWITCH_SIGMA_INFOTEXT
    assert option.default == 0.0
    assert option.component_args["maximum"] == speed.SWITCH_SIGMA_RANGE[1]


def test_odd_latent_sizes_run_on_an_even_coarse_grid_and_return_the_input_shape():
    model = _RecordingModel()
    x = _noise(1, 4, 131, 157)

    result = speed.sample_anima_speed(model, x, _anima_sigmas(12), disable=True)

    assert result.shape == x.shape
    assert all(h % 2 == 0 and w % 2 == 0 for h, w in model.shapes if (h, w) != (131, 157))
    assert model.shapes[0] != (131, 157)


def test_gaussian_denoiser_matches_full_resolution_flow_euler(monkeypatch):
    """With the closed-form Gaussian denoiser SPEED reproduces the full-res
    Euler sample from the same seed; skipping the Eq. 5-6 alignment does not."""

    noise = _noise(2, 4, 128, 128, dtype=torch.float64)
    sigmas = _anima_sigmas(30, dtype=torch.float64)
    reference = anima_samplers.sample_anima_flow_euler(_gaussian_model, noise.clone(), sigmas, disable=True)

    result = speed.sample_anima_speed(_RecordingModel(), noise.clone(), sigmas, disable=True)

    assert float((result - reference).abs().max()) < 0.05
    assert float(result.mean()) == pytest.approx(float(reference.mean()), abs=1e-3)
    assert float(result.std()) == pytest.approx(float(reference.std()), rel=0.02)

    monkeypatch.setattr(speed, "align", lambda sigma, _ratio: (1.0, sigma))
    unaligned = speed.sample_anima_speed(_RecordingModel(), noise.clone(), sigmas, disable=True)
    assert abs(float(unaligned.mean()) - float(reference.mean())) > 0.03


def test_stateful_guidance_restarts_at_the_resolution_switch():
    process = SimpleNamespace(extra_generation_params={})
    resets = []
    register_stateful_guidance(process, SimpleNamespace(reset=lambda: resets.append(1)))

    speed.sample_anima_speed(_RecordingModel(process), _noise(1, 4, 128, 128), _anima_sigmas(10), disable=True)

    assert len(resets) == 1


def test_img2img_falls_back_to_flow_euler_with_an_infotext_note():
    model = _RecordingModel()
    x = _noise(1, 4, 128, 128)
    sigmas = _anima_sigmas(10, start=0.6)

    result = speed.sample_anima_speed(model, x.clone(), sigmas, disable=True)

    expected = anima_samplers.sample_anima_flow_euler(_gaussian_model, x.clone(), sigmas, disable=True)
    assert torch.equal(result, expected)
    assert set(model.shapes) == {(128, 128)}
    note = model.p.extra_generation_params[CONFLICT_METADATA_KEY]
    assert "Anima Flow Euler" in note and "denoise < 1" in note
    assert SPEED_SAMPLER not in model.p.extra_generation_params


@pytest.mark.parametrize(
    "configure, reason",
    [
        (lambda model, unet: setattr(model, "mask", torch.ones(1)), "inpainting mask"),
        (lambda model, unet: set_feature_active(model.p, REGIONAL), REGIONAL),
        (lambda model, unet: setattr(unet, "controlnet_linked_list", object()), "ControlNet"),
        (lambda model, unet: unet.model_options.update(model_function_wrapper=lambda f, a: f(a["input"])), "model wrapper"),
        (lambda model, unet: unet.model_options["transformer_options"].update(patches={"attn1_patch": [object()]}), "attention patches"),
    ],
)
def test_resolution_bound_inputs_fall_back(configure, reason):
    unet = SimpleNamespace(controlnet_linked_list=None, model_options={"transformer_options": {}})
    process = SimpleNamespace(extra_generation_params={}, sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))
    model = _RecordingModel(process)
    sigmas = _anima_sigmas(10).tolist()
    assert speed.fallback_reason(model, sigmas) is None

    configure(model, unet)

    assert reason in speed.fallback_reason(model, sigmas)


def test_wrappers_that_work_at_any_size_do_not_force_a_fallback():
    def modulation(function, args):
        return function(args["input"])

    modulation.__resolution_independent__ = True
    unet = SimpleNamespace(controlnet_linked_list=None, model_options={"transformer_options": {}, "model_function_wrapper": modulation})
    process = SimpleNamespace(extra_generation_params={}, sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))

    assert speed.fallback_reason(_RecordingModel(process), _anima_sigmas(10).tolist()) is None


def test_reference_latents_fall_back(monkeypatch):
    from backend.args import dynamic_args

    monkeypatch.setattr(dynamic_args, "ref_latents", [torch.zeros(1, 16, 1, 8, 8)])
    assert speed.fallback_reason(_RecordingModel(), _anima_sigmas(10).tolist()) == "reference latents"


def test_small_or_single_step_passes_fall_back():
    assert speed.plan_speed(_anima_sigmas(10).tolist(), 96, 96)[0] is None  # 48 x 48 coarse < 512 x 384 px
    assert speed.plan_speed(_anima_sigmas(1).tolist(), 128, 128)[0] is None

    model = _RecordingModel()
    speed.sample_anima_speed(model, _noise(1, 4, 96, 96), _anima_sigmas(10), disable=True)
    assert set(model.shapes) == {(96, 96)}
    assert "512x384" in model.p.extra_generation_params[CONFLICT_METADATA_KEY]


def _process_with_transformer_options(transformer_options):
    unet = SimpleNamespace(controlnet_linked_list=None, model_options={"transformer_options": transformer_options})
    return SimpleNamespace(extra_generation_params={}, sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))


def test_the_aligned_sigma_is_published_as_sampling_sigmas_for_the_run():
    # The LoRA Stage Scheduler and artist windows read a model call's
    # progress from Forge's sampling_sigmas: the switch step's aligned sigma
    # must map to its own step, not an earlier one of the original schedule.
    from modules.anima_lora_support import sigma_progress

    sigmas = _anima_sigmas(30)
    options = {"sampling_sigmas": sigmas}
    process = _process_with_transformer_options(options)
    progress = []

    class ProgressModel(_RecordingModel):
        def __call__(self, x, sigma, **extra_args):
            progress.append(sigma_progress(sigma, options["sampling_sigmas"]))
            return super().__call__(x, sigma, **extra_args)

    speed.sample_anima_speed(ProgressModel(process), _noise(1, 4, 128, 128), sigmas, disable=True)

    plan, _ = speed.plan_speed(sigmas.tolist(), 128, 128)
    assert 0 < plan.step < 30
    assert progress == pytest.approx([k / 30 for k in range(30)])
    assert options["sampling_sigmas"] is sigmas


@pytest.mark.parametrize("published", [True, False])
def test_sampling_sigmas_are_restored_after_an_interrupt(monkeypatch, published):
    fake_shared = SimpleNamespace(state=SimpleNamespace(current_latent=None))
    monkeypatch.setitem(sys.modules, "modules.shared", fake_shared)
    monkeypatch.setattr(modules, "shared", fake_shared, raising=False)
    sigmas = _anima_sigmas(10)
    plan, _ = speed.plan_speed(sigmas.tolist(), 128, 128)
    options = {"sampling_sigmas": sigmas} if published else {}
    seen = []

    class Interrupted(Exception):
        pass

    class InterruptedModel(_RecordingModel):
        def __call__(self, x, sigma, **extra_args):
            seen.append(float(options["sampling_sigmas"][plan.step]))
            if len(seen) > plan.step:
                raise Interrupted
            return super().__call__(x, sigma, **extra_args)

    with pytest.raises(Interrupted):
        speed.sample_anima_speed(InterruptedModel(_process_with_transformer_options(options)), _noise(1, 4, 128, 128), sigmas, disable=True)

    assert seen == pytest.approx([plan.aligned_sigma] * (plan.step + 1))
    assert options == ({"sampling_sigmas": sigmas} if published else {})
    assert not published or options["sampling_sigmas"] is sigmas


def test_interrupt_in_the_coarse_stage_leaves_a_full_size_latent(monkeypatch):
    state = SimpleNamespace(current_latent=None)
    fake_shared = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "modules.shared", fake_shared)
    monkeypatch.setattr(modules, "shared", fake_shared, raising=False)

    class Interrupted(Exception):
        pass

    def model(x, sigma, **_extra_args):
        if state.current_latent is not None:
            raise Interrupted
        state.current_latent = _gaussian_model(x, sigma)
        return state.current_latent

    with pytest.raises(Interrupted):
        speed.sample_anima_speed(model, _noise(1, 4, 128, 128), _anima_sigmas(10), disable=True)

    assert state.current_latent.shape == (1, 4, 128, 128)


def test_interrupt_on_the_switch_step_leaves_a_full_size_latent(monkeypatch):
    # The sample is already full size when the switch step's denoiser call is
    # interrupted, but Forge raises before store_latent: the published latent
    # is still the coarse stage's last one
    state = SimpleNamespace(current_latent=None)
    fake_shared = SimpleNamespace(state=state)
    monkeypatch.setitem(sys.modules, "modules.shared", fake_shared)
    monkeypatch.setattr(modules, "shared", fake_shared, raising=False)
    sigmas = _anima_sigmas(10)
    plan, _ = speed.plan_speed(sigmas.tolist(), 128, 128)
    shapes = []

    class Interrupted(Exception):
        pass

    def model(x, sigma, **_extra_args):
        if len(shapes) == plan.step:
            raise Interrupted
        shapes.append(tuple(x.shape[-2:]))
        state.current_latent = _gaussian_model(x, sigma)
        return state.current_latent

    with pytest.raises(Interrupted):
        speed.sample_anima_speed(model, _noise(1, 4, 128, 128), sigmas, disable=True)

    assert plan.step > 0 and set(shapes) == {plan.size}
    assert state.current_latent.shape == (1, 4, 128, 128)


def test_the_automatic_switch_sigma_stays_within_the_documented_bounds():
    # Over every size the UI offers (8 px steps) that SPEED runs at: the
    # automatic switch depends on the short side only, at most 0.9 from a
    # 392 px short side up, and below 0.5 only from 1984 px (see SWITCH_SIGMA_RANGE)
    for short_px in range(64, 2049, 8):
        coarse = speed._coarse_size(short_px // 8, 2048 // 8)
        if coarse[0] * coarse[1] < speed._MIN_COARSE_AREA:
            continue
        sigma = speed.transition_sigma(short_px // 8, 2048 // 8)
        assert sigma == speed.transition_sigma(2048 // 8, short_px // 8)
        assert sigma <= 0.9 or short_px < 392, short_px
        assert round(sigma, 3) <= 0.903, short_px  # the narrowest supported strips
        assert (sigma < 0.5) == (short_px >= 1984), short_px
