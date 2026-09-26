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
