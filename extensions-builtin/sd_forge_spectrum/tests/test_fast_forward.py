"""Spectrum's cached-step path must reproduce the real Anima DiT exactly.

``_fast_forward`` re-runs only the timestep embedding, final layer, unpatchify
and the predictor's denoised conversion on the feature captured right before
``final_layer``.  Fed the feature from a real forward, it must give the same
denoised output as the full model (float32, CPU, tiny seeded DiT).
"""

import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import operations
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.nn.anima import Anima
from lib_spectrum.forecaster import _ensure_capture_hook, _fast_forward


def _tiny_anima():
    # Same construction as tests/test_anima_reference_output.py.
    generator = torch.Generator().manual_seed(1234)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        model = Anima(4, 4, 2, 1, model_channels=256, crossattn_emb_channels=64, adaln_lora_dim=32, num_blocks=2, num_heads=2)
    state = {k: torch.randn(v.shape, generator=generator) * 0.05 for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), generator


def _anima_predictor():
    # Anima's sampling settings (modules_forge/packages/huggingface_guess/model_list.py).
    return PredictionDiscreteFlow(SimpleNamespace(sampling_settings={"multiplier": 1.0, "shift": 3.0}))


def _real_denoised(dit, predictor, sigma, x, context):
    """What KModel.apply_model does for Anima, capturing the pre-final_layer feature."""
    state = SimpleNamespace(captured_feature=None)
    local = dit.final_layer._forge_spectrum_local
    local.state = state
    try:
        xc = predictor.calculate_input(sigma, x)
        timestep = predictor.timestep(sigma).float()
        model_output = dit(xc, timestep, context=context, control=None, transformer_options={}).float()
    finally:
        local.state = None
    return predictor.calculate_denoised(sigma, model_output, x), state.captured_feature


class FastForwardTests(unittest.TestCase):
    def setUp(self):
        self.dit, self.generator = _tiny_anima()
        self.predictor = _anima_predictor()
        _ensure_capture_hook(self.dit)

    def _check_matches_real_forward(self, x, sigma, context):
        with torch.inference_mode():
            expected, feature = _real_denoised(self.dit, self.predictor, sigma, x, context)
            self.assertIsNotNone(feature, "capture hook did not record the pre-final_layer feature")
            actual = _fast_forward(self.dit, self.predictor, sigma, x, feature)
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return feature, expected

    def test_fast_forward_reproduces_real_denoised(self):
        x = torch.randn(2, 4, 1, 16, 12, generator=self.generator)
        sigma = torch.tensor([0.8, 0.35])
        context = torch.randn(2, 7, 64, generator=self.generator)
        self._check_matches_real_forward(x, sigma, context)

    def test_fast_forward_crops_patch_padding_like_the_model(self):
        # Odd spatial size: the DiT pads to the patch size and crops back.
        x = torch.randn(2, 4, 1, 15, 11, generator=self.generator)
        sigma = torch.tensor([0.6, 0.6])
        context = torch.randn(2, 7, 64, generator=self.generator)
        self._check_matches_real_forward(x, sigma, context)

    def test_fast_forward_depends_on_sigma(self):
        x = torch.randn(2, 4, 1, 16, 12, generator=self.generator)
        sigma = torch.tensor([0.7, 0.7])
        context = torch.randn(2, 7, 64, generator=self.generator)
        feature, expected = self._check_matches_real_forward(x, sigma, context)
        other_sigma = torch.tensor([0.3, 0.3])
        with torch.inference_mode():
            other = _fast_forward(self.dit, self.predictor, other_sigma, x, feature)
        self.assertGreater((other - expected).abs().max().item(), 1e-2)

    def test_fast_forward_keeps_compute_dtype_with_bf16_storage(self):
        # Manual cast: bf16-stored weights, fp32 compute. The real forward runs
        # final_layer in fp32, so the cached step must not downcast the feature
        # to the parameters' storage dtype (it drifted ~1e-2; FP8 raised).
        self.dit.to(torch.bfloat16)
        x = torch.randn(2, 4, 1, 16, 12, generator=self.generator)
        sigma = torch.tensor([0.8, 0.35])
        context = torch.randn(2, 7, 64, generator=self.generator)
        feature, expected = self._check_matches_real_forward(x, sigma, context)
        self.assertEqual(feature.dtype, torch.float32)
        with torch.inference_mode():
            storage_dtype = _fast_forward(self.dit, self.predictor, sigma, x, feature.to(torch.bfloat16))
        self.assertGreater((storage_dtype - expected).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
