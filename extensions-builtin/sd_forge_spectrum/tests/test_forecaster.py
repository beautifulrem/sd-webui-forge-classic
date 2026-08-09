import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import ChebyshevForecaster, SpectrumNode


class ForecasterTests(unittest.TestCase):
    def test_reconstructs_linear_feature_trend(self):
        forecaster = ChebyshevForecaster(degree=2, history_size=6, lam=1e-5, steps=12, weight=0.5)
        for step in range(6):
            forecaster.update(step, torch.full((1, 2, 2), float(step)))
        prediction = forecaster.predict(6)
        self.assertTrue(forecaster.ready)
        self.assertTrue(torch.allclose(prediction, torch.full_like(prediction, 6.0), atol=2e-2))

    def test_shape_change_resets_history(self):
        forecaster = ChebyshevForecaster(degree=1, history_size=4, lam=0.1, steps=10, weight=0.5)
        forecaster.update(0, torch.zeros(1, 2))
        forecaster.update(1, torch.zeros(1, 3))
        self.assertEqual(len(forecaster.features), 1)
        self.assertEqual(forecaster.shape, (1, 3))

    def test_pc3_auxiliary_forward_bypasses_spectrum_state(self):
        class FakePatcher:
            def __init__(self):
                dit = SimpleNamespace(
                    final_layer=torch.nn.Identity(),
                    t_embedder=object(),
                    t_embedding_norm=object(),
                    unpatchify=object(),
                )
                self.model = SimpleNamespace(
                    diffusion_model=dit,
                    predictor=object(),
                )
                self.model_options = {}
                self.wrapper = None

            def clone(self):
                return self

            def set_model_unet_function_wrapper(self, wrapper):
                self.wrapper = wrapper

        process = SimpleNamespace(_anima_auxiliary_denoiser=True)
        patcher = SpectrumNode.patch(
            FakePatcher(),
            steps=10,
            weight=0.25,
            degree=1,
            lam=0.1,
            window_size=2,
            flex_window=0.0,
            warmup_steps=2,
            stop_caching_step=0.9,
            tail_actual_steps=1,
            history_size=3,
            schedule="Window",
            refresh_ratio=0.0,
            sea_beta=2.0,
            compat_policy="Conservative",
            verbose=False,
            sea_cache_dir="unused",
            sea_cache_context={},
            process=process,
        )
        sentinel = torch.tensor([123.0])

        result = patcher.wrapper(
            lambda *_args, **_kwargs: sentinel,
            {
                "input": torch.zeros(1),
                "timestep": torch.ones(1),
                "c": {},
            },
        )

        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
