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

    def test_non_anima_models_forecast_the_model_output(self):
        class FakePatcher:
            def __init__(self):
                self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
                self.model_options = {}
                self.wrapper = None

            def clone(self):
                return self

            def set_model_unet_function_wrapper(self, wrapper):
                self.wrapper = wrapper

        patcher = SpectrumNode.patch(
            FakePatcher(),
            steps=8,
            weight=1.0,
            degree=1,
            lam=1e-6,
            window_size=2,
            flex_window=0.0,
            warmup_steps=3,
            stop_caching_step=1.0,
            tail_actual_steps=1,
            history_size=3,
            schedule="Window",
            refresh_ratio=0.0,
            sea_beta=2.0,
            compat_policy="Conservative",
            verbose=False,
            sea_cache_dir="unused",
            sea_cache_context={},
            process=SimpleNamespace(),
        )
        calls = []

        def model_function(x, timestep, **_kwargs):
            calls.append(float(timestep[0]))
            return torch.full_like(x, float(len(calls)))

        outputs = []
        for step in range(5):
            sigma = torch.tensor([1.0 - step * 0.1])
            outputs.append(
                patcher.wrapper(
                    model_function,
                    {"input": torch.zeros(1, 1), "timestep": sigma, "c": {}, "cond_or_uncond": [0]},
                )
            )

        self.assertEqual(len(calls), 4)
        self.assertTrue(torch.allclose(outputs[3], torch.full((1, 1), 4.0), atol=1e-3))

    def test_wrappers_installed_earlier_in_the_pass_are_kept(self):
        class FakePatcher:
            def __init__(self, previous):
                self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
                self.model_options = {"model_function_wrapper": previous}
                self.wrapper = None

            def clone(self):
                return self

            def set_model_unet_function_wrapper(self, wrapper):
                self.wrapper = wrapper

        calls = []

        def regional(model_function, args):
            calls.append("regional")
            return model_function(args["input"], args["timestep"], **args["c"])

        patcher = SpectrumNode.patch(
            FakePatcher(regional),
            steps=4,
            weight=1.0,
            degree=1,
            lam=0.1,
            window_size=2,
            flex_window=0.0,
            warmup_steps=1,
            stop_caching_step=1.0,
            tail_actual_steps=1,
            history_size=3,
            schedule="Window",
            refresh_ratio=0.0,
            sea_beta=2.0,
            compat_policy="Conservative",
            verbose=False,
            sea_cache_dir="unused",
            sea_cache_context={},
            process=SimpleNamespace(),
        )
        patcher.wrapper(
            lambda x, t, **c: x,
            {"input": torch.zeros(1, 1), "timestep": torch.ones(1), "c": {}, "cond_or_uncond": [0]},
        )

        self.assertEqual(calls, ["regional"])

    def test_callable_force_flag_is_evaluated_per_step(self):
        class FakePatcher:
            def __init__(self):
                self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
                self.model_options = {}
                self.wrapper = None

            def clone(self):
                return self

            def set_model_unet_function_wrapper(self, wrapper):
                self.wrapper = wrapper

        patcher = SpectrumNode.patch(
            FakePatcher(),
            steps=8,
            weight=1.0,
            degree=1,
            lam=1e-6,
            window_size=2,
            flex_window=0.0,
            warmup_steps=3,
            stop_caching_step=1.0,
            tail_actual_steps=1,
            history_size=3,
            schedule="Window",
            refresh_ratio=0.0,
            sea_beta=2.0,
            compat_policy="Legacy / fastest",
            verbose=False,
            sea_cache_dir="unused",
            sea_cache_context={},
            process=SimpleNamespace(),
        )
        calls = []

        def model_function(x, timestep, **_kwargs):
            calls.append(round(float(timestep[0]), 2))
            return torch.full_like(x, float(len(calls)))

        def run(force_actual):
            calls.clear()
            for step in range(5):
                options = {"forge_spectrum_force_actual": force_actual}
                patcher.wrapper(
                    model_function,
                    {
                        "input": torch.zeros(1, 1),
                        "timestep": torch.tensor([1.0 - step * 0.1]),
                        "c": {"transformer_options": options},
                        "cond_or_uncond": [0],
                    },
                )
            return list(calls)

        self.assertEqual(len(run(lambda sigma: False)), 4)  # step 3 served from cache
        self.assertEqual(len(run(lambda sigma: sigma > 0.65)), 5)  # step 3 (sigma 0.7) forced


if __name__ == "__main__":
    unittest.main()
