import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.forecaster import ChebyshevForecaster


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


if __name__ == "__main__":
    unittest.main()
