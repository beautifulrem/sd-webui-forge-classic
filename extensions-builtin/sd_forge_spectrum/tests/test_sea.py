import math
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.sea import (
    count_refreshes,
    l1rel,
    sea_filter,
    solve_delta_for_refresh_ratio,
    window_refresh_fraction,
)


class SeaTests(unittest.TestCase):
    def test_filter_preserves_shape_and_finite_values(self):
        value = torch.randn(1, 4, 8, 10)
        filtered = sea_filter(value, sigma=0.5, beta=2.0)
        self.assertEqual(filtered.shape, value.shape)
        self.assertTrue(torch.isfinite(filtered).all())

    def test_relative_distance_uses_most_changed_batch_row(self):
        previous = torch.ones(2, 1, 1, 2)
        current = previous.clone()
        current[0] += 0.1
        current[1] += 2.0

        self.assertAlmostEqual(l1rel(current, previous), 2.0)

    def test_delta_solver_tracks_requested_refresh_count(self):
        distances = [0.1, 0.2, 0.05, 0.4, 0.1, 0.15]
        delta = solve_delta_for_refresh_ratio(distances, 0.5)
        self.assertTrue(math.isfinite(delta) and delta > 0)
        self.assertLessEqual(count_refreshes(distances, delta), round(len(distances) * 0.5))

    def test_window_fraction_is_bounded(self):
        fraction = window_refresh_fraction(30, 6, 3, 2.0, 0.25)
        self.assertTrue(0.0 <= fraction <= 1.0)


if __name__ == "__main__":
    unittest.main()
