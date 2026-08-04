from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.smc import SMCCFGState


class SMCCFGTests(unittest.TestCase):
    def test_alpha_zero_matches_standard_cfg(self):
        torch.manual_seed(44)
        cond = torch.randn(2, 4, 8, 8)
        uncond = torch.randn_like(cond)
        scale = 4.0
        actual = SMCCFGState(alpha=0.0).combine(cond, uncond, scale)
        expected = uncond + scale * (cond - uncond)
        self.assertTrue(torch.allclose(actual, expected))

    def test_state_resets_safely_after_shape_change(self):
        state = SMCCFGState(alpha=0.2)
        first = state.combine(torch.ones(1, 4, 8, 8), torch.zeros(1, 4, 8, 8), 4.0)
        second = state.combine(torch.ones(1, 4, 16, 16), torch.zeros(1, 4, 16, 16), 4.0)
        self.assertEqual(first.shape, (1, 4, 8, 8))
        self.assertEqual(second.shape, (1, 4, 16, 16))
        self.assertTrue(torch.isfinite(second).all())

    def test_previous_error_changes_later_control_surface(self):
        state = SMCCFGState(alpha=0.2, lam=5.0)
        uncond = torch.zeros(1, 1, 2, 2)
        first = state.combine(torch.ones_like(uncond), uncond, 4.0)
        second = state.combine(-torch.ones_like(uncond), uncond, 4.0)
        fresh = SMCCFGState(alpha=0.2, lam=5.0).combine(-torch.ones_like(uncond), uncond, 4.0)
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(torch.equal(second, fresh))


if __name__ == "__main__":
    unittest.main()
