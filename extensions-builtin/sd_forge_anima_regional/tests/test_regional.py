from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_regional.regional import conditioning_rows, parse_blocks, rectangle_token_mask


class RegionalTests(unittest.TestCase):
    def test_block_parser_clamps_to_model(self):
        self.assertEqual(parse_blocks("0-2,5,99", 8), {0, 1, 2, 5})

    def test_rectangle_mask_matches_anima_token_count(self):
        mask = rectangle_token_mask(
            latent_t=1,
            latent_h=128,
            latent_w=96,
            patch_t=1,
            patch_h=2,
            x=0.0,
            y=0.0,
            width=0.5,
            height=1.0,
            feather=0.02,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(mask.shape, (1, 64 * 48, 1))
        self.assertGreater(mask.sum(), 0)

    def test_only_conditional_rows_are_enabled(self):
        rows = conditioning_rows([0, 1], 4, "cpu", torch.float32)
        self.assertEqual(rows.flatten().tolist(), [1.0, 1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
