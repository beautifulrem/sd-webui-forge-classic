from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_regional.regional import conditioning_rows, rectangle_token_mask
from modules.anima_support import parse_block_list


class RegionalTests(unittest.TestCase):
    def test_block_parser_clamps_to_model(self):
        self.assertEqual(parse_block_list("0-2,5,99", 8, feature="Regional"), {0, 1, 2, 5})
        with self.assertRaisesRegex(ValueError, "Regional block selection is empty"):
            parse_block_list("99", 8, feature="Regional")

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


    def test_cross_attention_blend_matches_per_block_evaluation(self):
        from lib_anima_regional.regional import Region, RegionalState, _make_cross_attention_wrapper

        torch.manual_seed(0)
        regions = [
            Region(torch.randn(1, 3, 4), 0.0, 0.0, 0.5, 1.0, 0.8),
            Region(torch.randn(1, 3, 4), 0.5, 0.0, 0.5, 1.0, 0.0),  # zero strength
        ]
        state = RegionalState(regions, {0}, 0.02, 0.25, 0.0, 1.0)

        dit = type("Dit", (), {"patch_temporal": 1, "patch_spatial": 2})()
        state.prepare_masks(torch.zeros(2, 4, 1, 8, 8), dit)
        self.assertEqual(state.region_nonempty, [True, False])

        def attention(x, context=None, rope_emb=None, transformer_options={}):
            return x * context.mean(dim=(1, 2)).view(-1, 1, 1)

        x = torch.randn(2, 16, 4)
        context = torch.randn(2, 3, 4)
        options = {"cond_or_uncond": [0, 1]}
        wrapped = _make_cross_attention_wrapper(attention, state, 0)
        first = wrapped(x, context=context, transformer_options=options)
        second = wrapped(x, context=context, transformer_options=options)  # served from the call cache
        self.assertTrue(torch.equal(first, second))
        # The negative-prompt row is untouched by regional blending
        self.assertTrue(torch.equal(first[1], attention(x, context=context)[1]))
        state.end_call()
        self.assertIsNone(state.current_masks)
        self.assertEqual(state.call_cache, {})


if __name__ == "__main__":
    unittest.main()
