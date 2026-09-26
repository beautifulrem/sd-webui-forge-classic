from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_regional.regional import rectangle_token_mask
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

        seen_batches = []

        def attention(x, context=None, rope_emb=None, transformer_options={}):
            seen_batches.append(x.shape[0])
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
        # The positive row, by hand: 4x4 tokens, region 1 covers the left two
        # columns fully (feather 0.02 < half a token); region 2 is skipped.
        # There blend = 0.8 * (1 - 0.25) and the regional output replaces 60%.
        base_row = x[0] * context[0].mean()
        regional_row = x[0] * regions[0].conditioning.mean()
        inside = torch.tensor([1.0, 1.0, 0.0, 0.0]).repeat(4).view(16, 1)
        expected = base_row * (1 - 0.6 * inside) + regional_row * (0.6 * inside)
        self.assertTrue(torch.allclose(first[0], expected, atol=1e-6))
        # The regional forward runs on the positive-prompt row only
        self.assertEqual(seen_batches[:2], [2, 1])
        masks = state.current_masks
        state.end_call()
        self.assertIsNone(state.current_masks)
        # Masks are reused for the rest of the pass while the latent shape holds
        state.prepare_masks(torch.zeros(2, 4, 1, 8, 8), dit)
        self.assertIs(state.current_masks, masks)
        state.prepare_masks(torch.zeros(2, 4, 1, 8, 16), dit)
        self.assertIsNot(state.current_masks, masks)

    def test_rows_outside_the_markers_are_all_blended(self):
        from lib_anima_regional.regional import Region, RegionalState, _make_cross_attention_wrapper

        torch.manual_seed(0)
        state = RegionalState([Region(torch.randn(1, 3, 4), 0.0, 0.0, 1.0, 1.0, 1.0)], {0}, 0.02, 0.0, 0.0, 1.0)
        state.prepare_masks(torch.zeros(3, 4, 1, 8, 8), type("Dit", (), {"patch_temporal": 1, "patch_spatial": 2})())

        def attention(x, context=None, rope_emb=None, transformer_options={}):
            return x * context.mean(dim=(1, 2)).view(-1, 1, 1)

        x = torch.randn(3, 16, 4)
        wrapped = _make_cross_attention_wrapper(attention, state, 0)
        # Markers that do not divide the batch: every row is treated as positive
        out = wrapped(x, context=torch.randn(3, 3, 4), transformer_options={"cond_or_uncond": [0, 1]})
        self.assertTrue(torch.allclose(out, attention(x, context=state.regions[0].conditioning.expand(3, -1, -1))))
        # Negative-only call: the base output is returned untouched
        base_context = torch.randn(3, 3, 4)
        out = wrapped(x, context=base_context, transformer_options={"cond_or_uncond": [1]})
        self.assertTrue(torch.equal(out, attention(x, context=base_context)))


    def test_each_region_context_gets_its_own_negpip_mask(self):
        from lib_anima_regional.regional import Region, RegionalState, _make_cross_attention_wrapper
        from modules.anima_support import NEGPIP_CONTEXT_MASK_KEY, NEGPIP_MASK_KEY

        torch.manual_seed(0)
        region_mask = torch.tensor([[[1.0], [-1.0], [1.0]]])
        regions = [
            Region(torch.randn(1, 3, 4), 0.0, 0.0, 0.5, 1.0, 1.0, region_mask),
            Region(torch.randn(1, 3, 4), 0.5, 0.0, 0.5, 1.0, 1.0),  # no negative weights
        ]
        state = RegionalState(regions, {0}, 0.02, 0.0, 0.0, 1.0)
        state.prepare_masks(torch.zeros(4, 4, 1, 8, 8), type("Dit", (), {"patch_temporal": 1, "patch_spatial": 2})())
        seen = []

        def attention(x, context=None, rope_emb=None, transformer_options={}):
            seen.append(dict(transformer_options))
            return x * context.mean(dim=(1, 2)).view(-1, 1, 1)

        base_mask = -torch.ones(4, 3, 1)
        options = {"cond_or_uncond": [1, 0], NEGPIP_MASK_KEY: base_mask}
        _make_cross_attention_wrapper(attention, state, 0)(torch.randn(4, 16, 4), context=torch.randn(4, 3, 4), transformer_options=options)

        base, first, second = seen
        self.assertNotIn(NEGPIP_CONTEXT_MASK_KEY, base)
        # The region forwards run on the two positive rows, each with its own mask
        self.assertTrue(torch.equal(first[NEGPIP_CONTEXT_MASK_KEY], region_mask.expand(2, -1, -1)))
        self.assertIsNone(second[NEGPIP_CONTEXT_MASK_KEY])
        self.assertEqual(first["cond_or_uncond"], [0])


if __name__ == "__main__":
    unittest.main()
