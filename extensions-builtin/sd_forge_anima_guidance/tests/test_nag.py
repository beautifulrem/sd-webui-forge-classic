from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.nag import NAGAttentionModifier


def fake_attention(q, k, v, *, mask=None):
    result = q.flatten(2) + k.mean(dim=1, keepdim=True).flatten(2)
    if mask is not None:
        result = result + mask.reshape(mask.shape[0], -1).mean(1).reshape(-1, 1, 1)
    return result


class NAGTests(unittest.TestCase):
    def setUp(self):
        self.q = torch.zeros(2, 2, 1, 1)
        self.k = torch.tensor([[[[3.0]], [[3.0]]], [[[1.0]], [[1.0]]]])
        self.v = torch.zeros_like(self.k)
        self.options = {"cond_or_uncond": [0, 1], "sigmas": torch.tensor([0.5])}

    def test_changes_only_positive_rows(self):
        modifier = NAGAttentionModifier(scale=1.0, tau=100.0, alpha=1.0, sigma_start=0.0, sigma_end=1.0)
        base = fake_attention(self.q, self.k, self.v)
        actual = modifier(
            fake_attention, self.q, self.k, self.v,
            transformer_options=self.options, is_self_attention=False,
        )
        self.assertTrue(torch.equal(actual[1], base[1]))
        self.assertTrue(torch.all(actual[0] > base[0]))

    def test_self_attention_and_outside_range_are_noops(self):
        modifier = NAGAttentionModifier(scale=2.0, tau=2.5, alpha=0.5, sigma_start=0.0, sigma_end=0.4)
        base = fake_attention(self.q, self.k, self.v)
        self.assertTrue(torch.equal(modifier(fake_attention, self.q, self.k, self.v, transformer_options=self.options, is_self_attention=False), base))
        self.assertTrue(torch.equal(modifier(fake_attention, self.q, self.k, self.v, transformer_options=self.options, is_self_attention=True), base))

    def test_uses_negative_mask_for_negative_attention(self):
        modifier = NAGAttentionModifier(scale=1.0, tau=100.0, alpha=1.0, sigma_start=0.0, sigma_end=1.0)
        mask = torch.tensor([[[[10.0]]], [[[2.0]]]])
        actual = modifier(
            fake_attention, self.q, self.k, self.v, mask=mask,
            transformer_options=self.options, is_self_attention=False,
        )
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual.shape, (2, 2, 1))

    def test_reports_once_when_cfg_branches_are_not_batched(self):
        reports = []
        modifier = NAGAttentionModifier(
            scale=2.0,
            tau=2.5,
            alpha=0.5,
            sigma_start=0.0,
            sigma_end=1.0,
            on_unbatched=lambda: reports.append(True),
        )
        options = {"cond_or_uncond": [0], "sigmas": torch.tensor([0.5])}
        modifier(fake_attention, self.q[:1], self.k[:1], self.v[:1], transformer_options=options, is_self_attention=False)
        modifier(fake_attention, self.q[:1], self.k[:1], self.v[:1], transformer_options=options, is_self_attention=False)
        self.assertEqual(reports, [True])


if __name__ == "__main__":
    unittest.main()
