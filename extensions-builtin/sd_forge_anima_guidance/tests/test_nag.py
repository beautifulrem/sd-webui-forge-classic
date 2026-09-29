from __future__ import annotations

import pathlib
import sys
import unittest
import unittest.mock

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

    def test_sigma_window_is_read_once_per_model_call(self):
        from lib_anima_guidance import nag

        modifier = NAGAttentionModifier(scale=1.0, tau=100.0, alpha=1.0, sigma_start=0.0, sigma_end=1.0)
        calls = []
        original = nag.in_sigma_window
        nag.in_sigma_window = lambda *args: calls.append(args) or original(*args)
        try:
            for _ in range(3):  # three attention layers of one model call
                modifier(fake_attention, self.q, self.k, self.v, transformer_options=self.options, is_self_attention=False)
            next_call = dict(self.options, sigmas=torch.tensor([0.5]))
            modifier(fake_attention, self.q, self.k, self.v, transformer_options=next_call, is_self_attention=False)
        finally:
            nag.in_sigma_window = original
        self.assertEqual(len(calls), 2)

    def test_the_inner_attention_output_is_not_modified(self):
        modifier = NAGAttentionModifier(scale=1.0, tau=100.0, alpha=1.0, sigma_start=0.0, sigma_end=1.0)
        shared = fake_attention(self.q, self.k, self.v)
        before = shared.clone()

        def attention(q, k, v, mask=None):  # the full-batch call returns a shared buffer
            return shared if q.shape[0] == self.q.shape[0] else fake_attention(q, k, v, mask=mask)

        result = modifier(attention, self.q, self.k, self.v, transformer_options=self.options, is_self_attention=False)

        self.assertTrue(torch.equal(shared, before))
        self.assertFalse(torch.equal(result, before))

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
        # Hand computation with fake_attention (q = 0): a row's output is the
        # mean of its k plus the mean of its mask row.
        positive = 3.0 + 10.0  # positive k, positive mask
        negative = 1.0 + 2.0  # negative k with the negative row's mask
        guided = positive + 1.0 * (positive - negative)  # tau 100 does not clamp
        expected = torch.tensor([[[guided], [guided]], [[negative], [negative]]])
        self.assertTrue(torch.allclose(actual, expected))

    def test_blend_matches_nag_formula_with_tau_clamp_and_alpha(self):
        # NAG: z = z+ + s (z+ - z-), rescaled to at most tau x ||z+||_1,
        # then lerped with z+ by alpha.
        modifier = NAGAttentionModifier(scale=2.0, tau=1.5, alpha=0.25, sigma_start=0.0, sigma_end=1.0)
        q = torch.zeros(2, 1, 1, 2)
        k = torch.tensor([[[[1.0, 2.0]]], [[[3.0, -1.0]]]])

        def attention(q, k, v, *, mask=None):
            return k.flatten(2)

        actual = modifier(attention, q, k, torch.zeros_like(k), transformer_options=self.options, is_self_attention=False)
        z_pos, z_neg = torch.tensor([1.0, 2.0]), torch.tensor([3.0, -1.0])
        guided = z_pos + 2.0 * (z_pos - z_neg)  # (-3, 8): L1 11 vs 3 -> ratio 11/3 > 1.5
        normalized = guided * (1.5 / (11.0 / 3.0))
        expected_positive = z_pos + 0.25 * (normalized - z_pos)
        self.assertTrue(torch.allclose(actual[0, 0], expected_positive))
        self.assertTrue(torch.equal(actual[1, 0], z_neg))

    def test_active_follows_the_sigma_window(self):
        modifier = NAGAttentionModifier(scale=2.0, tau=2.5, alpha=0.5, sigma_start=0.0, sigma_end=0.4)
        self.assertFalse(modifier.active({"sigmas": torch.tensor([0.5])}))
        self.assertTrue(modifier.active({"sigmas": torch.tensor([0.3])}))
        self.assertFalse(modifier.active({"sigmas": torch.tensor([0.3]), "anima_nag_skip": "regional_context"}))
        idle = NAGAttentionModifier(scale=0.0, tau=2.5, alpha=0.5, sigma_start=0.0, sigma_end=1.0)
        self.assertFalse(idle.active({"sigmas": torch.tensor([0.3])}))

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

    def test_uncond_only_call_of_a_split_batch_is_not_reported(self):
        # Forge split the CFG batch for VRAM: the negative half has no rows
        # to guide, and the positive half applies NAG on its own.
        reports = []
        modifier = NAGAttentionModifier(
            scale=2.0, tau=2.5, alpha=0.5, sigma_start=0.0, sigma_end=1.0,
            on_unbatched=lambda: reports.append(True),
        )
        options = {"cond_or_uncond": [1], "sigmas": torch.tensor([0.5])}
        base = fake_attention(self.q[1:], self.k[1:], self.v[1:])
        actual = modifier(fake_attention, self.q[1:], self.k[1:], self.v[1:], transformer_options=options, is_self_attention=False)
        self.assertTrue(torch.equal(actual, base))
        self.assertEqual(reports, [])

    def test_positive_only_call_matches_batched_nag_via_negative_context(self):
        modifier = NAGAttentionModifier(scale=1.5, tau=2.0, alpha=0.7, sigma_start=0.0, sigma_end=1.0)
        batched = modifier(
            fake_attention, self.q, self.k, self.v,
            transformer_options=self.options, is_self_attention=False,
        )

        # CFG 1: only the positive row is run; the negative prompt's context
        # is projected with the layer's K/V (identity projection here).
        modifier.set_negative_context(torch.tensor([[[1.0]], [[1.0]]]).reshape(1, 2, 1))
        options = {
            "cond_or_uncond": [0],
            "sigmas": torch.tensor([0.5]),
            "anima_project_kv": lambda context: (context.unsqueeze(-1), torch.zeros_like(context).unsqueeze(-1)),
        }
        positive_only = modifier(
            fake_attention, self.q[:1], self.k[:1], self.v[:1],
            transformer_options=options, is_self_attention=False,
        )

        self.assertTrue(torch.allclose(positive_only, batched[:1]))

    def test_all_positive_rows_skip_the_copy_of_the_attention_output(self):
        # CFG 1 / Turbo: every row is positive, so the blend is the result;
        # copying ``base`` first would only be overwritten. A CFG batch
        # still copies (the negative rows keep the shared output).
        modifier = NAGAttentionModifier(scale=1.5, tau=2.0, alpha=0.7, sigma_start=0.0, sigma_end=1.0)
        base = torch.tensor([[[1.0, 2.0]], [[3.0, -1.0]]])
        negative = torch.tensor([[[0.5, 0.0]], [[-2.0, 1.0]]])
        original_clone = torch.Tensor.clone
        clones = []

        def counting_clone(tensor, *args, **kwargs):
            clones.append(tensor.shape)
            return original_clone(tensor, *args, **kwargs)

        with unittest.mock.patch.object(torch.Tensor, "clone", counting_clone):
            actual = modifier._blend(base, [0, 1], negative)
            self.assertEqual(clones, [])
            partial = modifier._blend(base, [0], negative[:1])
            self.assertEqual(clones, [base.shape])

        guided = base + 1.5 * (base - negative)
        ratio = guided.abs().sum(-1, keepdim=True) / base.abs().sum(-1, keepdim=True)
        expected = base.lerp(ratio.clamp(max=2.0) / ratio * guided, 0.7)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(partial[:1], expected[:1])
        self.assertTrue(torch.equal(partial[1:], base[1:]))
        self.assertTrue(torch.equal(base, torch.tensor([[[1.0, 2.0]], [[3.0, -1.0]]])))  # not written to

    def test_positive_only_call_applies_the_negative_prompts_negpip_mask(self):
        # `(tag:-1)` in the negative prompt: batched CFG projects the uncond V
        # from ``context * mask`` (K keeps the sign-restored token); the
        # positive-only path must project the stashed context the same way.
        def attention(q, k, v, *, mask=None):
            return q.flatten(2) + k.mean(dim=1, keepdim=True).flatten(2) + 10.0 * v.mean(dim=1, keepdim=True).flatten(2)

        modifier = NAGAttentionModifier(scale=1.5, tau=2.0, alpha=0.7, sigma_start=0.0, sigma_end=1.0)
        context = torch.tensor([1.0, 2.0]).reshape(1, 2, 1)
        negpip_mask = torch.tensor([1.0, -1.0]).reshape(1, 2, 1)
        k = torch.cat([self.k[:1], context.unsqueeze(-1)])
        v = torch.cat([torch.full_like(self.k[:1], 0.5), (context * negpip_mask).unsqueeze(-1)])
        batched = modifier(attention, self.q, k, v, transformer_options=self.options, is_self_attention=False)

        modifier.set_negative_context({"crossattn": context, "c_negpip_mask": negpip_mask})

        def project(context, value_context=None):  # identity K/V projection
            return context.unsqueeze(-1), (context if value_context is None else value_context).unsqueeze(-1)

        options = {"cond_or_uncond": [0], "sigmas": torch.tensor([0.5]), "anima_project_kv": project}
        positive_only = modifier(attention, self.q[:1], k[:1], v[:1], transformer_options=options, is_self_attention=False)

        self.assertTrue(torch.allclose(positive_only, batched[:1]))

    def test_negative_context_is_prepared_once_per_step_without_a_host_sync(self):
        from lib_anima_guidance import nag

        modifier = NAGAttentionModifier(scale=1.0, tau=2.0, alpha=1.0, sigma_start=0.0, sigma_end=1.0)
        context = torch.tensor([1.0, 2.0]).reshape(1, 2, 1)
        mask = torch.tensor([1.0, -1.0]).reshape(1, 2, 1)
        original = nag.split_conditioning
        nag.split_conditioning = lambda conds: self.fail("split_conditioning tests the mask on the host")
        try:
            modifier.set_negative_context({"crossattn": context, "c_negpip_mask": mask})
        finally:
            nag.split_conditioning = original

        projected = []

        def project(context, value_context=None):
            projected.append((context, value_context))
            return context.unsqueeze(-1), value_context.unsqueeze(-1)

        options = {"cond_or_uncond": [0], "sigmas": torch.tensor([0.5]), "anima_project_kv": project}
        for _ in range(3):  # three cross-attention layers of one step
            modifier(fake_attention, self.q[:1], self.k[:1], self.v[:1], transformer_options=options, is_self_attention=False)
        self.assertEqual(len(projected), 3)
        self.assertTrue(all(pair[0] is projected[0][0] and pair[1] is projected[0][1] for pair in projected))
        self.assertTrue(torch.equal(projected[0][1], context * mask))

        # The next step's context replaces the prepared one.
        modifier.set_negative_context({"crossattn": context * 2, "c_negpip_mask": torch.ones_like(mask)})
        modifier(fake_attention, self.q[:1], self.k[:1], self.v[:1], transformer_options=options, is_self_attention=False)
        self.assertTrue(torch.equal(projected[-1][1], context * 2))

    def test_dict_conditioning_uses_its_cross_attention_context(self):
        modifier = NAGAttentionModifier(scale=1.0, tau=2.0, alpha=1.0, sigma_start=0.0, sigma_end=1.0)
        context = torch.ones(1, 3, 4)
        modifier.set_negative_context({"crossattn": context})
        self.assertIs(modifier.negative_context, context)
        modifier.set_negative_context(None)
        self.assertIsNone(modifier.negative_context)


if __name__ == "__main__":
    unittest.main()
