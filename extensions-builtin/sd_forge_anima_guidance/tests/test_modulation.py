from __future__ import annotations

import pathlib
import sys
import threading
import types
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance.modulation import (
    ModulationPatch,
    _expand_rows,
    _modulation_pre_hook,
    clip_prompt_text,
    prepare_modulation_vectors,
    project_pooled,
)


def _adapter():
    return {
        "scales": torch.ones(2, 6),
        "text_embedder_clip.linear_1.weight": torch.eye(6, 3),
        "text_embedder_clip.linear_1.bias": torch.zeros(6),
        "text_embedder_clip.linear_2.weight": torch.eye(6),
        "text_embedder_clip.linear_2.bias": torch.zeros(6),
    }


class ModulationTests(unittest.TestCase):
    def test_projection_matches_two_layer_definition(self):
        pooled = torch.tensor([[1.0, -2.0, 3.0]])
        projected = project_pooled(pooled, _adapter())
        expected = torch.nn.functional.silu(torch.tensor([[1.0, -2.0, 3.0, 0.0, 0.0, 0.0]]))
        self.assertTrue(torch.allclose(projected, expected))

    def test_direction_is_applied_only_to_positive_base(self):
        adapter = _adapter()
        original = __import__("lib_anima_guidance.modulation", fromlist=["_load_adapter"])._load_adapter
        module = __import__("lib_anima_guidance.modulation", fromlist=["_load_adapter"])
        module._load_adapter = lambda _path: adapter
        try:
            positive, negative, _ = prepare_modulation_vectors(
                adapter_path="unused",
                base_positive=torch.tensor([[1.0, 0.0, 0.0]]),
                base_negative=torch.tensor([[0.0, 1.0, 0.0]]),
                direction_positive=torch.tensor([[0.0, 0.0, 1.0]]),
                direction_negative=torch.zeros(1, 3),
                weight=2.0,
            )
        finally:
            module._load_adapter = original
        self.assertTrue(torch.allclose(negative, project_pooled(torch.tensor([[0.0, 1.0, 0.0]]), adapter)))
        self.assertFalse(torch.allclose(positive, project_pooled(torch.tensor([[1.0, 0.0, 0.0]]), adapter)))

    def test_block_hook_adds_delta_without_mutating_input(self):
        class Block:
            _forge_anima_modulation_local = threading.local()

        block = Block()
        block._forge_anima_modulation_local.delta = torch.ones(2, 1, 6)

        current = torch.zeros(2, 1, 6)
        _, updated = _modulation_pre_hook(block, (), {"adaln_lora_B_T_3D": current})
        self.assertTrue(torch.equal(current, torch.zeros_like(current)))
        self.assertTrue(torch.equal(updated["adaln_lora_B_T_3D"], torch.ones_like(current)))

    def test_single_prompt_expands_to_batch(self):
        expanded = _expand_rows(torch.ones(1, 6), 3)
        self.assertEqual(expanded.shape, (3, 6))

    def test_spectrum_cache_safety_follows_the_wrapped_wrapper(self):
        class Patcher:
            def __init__(self, previous):
                dit = types.SimpleNamespace(
                    blocks=[torch.nn.Identity(), torch.nn.Identity()],
                    final_layer=types.SimpleNamespace(hidden_size=2),
                )
                self.model = types.SimpleNamespace(diffusion_model=dit)
                self.model_options = {} if previous is None else {"model_function_wrapper": previous}

            def clone(self):
                return self

            def set_model_unet_function_wrapper(self, wrapper):
                self.model_options["model_function_wrapper"] = wrapper

        def safe(model_function, args):
            return model_function(args["input"], args["timestep"], **args["c"])

        safe.__spectrum_cache_safe__ = True
        patch = ModulationPatch(torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(2, 6), 0, -1)
        for previous, expected in ((None, True), (safe, True), (lambda function, args: None, False)):
            patched, _, _ = patch.apply(Patcher(previous))
            self.assertIs(patched.model_options["model_function_wrapper"].__spectrum_cache_safe__, expected)

        safe.__resolution_independent__ = True
        for previous, expected in ((None, True), (safe, True), (lambda function, args: None, False)):
            patched, _, _ = patch.apply(Patcher(previous))
            self.assertIs(patched.model_options["model_function_wrapper"].__resolution_independent__, expected)


class _Patcher:
    def __init__(self, block_count, hidden_size=2):
        dit = types.SimpleNamespace(
            blocks=[torch.nn.Identity() for _ in range(block_count)],
            final_layer=types.SimpleNamespace(hidden_size=hidden_size),
        )
        self.model = types.SimpleNamespace(diffusion_model=dit)
        self.model_options = {}

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def _call(patched, branches, observe):
    """Run the installed wrapper once; ``observe`` sees the blocks' deltas."""

    def model_function(x, t, **c):
        observe(patched.model.diffusion_model.blocks)
        return x

    args = {"input": torch.zeros(len(branches), 4), "timestep": torch.zeros(len(branches)), "c": {}, "cond_or_uncond": branches}
    patched.model_options["model_function_wrapper"](model_function, args)


def _deltas_by_block(model_blocks, start=0, end=-1):
    # Adapter row r scales by r + 1, so each block's delta names its row.
    scales = torch.arange(1.0, 29.0).unsqueeze(1).repeat(1, 6)
    patch = ModulationPatch(torch.ones(1, 6), torch.zeros(1, 6), scales, start, end)
    patched, effective_start, effective_end = patch.apply(_Patcher(model_blocks))
    seen = {}

    def observe(blocks):
        for index, block in enumerate(blocks):
            delta = getattr(block._forge_anima_modulation_local, "delta", None)
            seen[index] = None if delta is None else int(delta.flatten()[0]) - 1

    _call(patched, [0], observe)
    return seen, (effective_start, effective_end)


# Gazingstars123/Anima-2.9B expand_manifest.json, as listed by the reference
# (ComfyUI-Spectrum-KSampler mod_guidance_layouts.py): the inserted blocks.
ANIMA_29B_INSERTED = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)


class ModulationBlockLayoutTests(unittest.TestCase):
    def test_same_depth_is_the_identity(self):
        seen, window = _deltas_by_block(28)
        self.assertEqual(seen, {index: index for index in range(28)})
        self.assertEqual(window, (0, 27))

    def test_2_9b_rows_go_to_the_preserved_blocks(self):
        seen, window = _deltas_by_block(40)
        preserved = [index for index in range(40) if index not in ANIMA_29B_INSERTED]
        self.assertEqual({index: seen[index] for index in preserved}, {index: row for row, index in enumerate(preserved)})
        self.assertTrue(all(seen[index] is None for index in ANIMA_29B_INSERTED))
        self.assertEqual(window, (0, 27))

    def test_3_8b_rows_go_to_the_preserved_blocks(self):
        from modules.anima_lora_support import ANIMA_BLOCK_MAPPINGS

        seen, _ = _deltas_by_block(52)
        mapping = ANIMA_BLOCK_MAPPINGS[(28, 52)]
        for index in range(52):
            original = mapping.index(mapping[index]) == index  # the first copy of its source block
            self.assertEqual(seen[index], mapping[index] if original else None, index)
        self.assertEqual(sum(row is not None for row in seen.values()), 28)

    def test_start_and_end_count_adapter_blocks(self):
        seen, window = _deltas_by_block(40, start=8, end=26)
        self.assertEqual(window, (8, 26))
        self.assertEqual(sorted(row for row in seen.values() if row is not None), list(range(8, 27)))

    def test_unknown_depth_or_width_is_rejected(self):
        with self.assertRaises(RuntimeError):
            ModulationPatch(torch.zeros(1, 6), torch.zeros(1, 6), torch.ones(28, 6), 0, -1).apply(_Patcher(30))
        with self.assertRaises(RuntimeError):
            ModulationPatch(torch.zeros(1, 9), torch.zeros(1, 9), torch.ones(28, 9), 0, -1).apply(_Patcher(40))

    def test_block_deltas_are_built_once_per_batch_layout(self):
        patch = ModulationPatch(torch.ones(1, 6), torch.zeros(1, 6), torch.ones(28, 6), 0, -1)
        patched, _, _ = patch.apply(_Patcher(28))
        deltas = []

        def observe(blocks):
            deltas.append(blocks[5]._forge_anima_modulation_local.delta)

        _call(patched, [0], observe)
        _call(patched, [0], observe)
        _call(patched, [0, 1], observe)
        self.assertIs(deltas[0], deltas[1])
        self.assertEqual(tuple(deltas[2].shape), (2, 1, 6))
        self.assertTrue(torch.equal(deltas[2][:, 0, 0], torch.tensor([1.0, 0.0])))  # cond row, uncond row


class ClipPromptTextTests(unittest.TestCase):
    def test_emphasis_and_prompt_editing_are_resolved_to_plain_text(self):
        self.assertEqual(clip_prompt_text("a (red:1.3) [cat:dog:5] BREAK ((tail)), [soft]", 10), "a red cat tail, soft")
        self.assertEqual(clip_prompt_text("[fe|]male", 10), "female")
        self.assertEqual(clip_prompt_text(r"\(literal\)", 10), "(literal)")

    def test_hires_pass_uses_its_own_first_step(self):
        # A fractional switch point counts over both passes in a hires pass.
        self.assertEqual(clip_prompt_text("a [b:.5] c", 10), "a  c")
        self.assertEqual(clip_prompt_text("a [b:.5] c", 10, 10), "a b c")


if __name__ == "__main__":
    unittest.main()
