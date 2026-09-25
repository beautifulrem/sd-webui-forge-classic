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


if __name__ == "__main__":
    unittest.main()
