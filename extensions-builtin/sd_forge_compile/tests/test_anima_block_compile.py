from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_block_compile import AnimaBlockCompileManager


class _Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])


class _KModel:
    def __init__(self, fail=False, run_blocks=False):
        self.diffusion_model = _Diffusion()
        self.fail = fail
        self.run_blocks = run_blocks
        self.seen = None

    def apply_model(self, value):
        self.seen = tuple(self.diffusion_model.blocks)
        if self.run_blocks:
            for block in self.diffusion_model.blocks:
                value = block(value)
        if self.fail:
            raise RuntimeError("test failure")
        return value


class AnimaBlockCompileTests(unittest.TestCase):
    def test_compiles_once_and_restores_original_blocks(self):
        model = _KModel()
        originals = tuple(model.diffusion_model.blocks)
        wrappers = [torch.nn.Sequential(block) for block in originals]
        with patch("anima_block_compile.torch.compile", side_effect=wrappers) as compile_mock:
            AnimaBlockCompileManager.install(model, {}, "test")
            self.assertEqual(model.apply_model(3), 3)
            self.assertEqual(model.apply_model(4), 4)
        self.assertEqual(compile_mock.call_count, 2)
        self.assertEqual(tuple(model.diffusion_model.blocks), originals)
        self.assertNotEqual(model.seen, originals)

    def test_restores_after_exception_and_remove_restores_method(self):
        model = _KModel(fail=True)
        originals = tuple(model.diffusion_model.blocks)
        with patch("anima_block_compile.torch.compile", side_effect=lambda block, **kwargs: torch.nn.Sequential(block)):
            AnimaBlockCompileManager.install(model, {}, "test")
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                model.apply_model(1)
        self.assertEqual(tuple(model.diffusion_model.blocks), originals)
        AnimaBlockCompileManager.remove(model)
        self.assertFalse(AnimaBlockCompileManager.is_installed(model))

    def test_compiler_failure_falls_back_to_eager_and_stays_eager(self):
        class BackendCompilerFailed(RuntimeError):
            pass

        BackendCompilerFailed.__module__ = "torch._dynamo.exc"
        model = _KModel()
        failures = []
        with patch("anima_block_compile.torch.compile", side_effect=BackendCompilerFailed("no compiler")) as compile_mock:
            AnimaBlockCompileManager.install(model, {}, "test", on_fallback=failures.append)
            self.assertEqual(model.apply_model(3), 3)
            self.assertEqual(model.apply_model(4), 4)
        self.assertEqual(compile_mock.call_count, 1)
        self.assertEqual(len(failures), 1)

    def test_lazy_compiler_failure_uses_current_handler_and_eager_blocks(self):
        class BackendCompilerFailed(RuntimeError):
            pass

        BackendCompilerFailed.__module__ = "torch._dynamo.exc"

        class LazyFailure(torch.nn.Module):
            def forward(self, value):
                raise BackendCompilerFailed("lazy failure")

        model = _KModel(run_blocks=True)
        first_task = []
        current_task = []
        with patch("anima_block_compile.torch.compile", side_effect=lambda block, **kwargs: LazyFailure()):
            AnimaBlockCompileManager.install(model, {}, "test", on_fallback=first_task.append)
            AnimaBlockCompileManager.update_fallback_handler(model, current_task.append)
            self.assertEqual(model.apply_model(torch.tensor(3)), torch.tensor(3))
        self.assertEqual(first_task, [])
        self.assertEqual(len(current_task), 1)
        self.assertEqual(AnimaBlockCompileManager.fallback_reason(model), "BackendCompilerFailed")


if __name__ == "__main__":
    unittest.main()
