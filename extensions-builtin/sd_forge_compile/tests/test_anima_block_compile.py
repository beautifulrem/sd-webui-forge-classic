from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_block_compile import (
    _EAGER_OPTION_KEYS,
    RECOMPILE_LIMIT,
    AnimaBlockCompileManager,
    _needs_eager_blocks,
    _recompile_limit_patch,
    needs_eager,
    patched_inside,
    without_inactive_modifiers,
)
from backend.nn.anima_attention import ANIMA_ATTENTION_MODIFIERS
from modules.forward_override import forward_overrides


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
            AnimaBlockCompileManager.install(model, {})
            self.assertEqual(model.apply_model(3), 3)
            self.assertEqual(model.apply_model(4), 4)
        self.assertEqual(compile_mock.call_count, 2)
        self.assertEqual(tuple(model.diffusion_model.blocks), originals)
        self.assertNotEqual(model.seen, originals)

    def test_restores_after_exception_and_remove_restores_method(self):
        model = _KModel(fail=True)
        originals = tuple(model.diffusion_model.blocks)
        with patch("anima_block_compile.torch.compile", side_effect=lambda block, **kwargs: torch.nn.Sequential(block)):
            AnimaBlockCompileManager.install(model, {})
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
            AnimaBlockCompileManager.install(model, {}, on_fallback=failures.append)
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
            AnimaBlockCompileManager.install(model, {}, on_fallback=first_task.append)
            AnimaBlockCompileManager.update_fallback_handler(model, current_task.append)
            self.assertEqual(model.apply_model(torch.tensor(3)), torch.tensor(3))
        self.assertEqual(first_task, [])
        self.assertEqual(len(current_task), 1)
        self.assertEqual(AnimaBlockCompileManager.fallback_reason(model), "BackendCompilerFailed")

    def test_a_runtime_error_in_the_compiled_blocks_is_not_a_compile_failure(self):
        # e.g. out of memory while the compiled blocks run: the error reaches
        # the caller every time, the compiled blocks are kept, and compile is
        # never handed to the eager fallback
        class OutOfMemory(torch.nn.Module):
            def forward(self, value):
                raise RuntimeError("CUDA out of memory")

        model = _KModel(run_blocks=True)
        failures = []
        with patch("anima_block_compile.torch.compile", side_effect=lambda block, **kwargs: OutOfMemory()) as compile_mock:
            AnimaBlockCompileManager.install(model, {}, on_fallback=failures.append)
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    model.apply_model(torch.tensor(3))
        self.assertEqual(failures, [])
        self.assertIsNone(AnimaBlockCompileManager.fallback_reason(model))
        self.assertEqual(compile_mock.call_count, 2)  # compiled once, kept for the next call


class NeedsEagerTests(unittest.TestCase):
    def test_each_per_generation_option_forces_eager_by_kwarg_and_position(self):
        root = torch.nn.Linear(2, 2)
        for key in _EAGER_OPTION_KEYS:
            with self.subTest(key=key):
                options = {key: object()}
                self.assertTrue(_needs_eager_blocks((), {"transformer_options": options}))
                self.assertTrue(_needs_eager_blocks((None,) * 5 + (options,), {}))
                self.assertTrue(needs_eager((), {"transformer_options": options}, root))
                for empty in (None, ()):
                    self.assertFalse(_needs_eager_blocks((), {"transformer_options": {key: empty}}))
        self.assertIn("optimized_attention_override", _EAGER_OPTION_KEYS)

    def test_other_options_and_missing_options_stay_compiled(self):
        self.assertFalse(_needs_eager_blocks((), {}))
        self.assertFalse(_needs_eager_blocks((None,) * 5, {}))
        self.assertFalse(_needs_eager_blocks((), {"transformer_options": {"block_index": 3, "sigmas": 1.0}}))

    def test_negpip_conditioning_kwarg_forces_eager(self):
        self.assertTrue(_needs_eager_blocks((), {"c_negpip_mask": torch.ones(1)}))
        self.assertFalse(_needs_eager_blocks((), {"c_negpip_mask": None}))

    def test_patched_inside_sees_overrides_below_the_root_only(self):
        root = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sequential(torch.nn.Linear(2, 2)))
        inner = root[1][0]
        outside = torch.nn.Linear(2, 2)
        self.assertFalse(patched_inside(root))
        with forward_overrides([(outside, lambda x: x)]):
            self.assertFalse(patched_inside(root))
        with forward_overrides([(inner, lambda x: x)]):
            self.assertTrue(patched_inside(root))
            self.assertTrue(needs_eager((), {}, root))
        self.assertFalse(patched_inside(root))

    def test_modifiers_idle_at_this_sigma_do_not_force_eager(self):
        class Windowed:
            def __init__(self, active):
                self.is_active = active

            def active(self, options):
                return self.is_active

        idle, busy = Windowed(False), Windowed(True)
        plain = object()  # no ``active``: always active
        for modifiers, remaining in (((idle,), ()), ((idle, busy), (busy,)), ((plain, idle), (plain,))):
            options = {ANIMA_ATTENTION_MODIFIERS: modifiers, "sigmas": 0.3}
            args, kwargs = without_inactive_modifiers((), {"transformer_options": options})
            self.assertEqual(kwargs["transformer_options"][ANIMA_ATTENTION_MODIFIERS], remaining)
            self.assertEqual(_needs_eager_blocks(args, kwargs), bool(remaining))
            self.assertEqual(options[ANIMA_ATTENTION_MODIFIERS], modifiers)  # caller's dict untouched
            args, kwargs = without_inactive_modifiers((None,) * 5 + (options, "tail"), {})
            self.assertEqual(args[5][ANIMA_ATTENTION_MODIFIERS], remaining)
            self.assertEqual(args[6], "tail")
        options = {ANIMA_ATTENTION_MODIFIERS: (busy,)}
        self.assertIs(without_inactive_modifiers((), {"transformer_options": options})[1]["transformer_options"], options)

    def test_compiled_blocks_run_while_a_modifier_is_outside_its_window(self):
        class Windowed:  # like NAG: active for sigma <= 0.4 only
            def active(self, options):
                return float(options["sigmas"]) <= 0.4

        nag = Windowed()
        model = _KModel()
        originals = tuple(model.diffusion_model.blocks)
        seen = []

        def apply_model(value, *args, transformer_options=None):
            seen.append((tuple(model.diffusion_model.blocks) != originals, transformer_options.get(ANIMA_ATTENTION_MODIFIERS)))
            return value

        model.apply_model = apply_model
        with patch("anima_block_compile.torch.compile", side_effect=lambda block, **kwargs: torch.nn.Sequential(block)):
            AnimaBlockCompileManager.install(model, {})
            model.apply_model(1, transformer_options={ANIMA_ATTENTION_MODIFIERS: [nag], "sigmas": torch.tensor([0.9])})
            model.apply_model(1, transformer_options={ANIMA_ATTENTION_MODIFIERS: [nag], "sigmas": torch.tensor([0.3])})
        self.assertEqual(seen, [(True, ()), (False, [nag])])

    def test_no_override_does_not_walk_the_model(self):
        root = torch.nn.Linear(2, 2)
        with patch.object(root, "modules", side_effect=AssertionError("walked the modules")):
            self.assertFalse(patched_inside(root))


def _skip_transformer_options_guards(guard_entries):
    # The compile script's guard_filter_fn (scripts/compile.py).
    return [("transformer_options" not in entry.name) for entry in guard_entries]


class _AttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, transformer_options={}):
        override = transformer_options.get("optimized_attention_override")
        if override is not None:
            return override(self.linear(x), transformer_options)
        return self.linear(x)


class _BlockIndexKModel:
    """Anima's loop: one options dict, ``block_index`` set per block."""

    def __init__(self):
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList([_AttentionBlock() for _ in range(3)])

    def apply_model(self, x, t=None, c_concat=None, c_crossattn=None, control=None, transformer_options={}):
        options = dict(transformer_options)
        outputs = []
        for block_index, block in enumerate(self.diffusion_model.blocks):
            options["block_index"] = block_index
            outputs.append(block(x, transformer_options=options))
        return torch.stack(outputs)


def _block_index_override(value, transformer_options):
    return value * 0 + transformer_options["block_index"]


class BlockIndexCompileTests(unittest.TestCase):
    def test_attention_override_sees_each_block_index(self):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        model = _BlockIndexKModel()
        config = dict(backend="eager", dynamic=False, fullgraph=False, options={"guard_filter_fn": _skip_transformer_options_guards})
        AnimaBlockCompileManager.install(model, config)
        x = torch.zeros(1, 4)
        with torch.no_grad():
            model.apply_model(x, transformer_options={})  # traced without the override
            options = {"optimized_attention_override": _block_index_override}
            by_kwarg = model.apply_model(x, transformer_options=options)
            by_position = model.apply_model(x, None, None, None, None, options)
        expected = torch.arange(3.0).reshape(3, 1, 1).expand(3, 1, 4)
        torch.testing.assert_close(by_kwarg, expected)
        torch.testing.assert_close(by_position, expected)


class RecompileLimitConfigTests(unittest.TestCase):
    def test_public_compiler_config_is_used_where_it_has_both_settings(self):
        from types import SimpleNamespace

        config, settings = _recompile_limit_patch()
        self.assertIs(config, torch.compiler.config)  # the pinned PyTorch has them
        self.assertEqual(settings, dict(recompile_limit=RECOMPILE_LIMIT, fail_on_recompile_limit_hit=True))
        with config.patch(**settings):
            self.assertEqual(torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT)  # an alias of Dynamo's
            self.assertTrue(torch._dynamo.config.fail_on_recompile_limit_hit)
        self.assertEqual(torch._dynamo.config.recompile_limit, 8)

        # Older installs: Dynamo's own config, then its pre-2.7 names
        with patch.object(torch.compiler, "config", SimpleNamespace(), create=True):
            config, settings = _recompile_limit_patch()
            self.assertIs(config, torch._dynamo.config)
            self.assertEqual(settings, dict(recompile_limit=RECOMPILE_LIMIT, fail_on_recompile_limit_hit=True))
            old_dynamo = SimpleNamespace(cache_size_limit=8, fail_on_cache_limit_hit=False)
            with patch.object(torch._dynamo, "config", old_dynamo):
                config, settings = _recompile_limit_patch()
            self.assertIs(config, old_dynamo)
            self.assertEqual(settings, dict(cache_size_limit=RECOMPILE_LIMIT, fail_on_cache_limit_hit=True))


if __name__ == "__main__":
    unittest.main()
