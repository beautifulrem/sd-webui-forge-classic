"""Modulation guidance under per-block compile, traced by real Dynamo (CPU, a
counting backend instead of Inductor) on a tiny Anima: the per-call hooks
keep the compiled blocks, where a forward override would run them eager."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "sd_forge_compile"))

from anima_block_compile import AnimaBlockCompileManager
from backend import memory_management, operations
from backend.nn.anima import Anima
from lib_anima_guidance.modulation import ModulationPatch

_BLOCKS = 4
_HIDDEN = 128


def _guard_filter(entries):
    # As scripts/compile.py: no guards on transformer_options (they change every step)
    return [("transformer_options" not in entry.name) for entry in entries]


class _CountingBackend(CompileCounter):
    """Counts compiled frames (``frame_count``) and runs of compiled graphs (``runs``)."""

    def __init__(self):
        super().__init__()
        self.runs = 0

    def __call__(self, gm, example_inputs):
        forward = super().__call__(gm, example_inputs)

        def run(*args):
            self.runs += 1
            return forward(*args)

        return run


class _KModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model

    def apply_model(self, x, t, context, transformer_options=None):
        return self.diffusion_model(x, t, context, transformer_options={} if transformer_options is None else transformer_options)


class _Patcher:
    def __init__(self, diffusion_model):
        self.model = type("Model", (), {"diffusion_model": diffusion_model})()
        self.model_options = {}

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def _anima():
    generator = torch.Generator().manual_seed(0)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        model = Anima(4, 4, 2, 1, model_channels=_HIDDEN, crossattn_emb_channels=32, adaln_lora_dim=16, num_blocks=_BLOCKS, num_heads=2)
    model.load_state_dict({key: torch.randn(value.shape, generator=generator) * 0.05 for key, value in model.state_dict().items()})
    return model.eval()


def _inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 4, 1, 8, 8, generator=generator), torch.tensor([0.3, 0.7]), torch.randn(2, 5, 32, generator=generator)


class ModulationUnderBlockCompileTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        # platform.mac_ver() (attention's upcast check on macOS) stats a file,
        # which Dynamo cannot trace: pin it so the graph is the same everywhere.
        mac_version = memory_management.mac_version()
        patcher = patch.object(memory_management, "mac_version", lambda: mac_version)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(torch._dynamo.reset)

        self.model = _anima()
        self.kmodel = _KModel(self.model)
        self.backend = _CountingBackend()
        self.fallbacks = []
        config = dict(backend=self.backend, dynamic=False, fullgraph=False, options={"guard_filter_fn": _guard_filter})
        AnimaBlockCompileManager.install(self.kmodel, config, on_fallback=self.fallbacks.append)

        generator = torch.Generator().manual_seed(9)
        positive, negative = (torch.randn(1, _HIDDEN * 3, generator=generator) for _ in range(2))
        scales = torch.rand(_BLOCKS, _HIDDEN * 3, generator=generator)
        patched, _, _ = ModulationPatch(positive, negative, scales, 1, -1).apply(_Patcher(self.model))
        self.wrapper = patched.model_options["model_function_wrapper"]

    def _call(self, seed, apply_model, modulated):
        x, t, context = _inputs(seed)
        if not modulated:
            return apply_model(x, t, context, transformer_options={})
        args = {"input": x, "timestep": t, "c": {"context": context}, "cond_or_uncond": [0, 1]}
        return self.wrapper(lambda x, t, **c: apply_model(x, t, c["context"], transformer_options={}), args)

    def _both(self, seed, modulated):
        eager = self._call(seed, self.model, modulated)
        runs = self.backend.runs
        compiled = self._call(seed, self.kmodel.apply_model, modulated)
        return eager, compiled, self.backend.runs - runs

    def test_modulated_calls_keep_the_compiled_blocks_without_recompiling(self):
        with torch.inference_mode():
            self._both(0, modulated=False)
            eager, compiled, runs = self._both(1, modulated=True)
            unmodulated, compiled_off, runs_off = self._both(1, modulated=False)
            frames = self.backend.frame_count
            for seed in (2, 3):
                again, compiled_again, runs_again = self._both(seed, modulated=True)
                torch.testing.assert_close(compiled_again, again, rtol=1e-5, atol=1e-5)
                self.assertEqual(runs_again, runs)
                self._both(seed, modulated=False)
        self.assertGreater((eager - unmodulated).abs().max().item(), 1e-3)  # the delta matters
        torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(compiled_off, unmodulated, rtol=1e-5, atol=1e-5)
        # Every block ran compiled: the modulated ones (1..3) also ran the hook's graph.
        self.assertEqual((runs, runs_off), (_BLOCKS + (_BLOCKS - 1), _BLOCKS))
        self.assertEqual(self.backend.frame_count, frames)  # toggling recompiles nothing
        self.assertEqual(self.fallbacks, [])
        self.assertTrue(all(not block._forward_pre_hooks for block in self.model.blocks))


if __name__ == "__main__":
    unittest.main()
