import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the DAVE / Cross-Attention Boost suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

negpip = harness.load_script(ROOT / "scripts" / "negpip.py", "negpip_under_test")

from backend.nn.anima import SelfCrossAttention  # noqa: E402
from backend.sampling import condition, sampling_function  # noqa: E402
from lib_negpip import anima as negpip_anima  # noqa: E402


class _Model:
    is_anima_engine = True

    def __init__(self, dit):
        self.forge_objects = SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=dit)))
        self.text_processing_engine_anima = SimpleNamespace(tokenize_line=lambda line: [])

    def is_webui_legacy_model(self):
        return False

    def get_learned_conditioning(self, prompt):
        return [torch.ones(2, 2) for _ in prompt]


class _DiT(torch.nn.Module):
    def forward(self, x, timesteps, context, padding_mask=None, **kwargs):
        return x


@pytest.fixture
def anima(monkeypatch):
    dit = _DiT()
    model = _Model(dit)
    monkeypatch.setattr(negpip_anima, "shared", SimpleNamespace(sd_model=model))  # the real one loads a checkpoint
    monkeypatch.setattr(negpip.NegPiP, "_patched", [False, False])
    monkeypatch.setattr(negpip, "on_cfg_denoiser", lambda callback: None)  # keep the global callback list clean
    originals = SelfCrossAttention.forward, condition.compile_conditions, sampling_function.compile_conditions
    yield model, dit, originals
    negpip_anima.patch_anima_negpip(negpip.NegPiP, unpatch=True)


def _process(model, prompt="a cat, (dog:-1)"):
    return SimpleNamespace(sd_model=model, prompts=[prompt], negative_prompts=[""], scripts=SimpleNamespace(scripts=[]), script_args=[], extra_generation_params={})


def _assert_restored(model, dit, originals):
    assert (SelfCrossAttention.forward, condition.compile_conditions, sampling_function.compile_conditions) == originals
    assert "get_learned_conditioning" not in model.__dict__ and "forward" not in dit.__dict__


@pytest.mark.parametrize("finish", ["cleanup", "postprocess"])
def test_anima_batch_hooks_the_model_and_finishing_unhooks_it(anima, finish):
    model, dit, originals = anima
    script = negpip.NegPiP()
    p = _process(model)

    script.process_batch(p)
    assert script.active and script.is_anima
    assert negpip.NegPiP._patched == [False, True]
    assert SelfCrossAttention.forward is not originals[0]
    assert "get_learned_conditioning" in model.__dict__ and "forward" in dit.__dict__
    assert p.extra_generation_params == {"NegPiP": True}
    assert p.cached_c == [None] * 3 and p.cached_uc == [None] * 3  # conditioning is encoded again, through the hook

    getattr(script, finish)(p)
    assert not script.active
    assert negpip.NegPiP._patched == [False, False]
    _assert_restored(model, dit, originals)

    script.cleanup(p)  # cleanup after postprocess (or twice) is harmless
    _assert_restored(model, dit, originals)


def test_a_batch_without_negative_weights_leaves_the_previous_hooks_undone(anima):
    model, dit, originals = anima
    script = negpip.NegPiP()
    script.process_batch(_process(model))
    assert script.active

    # The next batch (e.g. a later job) has no negative weight: the old hooks go
    p = _process(model, "a cat, (dog:1.2)")
    script.process_batch(p)
    assert not script.active and p.extra_generation_params == {}
    assert negpip.NegPiP._patched == [False, False]
    _assert_restored(model, dit, originals)
