from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import torch

sys.argv = sys.argv[:1]
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_block_compile import _FALLBACK_HANDLER_KEY, _FALLBACK_REASON_KEY, AnimaBlockCompileManager

SCRIPT = ROOT / "scripts" / "compile.py"
SPEC = importlib.util.spec_from_file_location("forge_compile_script_under_test", SCRIPT)
compile_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compile_script)


class _KModel:
    def __init__(self):
        self.diffusion_model = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Identity()]))

    def apply_model(self, value):
        return value


class _Process:
    def __init__(self, kmodel, anima=True):
        unet = SimpleNamespace(model=kmodel)
        self.sd_model = SimpleNamespace(is_anima_engine=anima, forge_objects=SimpleNamespace(unet=unet))
        self.extra_generation_params = {}


def _script():
    # __init__ only sets global dynamo options; keep them untouched for other tests.
    return compile_script.TorchCompileForForge.__new__(compile_script.TorchCompileForForge)


def _config(kmodel):
    return getattr(kmodel, compile_script._COMPILE_CONFIG_KEY, None)


def test_switching_from_per_block_to_whole_model_removes_the_block_compile():
    kmodel = _KModel()
    script = _script()
    script.process_batch(_Process(kmodel), "Anima per-block")
    assert AnimaBlockCompileManager.is_installed(kmodel)
    assert _config(kmodel) == "Anima per-block"

    script.process_batch(_Process(kmodel), "guard_filter_fn")
    assert not AnimaBlockCompileManager.is_installed(kmodel)
    assert getattr(kmodel.apply_model, compile_script._COMPILE_WRAPPER_KEY, False)
    assert _config(kmodel) == "guard_filter_fn"

    script.process_batch(_Process(kmodel), "Disable")
    assert kmodel.apply_model.__func__ is _KModel.apply_model
    assert _config(kmodel) is None


def test_per_block_is_ignored_on_non_anima_models():
    kmodel = _KModel()
    p = _Process(kmodel, anima=False)
    _script().process_batch(p, "Anima per-block")
    assert not AnimaBlockCompileManager.is_installed(kmodel)
    assert _config(kmodel) is None
    assert p.extra_generation_params["Torch compile"] == "Anima per-block ignored: non-Anima model"


def test_automatic_keeps_an_installed_block_compile_for_the_next_generation():
    kmodel = _KModel()
    script = _script()
    script.process_batch(_Process(kmodel), "Anima per-block")
    installed = kmodel.apply_model

    second = _Process(kmodel)
    script.process_batch(second, "Automatic")
    assert kmodel.apply_model is installed
    assert second.extra_generation_params == {}

    # A fallback reported later is recorded on the generation now running.
    getattr(kmodel, _FALLBACK_HANDLER_KEY)(RuntimeError("boom"))
    assert second.extra_generation_params["Torch compile"] == "Anima per-block fell back to eager: RuntimeError"

    # ... and one made earlier is recorded on every later generation.
    setattr(kmodel, _FALLBACK_REASON_KEY, "BackendCompilerFailed")
    third = _Process(kmodel)
    script.process_batch(third, "Automatic")
    assert kmodel.apply_model is installed
    assert third.extra_generation_params["Torch compile"] == "Anima per-block using eager fallback: BackendCompilerFailed"


def test_windows_precheck_follows_inductor_cxx_not_any_compiler_on_path(monkeypatch):
    on_path = set()
    monkeypatch.setattr(compile_script.sys, "platform", "win32")
    monkeypatch.setattr(compile_script.shutil, "which", lambda name: f"C:/bin/{name}.exe" if name in on_path else None)
    monkeypatch.delenv("CXX", raising=False)

    on_path.update({"g++", "clang-cl"})  # Inductor never runs these unless CXX says so
    assert not compile_script._windows_cpp_compiler_available()
    on_path.add("cl")
    assert compile_script._windows_cpp_compiler_available()

    monkeypatch.setenv("CXX", "clang-cl")
    on_path.clear()
    assert not compile_script._windows_cpp_compiler_available()
    on_path.add("clang-cl")
    assert compile_script._windows_cpp_compiler_available()


def test_per_block_without_windows_compiler_removes_compile_and_reports(monkeypatch):
    kmodel = _KModel()
    script = _script()
    script.process_batch(_Process(kmodel), "guard_filter_fn")
    assert _config(kmodel) == "guard_filter_fn"

    monkeypatch.setattr(compile_script.sys, "platform", "win32")
    monkeypatch.setattr(compile_script.shutil, "which", lambda name: None)
    p = _Process(kmodel)
    script.process_batch(p, "Anima per-block")
    assert not AnimaBlockCompileManager.is_installed(kmodel)
    assert kmodel.apply_model.__func__ is _KModel.apply_model
    assert _config(kmodel) is None
    assert "no C++ compiler found" in p.extra_generation_params["Torch compile"]
