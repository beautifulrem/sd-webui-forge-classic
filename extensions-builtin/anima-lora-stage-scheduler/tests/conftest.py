"""Before any test file is collected: pytest's arguments kept away from the
WebUI's argv parser and the WebUI modules importable (see
``prepare_imports`` of the repo's tests/remi_anima_lora_harness.py), so
every file runs the same alone, in the suite, and in any order. The
extension folder goes on sys.path, as Forge has it (modules/scripts.py), so
``lib_stage_scheduler`` imports from wherever pytest is started."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("remi_anima_lora_harness", Path(__file__).resolve().parents[3] / "tests" / "remi_anima_lora_harness.py")
_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_harness)
_harness.prepare_imports(Path(__file__).resolve().parents[1])


@pytest.fixture(autouse=True)
def fresh_extension_state(tmp_path, monkeypatch):
    """Every test starts without a run (the run state lives in
    lib_stage_scheduler.runtime, which all the tests share), with its own
    template store (seeded from the bundled templates.json under
    ``tmp_path``) and with online LoRA off."""
    from backend.args import dynamic_args
    from lib_stage_scheduler import runtime
    from modules import paths_internal

    for name in ("_ACTIVE_STATE", "_RELEASED_PATCHES", "_TEXT_ENCODER_SIGNATURE"):
        monkeypatch.setattr(runtime, name, None)
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    monkeypatch.setattr(dynamic_args, "online_lora", False)
