"""Before any test file is collected: pytest's arguments kept away from the
WebUI's argv parser, and ``modules.infotext_utils`` importable (see
``prepare_webui_imports`` of the repo's tests/extension_script_harness.py),
so every file runs the same alone, in the suite, and in any order."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("extension_script_harness", Path(__file__).resolve().parents[3] / "tests" / "extension_script_harness.py")
_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_harness)
_harness.prepare_webui_imports()


@pytest.fixture(autouse=True)
def no_run_in_progress(monkeypatch):
    """Every test starts without a run: the run state lives in
    lib_stage_scheduler.runtime, which all the tests share."""
    from lib_stage_scheduler import runtime

    for name in ("_ACTIVE_STATE", "_RELEASED_PATCHES", "_TEXT_ENCODER_SIGNATURE"):
        monkeypatch.setattr(runtime, name, None)
