"""Before any test file is collected: pytest's arguments kept away from the
WebUI's argv parser and the WebUI modules importable (see
``prepare_imports`` of the repo's tests/remi_anima_lora_harness.py), so
every file runs the same alone, in the suite, and in any order. The
extension folder goes on sys.path, as Forge has it (modules/scripts.py), so
``lib_artist_mixer`` imports from wherever pytest is started."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("remi_anima_lora_harness", Path(__file__).resolve().parents[3] / "tests" / "remi_anima_lora_harness.py")
_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_harness)
_harness.prepare_imports(Path(__file__).resolve().parents[1])


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Each test has its own data folder: the templates the panel's
    defaults and template controls read."""
    from modules import paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    return tmp_path
