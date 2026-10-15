"""Before any test file is collected: the WebUI modules importable and
pytest's arguments kept away from the WebUI's argv parser (see the repo's
tests/remi_extension_kit.py). The tests load the kit as ``kit``."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("remi_extension_kit", Path(__file__).resolve().parents[3] / "tests" / "remi_extension_kit.py")
kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kit)
kit.prepare_imports()

EXTENSION = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ext_kit():
    return kit
