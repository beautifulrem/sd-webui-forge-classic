"""Before any test file is collected: pytest's arguments kept away from the
WebUI's argv parser and the WebUI importable, with no Gradio (see
``prepare_imports`` of the repo's tests/remi_port_harness.py), so every file
runs the same alone, in the suite, and in any order."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("remi_port_harness", Path(__file__).resolve().parents[3] / "tests" / "remi_port_harness.py")
harness = sys.modules.setdefault("remi_port_harness", importlib.util.module_from_spec(_spec))
if not hasattr(harness, "prepare_imports"):
    _spec.loader.exec_module(harness)
harness.prepare_imports()
