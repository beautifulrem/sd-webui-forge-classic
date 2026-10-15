"""Before any test file is collected: pytest's arguments kept away from the
WebUI's argv parser and the extension folder on sys.path, as Forge has it (see
``prepare_imports`` of the repo's tests/remi_script_harness.py), so every file
runs the same alone, in the suite, and in any order. The tests import the
harness as ``remi_script_harness``."""

import importlib.util
import sys
from pathlib import Path

if "remi_script_harness" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("remi_script_harness", Path(__file__).resolve().parents[3] / "tests" / "remi_script_harness.py")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    sys.modules["remi_script_harness"] = _module
sys.modules["remi_script_harness"].prepare_imports(Path(__file__).resolve().parents[1])
