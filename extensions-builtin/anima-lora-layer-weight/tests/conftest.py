"""Before any test file is collected: pytest's arguments kept away from the
WebUI's argv parser and the WebUI modules importable (see
``prepare_imports`` of the repo's tests/remi_anima_lora_harness.py), so
every file runs the same alone, in the suite, and in any order."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("remi_anima_lora_harness", Path(__file__).resolve().parents[3] / "tests" / "remi_anima_lora_harness.py")
_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_harness)
_harness.prepare_imports(Path(__file__).resolve().parents[1])
