"""Before any test file is collected: the WebUI modules importable and
pytest's arguments kept away from the WebUI's argv parser (see the repo's
tests/remi_extension_kit.py). Nothing imports Gradio."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("remi_extension_kit", Path(__file__).resolve().parents[3] / "tests" / "remi_extension_kit.py")
_kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kit)
_kit.prepare_imports()
