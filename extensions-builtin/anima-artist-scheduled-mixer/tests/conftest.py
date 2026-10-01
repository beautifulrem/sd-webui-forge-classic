"""Before any test file is collected: pytest's arguments kept away from the
WebUI's argv parser, and ``modules.infotext_utils`` importable (see
``prepare_webui_imports`` of the repo's tests/extension_script_harness.py),
so every file runs the same alone, in the suite, and in any order. The
extension folder goes on sys.path, as Forge has it (modules/scripts.py), so
``lib_artist_mixer`` imports from wherever pytest is started."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("extension_script_harness", Path(__file__).resolve().parents[3] / "tests" / "extension_script_harness.py")
_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_harness)
_harness.prepare_webui_imports()
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
