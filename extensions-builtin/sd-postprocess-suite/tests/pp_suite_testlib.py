"""Post Processing Suite test helpers: the WebUI modules importable before collection
(see tests/legacy_ui_contract.py), the script loaded as Forge loads it.

Not named ``conftest``: test modules import it as ``pp_suite_testlib`` (unique), so
this suite can run in one pytest session with the other suites."""

import functools
import importlib.util
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]
SCRIPT = EXT / "scripts" / "postprocess_suite.py"

_spec = importlib.util.spec_from_file_location("legacy_ui_contract", EXT.parents[1] / "tests" / "legacy_ui_contract.py")
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)
contract.prepare_imports()


@functools.cache
def load_suite():
    spec = importlib.util.spec_from_file_location("postprocess_suite_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
