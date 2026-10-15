"""ADetailer test helpers: the WebUI modules importable before collection (as
tests/conftest.py does: pytest's arguments kept from the WebUI's argv parser),
model downloads off, and the script loaded as Forge loads it.

Not named ``conftest``: test modules import it as ``adetailer_testlib``
(unique), so this suite can run in one pytest session with the others."""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]
REPO = EXT.parents[1]
SCRIPT = EXT / "scripts" / "adetailer.py"
FIXTURES = EXT / "tests" / "fixtures"


def _load(path: Path, name: str):
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


contract = _load(REPO / "tests" / "legacy_ui_contract.py", "legacy_ui_contract")
kit = _load(REPO / "tests" / "remi_extension_kit.py", "remi_extension_kit")
contract.prepare_imports()  # sys.argv = sys.argv[:1]; repo and bundled packages on sys.path
kit.prepare_imports()
for path in (str(EXT),):
    if path not in sys.path:
        sys.path.insert(0, path)

from modules import shared  # noqa: E402

# Never download detector models from tests (lib_adetailer reads it on import)
shared.cmd_opts.ad_no_huggingface = True


@functools.cache
def load_adetailer():
    """The script module (scans ``models/adetailer`` once)."""

    return kit.load_script(SCRIPT, "adetailer_under_test")
