"""The preset's regional.* values are the panel's own defaults."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the DAVE / Cross-Attention Boost suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


@pytest.fixture
def regional(monkeypatch):
    # The panel builds PasteField(component, key) pairs; other suites of this
    # folder may leave a stub behind that takes one argument
    stub = types.ModuleType("modules.infotext_utils")
    stub.PasteField = lambda component, target: (component, target)
    monkeypatch.setitem(sys.modules, "modules.infotext_utils", stub)
    return harness.load_script(ROOT / "scripts" / "anima_regional.py", "anima_regional_presets_under_test")


def test_apply_preset_turns_the_panel_off_at_its_defaults(regional):
    # Region prompts are user content: the preset leaves them alone
    for tab in ("txt2img", "img2img"):
        harness.assert_registers_preset(regional.AnimaRegionalScript(), "regional", tab, unowned_labels=("Regional prompt",))
