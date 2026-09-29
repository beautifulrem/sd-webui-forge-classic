"""The preset's guidance.* values are the panel's own defaults."""

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

# User content the preset leaves alone: CLIP modulation prompts and local file paths
UNOWNED = (
    "Base CLIP prompt override",
    "Positive modulation direction",
    "Negative modulation direction",
    "Local adapter .pt path",
    "Local CLIP-L .safetensors path",
)


@pytest.fixture
def guidance(monkeypatch):
    # The panel builds PasteField(component, key) pairs; the real module needs
    # the full WebUI install, and no stub is left behind for the other suites
    stub = types.ModuleType("modules.infotext_utils")
    stub.PasteField = lambda component, target: (component, target)
    monkeypatch.setitem(sys.modules, "modules.infotext_utils", stub)
    return harness.load_script(ROOT / "scripts" / "anima_guidance.py", "anima_guidance_presets_under_test")


def test_apply_preset_turns_the_panel_off_at_its_defaults(guidance):
    for tab in ("txt2img", "img2img"):
        # The panel's controls follow SCRIPT_ARGS (the API's positional order), not the preset table
        harness.assert_registers_preset(guidance.AnimaGuidanceScript(), "guidance", tab, unowned_labels=UNOWNED, preset_order=False)


def test_script_args_name_the_registered_preset_fields(guidance):
    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET, preset_controls

    harness.build_ui(guidance.AnimaGuidanceScript())
    names, _, _ = preset_controls("txt2img")
    assert list(names) == [preset for _, _, preset, _ in guidance.SCRIPT_ARGS if preset is not None]
    # An API call that omits a later argument gets the preset's default for it
    for _, _, preset, default in guidance.SCRIPT_ARGS:
        if default is not guidance._REQUIRED:
            assert default == SAFE_BASE_AESTHETIC_PRESET[preset], preset
            assert type(default) is type(SAFE_BASE_AESTHETIC_PRESET[preset]), preset
