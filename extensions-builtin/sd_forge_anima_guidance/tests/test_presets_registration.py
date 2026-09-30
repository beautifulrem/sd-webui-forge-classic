"""The preset's guidance.* values are the panel's own defaults."""

import importlib.util
import sys
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
def guidance():
    # modules.infotext_utils (the panel's PasteField): set up by conftest.py for the whole session
    return harness.load_script(ROOT / "scripts" / "anima_guidance.py", "anima_guidance_presets_under_test")


def test_apply_preset_turns_the_panel_off_at_its_defaults(guidance):
    for tab in ("txt2img", "img2img"):
        # The preset table lists the guidance controls in the panel's order
        harness.assert_registers_preset(guidance.AnimaGuidanceScript(), "guidance", tab, unowned_labels=UNOWNED)


def test_script_args_name_the_registered_preset_fields(guidance):
    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET, preset_controls

    harness.build_ui(guidance.AnimaGuidanceScript())
    names, _, _ = preset_controls("txt2img")
    assert list(names) == [preset for _, _, preset, _ in guidance.SCRIPT_ARGS if preset is not None]
    # An argument list stored before a later control was added gets the preset's default for it
    for _, _, preset, default in guidance.SCRIPT_ARGS:
        if default is not guidance._REQUIRED:
            assert default == SAFE_BASE_AESTHETIC_PRESET[preset], preset
            assert type(default) is type(SAFE_BASE_AESTHETIC_PRESET[preset]), preset


def test_local_path_boxes_are_hidden_under_hide_ui_dir_config(guidance, monkeypatch):
    # Server paths are privileged, like upstream's batch img2img directories.
    for hidden in (False, True):
        monkeypatch.setitem(guidance.shared.hide_dirs, "visible", not hidden)
        controls = dict(zip(guidance.SCRIPT_ARG_NAMES, harness.build_ui(guidance.AnimaGuidanceScript())))
        for name in ("adapter_path", "clip_path"):
            assert controls[name].visible is not hidden, name


def test_every_cfg_mode_is_offered_and_pasted_back(guidance):
    from modules.anima_feature_conflicts import GUIDANCE_MODES, TCFG_GUIDANCE
    from modules.anima_presets import preset_controls

    script = guidance.AnimaGuidanceScript()
    controls = dict(zip(guidance.SCRIPT_ARG_NAMES, harness.build_ui(script)))
    choices = [choice[0] if isinstance(choice, (tuple, list)) else choice for choice in controls["guidance_mode"].choices]
    assert TCFG_GUIDANCE in GUIDANCE_MODES
    assert choices == list(GUIDANCE_MODES)
    # One infotext key writes and pastes back the mode, and the preset owns it.
    assert "Anima CFG guidance mode" in script.paste_field_names
    names, components, _ = preset_controls("txt2img")
    assert components[list(names).index("guidance.mode")] is controls["guidance_mode"]
