"""The preset's guidance.* values are the panel's own defaults."""

from pathlib import Path

import pytest
import remi_script_harness as harness

ROOT = Path(__file__).resolve().parents[1]

# User content the preset leaves alone: CLIP modulation prompts and local file paths
UNOWNED = ("modulation_base", "modulation_positive", "modulation_negative", "adapter_path", "clip_path")


@pytest.fixture(scope="module")
def guidance():
    return harness.load_script(ROOT / "scripts" / "anima_guidance.py", "anima_guidance_presets_under_test")


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_apply_preset_turns_the_panel_off_at_its_defaults(guidance, tab):
    # The preset table lists the guidance controls in the panel's order
    harness.assert_registers_preset(harness.script_layout(guidance.AnimaGuidanceScript(), tab), "guidance", unowned=UNOWNED)


def test_script_args_name_the_bound_preset_fields(guidance):
    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET

    layout = harness.script_layout(guidance.AnimaGuidanceScript())
    bound = {name: field for field, name in harness.preset_fields(layout).items()}
    assert [bound.get(name) for name in layout.names] == [preset for _, _, preset, _ in guidance.SCRIPT_ARGS]
    # An argument list stored before a later control was added gets the preset's default for it
    for _, _, preset, default in guidance.SCRIPT_ARGS:
        if default is not guidance._REQUIRED:
            assert default == SAFE_BASE_AESTHETIC_PRESET[preset], preset
            assert type(default) is type(SAFE_BASE_AESTHETIC_PRESET[preset]), preset


def test_local_path_boxes_are_hidden_under_hide_ui_dir_config(guidance, monkeypatch):
    # Server paths are privileged, like upstream's batch img2img directories.
    for hidden in (False, True):
        monkeypatch.setattr(guidance.shared.cmd_opts, "hide_ui_dir_config", hidden)
        layout = harness.script_layout(guidance.AnimaGuidanceScript())
        for name in ("adapter_path", "clip_path"):
            assert layout.by_name[name].hidden is hidden, name
            # Still a script argument: hidden params are submitted
            assert name in layout.names


def test_every_cfg_mode_is_offered_and_pasted_back(guidance):
    from modules.anima_feature_conflicts import GUIDANCE_MODES, ROLE_DCW, ROLE_GUIDANCE_MODE, ROLE_MOMENTUM, TCFG_GUIDANCE
    from remi.ui import Context

    layout = harness.script_layout(guidance.AnimaGuidanceScript())
    mode = layout.by_name["guidance_mode"]
    assert TCFG_GUIDANCE in GUIDANCE_MODES
    assert mode.values(Context()) == list(GUIDANCE_MODES)
    # One infotext key writes and pastes back the mode, and the preset owns it.
    assert layout.infotext["Anima CFG guidance mode"] == "guidance_mode"
    # The conflict roles the core __conflicts__ action locks, found by preset field
    fields = harness.preset_fields(layout)
    assert (fields[ROLE_GUIDANCE_MODE], fields[ROLE_MOMENTUM], fields[ROLE_DCW]) == ("guidance_mode", "momentum_enable", "dcw_enable")
