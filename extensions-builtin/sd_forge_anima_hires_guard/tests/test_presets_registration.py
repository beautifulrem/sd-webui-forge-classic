"""The preset's hires_guard.* values are the panel's own defaults."""

from pathlib import Path

import remi_script_harness as harness

ROOT = Path(__file__).resolve().parents[1]
hires_guard = harness.load_script(ROOT / "scripts" / "anima_hires_guard.py", "anima_hires_guard_presets_under_test")


def test_apply_preset_turns_the_panel_off_at_its_defaults():
    harness.assert_registers_preset(harness.script_layout(hires_guard.AnimaHiresGuardScript()), "hires_guard")


def test_preset_limits_are_those_of_its_safety_preset():
    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET as preset

    # The sliders only apply to "Custom"; they start at the chosen preset's limits
    limits = hires_guard.PRESETS[preset["hires_guard.preset"]]
    assert (preset["hires_guard.base_mp"], preset["hires_guard.hires_mp"], preset["hires_guard.max_upscale"]) == (
        limits.base_megapixels,
        limits.hires_megapixels,
        limits.max_upscale,
    )


def test_img2img_has_no_panel():
    # txt2img only: no layout, so no params, preset bindings or infotext keys on img2img
    assert harness.script_layout(hires_guard.AnimaHiresGuardScript(), "img2img") is None
