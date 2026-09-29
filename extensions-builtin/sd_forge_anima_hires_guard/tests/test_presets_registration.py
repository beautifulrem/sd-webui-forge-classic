"""The preset's hires_guard.* values are the panel's own defaults."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the DAVE / Cross-Attention Boost suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

hires_guard = harness.load_script(ROOT / "scripts" / "anima_hires_guard.py", "anima_hires_guard_presets_under_test")


def test_apply_preset_turns_the_panel_off_at_its_defaults():
    harness.assert_registers_preset(hires_guard.AnimaHiresGuardScript(), "hires_guard")


def test_preset_limits_are_those_of_its_safety_preset():
    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET as preset

    # The sliders only apply to "Custom"; they start at the chosen preset's limits
    limits = hires_guard.PRESETS[preset["hires_guard.preset"]]
    assert (preset["hires_guard.base_mp"], preset["hires_guard.hires_mp"], preset["hires_guard.max_upscale"]) == (
        limits.base_megapixels,
        limits.hires_megapixels,
        limits.max_upscale,
    )


def test_img2img_registers_nothing():
    from modules.anima_presets import preset_controls

    assert harness.build_ui(hires_guard.AnimaHiresGuardScript(), "img2img") == []
    assert preset_controls("img2img") == ((), (), ())
