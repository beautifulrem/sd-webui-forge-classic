"""The panel's params() against the legacy Gradio ui() it replaces, and its schema."""

from pathlib import Path

import pytest
import remi_script_harness as harness

from remi import ui

ROOT = Path(__file__).resolve().parents[1]
hires_guard = harness.load_script(ROOT / "scripts" / "anima_hires_guard.py", "anima_hires_guard_params_under_test")


def _layout(tab="txt2img"):
    return harness.script_layout(hires_guard.AnimaHiresGuardScript(), tab)


def test_params_match_the_legacy_ui():
    harness.assert_matches_legacy(_layout(), ROOT, "txt2img")
    # The legacy img2img panel had no controls; the script is not shown there
    assert harness.legacy_snapshot(ROOT, "img2img")["inputs"] == [] and _layout("img2img") is None


def test_positional_args_follow_the_hook_signature():
    import inspect

    layout = _layout()
    hook = inspect.signature(hires_guard.AnimaHiresGuardScript.after_model_load)
    assert layout.names == [name for name in hook.parameters if name not in ("self", "process", "kwargs")]
    assert layout.names == [name for name, _key in hires_guard.FIELDS]


def test_schema_export():
    schema = harness.schema_json(_layout())
    assert schema["type"] == "section" and schema["enable"]["name"] == "enabled" and schema["enable"]["preset"] == "hires_guard.enabled"


@pytest.mark.parametrize(("preset", "editable"), [("Balanced", False), ("Low VRAM", False), ("Detail", False), ("Custom", True)])
def test_limit_sliders_are_editable_with_the_custom_preset_only(preset, editable):
    layout = _layout()
    rows = [node for node in layout.root.walk() if isinstance(node, ui.Row) and "base_mp" in [getattr(child, "name", None) for child in node.children]]
    assert len(rows) == 1
    assert ui.evaluate(rows[0].enabled_if, {"preset": preset}) is editable


def test_gradio_free():
    harness.assert_gradio_free(ROOT, ("scripts/anima_hires_guard.py",))


def test_zh_cn_catalog_covers_the_panel():
    harness.assert_localized(ROOT, [_layout()])
