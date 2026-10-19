"""The ``params()`` panel against the legacy Gradio ``ui()`` contract
(``fixtures/legacy_ui_<tab>.json``, snapshot before the port), its schema
export, and the region editor's argument slot."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "anima_regional.py"
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("remi_panel_harness", ROOT.parents[1] / "tests" / "remi_panel_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

regional = harness.load_script(SCRIPT, "anima_regional_params_under_test")
REGION_FIELDS = ("active", "prompt", "x", "y", "width", "height", "strength")


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_keep_the_legacy_contract(tab):
    legacy = harness.load_fixture(ROOT / "tests" / "fixtures" / f"legacy_ui_{tab}.json")
    layout = harness.script_layout(regional.AnimaRegionalScript(), tab)
    # Intentional: the region editor widget is appended as the last argument
    assert harness.contract_differences(legacy, layout) == {"region_editor": {"added": "custom"}}
    assert layout.names == [*regional.SCRIPT_ARG_NAMES, "region_editor"]


def test_the_switch_is_a_member_of_the_spatial_group():
    from modules.anima_feature_conflicts import EXCLUSIVE_GROUPS, SPATIAL_GROUP

    layout = harness.script_layout(regional.AnimaRegionalScript())
    assert layout.exclusive == {SPATIAL_GROUP: {"regional": "enable"}}
    assert "regional" in EXCLUSIVE_GROUPS[SPATIAL_GROUP]


def test_region_prompts_use_the_prompt_editor():
    layout = harness.script_layout(regional.AnimaRegionalScript())
    assert all(layout.by_name[f"region{index}_prompt"].prompt for index in regional.REGIONS)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_schema_export(tab):
    layout = harness.script_layout(regional.AnimaRegionalScript(), tab)
    schema = layout.schema()
    json.dumps(schema, allow_nan=False)
    assert schema["type"] == "section" and schema["enable"]["exclusive"] == {"group": "anima_spatial_conditioning", "member": "regional"}
    editor = next(node for node in schema["children"] if node["type"] == "custom")
    assert editor["widget"] == "region-boxes" and editor["default"] is None
    assert editor["props"] == {"regions": ["region1", "region2", "region3"]}
    # Every region param the widget binds exists, with the bounds it edits within
    for prefix in editor["props"]["regions"]:
        for field in ("x", "y", "width", "height"):
            node = layout.by_name[f"{prefix}_{field}"]
            assert (node.minimum, node.maximum, node.step) == (0.0, 1.0, 0.01)
        assert {f"{prefix}_{field}" for field in REGION_FIELDS} <= set(layout.names)
    groups = [node for node in schema["children"] if node["type"] == "group"]
    assert [(group["title"], group["open"]) for group in groups] == [("Region 1", True), ("Region 2", True), ("Region 3", False)]


def test_argument_lists_with_and_without_the_editor_slot():
    _p, unet = harness.make_process()
    unet.model.predictor = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
    layout = harness.script_layout(regional.AnimaRegionalScript())
    args = layout.values_to_args({"enable": True, "region1_prompt": "a cat", "region_editor": {"anything": 1}})
    assert len(args) == len(regional.SCRIPT_ARGS) + 1
    for given in (args, args[:-1], [*args[:-1], None]):
        assert regional._arguments(given).region1_prompt == "a cat"
    # A scalar in the editor slot is a misaligned list (one stray value), not an editor value
    with pytest.raises(TypeError, match="misaligned"):
        regional._arguments([*args[:-1], 0.5])


def test_every_string_has_a_zh_cn_translation():
    layouts = [harness.script_layout(regional.AnimaRegionalScript(), tab) for tab in ("txt2img", "img2img")]
    harness.assert_localized(layouts, ROOT / "localizations" / "zh_CN.json")


def test_the_extension_is_ported():
    harness.assert_ported(SCRIPT)
