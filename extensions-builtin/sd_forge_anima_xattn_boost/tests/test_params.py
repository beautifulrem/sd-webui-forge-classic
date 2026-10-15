"""The ``params()`` panel against the legacy Gradio ``ui()`` contract
(``fixtures/legacy_ui_<tab>.json``, snapshot before the port) and its
schema export."""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "anima_xattn_boost.py"
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("remi_panel_harness", ROOT.parents[1] / "tests" / "remi_panel_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

boost = harness.load_script(SCRIPT, "anima_xattn_boost_params_under_test")


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_keep_the_legacy_contract(tab):
    legacy = harness.load_fixture(ROOT / "tests" / "fixtures" / f"legacy_ui_{tab}.json")
    layout = harness.script_layout(boost.AnimaXattnBoostScript(), tab)
    assert harness.contract_differences(legacy, layout) == {}
    assert layout.names == ["enabled", "strength", "band", "renorm"]


def test_params_follow_the_processing_signature():
    layout = harness.script_layout(boost.AnimaXattnBoostScript())
    hook = inspect.signature(boost.AnimaXattnBoostScript.process_before_every_sampling)
    positional = [name for name, parameter in hook.parameters.items() if parameter.kind is parameter.POSITIONAL_OR_KEYWORD][2:]
    assert len(positional) == len(layout.inputs)
    assert layout.values_to_args({}) == [False, 2.0, 0.85, 0.5]


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_schema_export(tab):
    layout = harness.script_layout(boost.AnimaXattnBoostScript(), tab)
    schema = layout.schema()
    json.dumps(schema, allow_nan=False)
    assert schema["type"] == "section" and schema["enable"]["name"] == "enabled" and schema["enable"]["preset"] == "xattn_boost.enabled"
    assert [node["type"] for node in schema["children"]] == ["markdown", "row"]
    assert {node["name"]: node["infotext"] for node in schema["children"][1]["children"]} == {node.name: node.infotext for node in layout.inputs[1:]}


def test_every_string_has_a_zh_cn_translation():
    layouts = [harness.script_layout(boost.AnimaXattnBoostScript(), tab) for tab in ("txt2img", "img2img")]
    harness.assert_localized(layouts, ROOT / "localizations" / "zh_CN.json")


def test_the_extension_is_ported():
    harness.assert_ported(SCRIPT)
