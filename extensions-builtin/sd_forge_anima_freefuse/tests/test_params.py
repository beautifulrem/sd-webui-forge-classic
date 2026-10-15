"""The ``params()`` panel against the legacy Gradio ``ui()`` contract
(``fixtures/legacy_ui_<tab>.json``, snapshot before the port), its schema
export, and the conflict wiring that replaced the Gradio listeners."""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "anima_freefuse.py"
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("remi_panel_harness", ROOT.parents[1] / "tests" / "remi_panel_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

freefuse = harness.load_script(SCRIPT, "anima_freefuse_params_under_test")


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_keep_the_legacy_contract(tab):
    legacy = harness.load_fixture(ROOT / "tests" / "fixtures" / f"legacy_ui_{tab}.json")
    layout = harness.script_layout(freefuse.AnimaFreeFuseScript(), tab)
    assert harness.contract_differences(legacy, layout) == {}
    assert tuple(layout.names) == freefuse.SCRIPT_ARG_NAMES


def test_the_hooks_take_every_param():
    layout = harness.script_layout(freefuse.AnimaFreeFuseScript())
    assert len(freefuse._arguments(layout.values_to_args({})).__dict__) == len(layout.inputs)
    for hook in (freefuse.AnimaFreeFuseScript.process, freefuse.AnimaFreeFuseScript.process_before_every_sampling):
        assert list(inspect.signature(hook).parameters)[2] == "enable"  # after self, p; the rest is *values


def test_conflicts_are_declared_not_wired():
    from modules.anima_feature_conflicts import CONFLICT_ROLES, ROLE_FREEFUSE, SPATIAL_GROUP

    layout = harness.script_layout(freefuse.AnimaFreeFuseScript())
    # One-of-three with Regional and Artist Mixer
    assert layout.exclusive == {SPATIAL_GROUP: {"freefuse": "enable"}}
    # The Refiner rule reads and locks the switch through its preset field
    assert ROLE_FREEFUSE in CONFLICT_ROLES and layout.presets[ROLE_FREEFUSE][0] == "enable"
    assert "after_component" not in vars(freefuse.AnimaFreeFuseScript)


def test_subject_texts_follow_their_switch():
    layout = harness.script_layout(freefuse.AnimaFreeFuseScript())
    for index in freefuse.SUBJECTS:
        for suffix in ("selector", "concept"):
            assert layout.by_name[f"subject{index}_{suffix}"].enabled_if == {"param": f"subject{index}_active"}


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_schema_export(tab):
    layout = harness.script_layout(freefuse.AnimaFreeFuseScript(), tab)
    schema = layout.schema()
    json.dumps(schema, allow_nan=False)
    assert schema["type"] == "section" and schema["enable"]["preset"] == "freefuse.enabled"
    assert schema["enable"]["exclusive"] == {"group": "anima_spatial_conditioning", "member": "freefuse"}
    groups = [node["title"] for node in schema["children"] if node["type"] == "group"]
    assert groups == ["Phase 1: automatic mask collection", "Phase 2: LoRA routing and attention isolation"]


def test_preset_fields():
    harness.assert_preset_fields(
        harness.script_layout(freefuse.AnimaFreeFuseScript()),
        "freefuse",
        unowned=(*(f"subject{index}_{suffix}" for index in freefuse.SUBJECTS for suffix in ("active", "selector", "concept")), "background"),
    )


def test_every_string_has_a_zh_cn_translation():
    layouts = [harness.script_layout(freefuse.AnimaFreeFuseScript(), tab) for tab in ("txt2img", "img2img")]
    harness.assert_localized(layouts, ROOT / "localizations" / "zh_CN.json")


def test_the_extension_is_ported():
    harness.assert_ported(SCRIPT)
