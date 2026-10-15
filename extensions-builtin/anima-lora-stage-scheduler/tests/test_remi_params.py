"""The ported panel (``params()``) against the Gradio ``ui()`` it replaces
(tests/fixtures/legacy_ui_<tab>.json, recorded before the port), its schema
and the Gradio-free load."""

import importlib.util
import json
import re
from pathlib import Path

import pytest
from lib_stage_scheduler import options

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "anima_lora_stage_scheduler.py"
_spec = importlib.util.spec_from_file_location("remi_anima_lora_harness", ROOT.parents[1] / "tests" / "remi_anima_lora_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)
module = harness.load_script(SCRIPT, "anima_lora_stage_scheduler_params")

# Choices keep their canonical (recorded) values; the English labels the
# Gradio panel used as values in its en mode are now the choice labels.
TARGET_MODE = {"choices": options.TARGET_MODE_CHOICES, "default": options.MODE_AUTO_ANIMA}
CHANGES = {
    "base_target_mode": {**TARGET_MODE, "paste": True},
    "hr_target_mode": {**TARGET_MODE, "paste": True},
    "hr_disable_mode": {"choices": options.DISABLE_CHOICES, "default": options.DISABLE_NONE, "paste": True},
    # Paste functions read the recorded codes (the on_infotext_pasted callback is gone)
    **{f"{prefix}_{name}": {"paste": True} for prefix in ("base", "hr") for name in ("target_loras", "template", "lora_templates")},
    "hr_disable_tags": {"paste": True},
}


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(tab):
    """Every legacy argument is a param of the same name, except each
    panel's six timing controls, which are one stage-timeline value."""

    layout = harness.script_layout(module.Script(), tab)
    legacy = dict(zip(options.SCRIPT_ARG_NAMES, harness.legacy_snapshot(ROOT, tab)["inputs"], strict=True))
    ported = {entry["name"]: entry for entry in harness.describe(layout)}
    plain = [name for name in options.PARAM_NAMES if not name.endswith("_timing")]
    harness.assert_entries([ported[name] for name in plain], [legacy[name] for name in plain], CHANGES)

    for prefix in ("base", "hr"):
        timing = ported[f"{prefix}_timing"]
        recorded = [legacy[f"{prefix}_{arg}"] for arg in options.TIMING_ARGS]
        assert timing["kind"] == "custom" and timing["widget"] == "stage-timeline"
        assert timing["default"] == dict(zip(options.TIMING_KEYS, (entry["default"] for entry in recorded)))
        # Pasted from the recorded keys of all six (the first names the value)
        assert timing["infotext"] == recorded[0]["infotext"] and timing["paste"]
        assert all(entry["kind"] in ("toggle", "slider", "choice") for entry in recorded)
        assert [entry["maximum"] for entry in recorded[1:5]] == [1.0, 1.0, 1.0, 2.0]


def test_the_schema_exports_the_timelines():
    layout = harness.script_layout(module.Script())
    schema = harness.schema_json(layout)
    assert schema["enable"]["name"] == "enabled" and schema["title"] == "Anima LoRA Stage Scheduler"
    text = json.dumps(schema, ensure_ascii=False)
    assert text.count('"widget": "stage-timeline"') == 2
    template = _find(schema, "base_template")
    # A user's selection only (Gradio's .input): a pasted template comes with its own timing
    assert template["on_input"] == "base_template_changed" and "on_change" not in template
    assert {"value": options.PRESET_STYLE_ALLOW, "label": "Style stage only"} in template["choices"]
    assert _find(schema, "base_target_loras")["visible_if"] == {"param": "base_target_mode", "eq": options.MODE_TARGETS}
    timeline = _find(schema, "hr_timing")
    assert timeline["props"]["template_param"] == "hr_template" and options.PRESET_STYLE_ALLOW in timeline["props"]["templates"]


@pytest.mark.parametrize("tab", harness.TABS)
def test_the_gradio_layout_cut_to_the_api_slots_is_refused(tab):
    """The API gives the script its params() slots only: a request built for
    the Gradio panel arrives cut to them and must fail clearly."""
    legacy = [entry["default"] for entry in harness.legacy_snapshot(ROOT, tab)["inputs"]]
    with pytest.raises(TypeError, match=r"argument 6 \(base_timing\) is a timing object, got bool"):
        options._arguments(legacy[: len(options.PARAM_NAMES)])


def test_presets_bind_the_switch():
    layout = harness.script_layout(module.Script())
    assert harness.preset_bindings(layout) == {"lora_stage.enabled": ("enabled", False)}


def test_a_timing_value_is_bounded_and_codes_read_as_curves():
    from remi import ui

    layout = harness.script_layout(module.Script())
    coerced = layout.coerce({"base_timing": {"auto": "False", "start": -1, "end": 3, "peak": 0.5, "strength": 9, "curve": "back"}})["base_timing"]
    assert coerced == {"auto": False, "start": 0.0, "end": 1.0, "peak": 0.5, "strength": 4.0, "curve": options.CURVE_BACK}
    with pytest.raises(ui.ValidationError):
        layout.coerce({"base_timing": [0.1, 0.2]})


def test_the_extension_is_gradio_free():
    assert harness.gradio_imports(ROOT) == []
    harness.assert_loads_without_gradio(SCRIPT)


def test_the_zh_cn_catalog_covers_every_string():
    catalog = json.loads((ROOT / "localizations" / "zh_CN.json").read_text(encoding="utf8"))
    layout = harness.script_layout(module.Script())
    # The action toasts too (the SPA translates them, placeholders and all)
    strings = _strings(harness.schema_json(layout)) | {module.Script().title()} | set(options.TEXT.values())
    missing = sorted(text for text in strings if text not in catalog and not re.search("[　-鿿]", text))
    assert missing == []
    # The canonical (Chinese) values are shown by their English labels, all translated
    labels = [*options.TEMPLATE_LABELS_EN.values(), *options.CURVE_LABELS_EN.values(), *options.STAGE_LABELS_EN.values(), *options.TARGET_MODE_EN.values(), *options.DISABLE_EN.values()]
    assert [label for label in labels if label not in catalog] == []


def _find(node, name):
    if isinstance(node, dict):
        if node.get("name") == name:
            return node
        for key in ("children", "enable"):
            found = _find(node.get(key), name)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find(item, name)
            if found is not None:
                return found
    return None


def _strings(node) -> set[str]:
    """Every translatable string of a schema tree (the widgets' props included)."""

    found: set[str] = set()
    if isinstance(node, dict):
        for key in ("label", "info", "title", "text", "placeholder"):
            if isinstance(node.get(key), str) and node[key] and node.get("type") != "html":
                found.add(node[key])
        for choice in node.get("choices", ()):
            if isinstance(choice, dict) and isinstance(choice.get("label"), str):
                found.add(choice["label"])
        for key in ("children", "enable", "props", "curves", "stages", "builder"):
            if key in node:
                found |= _strings(node[key])
    elif isinstance(node, list):
        for item in node:
            found |= _strings(item)
    return found
