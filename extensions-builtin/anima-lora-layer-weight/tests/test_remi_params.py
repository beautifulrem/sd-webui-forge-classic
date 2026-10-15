"""The ported panel (``params()``) against the Gradio ``ui()`` it replaces
(tests/fixtures/legacy_ui_<tab>.json, recorded before the port), its schema,
actions and the Gradio-free load."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "anima_lora_layer_weight.py"
_spec = importlib.util.spec_from_file_location("remi_anima_lora_harness", ROOT.parents[1] / "tests" / "remi_anima_lora_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)
module = harness.load_script(SCRIPT, "anima_lora_layer_weight_params")

# The three weight texts are edited by the block-weight-grid widget; their
# value is still the weight text, so processing and infotext are unchanged.
WIDGET = {"kind": "custom", "widget": "block-weight-grid"}
CHANGES = {f"{kind}_{name}": WIDGET for kind in module.ADAPTER_KINDS for name in ("block_weights", "module_weights", "text_layer_weights")}


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(tab):
    layout = harness.script_layout(module.Script(), tab)
    assert layout.names == list(module.SCRIPT_ARG_NAMES)
    harness.assert_parity(layout, ROOT, tab, CHANGES)


def test_the_schema_exports_and_names_the_widgets():
    layout = harness.script_layout(module.Script())
    schema = harness.schema_json(layout)
    assert schema["type"] == "section" and schema["title"] == "Anima LoRA/LoKR Layer Weight"
    text = json.dumps(schema)
    assert text.count('"widget": "block-weight-grid"') == 6
    tabs = next(node for node in schema["children"] if node["type"] == "tabs")
    assert [tab["title"] for tab in tabs["children"]] == ["LoRA", "LoKR"]
    blocks = next(node for node in tabs["children"][0]["children"] if node.get("name") == "lora_block_weights")
    assert blocks["props"]["layouts"] == [28, 40, 52] and len(blocks["props"]["mappings"]["52"]) == 52
    assert blocks["visible_if"] == {"param": "lora_block_enabled"}
    assert blocks["props"]["presets"][0] == {"label": "Flat 1.0", "value": "all=1.0"}


def test_presets_bind_the_switches():
    layout = harness.script_layout(module.Script())
    assert harness.preset_bindings(layout) == {"lora_layer.lora_enabled": ("lora_enabled", False), "lora_layer.lokr_enabled": ("lokr_enabled", False)}


def test_weight_texts_validate_as_text():
    from remi import ui

    layout = harness.script_layout(module.Script())
    assert layout.coerce({"lora_block_weights": "0-9=0.5", "lora_text_layer_weights": None})["lora_text_layer_weights"] == ""
    with pytest.raises(ui.ValidationError):
        layout.coerce({"lora_block_weights": [0.5, 1.0]})


def test_the_zh_ui_preset_scopes_of_the_gradio_panel_are_accepted():
    layout = harness.script_layout(module.Script())
    coerced = layout.coerce({"lora_preset_scope": module.PRESET_SCOPE_ALL, "lokr_preset_scope": "仅目标列表"})
    assert (coerced["lora_preset_scope"], coerced["lokr_preset_scope"]) == (module.PRESET_SCOPE_ALL_EN, module.PRESET_SCOPE_TARGETS_EN)
    assert module._preset_target_mode(coerced["lokr_preset_scope"]) == "targets"


def test_the_module_grid_has_a_row_for_every_module_a_run_weights():
    schema = harness.schema_json(harness.script_layout(module.Script()))
    grid = next(node for node in _walk(schema) if node.get("name") == "lora_module_weights")
    from modules.anima_lora_support import module_name_from_block_tail

    names = {module_name_from_block_tail(tail) for tail in ("self_attn.q", "cross_attn.k", "mlp.layer1", "adaln_modulation.1", "norm1")}
    assert set(grid["props"]["modules"]) == names


def _walk(node):
    if isinstance(node, dict):
        yield node
        for child in node.get("children", ()):
            yield from _walk(child)


def test_apply_preset_rewrites_the_core_prompt(monkeypatch):
    from remi import ui

    monkeypatch.setattr(module, "_prompt_lora_is_controlled", lambda name, targets, kind="lora", misses=None: kind == "lora")
    layout = harness.script_layout(module.Script())
    values = layout.defaults()
    ctx = ui.ActionContext(tab="txt2img", owner=layout.owner, core={"prompt": "a <lora:cat:0.6>"})
    patch = ui.call_action(module.Script(), "lora_apply_preset", ctx, values)
    assert patch["core.prompt"]["value"] == "a <lora:cat:blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,cross_attn=1.0,mlp=1.0,adaln=1.0>"
    assert patch["__toast__"]["level"] == "success"

    untouched = ui.call_action(module.Script(), "lokr_apply_preset", ctx, values)
    assert "core.prompt" not in untouched and untouched["__toast__"]["level"] == "info"


def test_editing_the_presets_updates_the_preset_choices():
    from remi import ui

    ctx = ui.ActionContext(tab="txt2img")
    patch = ui.call_action(module.Script(), "lokr_presets_changed", ctx, {"lokr_preset_text": "Mine=blocks=all=0.5\nOther=blocks=0=1", "lokr_preset_name": "gone"})
    entry = patch["lokr_preset_name"]
    assert [choice["value"] for choice in entry["choices"]] == ["gone", "mine", "other"] and entry["value"] == "gone"
    patch = ui.call_action(module.Script(), "lokr_presets_changed", ctx, {"lokr_preset_text": "Mine=blocks=all=0.5", "lokr_preset_name": ""})
    assert patch["lokr_preset_name"]["value"] == "mine"


def test_the_extension_is_gradio_free():
    assert harness.gradio_imports(ROOT) == []
    harness.assert_loads_without_gradio(SCRIPT)


def test_the_zh_cn_catalog_covers_every_string():
    catalog = json.loads((ROOT / "localizations" / "zh_CN.json").read_text(encoding="utf8"))
    layout = harness.script_layout(module.Script())
    strings = _strings(harness.schema_json(layout)) | {module.TOAST_NO_MATCH, module.TOAST_APPLIED}
    # User content in Chinese (the default preset names) is not translated
    missing = sorted(text for text in strings if text not in catalog and not re.search("[\u3000-\u9fff]", text))
    assert missing == []


def _strings(node) -> set[str]:
    """Every translatable string of a schema tree: labels, info, titles,
    markdown, choice labels, placeholders and the widgets' own labels."""

    found: set[str] = set()
    if isinstance(node, dict):
        for key in ("label", "info", "title", "text", "placeholder"):
            if isinstance(node.get(key), str) and node[key] and node.get("type") != "html":
                found.add(node[key])
        for choice in node.get("choices", ()):
            if isinstance(choice, dict) and isinstance(choice.get("label"), str):
                found.add(choice["label"])
        for key in ("children", "enable", "props", "presets", "bands"):
            if key in node:
                found |= _strings(node[key])
    elif isinstance(node, list):
        for item in node:
            found |= _strings(item)
    return found
