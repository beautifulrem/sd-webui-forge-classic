"""A layer-weighted run's infotext pastes back into the panel it was run
from, through the WebUI's paste and through the API's infotext replay."""

import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_layer_weight.py"
REPO = Path(__file__).parents[3]
_MODULE = []


def _extract(path: Path, wanted: set, namespace: dict) -> dict:
    """The top-level functions, classes and assignments named in ``wanted``
    from ``path``, whose module needs the full WebUI install to import."""

    tree = ast.parse(path.read_text(encoding="utf8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted)
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in wanted)
    ]
    exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)  # noqa: S102 - our own source file
    assert wanted <= set(namespace)
    return namespace


# re_param_code, quote, unquote, _parse_info and PasteField of modules/infotext_utils.py
INFOTEXT = _extract(
    REPO / "modules" / "infotext_utils.py",
    {"re_param_code", "quote", "unquote", "_parse_info", "PasteField"},
    {"json": json, "gr": gr, "Any": object, "type_of_gr_update": type(gr.skip()), "errors": SimpleNamespace(report=print)},
)
# The API's conversion of a paste field's value (modules/api/api.py)
pasted_value = _extract(REPO / "modules" / "api" / "api.py", {"pasted_value"}, {})["pasted_value"]


@pytest.fixture
def module(monkeypatch):
    if not _MODULE:
        import modules

        if "modules.infotext_utils" not in sys.modules:
            # Its image helpers need the full WebUI install; the script only needs PasteField
            stub = types.ModuleType("modules.infotext_utils")
            stub.PasteField = INFOTEXT["PasteField"]
            modules.infotext_utils = sys.modules["modules.infotext_utils"] = stub
        spec = importlib.util.spec_from_file_location("anima_lora_layer_weight_infotext", SCRIPT)
        _MODULE.append(importlib.util.module_from_spec(spec))
        spec.loader.exec_module(_MODULE[0])
    module = _MODULE[0]
    monkeypatch.setattr(module, "install_patch", lambda: None)
    monkeypatch.setattr(module, "_prompt_lora_is_controlled", lambda *args: True)
    monkeypatch.setattr(module, "_ACTIVE_RULE", None)
    monkeypatch.setattr(module, "_LAST_SIGNATURE", None)
    monkeypatch.setattr(module, "_LAST_TEXT_ENCODER_SIGNATURE", None)
    return module


def _language(module, monkeypatch, language):
    from modules import shared

    monkeypatch.setattr(shared, "opts", SimpleNamespace(**{module.LANGUAGE_OPTION: language}), raising=False)


def _build_ui(module):
    from modules.anima_presets import reset_preset_controls

    reset_preset_controls("txt2img")
    script = module.Script()
    script.tabname, script.is_txt2img, script.is_img2img = "txt2img", True, False
    with gr.Blocks():
        controls = script.ui(False)
    return script, controls


def _run(module, args):
    """The infotext parameters ``before_process_batch`` records for the script
    args ``args``, and the rule the batch runs."""

    p = SimpleNamespace(prompts=["<lora:cat:0.6> <lora:fox:0.5>"], extra_generation_params={})
    module.Script().before_process_batch(p, *args)
    return p.extra_generation_params, module._ACTIVE_RULE


def _parse(params) -> dict:
    """``params`` as processing.create_infotext writes them, read back as
    parse_generation_parameters does."""

    line = "Steps: 20, Sampler: Euler, " + ", ".join(f"{key}: {INFOTEXT['quote'](value)}" for key, value in params.items())
    return {key: INFOTEXT["unquote"](value) for key, value in re.findall(INFOTEXT["re_param_code"], line)}


def _paste(script, params):
    """What pasting ``params`` in the WebUI sets each control to (None: left alone)."""

    updates = [INFOTEXT["_parse_info"](field.component, field.label, _parse(params)) for field in script.infotext_fields]
    return [update.get("value") if "value" in update else None for update in updates]


def _api_replay(script, controls, params):
    """The script args an API request with ``params`` as its infotext runs:
    the controls' defaults, each mentioned field replaced as
    Api.apply_infotext's get_field_value does (field function or label, then
    pasted_value to the control's value type). No on_infotext_pasted callback runs."""

    parsed = _parse(params)
    indexes = {id(control): index for index, control in enumerate(controls)}
    args = [control.value for control in controls]
    for field in script.infotext_fields:
        value = field.function(parsed) if field.function else parsed.get(field.label)
        if value is not None:
            args[indexes[id(field.component)]] = pasted_value(value, type(field.component.value))
    return args


def _args(module, scope_label):
    """A run of every tab control away from its default. ``scope_label``: the
    preset scope label of the UI language (a preset control: not recorded)."""

    lora = [True, "cat, sub/dog.safetensors", True, "0-9=0.6, 10-18=1.0, 19-27=0.8", True, "self_attn=0.9\nmlp=0.7", True, "0-13=0.3,all=0.9", True, "强画风", scope_label, "mine=blocks=0-9=0.5"]
    lokr = [True, "", False, "0-5=0.2", True, "cross_attn=0.5, mlp=0.8", False, "", True, "保构图", scope_label, "other=blocks=all=0.9"]
    return lora + lokr


# Where each tab's recorded controls sit in the script args (the preset slots follow them)
LORA, LOKR = slice(0, 8), slice(12, 20)


def _recorded(args):
    """The tab controls of the script args ``args``, as the paste fields list them."""
    return args[LORA] + args[LOKR]


def _scope(module, language):
    return module.LANG[language]["scope_targets"]


def _config(rule):
    """What a rule weights: per tab, its switches, targets and parsed weights."""
    tabs = (rule.lora_rule, rule.lokr_rule)
    return [(tab.enabled, tab.targets, tab.block_enabled, tab.block_weights, tab.module_enabled, tab.module_weights, tab.text_layer_enabled, tab.text_layer_weights) for tab in tabs], rule.text_encoder_signature


@pytest.mark.parametrize("run_language, paste_language", [("zh", "zh"), ("en", "zh"), ("zh", "en"), ("en", "en")])
def test_a_runs_infotext_restores_every_control(module, monkeypatch, run_language, paste_language):
    _language(module, monkeypatch, run_language)
    args = _args(module, _scope(module, run_language))
    params, _ = _run(module, args)
    _language(module, monkeypatch, paste_language)
    script, controls = _build_ui(module)

    assert [field.component for field in script.infotext_fields] == _recorded(controls)
    # Plain keys, so the "Send to" buttons carry the controls
    assert script.paste_field_names == list(module.INFOTEXT_KEYS) == [field.label for field in script.infotext_fields]
    assert all(type(name) is str for name in script.paste_field_names)
    assert set(params) <= set(module.INFOTEXT_KEYS)

    expected = _recorded(args)
    expected[11], expected[15] = None, None  # LoKR block / Qwen text with its switch off: not recorded, left alone
    assert _paste(script, params) == expected


def test_the_infotext_records_one_stable_key_per_control(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    zh, _ = _run(module, _args(module, _scope(module, "zh")))
    _language(module, monkeypatch, "en")
    en, _ = _run(module, _args(module, _scope(module, "en")))

    assert zh == en == {
        "Anima LoRA layer lora enabled": True,
        "Anima LoRA layer lora targets": "cat, sub/dog.safetensors",
        "Anima LoRA layer lora blocks enabled": True,
        "Anima LoRA layer lora blocks": "0-9=0.6, 10-18=1.0, 19-27=0.8",
        "Anima LoRA layer lora modules enabled": True,
        "Anima LoRA layer lora modules": "self_attn=0.9\nmlp=0.7",
        "Anima LoRA layer lora qwen enabled": True,
        "Anima LoRA layer lora qwen": "0-13=0.3,all=0.9",
        "Anima LoRA layer lokr enabled": True,
        "Anima LoRA layer lokr blocks enabled": False,
        "Anima LoRA layer lokr modules enabled": True,
        "Anima LoRA layer lokr modules": "cross_attn=0.5, mlp=0.8",
        "Anima LoRA layer lokr qwen enabled": False,
    }
    assert "Anima LoRA/LoKR Layer Weight" not in zh  # the old signature line: nothing reads it


def test_the_pasted_panel_runs_the_same_weights(module, monkeypatch):
    _language(module, monkeypatch, "en")
    args = _args(module, _scope(module, "en"))
    params, rule = _run(module, args)
    config = _config(rule)
    script, controls = _build_ui(module)

    # Into a fresh panel: the controls left alone keep their defaults
    pasted = [control.value for control in controls]
    for index, value in zip([*range(LORA.start, LORA.stop), *range(LOKR.start, LOKR.stop)], _paste(script, params), strict=True):
        if value is not None:
            pasted[index] = value
    pasted_params, pasted_rule = _run(module, pasted)

    assert pasted_params == params
    assert _config(pasted_rule) == config
    assert pasted_rule.lora_rule.targets and pasted_rule.lora_rule.block_weights.factor(15) == 1.0 and pasted_rule.lora_rule.module_weights["mlp"] == 0.7
    assert pasted_rule.lokr_rule.targets == set() and pasted_rule.lokr_rule.module_weights["cross_attn"] == 0.5


def test_the_api_infotext_replay_runs_the_same_weights(module, monkeypatch):
    """The API replays an infotext without the on_infotext_pasted callbacks:
    its script args are the recorded values themselves, converted to each
    control's value type. A recorded "False" must not become bool("False")."""

    _language(module, monkeypatch, "zh")
    args = _args(module, _scope(module, "zh"))
    params, rule = _run(module, args)
    config = _config(rule)
    script, controls = _build_ui(module)

    replayed = _api_replay(script, controls, params)
    replayed_params, replayed_rule = _run(module, replayed)

    assert replayed_params == params
    assert _config(replayed_rule) == config
    assert replayed[LOKR][2] is False and replayed[LOKR][6] is False  # the LoKR block / Qwen switches stay off
    assert replayed[LOKR][1] == ""  # the empty target list stays empty: every Anima LoKR


def test_the_run_reads_recorded_switch_texts(module):
    """An API caller may pass the recorded switch values as text."""

    args = _args(module, "")
    params, rule = _run(module, args)
    as_text = [str(value) if isinstance(value, bool) else value for value in args]
    text_params, text_rule = _run(module, as_text)

    assert (text_params, _config(text_rule)) == (params, _config(rule))
    assert module._switch("False") is False and module._switch(" true ") is True and module._switch("") is False


def test_a_disabled_tab_records_its_switch_alone(module, monkeypatch):
    _language(module, monkeypatch, "en")
    args = _args(module, _scope(module, "en"))
    args[12] = False  # LoKR off
    params, _ = _run(module, args)
    script, controls = _build_ui(module)

    assert {key for key in params if " lokr " in key} == {"Anima LoRA layer lokr enabled"}
    assert _paste(script, params)[8:] == [False] + [None] * 7  # its other controls stay as they are
    replayed = _api_replay(script, controls, params)
    assert replayed[12] is False and replayed[LOKR][1:] == [control.value for control in controls[LOKR][1:]]

    both_off = [False, *args[1:12], False, *args[13:]]
    assert _run(module, both_off)[0] == {}  # a run the panel did not touch records nothing


def test_an_old_signature_only_infotext_leaves_the_panel_alone(module, monkeypatch):
    _language(module, monkeypatch, "en")
    script, controls = _build_ui(module)
    old = {
        "Anima LoRA/LoKR Layer Weight": "lora[lora|1|cat|target_mode=legacy|block=1|0-9=0.6|module=0|self_attn=1.0|qwen=0||]|lokr[lokr|0||target_mode=legacy|block=0||module=0||qwen=0||]",
        "Anima LoRA/LoKR Layer Weight skipped": "cat.safetensors",
    }

    assert _paste(script, old) == [None] * len(module.INFOTEXT_KEYS)
    assert _api_replay(script, controls, old) == [control.value for control in controls]
