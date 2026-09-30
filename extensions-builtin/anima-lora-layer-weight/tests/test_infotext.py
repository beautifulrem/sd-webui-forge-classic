"""A layer-weighted run's infotext pastes back into the panel it was run
from, through the WebUI's paste and through the API's infotext replay."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_layer_weight.py"
# Shared with the other panels' suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", Path(__file__).parents[3] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)
_MODULE = []


@pytest.fixture
def module(monkeypatch):
    if not _MODULE:
        _MODULE.append(harness.load_script(SCRIPT, "anima_lora_layer_weight_infotext"))
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


def test_the_script_args_table_is_the_ui_and_hook_order(module, monkeypatch):
    """ui() returns its controls in SCRIPT_ARG_NAMES order (each control's
    elem_id carries its own name), and before_process_batch reads its
    positional args by those names."""

    # The API's positional order: API callers pass these slots as they are
    tab = ["enabled", "target_loras", "block_enabled", "block_weights", "module_enabled", "module_weights", "text_layer_enabled", "text_layer_weights", "preset_enabled", "preset_name", "preset_scope", "preset_text"]
    assert list(module.SCRIPT_ARG_NAMES) == [f"lora_{name}" for name in tab] + [f"lokr_{name}" for name in tab]

    _language(module, monkeypatch, "en")
    script, controls = _build_ui(module)
    assert [control.elem_id for control in controls] == [script.elem_id(name) for name in module.SCRIPT_ARG_NAMES]
    assert [field.component.elem_id for field in harness.paste_fields(script)] == [script.elem_id(f"{kind}_{name}") for kind in module.ADAPTER_KINDS for name, field, _ in module.TAB_ARGS if field]

    read = []

    class Read(Exception):
        pass

    def active_rule(tabs, inline_rules, extra_generation_params=None):
        read.extend(getattr(tabs[kind], name) for kind in module.ADAPTER_KINDS for name in module.TAB_ARG_NAMES)
        raise Read

    monkeypatch.setattr(module, "ActiveRule", active_rule)
    args = [f"arg {index}" for index in range(len(module.SCRIPT_ARG_NAMES))]
    args[0] = args[12] = False  # both tabs off: no prompt rewrite
    with pytest.raises(Read):
        module.Script().before_process_batch(SimpleNamespace(prompts=[], extra_generation_params={}), *args, batch_number=0)
    assert read == args
    with pytest.raises(TypeError, match="lokr_preset_text"):
        module.Script().before_process_batch(SimpleNamespace(prompts=[]), *args[:-1])
    with pytest.raises(TypeError, match="takes 24 script arguments"):
        module.Script().before_process_batch(SimpleNamespace(prompts=[]), *args, "not a script arg")


@pytest.mark.parametrize("run_language, paste_language",[("zh", "zh"), ("en", "zh"), ("zh", "en"), ("en", "en")])
def test_a_runs_infotext_restores_every_control(module, monkeypatch, run_language, paste_language):
    _language(module, monkeypatch, run_language)
    args = _args(module, _scope(module, run_language))
    params, _ = _run(module, args)
    _language(module, monkeypatch, paste_language)
    script, controls = _build_ui(module)

    assert [field.component for field in harness.paste_fields(script)] == _recorded(controls)
    # Plain keys, so the "Send to" buttons carry the controls
    assert script.paste_field_names == list(module.INFOTEXT_KEYS) == [field.label for field in harness.paste_fields(script)]
    assert all(type(name) is str for name in script.paste_field_names)
    assert harness.send_to_components(script) == _recorded(controls)
    assert set(params) <= set(module.INFOTEXT_KEYS)

    expected = _recorded(args)
    expected[11], expected[15] = None, None  # LoKR block / Qwen text with its switch off: not recorded, left alone
    assert harness.ui_paste(script, params) == expected


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
    pasted = harness.pasted_args(script, controls, params)
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

    replayed = harness.api_replay(script, controls, params)
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
    assert harness.ui_paste(script, params)[8:] == [False] + [None] * 7  # its other controls stay as they are
    replayed = harness.api_replay(script, controls, params)
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

    assert harness.ui_paste(script, old) == [None] * len(module.INFOTEXT_KEYS)
    assert harness.api_replay(script, controls, old) == [control.value for control in controls]
