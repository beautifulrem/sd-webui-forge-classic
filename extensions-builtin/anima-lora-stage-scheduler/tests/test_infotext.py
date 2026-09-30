"""A scheduled run's infotext pastes back into the panel it was run from."""

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
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_stage_scheduler.py"
INFOTEXT_UTILS = Path(__file__).parents[3] / "modules" / "infotext_utils.py"
_MODULE = []


def _infotext_helpers() -> dict:
    """``re_param_code``, ``quote``, ``unquote`` and ``_parse_info`` from
    modules/infotext_utils.py, whose import needs the full WebUI install."""

    tree = ast.parse(INFOTEXT_UTILS.read_text(encoding="utf8"))
    wanted = {"re_param_code", "quote", "unquote", "_parse_info"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in wanted)
    ]
    namespace = {"json": json, "gr": gr, "Any": object, "type_of_gr_update": type(gr.skip()), "errors": SimpleNamespace(report=print)}
    exec(compile(ast.Module(nodes, []), str(INFOTEXT_UTILS), "exec"), namespace)  # noqa: S102 - our own source file
    assert wanted <= set(namespace)
    return namespace


INFOTEXT = _infotext_helpers()


@pytest.fixture
def module(tmp_path, monkeypatch):
    from backend.args import dynamic_args
    from modules import paths_internal

    # _TEMPLATES.path() seeds <data dir>/extension-data with the bundled templates
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    monkeypatch.setattr(dynamic_args, "online_lora", False)
    if not _MODULE:
        import modules

        if "modules.infotext_utils" not in sys.modules:
            # Its image helpers need the full WebUI install; the script only needs PasteField
            stub = types.ModuleType("modules.infotext_utils")
            stub.PasteField = lambda component, target: (component, target)
            modules.infotext_utils = sys.modules["modules.infotext_utils"] = stub
        spec = importlib.util.spec_from_file_location("anima_lora_stage_scheduler_infotext", SCRIPT)
        _MODULE.append(importlib.util.module_from_spec(spec))
        spec.loader.exec_module(_MODULE[0])
    monkeypatch.setattr(_MODULE[0], "_TEXT_ENCODER_SIGNATURE", None, raising=False)
    return _MODULE[0]


def _language(module, monkeypatch, language):
    from modules import shared

    monkeypatch.setattr(shared, "opts", SimpleNamespace(**{module.LANGUAGE_OPTION: language}), raising=False)


def _build_ui(module):
    from modules.anima_presets import reset_preset_controls

    reset_preset_controls("txt2img")
    script = module.Script()
    script.tabname, script.is_txt2img, script.is_img2img = "txt2img", True, False
    with gr.Blocks() as blocks:
        controls = script.ui(False)
    return script, controls, blocks


def _run(module, args, enable_hr=True):
    """The infotext parameters ``process`` records for the script args ``args``."""

    p = SimpleNamespace(extra_generation_params={}, enable_hr=enable_hr, distilled_cfg_scale=3.0, hr_distilled_cfg=5.0)
    module.Script().process(p, *args)
    state = p._anima_lora_stage_scheduler_state
    module._cleanup_runtime(p)
    return p.extra_generation_params, state


def _paste(module, script, params):
    """What pasting ``params`` sets each control to (None: left alone): the
    infotext line as processing.create_infotext writes it, read back as
    parse_generation_parameters and connect_paste do."""

    line = "Steps: 20, Sampler: Euler, " + ", ".join(f"{key}: {INFOTEXT['quote'](value)}" for key, value in params.items())
    parsed = {key: INFOTEXT["unquote"](value) for key, value in re.findall(INFOTEXT["re_param_code"], line)}
    module._translate_pasted_infotext(line, parsed)
    updates = [INFOTEXT["_parse_info"](component, key, parsed) for component, key in script.infotext_fields]
    return [update.get("value") if "value" in update else None for update in updates]


def _args(module, labels):
    """A run of every panel control away from its default. ``labels``: the
    target scope / disable mode labels of the UI language."""

    targets, prompt_all = labels(module.MODE_TARGETS), labels(module.MODE_PROMPT_ALL)
    style_start, style_end, style_peak, _ = module._auto_stage_values(module.STAGE_STYLE, 5.0)
    return [
        True,
        # Base: target list, manual timing, a per-LoRA template
        targets,
        "mylora, other",
        "0.8",
        module.PRESET_CHARACTER_ALLOW,
        "mylora=" + module.PRESET_COMPOSITION_ALLOW,
        False,
        0.3,
        0.7,
        0.4,
        1.7,
        module.CURVE_TRIANGLE,
        True,
        # Hires: independent, block weights, auto timing at the Hires Shift
        prompt_all,
        "",
        "blocks=0-18=1,19-27=0.2",
        module.PRESET_STYLE_ALLOW,
        "",
        True,
        round(style_start, 4),
        round(style_end, 4),
        round(style_peak, 4),
        1.0,
        module.CURVE_BACK,
        labels(module.DISABLE_SELECTED),
        "<lora:a:0.5>\n<lora:b:blocks=0-9=0.8,10-27=1>",
    ]


def _zh(choice):
    """The zh UI's label of a target scope or disable mode: the choice itself."""
    return choice


def _en(choice):
    """The en UI's label of a target scope or disable mode."""
    return {**_MODULE[0].TARGET_MODE_EN, **_MODULE[0].DISABLE_EN}[choice]


@pytest.mark.parametrize("run_language, paste_language", [("zh", "zh"), ("en", "zh"), ("zh", "en"), ("en", "en")])
def test_a_runs_infotext_restores_every_control(module, monkeypatch, run_language, paste_language):
    labels = {"zh": _zh, "en": _en}
    _language(module, monkeypatch, run_language)
    params, _ = _run(module, _args(module, labels[run_language]))
    _language(module, monkeypatch, paste_language)
    script, controls, _ = _build_ui(module)

    assert [component for component, _ in script.infotext_fields] == controls
    assert script.paste_field_names == list(module.INFOTEXT_KEYS) == [key for _, key in script.infotext_fields]
    assert set(params) <= set(module.INFOTEXT_KEYS)
    # Every control is back where the run had it, labelled in the pasting UI's language
    assert _paste(module, script, params) == _args(module, labels[paste_language])


def test_the_infotext_records_stable_values_whatever_the_ui_language(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    zh, _ = _run(module, _args(module, _zh))
    _language(module, monkeypatch, "en")
    en, _ = _run(module, _args(module, _en))

    assert zh == en
    key = module._panel_key
    assert (zh[key("base", "scope")], zh[key("hires", "scope")], zh[module.INFOTEXT_HIRES_DISABLE]) == ("targets", "all", "selected")
    assert (zh[key("base", "curve")], zh[key("hires", "curve")]) == ("triangle", "back")
    assert (zh[key("base", "weight")], zh[key("base", "strength")], zh[key("base", "per-LoRA templates")]) == ("0.8", 1.7, "mylora=" + module.PRESET_COMPOSITION_ALLOW)


def test_the_pasted_panel_runs_the_same_schedule_per_lora_templates_included(module, monkeypatch):
    _language(module, monkeypatch, "en")
    params, state = _run(module, _args(module, _en))
    script, _, _ = _build_ui(module)
    pasted, pasted_state = _run(module, _paste(module, script, params))

    assert pasted == params
    for config, pasted_config in ((state.base, pasted_state.base), (state.hires, pasted_state.hires)):
        for filename, names in (("mylora.safetensors", {"mylora"}), ("other.safetensors", {"other"})):
            before, after = config.config_for_file(filename, names), pasted_config.config_for_file(filename, names)
            assert (after.template_name, after.stage, after.parts, after.curve) == (before.template_name, before.stage, before.parts, before.curve)
            assert (after.start, after.end, after.peak, after.stage_strength) == pytest.approx((before.start, before.end, before.peak, before.stage_strength))
        assert (pasted_config.target_mode, pasted_config.targets, pasted_config.panel_weight) == (config.target_mode, config.targets, config.panel_weight)

    # mylora runs on its own template's (composition) window, not the panel's
    mylora = pasted_state.base.config_for_file("mylora.safetensors", {"mylora"})
    assert mylora.template_name == module.PRESET_COMPOSITION_ALLOW
    assert (mylora.start, mylora.end, mylora.peak) == module._auto_stage_values(module.STAGE_COMPOSITION, 3.0)[:3]
    assert (pasted_state.base.start, pasted_state.base.end) == (0.3, 0.7)
    assert pasted_state.hr_disable_tags == ["<lora:a:0.5>", "<lora:b:blocks=0-9=0.8,10-27=1>"]


def test_text_a_run_ignored_pastes_empty_and_no_hires_leaves_the_hires_section_alone(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    args = _args(module, _zh)
    args[1] = module.MODE_AUTO_ANIMA  # the target list is ignored
    args[5] = ""
    params, _ = _run(module, args, enable_hr=False)
    script, controls, _ = _build_ui(module)
    pasted = _paste(module, script, params)

    key = module._panel_key
    assert key("base", "targets") not in params and key("base", "per-LoRA templates") not in params
    assert not any(" hires " in name for name in params)
    assert pasted[:12] == [*args[:2], "", *args[3:12]]  # the ignored target list is emptied
    assert pasted[12:] == [None] * 14  # no Hires. fix pass: the hires controls stay


def test_a_shared_hires_panel_records_only_its_switch_and_disable_mode(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    args = _args(module, _zh)
    args[12], args[24], args[25] = False, module.DISABLE_ALL, "<lora:a:0.5>"
    params, _ = _run(module, args)
    script, _, _ = _build_ui(module)
    pasted = _paste(module, script, params)

    assert params[module.INFOTEXT_HIRES_INDEPENDENT] is False and params[module.INFOTEXT_HIRES_DISABLE] == "all"
    assert module.INFOTEXT_HIRES_DISABLE_TAGS not in params  # ignored when disabling every LoRA
    assert pasted[12] is False and pasted[13:24] == [None] * 11
    assert pasted[24:] == [module.DISABLE_ALL, ""]

    args[24] = module.DISABLE_NONE
    params, _ = _run(module, args)
    assert module.INFOTEXT_HIRES_DISABLE not in params
    assert _paste(module, script, params)[24:] == [module.DISABLE_NONE, ""]


def test_an_old_summary_only_infotext_leaves_the_panel_alone(module, monkeypatch):
    _language(module, monkeypatch, "en")
    script, _, _ = _build_ui(module)
    old = {
        "Anima LoRA Stage Scheduler": f"base={module.PRESET_STYLE_ALLOW}:0.62-1.00, hires={module.PRESET_STYLE_ALLOW}:0.70-1.00",
        "Anima Hires LoRA Disable": module.DISABLE_ALL,
    }

    assert _paste(module, script, old) == [None] * len(module.INFOTEXT_KEYS)


def test_a_template_missing_here_leaves_its_dropdown_alone(module, monkeypatch, caplog):
    _language(module, monkeypatch, "zh")
    module._TEMPLATES.edit(lambda data: (data.update(mine={"auto": True, "stage": module.STAGE_STYLE}), ("mine", "")))
    args = _args(module, _zh)
    args[4] = "mine"
    params, _ = _run(module, args)
    module._TEMPLATES.edit(lambda data: (data.pop("mine"), (None, "")))
    script, _, _ = _build_ui(module)

    pasted = _paste(module, script, params)
    assert pasted[4] is None and "mine" in caplog.text
    assert pasted[:4] == args[:4] and pasted[5:] == args[5:]


def test_template_dropdowns_reapply_on_user_selection_only(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    script, controls, blocks = _build_ui(module)
    templates = {controls[4]._id, controls[16]._id}
    triggers = {(target, event) for dependency in blocks.config["dependencies"] for target, event in dependency["targets"]}

    assert {(template, "input") for template in templates} <= triggers
    assert not {(template, "change") for template in templates} & triggers


def test_template_management_reapplies_the_templates_it_changes(module):
    module._TEMPLATES.edit(lambda data: (data.update(mine={"auto": False, "stage": module.STAGE_STYLE, "start": 0.1, "end": 0.3, "peak": 0.2, "strength": 1.5, "curve": module.CURVE_HOLD}), ("mine", "")))
    builtin = next(iter(module.BUILTIN_TEMPLATES))
    rename = module._reapplying_changed_templates(module._rename_template_ui)
    delete = module._reapplying_changed_templates(module._delete_template_ui)

    base, hires, _, _, *timings = rename("mine", "renamed", "mine", builtin, 3.0, 5.0)
    assert (base["value"], "value" in hires) == ("renamed", False)
    assert [update["value"] for update in timings[:6]] == [False, 0.1, 0.3, 0.2, 1.5, module.CURVE_HOLD]
    assert all("value" not in update for update in timings[6:])  # the hires panel's template stayed

    base, hires, _, _, *timings = delete("renamed", builtin, "renamed", 3.0, 5.0)
    assert hires["value"] == builtin  # fell back
    assert all("value" not in update for update in timings[:6])
    assert [update["value"] for update in timings[6:]] == [value["value"] for value in module._apply_template_ui(builtin, 5.0)]
