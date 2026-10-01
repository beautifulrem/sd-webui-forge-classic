"""A scheduled run's infotext pastes back into the panel it was run from."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
from lib_stage_scheduler import options, runtime, templates

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_stage_scheduler.py"
# Shared with the other panels' suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", Path(__file__).parents[3] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)
_MODULE = []


@pytest.fixture
def module(tmp_path, monkeypatch):
    from backend.args import dynamic_args
    from modules import paths_internal

    # _TEMPLATES.path() seeds <data dir>/extension-data with the bundled templates
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    monkeypatch.setattr(dynamic_args, "online_lora", False)
    if not _MODULE:
        _MODULE.append(harness.load_script(SCRIPT, "anima_lora_stage_scheduler_infotext"))
    monkeypatch.setattr(runtime, "_TEXT_ENCODER_SIGNATURE", None, raising=False)
    return _MODULE[0]


def _language(module, monkeypatch, language):
    from modules import shared

    monkeypatch.setattr(shared, "opts", SimpleNamespace(**{options.LANGUAGE_OPTION: language}), raising=False)


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
    runtime._cleanup_runtime(p)
    return p.extra_generation_params, state


def _paste(module, script, params):
    """What pasting ``params`` sets each control to (None: left alone),
    through the panel's on_infotext_pasted callback."""

    return harness.ui_paste(script, params, callbacks=[runtime._translate_pasted_infotext])


def _schedule(module, state):
    """What a run schedules: each pass's panel as the LoRAs mylora (which has
    a per-LoRA template in _args) and other run it, and the Hires. fix
    disabling (its tags only when it disables those). Times rounded as the
    infotext records them."""

    def config(pass_config):
        return (
            pass_config.target_mode,
            pass_config.targets,
            pass_config.panel_weight,
            pass_config.lora_templates,
            pass_config.template_name,
            pass_config.auto_timing,
            pass_config.stage,
            pass_config.parts,
            pass_config.ban_window,
            *(round(value, 4) for value in (pass_config.start, pass_config.end, pass_config.peak, pass_config.stage_strength, pass_config.shift)),
            pass_config.curve,
        )

    files = (("mylora.safetensors", {"mylora"}), ("other.safetensors", {"other"}))
    passes = [[config(pass_config.config_for_file(filename, names)) for filename, names in files] for pass_config in (state.base, state.hires)]
    return passes, state.hr_disable_mode, state.hr_disable_tags if state.hr_disable_mode == options.DISABLE_SELECTED else None


def _args(module, labels):
    """A run of every panel control away from its default. ``labels``: the
    target scope / disable mode labels of the UI language."""

    targets, prompt_all = labels(options.MODE_TARGETS), labels(options.MODE_PROMPT_ALL)
    style_start, style_end, style_peak, _ = templates._auto_stage_values(options.STAGE_STYLE, 5.0)
    return [
        True,
        # Base: target list, manual timing, a per-LoRA template
        targets,
        "mylora, other",
        "0.8",
        options.PRESET_CHARACTER_ALLOW,
        "mylora=" + options.PRESET_COMPOSITION_ALLOW,
        False,
        0.3,
        0.7,
        0.4,
        1.7,
        options.CURVE_TRIANGLE,
        True,
        # Hires: independent, block weights, auto timing at the Hires Shift
        prompt_all,
        "",
        "blocks=0-18=1,19-27=0.2",
        options.PRESET_STYLE_ALLOW,
        "",
        True,
        round(style_start, 4),
        round(style_end, 4),
        round(style_peak, 4),
        1.0,
        options.CURVE_BACK,
        labels(options.DISABLE_SELECTED),
        "<lora:a:0.5>\n<lora:b:blocks=0-9=0.8,10-27=1>",
    ]


def _zh(choice):
    """The zh UI's label of a target scope or disable mode: the choice itself."""
    return choice


def _en(choice):
    """The en UI's label of a target scope or disable mode."""
    return {**options.TARGET_MODE_EN, **options.DISABLE_EN}[choice]


@pytest.mark.parametrize("run_language, paste_language", [("zh", "zh"), ("en", "zh"), ("zh", "en"), ("en", "en")])
def test_a_runs_infotext_restores_every_control(module, monkeypatch, run_language, paste_language):
    labels = {"zh": _zh, "en": _en}
    _language(module, monkeypatch, run_language)
    params, _ = _run(module, _args(module, labels[run_language]))
    _language(module, monkeypatch, paste_language)
    script, controls, _ = _build_ui(module)

    assert [component for component, _ in script.infotext_fields] == controls
    assert script.paste_field_names == list(options.INFOTEXT_KEYS) == [key for _, key in script.infotext_fields]
    assert set(params) <= set(options.INFOTEXT_KEYS)
    # Every control is back where the run had it, labelled in the pasting UI's language
    assert _paste(module, script, params) == _args(module, labels[paste_language])


def test_ui_and_process_name_the_script_args_in_table_order(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    script, controls, _ = _build_ui(module)
    panel = dict(options.PANEL_ARGS)

    def elem_id(name):
        # A panel control's elem_id is "base_*" / "hires_*" ("auto" is "auto_timing")
        prefix, _, field = name.partition("_")
        if field not in panel:
            return name
        return f"{'hires' if prefix == 'hr' else prefix}_{'auto_timing' if field == 'auto' else field}"

    assert [control.elem_id for control in controls] == [script.elem_id(elem_id(name)) for name in options.SCRIPT_ARG_NAMES]

    # process gives each panel's _build_pass_config the args of that panel's names
    calls = []
    build = runtime._build_pass_config
    monkeypatch.setattr(runtime, "_build_pass_config", lambda *panel_args: calls.append(panel_args) or build(*panel_args))
    args = _args(module, _zh)
    named = dict(zip(options.SCRIPT_ARG_NAMES, args, strict=True))
    _run(module, args)
    for prefix, shift, call in zip(("base", "hr"), (3.0, 5.0), calls, strict=True):
        values = [named[f"{prefix}_{name}"] for name in panel]
        assert call == (*values[:6], None, *values[6:], shift)

    with pytest.raises(TypeError, match=f"takes {len(args)} script arguments, got {len(args) - 1}"):
        module.Script().process(SimpleNamespace(), *args[:-1])
    with pytest.raises(TypeError, match=f"takes {len(args)} script arguments, got {len(args) + 1}"):
        module.Script().process(SimpleNamespace(), *args, "not a script arg")


def test_each_panels_sliders_start_on_its_default_templates_window(module, monkeypatch):
    # Unticking auto timing keeps the sliders: they must hold the window of the
    # template the dropdown shows, in the Hires. fix panel as in the base one
    _language(module, monkeypatch, "zh")
    script, controls, _ = _build_ui(module)
    named = dict(zip(options.SCRIPT_ARG_NAMES, controls, strict=True))
    for prefix in ("base", "hr"):
        stage = templates._template_stage_from_name(named[f"{prefix}_template"].value)
        window = tuple(named[f"{prefix}_{field}"].value for field in ("start", "end", "peak"))
        assert window == options.STAGE_DEFAULT_TIMINGS[stage], prefix


def test_the_infotext_records_stable_values_whatever_the_ui_language(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    zh, _ = _run(module, _args(module, _zh))
    _language(module, monkeypatch, "en")
    en, _ = _run(module, _args(module, _en))

    assert zh == en
    key = options._panel_key
    assert (zh[key("base", "scope")], zh[key("hires", "scope")], zh[options.INFOTEXT_HIRES_DISABLE]) == ("targets", "all", "selected")
    assert (zh[key("base", "curve")], zh[key("hires", "curve")]) == ("triangle", "back")
    assert (zh[key("base", "weight")], zh[key("base", "strength")], zh[key("base", "per-LoRA templates")]) == ("0.8", 1.7, "mylora=" + options.PRESET_COMPOSITION_ALLOW)


def test_the_pasted_panel_runs_the_same_schedule_per_lora_templates_included(module, monkeypatch):
    _language(module, monkeypatch, "en")
    params, state = _run(module, _args(module, _en))
    script, _, _ = _build_ui(module)
    pasted, pasted_state = _run(module, _paste(module, script, params))

    assert pasted == params
    assert _schedule(module, pasted_state) == _schedule(module, state)

    # mylora runs on its own template's (composition) window, not the panel's
    mylora = pasted_state.base.config_for_file("mylora.safetensors", {"mylora"})
    assert mylora.template_name == options.PRESET_COMPOSITION_ALLOW
    assert (mylora.start, mylora.end, mylora.peak) == templates._auto_stage_values(options.STAGE_COMPOSITION, 3.0)[:3]
    assert (pasted_state.base.start, pasted_state.base.end) == (0.3, 0.7)
    assert pasted_state.hr_disable_tags == ["<lora:a:0.5>", "<lora:b:blocks=0-9=0.8,10-27=1>"]


@pytest.mark.parametrize("run_language, api_language", [("zh", "en"), ("en", "zh"), ("en", "en")])
@pytest.mark.parametrize("hires", ["independent", "shared"])
def test_the_api_replays_the_infotext_into_the_same_schedule(module, monkeypatch, run_language, api_language, hires):
    # The API sets the script args from the recorded values as they are (the
    # scope, curve and disable mode codes), without on_infotext_pasted; its
    # pasted_value reads a recorded "False" as False
    labels = {"zh": _zh, "en": _en}[run_language]
    _language(module, monkeypatch, run_language)
    args = _args(module, labels)
    if hires == "shared":
        args[12], args[24] = False, labels(options.DISABLE_ALL)
    params, state = _run(module, args)
    _language(module, monkeypatch, api_language)
    script, controls, _ = _build_ui(module)
    replayed, replayed_state = _run(module, harness.api_replay(script, controls, params))

    assert replayed == params
    assert _schedule(module, replayed_state) == _schedule(module, state)
    assert (replayed_state.base.target_mode, replayed_state.base.curve) == (options.MODE_TARGETS, options.CURVE_TRIANGLE)
    # Recorded "False" checkboxes: base manual timing; a shared Hires panel
    assert params[options._panel_key("base", "auto timing")] is False and replayed_state.base.auto_timing is False
    assert (replayed_state.base.start, replayed_state.base.end) == (0.3, 0.7)
    if hires == "shared":
        assert params[options.INFOTEXT_HIRES_INDEPENDENT] is False
        assert replayed_state.hires.template_name == replayed_state.base.template_name and replayed_state.hires.target_mode == options.MODE_TARGETS
    assert replayed_state.hr_disable_mode == (options.DISABLE_SELECTED if hires == "independent" else options.DISABLE_ALL)


def test_the_run_takes_each_choice_as_its_code_its_zh_label_or_its_en_label(module):
    for codes, labels, normalize in (
        (options.TARGET_MODE_CODES, options.TARGET_MODE_EN, options._normalize_target_mode),
        (options.DISABLE_CODES, options.DISABLE_EN, options._normalize_disable_mode),
        (options.CURVE_CODES, {}, options._normalize_curve),
    ):
        for choice, code in codes.items():
            assert normalize(code) == normalize(choice) == normalize(labels.get(choice, choice)) == choice
    assert (options._normalize_target_mode("?"), options._normalize_disable_mode(None), options._normalize_curve("")) == (options.MODE_AUTO_ANIMA, options.DISABLE_NONE, options.CURVE_SMOOTH)


def test_text_a_run_ignored_pastes_empty_and_no_hires_leaves_the_hires_section_alone(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    args = _args(module, _zh)
    args[1] = options.MODE_AUTO_ANIMA  # the target list is ignored
    args[5] = ""
    params, _ = _run(module, args, enable_hr=False)
    script, controls, _ = _build_ui(module)
    pasted = _paste(module, script, params)

    key = options._panel_key
    assert key("base", "targets") not in params and key("base", "per-LoRA templates") not in params
    assert not any(" hires " in name for name in params)
    assert pasted[:12] == [*args[:2], "", *args[3:12]]  # the ignored target list is emptied
    assert pasted[12:] == [None] * 14  # no Hires. fix pass: the hires controls stay


def test_a_shared_hires_panel_records_only_its_switch_and_disable_mode(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    args = _args(module, _zh)
    args[12], args[24], args[25] = False, options.DISABLE_ALL, "<lora:a:0.5>"
    params, _ = _run(module, args)
    script, _, _ = _build_ui(module)
    pasted = _paste(module, script, params)

    assert params[options.INFOTEXT_HIRES_INDEPENDENT] is False and params[options.INFOTEXT_HIRES_DISABLE] == "all"
    assert options.INFOTEXT_HIRES_DISABLE_TAGS not in params  # ignored when disabling every LoRA
    assert pasted[12] is False and pasted[13:24] == [None] * 11
    assert pasted[24:] == [options.DISABLE_ALL, ""]

    args[24] = options.DISABLE_NONE
    params, _ = _run(module, args)
    assert options.INFOTEXT_HIRES_DISABLE not in params
    assert _paste(module, script, params)[24:] == [options.DISABLE_NONE, ""]


def test_an_old_summary_only_infotext_leaves_the_panel_alone(module, monkeypatch):
    _language(module, monkeypatch, "en")
    script, _, _ = _build_ui(module)
    old = {
        "Anima LoRA Stage Scheduler": f"base={options.PRESET_STYLE_ALLOW}:0.62-1.00, hires={options.PRESET_STYLE_ALLOW}:0.70-1.00",
        "Anima Hires LoRA Disable": options.DISABLE_ALL,
    }

    assert _paste(module, script, old) == [None] * len(options.INFOTEXT_KEYS)


def test_a_template_missing_here_leaves_its_dropdown_alone(module, monkeypatch, caplog):
    _language(module, monkeypatch, "zh")
    templates._TEMPLATES.edit(lambda data: (data.update(mine={"auto": True, "stage": options.STAGE_STYLE}), ("mine", "")))
    args = _args(module, _zh)
    args[4] = "mine"
    params, _ = _run(module, args)
    templates._TEMPLATES.edit(lambda data: (data.pop("mine"), (None, "")))
    script, _, _ = _build_ui(module)

    pasted = _paste(module, script, params)
    assert pasted[4] is None and "mine" in caplog.text
    assert pasted[:4] == args[:4] and pasted[5:] == args[5:]


def test_template_dropdowns_reapply_on_user_selection_only(module, monkeypatch):
    _language(module, monkeypatch, "zh")
    script, controls, blocks = _build_ui(module)
    template_ids = {controls[4]._id, controls[16]._id}
    triggers = {(target, event) for dependency in blocks.config["dependencies"] for target, event in dependency["targets"]}

    assert {(template, "input") for template in template_ids} <= triggers
    assert not {(template, "change") for template in template_ids} & triggers


def test_template_management_reapplies_the_templates_it_changes(module):
    templates._TEMPLATES.edit(lambda data: (data.update(mine={"auto": False, "stage": options.STAGE_STYLE, "start": 0.1, "end": 0.3, "peak": 0.2, "strength": 1.5, "curve": options.CURVE_HOLD}), ("mine", "")))
    builtin = next(iter(options.BUILTIN_TEMPLATES))
    rename = templates._reapplying_changed_templates(templates._rename_template_ui)
    delete = templates._reapplying_changed_templates(templates._delete_template_ui)

    base, hires, _, _, *timings = rename("mine", "renamed", "mine", builtin, 3.0, 5.0)
    assert (base["value"], "value" in hires) == ("renamed", False)
    assert [update["value"] for update in timings[:6]] == [False, 0.1, 0.3, 0.2, 1.5, options.CURVE_HOLD]
    assert all("value" not in update for update in timings[6:])  # the hires panel's template stayed

    base, hires, _, _, *timings = delete("renamed", builtin, "renamed", 3.0, 5.0)
    assert hires["value"] == builtin  # fell back
    assert all("value" not in update for update in timings[:6])
    assert [update["value"] for update in timings[6:]] == [value["value"] for value in templates._apply_template_ui(builtin, 5.0)]
