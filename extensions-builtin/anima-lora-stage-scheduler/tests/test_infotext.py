"""A scheduled run's infotext pastes back into the panel it was run from:
the SPA paste, ``/infotext/parse`` and the API's infotext replay all read it
through the panel's ``remi.ui`` layout."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from lib_stage_scheduler import options, runtime, templates

SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_stage_scheduler.py"
# Shared with the other Anima LoRA panels' suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("remi_anima_lora_harness", Path(__file__).parents[3] / "tests" / "remi_anima_lora_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)
_MODULE = []


@pytest.fixture
def module():
    if not _MODULE:
        _MODULE.append(harness.load_script(SCRIPT, "anima_lora_stage_scheduler_infotext"))
    return _MODULE[0]


def _layout(module):
    return harness.script_layout(module.Script(), "txt2img")


def _run(module, args, enable_hr=True):
    """The infotext parameters ``process`` records for the script args ``args``."""

    p = SimpleNamespace(extra_generation_params={}, enable_hr=enable_hr, distilled_cfg_scale=3.0, hr_distilled_cfg=5.0)
    module.Script().process(p, *args)
    state = p._anima_lora_stage_scheduler_state
    runtime._cleanup_runtime(p)
    return p.extra_generation_params, state


def _schedule(state):
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
            None if pass_config.parts else pass_config.curve,  # the parts' own curves run instead
        )

    files = (("mylora.safetensors", {"mylora"}), ("other.safetensors", {"other"}))
    passes = [[config(pass_config.config_for_file(filename, names)) for filename, names in files] for pass_config in (state.base, state.hires)]
    return passes, state.hr_disable_mode, state.hr_disable_tags if state.hr_disable_mode == options.DISABLE_SELECTED else None


def _timing(auto, start, end, peak, strength, curve):
    return {"auto": auto, "start": start, "end": end, "peak": peak, "strength": strength, "curve": curve}


def _values():
    """Named params of a run with every panel control away from its default."""

    style_start, style_end, style_peak, _ = templates._auto_stage_values(options.STAGE_STYLE, 5.0)
    return {
        "enabled": True,
        # Base: target list, manual timing, a per-LoRA template
        "base_target_mode": options.MODE_TARGETS,
        "base_target_loras": "mylora, other",
        "base_panel_weight": "0.8",
        "base_template": options.PRESET_CHARACTER_ALLOW,
        "base_lora_templates": "mylora=" + options.PRESET_COMPOSITION_ALLOW,
        "base_timing": _timing(False, 0.3, 0.7, 0.4, 1.7, options.CURVE_TRIANGLE),
        "hr_independent": True,
        # Hires: independent, block weights, auto timing at the Hires Shift
        "hr_target_mode": options.MODE_PROMPT_ALL,
        "hr_target_loras": "",
        "hr_panel_weight": "blocks=0-18=1,19-27=0.2",
        "hr_template": options.PRESET_STYLE_ALLOW,
        "hr_lora_templates": "",
        "hr_timing": _timing(True, round(style_start, 4), round(style_end, 4), round(style_peak, 4), 1.0, options.CURVE_BACK),
        "hr_disable_mode": options.DISABLE_SELECTED,
        "hr_disable_tags": "<lora:a:0.5>\n<lora:b:blocks=0-9=0.8,10-27=1>",
    }


def _args(module, **changes):
    return _layout(module).values_to_args({**_values(), **changes})


def _legacy_args(values):
    """The same panel as the 26 positional arguments of the Gradio panel."""

    args = []
    for name in options.SCRIPT_ARG_NAMES:
        prefix, _, arg = name.partition("_")
        if arg in options.TIMING_ARGS and prefix in ("base", "hr"):
            args.append(values[f"{prefix}_timing"][options.TIMING_KEYS[options.TIMING_ARGS.index(arg)]])
        else:
            args.append(values[name])
    return args


def test_a_runs_infotext_restores_every_control(module):
    params, _ = _run(module, _args(module))
    layout = _layout(module)
    assert set(params) <= set(options.INFOTEXT_KEYS)
    assert set(layout.infotext.values()) == set(options.PARAM_NAMES)

    # Every control is back where the run had it; the Hires curve the run did
    # not use (its template's parts bring their own) reads as smooth
    expected = _values()
    expected["hr_timing"] = {**expected["hr_timing"], "curve": options.CURVE_SMOOTH}
    assert harness.pasted(layout, params) == expected


def test_the_curve_is_recorded_when_the_run_uses_it(module):
    # The panel's curve shapes the run only where no template parts run and
    # the window applies the LoRA (a ban window is all or nothing): exactly
    # there does another curve change the strength, and the curve recorded
    templates._TEMPLATES.edit(lambda data: data.update(partless={"auto": True, "stage": options.STAGE_STYLE}, curved={"auto": True, "stage": options.STAGE_STYLE, "curve": options.CURVE_FRONT}, banning={"auto": False, "parts": [options.PRESET_STYLE_BAN]}))
    key = options._panel_key

    def run(template, auto, per_lora="", curve=options.CURVE_HOLD):
        args = _args(module, base_template=template, base_lora_templates=per_lora, base_timing=_timing(auto, 0.3, 0.7, 0.4, 1.0, curve))
        params, state = _run(module, args)
        config = state.base.config_for_file("mylora.safetensors", {"mylora"})
        factors = [runtime._stage_curve_factor(config, at / 20) for at in range(21)]
        return key("base", "curve") in params, factors

    cases = [
        ("partless", True, ""),
        (options.PRESET_STYLE_ALLOW, False, ""),
        (options.PRESET_STYLE_ALLOW, True, ""),
        (options.PRESET_STYLE_BAN, False, ""),
        (options.PRESET_STYLE_ALLOW, True, "mylora=partless"),  # a per-LoRA template on the panel's curve
        (options.PRESET_STYLE_ALLOW, True, "mylora=curved"),  # ... with a curve of its own
        (options.PRESET_STYLE_ALLOW, True, "mylora=banning"),  # ... banning in a manual window
    ]
    for template, auto, per_lora in cases:
        recorded, hold = run(template, auto, per_lora)
        used = hold != run(template, auto, per_lora, options.CURVE_FRONT)[1]
        assert recorded is used, (template, auto, per_lora)
    assert [run(*case)[0] for case in cases] == [True, True, False, False, True, False, False]


def test_params_and_process_name_the_script_args_in_table_order(module, monkeypatch):
    layout = _layout(module)
    assert layout.names == list(options.PARAM_NAMES)

    # process gives each panel's _build_pass_config the args of that panel's names
    calls = []
    build = runtime._build_pass_config
    monkeypatch.setattr(runtime, "_build_pass_config", lambda *panel_args: calls.append(panel_args) or build(*panel_args))
    values = _values()
    args = layout.values_to_args(values)
    _run(module, args)
    for prefix, shift, call in zip(("base", "hr"), (3.0, 5.0), calls, strict=True):
        timing = values[f"{prefix}_timing"]
        panel = [values[f"{prefix}_{name}"] for name in ("target_mode", "target_loras", "panel_weight", "template", "lora_templates")]
        assert call == (*panel, timing["auto"], None, timing["start"], timing["end"], timing["peak"], timing["strength"], timing["curve"], shift)

    with pytest.raises(TypeError, match=f"takes {len(args)} script arguments, got {len(args) - 1}"):
        module.Script().process(SimpleNamespace(), *args[:-1])
    with pytest.raises(TypeError, match=f"takes {len(args)} script arguments, got {len(args) + 1}"):
        module.Script().process(SimpleNamespace(), *args, "not a script arg")


def test_the_gradio_panels_26_positional_args_run_the_same_schedule(module):
    """The Gradio panel passed each timing control on its own: the folded
    timing value runs exactly the same schedule."""

    values = _values()
    params, state = _run(module, _layout(module).values_to_args(values))
    legacy_params, legacy_state = _run(module, _legacy_args(values))
    assert legacy_params == params and _schedule(legacy_state) == _schedule(state)


def test_each_panel_starts_on_its_default_templates_window(module):
    # Unticking auto timing keeps the window: it must be the window of the
    # template the dropdown shows, in the Hires. fix panel as in the base one
    defaults = _layout(module).defaults()
    for prefix in ("base", "hr"):
        stage = templates._template_stage_from_name(defaults[f"{prefix}_template"])
        timing = defaults[f"{prefix}_timing"]
        assert (timing["start"], timing["end"], timing["peak"]) == options.STAGE_DEFAULT_TIMINGS[stage], prefix
        assert timing["auto"] is True and timing["strength"] == 1.0 and timing["curve"] == options.CURVE_SMOOTH


def test_the_infotext_records_stable_values(module):
    params, _ = _run(module, _args(module))
    key = options._panel_key
    assert (params[key("base", "scope")], params[key("hires", "scope")], params[options.INFOTEXT_HIRES_DISABLE]) == ("targets", "all", "selected")
    assert params[key("base", "curve")] == "triangle" and key("hires", "curve") not in params
    assert (params[key("base", "weight")], params[key("base", "strength")], params[key("base", "per-LoRA templates")]) == ("0.8", 1.7, "mylora=" + options.PRESET_COMPOSITION_ALLOW)


@pytest.mark.parametrize("hires", ["independent", "shared"])
@pytest.mark.parametrize("start", ["defaults", "edited"])
def test_the_pasted_panel_runs_the_same_schedule_per_lora_templates_included(module, hires, start):
    # Into a fresh panel (the API's infotext replay) or one with other values (the SPA paste)
    changes = {} if hires == "independent" else {"hr_independent": False, "hr_disable_mode": options.DISABLE_ALL}
    params, state = _run(module, _args(module, **changes))
    begin = None if start == "defaults" else {"base_panel_weight": "0.3", "hr_template": options.PRESET_CHARACTER_BAN}
    pasted, pasted_state = _run(module, harness.replayed_args(_layout(module), params, begin))

    assert pasted == params
    assert _schedule(pasted_state) == _schedule(state)

    # mylora runs on its own template's (composition) window, not the panel's
    mylora = pasted_state.base.config_for_file("mylora.safetensors", {"mylora"})
    assert mylora.template_name == options.PRESET_COMPOSITION_ALLOW
    assert (mylora.start, mylora.end, mylora.peak) == templates._auto_stage_values(options.STAGE_COMPOSITION, 3.0)[:3]
    assert (pasted_state.base.start, pasted_state.base.end) == (0.3, 0.7)
    assert (pasted_state.base.target_mode, pasted_state.base.curve) == (options.MODE_TARGETS, options.CURVE_TRIANGLE)
    # Recorded "False" switches: base manual timing; a shared Hires panel
    assert params[options._panel_key("base", "auto timing")] is False and pasted_state.base.auto_timing is False
    if hires == "shared":
        assert params[options.INFOTEXT_HIRES_INDEPENDENT] is False
        assert pasted_state.hires.template_name == pasted_state.base.template_name and pasted_state.hires.target_mode == options.MODE_TARGETS
        assert pasted_state.hr_disable_mode == options.DISABLE_ALL
    else:
        assert pasted_state.hr_disable_tags == ["<lora:a:0.5>", "<lora:b:blocks=0-9=0.8,10-27=1>"]


def test_the_spa_paste_keeps_the_recorded_timing(module):
    """The SPA paste runs the pasted params' on_change actions: none of them
    may put the pasted timing back on its template's defaults."""

    params, _ = _run(module, _args(module))
    begin = {"base_template": options.PRESET_STYLE_BAN, "hr_template": options.PRESET_CHARACTER_BAN}
    values = harness.spa_paste(module.Script(), _layout(module), params, begin, core={"distilled_cfg_scale": 3.0, "hr_distilled_cfg": 5.0})
    assert values["base_template"] == options.PRESET_CHARACTER_ALLOW
    assert values["base_timing"] == _values()["base_timing"]
    assert values["hr_timing"] == {**_values()["hr_timing"], "curve": options.CURVE_SMOOTH}


def test_the_run_takes_each_choice_as_its_code_its_value_or_its_english_label():
    for codes, labels, normalize in (
        (options.TARGET_MODE_CODES, options.TARGET_MODE_EN, options._normalize_target_mode),
        (options.DISABLE_CODES, options.DISABLE_EN, options._normalize_disable_mode),
        (options.CURVE_CODES, {}, options._normalize_curve),
    ):
        for choice, code in codes.items():
            assert normalize(code) == normalize(choice) == normalize(labels.get(choice, choice)) == choice
    assert (options._normalize_target_mode("?"), options._normalize_disable_mode(None), options._normalize_curve("")) == (options.MODE_AUTO_ANIMA, options.DISABLE_NONE, options.CURVE_SMOOTH)


def test_text_a_run_ignored_pastes_empty_and_no_hires_leaves_the_hires_section_alone(module):
    # The target list is ignored outside the target-list scope
    params, _ = _run(module, _args(module, base_target_mode=options.MODE_AUTO_ANIMA, base_lora_templates=""), enable_hr=False)
    pasted = harness.pasted(_layout(module), params)

    key = options._panel_key
    assert key("base", "targets") not in params and key("base", "per-LoRA templates") not in params
    assert not any(" hires " in name for name in params)
    assert pasted["base_target_loras"] == "" and pasted["base_lora_templates"] == ""  # emptied
    assert pasted["base_target_mode"] == options.MODE_AUTO_ANIMA
    assert not any(name.startswith("hr_") for name in pasted)  # no Hires. fix pass: the hires controls stay


def test_a_shared_hires_panel_records_only_its_switch_and_disable_mode(module):
    params, _ = _run(module, _args(module, hr_independent=False, hr_disable_mode=options.DISABLE_ALL, hr_disable_tags="<lora:a:0.5>"))
    pasted = harness.pasted(_layout(module), params)

    assert params[options.INFOTEXT_HIRES_INDEPENDENT] is False and params[options.INFOTEXT_HIRES_DISABLE] == "all"
    assert options.INFOTEXT_HIRES_DISABLE_TAGS not in params  # ignored when disabling every LoRA
    hires = {name: value for name, value in pasted.items() if name.startswith("hr_")}
    assert hires == {"hr_independent": False, "hr_disable_mode": options.DISABLE_ALL, "hr_disable_tags": ""}

    params, _ = _run(module, _args(module, hr_independent=False, hr_disable_mode=options.DISABLE_NONE))
    assert options.INFOTEXT_HIRES_DISABLE not in params
    pasted = harness.pasted(_layout(module), params)
    assert (pasted["hr_disable_mode"], pasted["hr_disable_tags"]) == (options.DISABLE_NONE, "")


def test_an_old_summary_only_infotext_leaves_the_panel_alone(module):
    old = {
        "Anima LoRA Stage Scheduler": f"base={options.PRESET_STYLE_ALLOW}:0.62-1.00, hires={options.PRESET_STYLE_ALLOW}:0.70-1.00",
        "Anima Hires LoRA Disable": options.DISABLE_ALL,
    }
    assert harness.pasted(_layout(module), old) == {}


def test_a_template_missing_here_leaves_its_dropdown_alone(module, caplog):
    templates._TEMPLATES.edit(lambda data: (data.update(mine={"auto": True, "stage": options.STAGE_STYLE}), ("mine", "")))
    params, _ = _run(module, _args(module, base_template="mine"))
    templates._TEMPLATES.edit(lambda data: (data.pop("mine"), (None, "")))

    pasted = harness.pasted(_layout(module), params)
    assert "base_template" not in pasted and "mine" in caplog.text
    assert pasted["base_target_loras"] == "mylora, other" and pasted["hr_template"] == options.PRESET_STYLE_ALLOW


def test_a_template_deleted_from_the_other_tab_runs_as_the_default(module, caplog):
    """Templates are shared by both tabs; a form still showing a deleted
    one submits it: the job runs the default template, with a warning."""
    layout = _layout(module)
    coerced = layout.coerce({**layout.defaults(), "base_template": "deleted elsewhere"})
    assert coerced["base_template"] == layout.defaults()["base_template"] and "deleted elsewhere" in caplog.text


def test_selecting_a_template_applies_its_timing(module):
    from lib_stage_scheduler import ui
    from remi import ui as remi_ui

    ctx = remi_ui.ActionContext(tab="txt2img", core={"distilled_cfg_scale": 3.0, "hr_distilled_cfg": 6.0})
    values = {"base_template": options.PRESET_COMPOSITION_ALLOW, "hr_template": options.PRESET_STYLE_ALLOW}
    script = module.Script()
    patch = remi_ui.call_action(script, "base_template_changed", ctx, values)
    assert patch == {"base_timing": {"value": ui.apply_template(options.PRESET_COMPOSITION_ALLOW, 3.0)}}
    patch = remi_ui.call_action(script, "hr_template_changed", ctx, values)
    assert patch["hr_timing"]["value"] == ui.apply_template(options.PRESET_STYLE_ALLOW, 6.0)
    # img2img has no Hires Shift: the base one
    patch = remi_ui.call_action(script, "hr_template_changed", remi_ui.ActionContext(tab="img2img", core={"distilled_cfg_scale": 4.0}), values)
    assert patch["hr_timing"]["value"] == ui.apply_template(options.PRESET_STYLE_ALLOW, 4.0)


def test_template_actions_report_and_refresh_both_panels(module):
    from remi import ui as remi_ui

    script = module.Script()
    values = _layout(module).defaults()

    def call(action, **extra):
        return remi_ui.call_action(script, action, remi_ui.ActionContext(tab="txt2img", core={"distilled_cfg_scale": 3.0}, extra=extra), values)

    saved = call("template_save", name="mine", auto=False, panel="base")
    # A message the SPA translates, then fills in
    assert saved["__toast__"] == {"level": "success", "message": {"text": "Saved template: {{name}}", "params": {"name": "mine"}}}
    assert "mine" in [choice["value"] for choice in saved["base_template"]["choices"]] and "mine" in saved["__outputs__"]["templates"]
    assert templates._template_config("mine")["auto"] is False

    combo = call("template_combo", name="combo", presets=[options.PRESET_STYLE_BAN, options.PRESET_CHARACTER_ALLOW])
    assert combo["__toast__"]["level"] == "success"
    assert [part["mode"] for part in templates._template_parts(templates._template_config("combo"))] == [options.PART_BAN, options.PART_ALLOW]

    refused = call("template_rename", old="mine", new=options.PRESET_STYLE_ALLOW)
    assert refused == {"__toast__": {"level": "warning", "message": options._t("builtin_name_taken")}}

    values["base_template"] = "mine"
    renamed = call("template_rename", old="mine", new="ours")
    assert renamed["base_template"]["value"] == "ours" and renamed["base_timing"]["value"]["auto"] is False
    deleted = call("template_delete", name="combo")
    assert "combo" not in deleted["__outputs__"]["templates"]
