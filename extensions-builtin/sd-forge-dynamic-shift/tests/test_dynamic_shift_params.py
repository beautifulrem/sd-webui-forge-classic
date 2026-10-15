"""Dynamic Shift's ``params()``: the Gradio panel's contract, conflict locks, infotext and processing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "dynamic_shift.py"


@pytest.fixture(scope="module")
def env(ext_kit):
    from modules import scripts
    from remi.testing import webui_state

    with webui_state():
        module = ext_kit.load_script(SCRIPT, "dynamic_shift_under_test")
    data = scripts.ScriptClassData(module.DynamicShiftScript, str(SCRIPT), str(EXTENSION), module)
    with webui_state(script_classes=[data]):
        yield SimpleNamespace(kit=ext_kit, module=module)


def _curve_change(module):
    # Intended: the curve choice is the `shift-curve` widget (choice + live schedule plot)
    return {"kind": "custom", "choices": None, "widget": "shift-curve", "props": {"choices": module.CURVES, "start": "shift_start", "end": "shift_end", "steps": "sampler.steps"}}


def test_matches_the_legacy_panel_txt2img(env):
    _script, layout = env.kit.layout_of(env.module.DynamicShiftScript, "txt2img")
    env.kit.assert_matches_snapshot(layout, env.kit.load_snapshot(SCRIPT, "txt2img"), {3: _curve_change(env.module)})


def test_matches_the_legacy_panel_img2img(env):
    _script, layout = env.kit.layout_of(env.module.DynamicShiftScript, "img2img")
    # Intended: img2img's unrendered Hires. fix params get txt2img's labels and slider bounds
    # (Gradio's invisible sliders had the default 0-100 / 1)
    shift = {"minimum": 0.1, "maximum": 12.0, "step": 0.05}
    changes = {
        3: _curve_change(env.module),
        4: {"label": "Hires fix"},
        5: {"label": "Hires shift @ first step", **shift},
        6: {"label": "Hires shift @ last step", **shift},
    }
    env.kit.assert_matches_snapshot(layout, env.kit.load_snapshot(SCRIPT, "img2img"), changes)
    assert all(layout.by_name[name].hidden for name in ("hr_mode", "hr_start", "hr_end"))


def test_hires_shifts_show_for_custom_mode_only(env):
    from remi.ui import evaluate

    _script, layout = env.kit.layout_of(env.module.DynamicShiftScript, "txt2img")
    row = next(node for node in layout.root.walk() if getattr(node, "visible_if", None) and "hr_start" in [getattr(child, "name", None) for child in node.children])
    assert evaluate(row.visible_if, lambda ref: {"hr_mode": "Custom"}.get(ref)) is True
    assert evaluate(row.visible_if, lambda ref: {"hr_mode": "Same as first pass"}.get(ref)) is False


def test_compatibility_notes_follow_the_sampler(env):
    from remi.ui import evaluate

    module = env.module
    _script, layout = env.kit.layout_of(module.DynamicShiftScript, "txt2img")
    notes = {node.text: node.visible_if for node in layout.root.walk() if node.type == "markdown"}

    def shown(text, sampler, enabled):
        values = {"sampler.sampler_name": sampler, "enabled": enabled}
        return evaluate(notes[text], values.get)

    flow = sorted(module.FLOW_SAMPLERS)[0]
    assert shown(module.NOTE_FLOW_SAMPLER, flow, False) and not shown(module.NOTE_FLOW_SAMPLER, "Euler", True)
    assert shown(module.NOTE_SCHEDULE, "Euler", True) and not shown(module.NOTE_SCHEDULE, "Euler", False)
    assert not shown(module.NOTE_SCHEDULE, flow, True)


def test_conflict_locks_through_the_core_action(env):
    from remi.core.conflicts import conflict_patch
    from remi.core.tabs import tab_schema

    assert "dynamic_shift.enabled" in tab_schema("txt2img")["conflict_params"]
    flow = sorted(env.module.FLOW_SAMPLERS)[0]
    patch = conflict_patch("txt2img", {"sampler": {"sampler_name": flow}, "dynamic_shift": {"enabled": True}}, "sampler.sampler_name")
    assert patch["dynamic_shift.enabled"]["value"] is False and patch["dynamic_shift.enabled"]["enabled"] is False
    patch = conflict_patch("txt2img", {"sampler": {"sampler_name": "Euler", "scheduler": "Karras"}, "dynamic_shift": {"enabled": True}}, "dynamic_shift.enabled")
    assert patch["sampler.scheduler"]["value"] == "Automatic" and patch["sampler.scheduler"]["enabled"] is False


def test_infotext_paste(env):
    _script, layout = env.kit.layout_of(env.module.DynamicShiftScript, "txt2img")
    values, problems = layout.paste({"DynShift enabled": "True", "DynShift start": "4.5", "DynShift curve": "Linear", "DynShift hires mode": "Custom", "DynShift hires start": "2"})
    assert not problems
    assert values == {"enabled": True, "shift_start": 4.5, "curve": "Linear", "hr_mode": "Custom", "hr_start": 2.0}
    # As in Gradio: an infotext without the key switches the panel off
    assert layout.paste({})[0] == {"enabled": False}
    # An unknown curve runs as "Linear", as process() always treated it (Gradio
    # passed any dropdown value through); the form shows the curve that runs
    values, problems = layout.paste({"DynShift curve": "Wobbly"})
    assert not problems and values["curve"] == "Linear"


def test_unknown_curve_falls_back_to_linear(env):
    _script, layout = env.kit.layout_of(env.module.DynamicShiftScript, "txt2img")
    assert layout.values_to_args({"curve": "Sigmoid"})[3] == "Linear"
    assert layout.values_to_args({"curve": None})[3] == "Linear"
    assert layout.values_to_args({"curve": "Exponential"})[3] == "Exponential"


def test_process_installs_the_schedule(env):
    script, layout = env.kit.layout_of(env.module.DynamicShiftScript, "txt2img")
    args = layout.values_to_args({"enabled": True, "shift_start": 4.0, "shift_end": 1.0, "curve": "Linear"})
    p = SimpleNamespace(
        sd_model=SimpleNamespace(use_shift=True),
        sampler_name="Euler",
        scheduler="Karras",
        sampler_noise_scheduler_override=None,
        extra_generation_params={},
        enable_hr=False,
    )
    script.process(p, *args)
    assert p.scheduler == "Automatic"
    assert p.extra_generation_params["DynShift replaced scheduler"] == "Karras"
    sigmas = p.sampler_noise_scheduler_override(10)
    assert len(sigmas) == 11 and float(sigmas[-1]) == 0.0
    # What it wrote pastes back to the same values
    values, problems = layout.paste({key: str(value) for key, value in p.extra_generation_params.items()})
    assert not problems and layout.values_to_args(values) == args


def test_schema_export(env):
    from remi.core.tabs import tab_schema
    from remi.ui.schema import dumps

    for tab in ("txt2img", "img2img"):
        schema = json.loads(dumps(tab_schema(tab)))
        entry = next(script for script in schema["scripts"] if script["id"] == "dynamic_shift")
        assert entry["section"] == "sampler" and entry["args"] == ["enabled", "shift_start", "shift_end", "curve", "hr_mode", "hr_start", "hr_end"]
        assert entry["layout"]["enable"]["preset"] == "dynamic_shift.enabled"


def test_localized(env):
    strings = set()
    for tab in ("txt2img", "img2img"):
        script, layout = env.kit.layout_of(env.module.DynamicShiftScript, tab)
        strings |= env.kit.schema_strings(layout.schema())
    env.kit.assert_localized(EXTENSION, strings, extra={script.title(), "[DynShift] Start", "[DynShift] End", "[DynShift] Curve"})


def test_standalone(ext_kit):
    ext_kit.assert_standalone(EXTENSION, [SCRIPT])
