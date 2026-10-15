"""The X/Y/Z plot on ``params()``: the legacy ``ui()`` contract, the run()
arguments, its actions (axis switch, text mode, fill, swap, values list),
infotext paste and the ``axis_options`` API other extensions extend."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from remi import ui
from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "legacy_ui"

_spec = importlib.util.spec_from_file_location("legacy_ui_contract", ROOT / "tests" / "legacy_ui_contract.py")
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)

DROPDOWNS = (2, 5, 8)
"""The values dropdowns are the ``xyz-values`` widget (choices follow the axis)."""


def load_xyz():
    """A fresh module (its own ``axis_options``), registered as Forge does."""

    path = ROOT / "scripts" / "xyz_grid.py"
    spec = importlib.util.spec_from_file_location(f"xyz_grid_under_test_{len(sys.modules)}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def xyz(webui):
    return load_xyz()


def make(module, tab="txt2img"):
    script = module.Script()
    script.is_img2img = tab == "img2img"
    script.layout = ui.ScriptLayout(script.params(script.is_img2img), owner="x_y_z_plot", order=script.param_order)
    return script


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_keep_the_legacy_contract(xyz, tab):
    script = make(xyz, tab)
    snapshot = contract.load_snapshot(FIXTURES / f"xyz_grid_{tab}.json")
    expected = {position: {"kind": "custom", "default": [], "infotext": None, "paste": True} for position in DROPDOWNS}
    assert contract.differences(script.layout, snapshot, expected=expected) == []
    assert [node.name for node in script.layout.inputs][:3] == ["x_type", "x_values", "x_values_dropdown"]


def test_named_values_become_the_legacy_run_arguments(xyz):
    script = make(xyz)
    labels = [x.label for x in script.current_axis_options]
    args = script.layout.values_to_args({"x_type": "Steps", "x_values": "10-30 (+10)", "y_type": "Sampler", "y_values_dropdown": ["Euler", "Euler a"], "csv_mode": False})
    assert args == [labels.index("Steps"), "10-30 (+10)", [], labels.index("Sampler"), "", ["Euler", "Euler a"], 0, "", [], True, False, False, False, False, False, False, 0, 0, False]
    # The defaults are the old UI's: X = Seed, Y and Z = Nothing
    defaults = script.layout.values_to_args({})
    assert defaults[0] == labels.index("Seed") and defaults[3] == defaults[6] == 0
    with pytest.raises(ui.ValidationError):
        script.layout.values_to_args({"x_type": "Not an axis"})
    with pytest.raises(ui.ValidationError):
        script.layout.values_to_args({"x_values_dropdown": "Euler"})  # a list, not text


def test_axes_added_later_are_selectable(xyz):
    """Extensions (ADetailer, Detail Daemon, Dynamic Shift) extend ``axis_options``."""

    script = make(xyz)
    xyz.axis_options.append(xyz.AxisOption("[Test] Strength", float, xyz.apply_field("test_strength")))
    xyz.axis_options.append(xyz.AxisOptionImg2Img("[Test] img2img only", float, xyz.apply_field("x")))
    labels = script.layout.by_name["x_type"].values(ui.Context())
    assert "[Test] Strength" in labels and "[Test] img2img only" not in labels
    args = script.layout.values_to_args({"x_type": "[Test] Strength"})
    assert script.current_axis_options[args[0]].label == "[Test] Strength"


def test_visibility_follows_the_axis_and_the_text_mode(xyz):
    script = make(xyz)
    nodes = script.layout.by_name
    values = {"x_type": "Sampler", "csv_mode": False}
    get = values.get
    assert ui.evaluate(nodes["x_values_dropdown"].visible_if, get) and not ui.evaluate(nodes["x_values"].visible_if, get)
    values["csv_mode"] = True
    assert not ui.evaluate(nodes["x_values_dropdown"].visible_if, get) and ui.evaluate(nodes["x_values"].visible_if, get)
    values.update(x_type="Steps", csv_mode=False)
    assert not ui.evaluate(nodes["x_values_dropdown"].visible_if, get) and ui.evaluate(nodes["x_values"].visible_if, get)


def call(script, action, values, **extra):
    return ui.call_action(script, action, ui.ActionContext(tab="txt2img", owner="x_y_z_plot", extra=extra), values)


def test_switching_the_axis_moves_values_between_text_and_dropdown(xyz, monkeypatch):
    script = make(xyz)
    sampler = next(x for x in xyz.axis_options if x.label == "Sampler" and not getattr(x, "is_img2img", False))
    monkeypatch.setattr(sampler, "choices", lambda: ["Euler", "Euler a", "DDIM"])

    patch = call(script, "axis_x", {"x_type": "Sampler", "x_values": "Euler a, Nope, DDIM", "x_values_dropdown": []})
    # The axis' fresh list comes along (the old UI re-read it on every axis change)
    assert patch == {"x_values": {"value": ""}, "x_values_dropdown": {"value": ["Euler a", "DDIM"]}, "__outputs__": {"axis_choices/Sampler": ["Euler", "Euler a", "DDIM"]}}

    patch = call(script, "text_mode", {"x_type": "Sampler", "x_values_dropdown": ["Euler", "DDIM"], "csv_mode": True})
    assert patch["x_values"] == {"value": "Euler,DDIM"} and patch["x_values_dropdown"] == {"value": []}
    # Axes without choices keep their text
    assert patch["y_values"] == {"value": ""}

    assert call(script, "axis_x", {"x_type": "Steps", "x_values": "1-3"}) == {"x_values": {"value": "1-3"}, "x_values_dropdown": {"value": []}}


def test_fill_swap_and_values_list(xyz, monkeypatch):
    script = make(xyz)
    schedule = next(x for x in xyz.axis_options if x.label == "Schedule type")
    monkeypatch.setattr(schedule, "choices", lambda: ["Automatic", "Karras"])

    fresh = {"__outputs__": {"axis_choices/Schedule type": ["Automatic", "Karras"]}}
    assert call(script, "fill_y", {"y_type": "Schedule type"}) == {**fresh, "y_values_dropdown": {"value": ["Automatic", "Karras"]}}
    assert call(script, "fill_y", {"y_type": "Schedule type", "csv_mode": True}) == {**fresh, "y_values": {"value": "Automatic,Karras"}}
    assert call(script, "fill_y", {"y_type": "Steps"}) == {}

    patch = call(script, "swap_xz", {"x_type": "Steps", "x_values": "1,2", "z_type": "Schedule type", "z_values_dropdown": ["Karras"]})
    assert patch["x_type"] == {"value": "Schedule type"} and patch["x_values_dropdown"] == {"value": ["Karras"]} and patch["x_values"] == {"value": ""}
    assert patch["z_type"] == {"value": "Steps"} and patch["z_values"] == {"value": "1,2"} and patch["z_values_dropdown"] == {"value": []}

    listed = call(script, "axis_choices", {})["__outputs__"]["axis_choices"]
    assert listed["Schedule type"] == ["Automatic", "Karras"] and "Steps" not in listed
    json.dumps(listed)


def test_a_failing_axis_list_does_not_break_the_others(xyz, monkeypatch):
    script = make(xyz)
    sampler = next(x for x in xyz.axis_options if x.label == "Sampler" and not getattr(x, "is_img2img", False))
    schedule = next(x for x in xyz.axis_options if x.label == "Schedule type")

    def broken():
        raise RuntimeError("model scan failed")

    monkeypatch.setattr(sampler, "choices", broken)
    monkeypatch.setattr(schedule, "choices", lambda: ["Automatic", "Karras"])
    monkeypatch.setattr(xyz.errors, "report", lambda *args, **kwargs: None)

    listed = call(script, "axis_choices", {})["__outputs__"]["axis_choices"]
    assert listed["Sampler"] is None and listed["Schedule type"] == ["Automatic", "Karras"]
    # Switching to the broken axis keeps the typed values and reports its list as failed
    patch = call(script, "axis_x", {"x_type": "Sampler", "x_values": "Euler", "x_values_dropdown": ["DDIM"]})
    assert patch == {"x_values": {"value": "Euler"}, "x_values_dropdown": {"value": ["DDIM"]}, "__outputs__": {"axis_choices/Sampler": None}}
    assert call(script, "fill_x", {"x_type": "Sampler"}) == {"__outputs__": {"axis_choices/Sampler": None}}


def test_list_axes_added_after_the_layout_act_as_text_axes(xyz):
    """The dropdown visibility is static JSON built with the layout: a list
    axis registered later (after before_ui) keeps its text box, in the UI
    and in run(), so typed values never move into a hidden dropdown."""

    script = make(xyz)
    late = xyz.AxisOption("[Test] Late list", str, xyz.apply_field("test_late"), choices=lambda: ["a", "b"])
    xyz.axis_options.append(late)
    nodes = script.layout.by_name
    get = {"x_type": "[Test] Late list", "csv_mode": False}.get
    assert ui.evaluate(nodes["x_values"].visible_if, get) and not ui.evaluate(nodes["x_values_dropdown"].visible_if, get)
    assert not script.is_listed(late) and "[Test] Late list" not in call(script, "axis_choices", {})["__outputs__"]["axis_choices"]
    assert call(script, "axis_x", {"x_type": "[Test] Late list", "x_values": "a, b"}) == {"x_values": {"value": "a, b"}, "x_values_dropdown": {"value": []}}
    # Registered before the layout (on_before_ui, as ADetailer / Dynamic Shift do): a dropdown axis
    rebuilt = make(xyz)
    assert rebuilt.is_listed(late)
    get = {"x_type": "[Test] Late list", "csv_mode": False}.get
    assert ui.evaluate(rebuilt.layout.by_name["x_values_dropdown"].visible_if, get)


def test_infotext_paste(xyz):
    script = make(xyz)
    values, problems = script.layout.paste({"X Type": "Sampler", "X Values": "Euler, \"DPM++ 2M, Karras\"", "Y Type": "Steps", "Y Values": "10,20"})
    assert problems == []
    assert values == {"x_type": "Sampler", "x_values": 'Euler, "DPM++ 2M, Karras"', "x_values_dropdown": ["Euler", "DPM++ 2M, Karras"], "y_type": "Steps", "y_values": "10,20", "y_values_dropdown": ["10", "20"]}
    # An infotext without the plot leaves it alone
    assert script.layout.paste({"Steps": "20"}) == ({}, [])


def test_the_runner_hands_run_the_index(webui):
    from modules import scripts

    module = load_xyz()
    data = scripts.ScriptClassData(module.Script, str(ROOT / "scripts" / "xyz_grid.py"), str(ROOT), module)
    with webui_state(script_classes=[data]) as state:
        runner = state.txt2img
        plot = runner.by_id["x_y_z_plot"]
        assert not plot.layout.legacy
        args = runner.script_args_from_values("X/Y/Z plot", {"x_y_z_plot": {"x_type": "CFG Scale", "x_values": "4, 7"}})
        x_type, x_values = args[plot.args_from : plot.args_from + 2]
        assert plot.current_axis_options[x_type].label == "CFG Scale" and x_values == "4, 7"
        entry = next(entry for entry in runner.scripts_schema() if entry["id"] == "x_y_z_plot")
        json.dumps(entry, allow_nan=False)
        assert entry["kind"] == "selectable" and entry["args"][:3] == ["x_type", "x_values", "x_values_dropdown"]


def test_no_gradio():
    contract.assert_gradio_free(ROOT / "scripts" / "xyz_grid.py")
