"""The params() path of modules/scripts.py and the ported processing scripts.

The processing scripts keep their legacy Gradio ``ui()`` until the Gradio UI
is deleted; their ``params()`` must declare the same arguments in the same
order, with the same labels, defaults, bounds, choices and infotext keys.
"""

import json

import pytest

from remi import ui
from remi.testing import processing_script_classes, webui_state

gr = pytest.importorskip("gradio")


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


def _script(cls, tab):
    script = cls()
    script.tabname = tab
    script.is_img2img = tab == "img2img"
    script.is_txt2img = not script.is_img2img
    script.filename = f"{cls.__name__}.py"
    return script


def _kind(control):
    from modules.ui_components import InputAccordionImpl

    return {
        InputAccordionImpl: ui.Toggle,
        gr.Checkbox: ui.Toggle,
        gr.Slider: ui.Slider,
        gr.Number: (ui.Seed, ui.Number),
        gr.Dropdown: ui.Choice,
        gr.Textbox: ui.Text,
    }[type(control)]


PARAMS_SAMPLES = [
    {},
    {"Variation seed": "5", "Seed resize from-1": "512", "Refiner": "x", "Sampler": "Euler", "Schedule type": "Karras"},
    {"Sampler": "DPM++ 2M Karras", "Hires sampler": "Euler"},
]


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_match_the_legacy_ui(webui, tab):
    for data in processing_script_classes():
        script = _script(data.script_class, tab)
        with gr.Blocks():
            controls = list(script.ui(script.is_img2img) or [])
        layout = ui.ScriptLayout(script.params(script.is_img2img), owner="x", order=script.param_order)
        ctx = ui.Context()
        name = data.script_class.__name__
        assert len(controls) == len(layout.inputs), name

        for control, node in zip(controls, layout.inputs):
            where = f"{name}.{node.name}"
            assert isinstance(node, _kind(control)), where
            assert (control.label or "") == node.label, where
            if callable(node.default):  # follows the UI preset (the old UI updated it on page load)
                assert node.default() == node.resolve_default(ctx) or node.resolve_default(ctx) in node.values(ctx), where
            else:
                assert control.value == node.resolve_default(ctx), where
            for bound in ("minimum", "maximum", "step"):
                if isinstance(node, ui.Slider):
                    assert getattr(control, bound) == getattr(node, bound), where
            if isinstance(node, ui.Choice):
                assert [choice[1] if isinstance(choice, tuple) else choice for choice in control.choices] == node.values(ctx), where

        fields = list(script.infotext_fields or [])
        assert len(fields) == sum(1 for node in layout.inputs if node.infotext or node.paste), name
        for component, target in fields:
            node = layout.inputs[next(index for index, control in enumerate(controls) if control is component)]
            if isinstance(target, str):
                assert node.infotext == target, f"{name}.{node.name}"
            else:
                assert node.paste is not None, f"{name}.{node.name}"
                for params in PARAMS_SAMPLES:
                    assert node.paste(params) == target(params), f"{name}.{node.name} {params}"


def test_runner_builds_layouts_ids_and_slots_without_a_ui(webui):
    runner = webui.txt2img
    assert not runner.legacy_ui_built
    assert [script.script_id for script in runner.scripts] == ["seed", "sampler", "refiner", "comments"]
    assert all(script.layout is not None for script in runner.scripts)
    total = runner.ensure_arg_slots()
    # The Gradio build order: sampler, accordions (Refiner), seed, then the scripts section
    slots = {script.script_id: (script.args_from, script.args_to) for script in runner.scripts}
    assert slots == {"sampler": (1, 4), "refiner": (4, 9), "seed": (9, 15), "comments": (15, 15)}
    assert total == 15
    assert runner.by_id["sampler"].api_info.args[0].label == "Sampling Steps"


def test_named_values_to_script_args_and_back(webui):
    runner = webui.txt2img
    args = runner.script_args_from_values("None", {"seed": {"seed": "42", "subseed_show": True}, "sampler": {"steps": 30, "scheduler": "Karras"}, "refiner": {"cfg": 99}})
    assert args[0] == 0
    assert args[1:4] == [30, runner.by_id["sampler"].layout.defaults()["sampler_name"], "Karras"]
    assert args[4:9] == [False, "None", 0.875, 24.0, args[8]]
    assert args[9:15] == [42, True, -1, 0.0, 0, 0]
    selected, values = runner.script_values_from_args(args)
    assert selected == "None"
    assert values["seed"]["seed"] == 42 and values["sampler"]["steps"] == 30 and values["refiner"]["cfg"] == 24.0


def test_named_values_report_every_error(webui):
    runner = webui.txt2img
    with pytest.raises(ui.ValidationError) as caught:
        runner.script_args_from_values("Nope", {"seed": {"seed": "x"}, "sampler": {"bogus": 1}, "ghost": {}})
    errors = {(error["owner"], error["param"]) for error in caught.value.to_json()["errors"]}
    assert errors == {("core", "script")}
    with pytest.raises(ui.ValidationError) as caught:
        runner.script_args_from_values(None, {"seed": {"seed": "x"}, "sampler": {"bogus": 1}, "ghost": {}})
    errors = {(error["owner"], error["param"]) for error in caught.value.to_json()["errors"]}
    assert errors == {("ghost", None), ("seed", "seed"), ("sampler", "bogus")}


def test_scripts_schema(webui):
    entries = webui.txt2img.scripts_schema()
    json.dumps(entries, allow_nan=False)
    seed = next(entry for entry in entries if entry["id"] == "seed")
    assert seed["kind"] == "alwayson" and seed["section"] == "seed" and not seed["legacy"]
    assert seed["args"] == ["seed", "subseed_show", "subseed", "subseed_strength", "seed_resize_from_w", "seed_resize_from_h"]
    assert seed["layout"]["type"] == "group" and seed["defaults"]["seed"] == -1
    sampler = next(entry for entry in entries if entry["id"] == "sampler")
    assert sampler["args"] == ["steps", "sampler_name", "scheduler"]
    assert [child["name"] for child in sampler["layout"]["children"]] == ["sampler_name", "scheduler", "steps"]


def test_paste_values_and_preset_bindings(webui):
    values, problems = webui.txt2img.paste_values({"Seed": "7", "Steps": "25", "Sampler": "Euler", "Schedule type": "Simple", "Refiner switch at": "0.5"})
    assert values["seed"] == {"seed": 7, "subseed_show": False}
    assert values["sampler"] == {"steps": 25, "sampler_name": "Euler", "scheduler": "Simple"}
    assert values["refiner"] == {"enable": False, "switch_at": 0.5}
    assert problems == []
    assert webui.txt2img.preset_bindings() == {"native.sampler": [("sampler", "sampler_name")], "native.scheduler": [("sampler", "scheduler")], "native.steps": [("sampler", "steps")], "native.refiner": [("refiner", "enable")]}


class _Ported:
    """A selectable script ported to params()."""

    def title(self):
        return "Loopback Demo"

    def params(self, is_img2img):
        return ui.Group(ui.Slider("loops", 1, 8, 2, 1), ui.Toggle("final", True))

    def run(self, p, loops, final):
        return (loops, final)


class _Legacy:
    """A selectable script still on ui()."""

    def title(self):
        return "Legacy Demo"

    def ui(self, is_img2img):
        return [gr.Slider(1, 10, value=3, label="Count"), gr.Textbox(value="hi", label="Word")]

    def run(self, p, count, word):
        return (count, word)


def _runner_with_selectables():
    from modules import scripts

    ported = type("PortedDemo", (_Ported, scripts.Script), {})
    legacy = type("LegacyDemo", (_Legacy, scripts.Script), {})
    runner = scripts.ScriptRunner()
    saved = list(scripts.scripts_data)
    scripts.scripts_data[:] = [*processing_script_classes(), scripts.ScriptClassData(ported, "ported.py", "", None), scripts.ScriptClassData(legacy, "legacy.py", "", None)]
    try:
        runner.initialize_scripts(is_img2img=False)
    finally:
        scripts.scripts_data[:] = saved
    return runner


def test_selectable_scripts(webui, monkeypatch):
    from types import SimpleNamespace

    from modules import shared

    monkeypatch.setattr(shared, "total_tqdm", SimpleNamespace(clear=lambda: None))
    runner = _runner_with_selectables()
    assert runner.selector_choices() == ["None", "Loopback Demo", "Legacy Demo"]
    assert runner.by_id["loopback_demo"].layout is not None and runner.by_id["legacy_demo"].layout is None

    args = runner.script_args_from_values("Loopback Demo", {"loopback_demo": {"loops": 5}, "legacy_demo": {"#0": 9}})
    ported, legacy = runner.by_id["loopback_demo"], runner.by_id["legacy_demo"]
    assert args[0] == 1
    assert args[ported.args_from : ported.args_to] == [5, True]
    # Unselected selectable scripts get their defaults
    assert args[legacy.args_from : legacy.args_to] == [3, "hi"]
    assert runner.run(None, *args) == (5, True)

    args = runner.script_args_from_values("Legacy Demo", {"legacy_demo": [7, "yo"]})
    assert args[0] == 2 and args[legacy.args_from : legacy.args_to] == [7, "yo"]
    assert runner.run(None, *args) == (7, "yo")

    entries = {entry["id"]: entry for entry in runner.scripts_schema()}
    assert entries["loopback_demo"]["visible_if"] == {"param": "core.script", "eq": "Loopback Demo"}
    assert entries["legacy_demo"]["legacy"] and entries["legacy_demo"]["args"] == ["#0", "#1"]
    assert entries["legacy_demo"]["defaults"] == {"#0": 3, "#1": "hi"}


def test_the_legacy_gradio_ui_renders_ported_scripts_in_place(webui):
    """While the Gradio UI exists it owns the slots; a params()-only script
    gets an approximate rendering there, and the new path follows its slots."""

    from modules import scripts

    runner = _runner_with_selectables()
    previous = scripts.scripts_current
    scripts.scripts_current = runner
    try:
        with gr.Blocks():
            runner.prepare_ui()
            for category in scripts.ordered_ui_categories():
                if category == "scripts":
                    runner.setup_ui()
                runner.setup_ui_for_section(category)
    finally:
        scripts.scripts_current = previous
    assert runner.legacy_ui_built
    ported = runner.by_id["loopback_demo"]
    assert [control.value for control in ported.controls] == [2, True]
    assert runner.ensure_arg_slots() == len(runner.inputs)
    args = runner.script_args_from_values("Loopback Demo", {"loopback_demo": {"loops": 4}})
    assert len(args) == len(runner.inputs)
    assert args[ported.args_from : ported.args_to] == [4, True]
    for script in runner.scripts:
        assert runner.inputs[script.args_from : script.args_to] == list(script.controls or []), script.script_id


# region Unported scripts and failures


def _legacy_plot():
    """An unported selectable script shaped like the old X/Y/Z plot's ``ui()``
    (the plot itself is ported): a ``type="index"`` dropdown, a textbox, a
    multiselect, a checkbox and a slider."""

    from modules import scripts

    class LegacyPlot(scripts.Script):
        def title(self):
            return "X/Y/Z plot"

        def ui(self, is_img2img):
            return [
                gr.Dropdown(label="X type", choices=["Nothing", "Seed", "Steps"], value="Seed", type="index"),
                gr.Textbox(label="X values", lines=1),
                gr.Dropdown(label="X values", multiselect=True, choices=[], visible=False),
                gr.Checkbox(label="Draw legend", value=True),
                gr.Slider(label="Grid Margins", minimum=0, maximum=500, value=0, step=2),
            ]

        def run(self, p, *args):
            return args

    return scripts.ScriptClassData(LegacyPlot, "legacy_plot.py", "", None)


def test_unported_scripts_get_what_gradio_passed():
    """A script on ``ui()`` (``type="index"`` dropdowns, checkboxes, numbers):
    its params are typed, and it receives indices, not labels."""

    with webui_state(script_classes=[_legacy_plot()]) as state:
        runner = state.txt2img
        xyz = runner.by_id["x_y_z_plot"]
        layout = runner.layout_for(xyz)
        x_type = layout.by_name["#0"]
        assert layout.legacy and isinstance(x_type, ui.Choice) and x_type.returns == "index"
        controls = xyz.controls
        expected = [control.preprocess(control.value) if hasattr(control, "preprocess") else control.value for control in controls]

        args = runner.script_args_from_values("X/Y/Z plot", {})
        assert args[0] == 1 and args[xyz.args_from : xyz.args_to] == expected
        assert args[xyz.args_from] == x_type.values(ui.Context()).index("Seed")
        # The schema default sent back unchanged is the same as leaving it out
        defaults = runner.scripts_schema()[-1]["defaults"]
        assert runner.script_args_from_values("X/Y/Z plot", {"x_y_z_plot": defaults})[xyz.args_from : xyz.args_to] == expected

        steps = x_type.values(ui.Context()).index("Steps")
        assert runner.script_args_from_values("X/Y/Z plot", {"x_y_z_plot": {"#0": "Steps"}})[xyz.args_from] == steps
        with pytest.raises(ui.ValidationError):
            runner.script_args_from_values("X/Y/Z plot", {"x_y_z_plot": {"#0": "Not an axis"}})
        json.dumps(runner.scripts_schema(), allow_nan=False)


def test_legacy_controls_map_to_typed_params():
    from remi.ui.legacy import legacy_layout

    with gr.Blocks():
        controls = [
            gr.Dropdown(["a", "b", "c"], value="b", type="index", label="Pick"),
            gr.Radio(["x", "y"], value=None, type="index", label="Radio"),
            gr.CheckboxGroup(["p", "q"], value=["q"], type="index", label="Parts"),
            gr.Number(value=2.5, precision=0, label="Count"),
            gr.Dropdown(["s"], value="custom", allow_custom_value=True, label="Free"),
            gr.State({"k": 1}),
            gr.Checkbox(value=True, label="On"),
        ]
    layout = legacy_layout("legacy", controls)
    assert [node.type for node in layout.inputs] == ["choice", "choice", "multichoice", "number", "choice", "state", "toggle"]
    assert layout.values_to_args({}) == [1, None, [1], 2, "custom", {"k": 1}, True]
    assert layout.values_to_args({"#0": "c", "#1": "y", "#2": ["p", "q"], "#3": 7.4, "#4": "other"})[:5] == [2, 1, [0, 1], 7, "other"]
    assert layout.defaults()["#0"] == "b" and layout.defaults()["#5"] == {"k": 1}


class _BrokenParams:
    def title(self):
        return "Broken Params"

    def show(self, is_img2img):
        from modules import scripts

        return scripts.AlwaysVisible

    def params(self, is_img2img):
        raise KeyError("dave.nope")

    def process(self, p, *args):
        raise AssertionError("a disabled script must not run")


class _BrokenUi:
    def title(self):
        return "Broken UI"

    def show(self, is_img2img):
        from modules import scripts

        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return [][1]

    def process(self, p, *args):
        raise AssertionError("a disabled script must not run")


class _Core:
    def title(self):
        return "Core"

    def show(self, is_img2img):
        from modules import scripts

        return scripts.AlwaysVisible

    def params(self, is_img2img):
        return ui.Toggle("on")


def test_scripts_whose_params_cannot_be_read_are_disabled():
    from modules import scripts

    classes = [type(name, (base, scripts.Script), {}) for name, base in (("BrokenParams", _BrokenParams), ("BrokenUi", _BrokenUi), ("CoreScript", _Core))]
    with webui_state(script_classes=classes) as state:
        runner = state.txt2img
        assert "broken_params" not in runner.by_id and [reason for _script, reason in runner.disabled_scripts] == ["params() failed: 'dave.nope'"]
        runner.ensure_arg_slots()
        assert "broken_ui" not in runner.by_id and [reason for _script, reason in runner.disabled_scripts][1] == "ui() failed"
        ids = [entry["id"] for entry in runner.scripts_schema()]
        assert "broken_params" not in ids and "broken_ui" not in ids
        assert all(script.title() not in ("Broken Params", "Broken UI") for script in runner.alwayson_scripts)
        # A script cannot take a reserved owner id
        assert "core_2" in ids and "core" not in runner.by_id
        assert runner.script_args_from_values("None", {"core_2": {"on": True}})[runner.by_id["core_2"].args_from] is True


def test_script_values_must_be_objects(webui):
    runner = webui.txt2img
    with pytest.raises(ui.ValidationError) as caught:
        runner.script_args_from_values("None", "abc")
    assert caught.value.to_json()["errors"][0]["owner"] is None
    with pytest.raises(ui.ValidationError) as caught:
        runner.script_args_from_values("None", {"seed": "x"})
    assert caught.value.to_json()["errors"][0]["owner"] == "seed"


# endregion
