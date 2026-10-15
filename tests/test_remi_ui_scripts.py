"""The params() path of modules/scripts.py and the built-in processing scripts.

Their ``params()`` declare the arguments their former Gradio ``ui()`` did, in
the same order, with the same labels, defaults, bounds, choices and infotext
keys (``tests/fixtures/legacy_ui/processing_scripts.json``, recorded from
that ``ui()`` before it was removed).
"""

import json
from pathlib import Path

import pytest

from remi import ui
from remi.testing import processing_script_classes, webui_state

SNAPSHOT = json.loads((Path(__file__).parent / "fixtures" / "legacy_ui" / "processing_scripts.json").read_text(encoding="utf-8"))
KINDS = {"Toggle": ui.Toggle, "Slider": ui.Slider, "Number": (ui.Seed, ui.Number), "Choice": ui.Choice, "Text": ui.Text}


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


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_match_the_legacy_ui(webui, tab):
    recorded = {entry["script"]: entry["inputs"] for entry in SNAPSHOT[tab]}
    classes = processing_script_classes()
    assert sorted(recorded) == sorted(data.script_class.__name__ for data in classes)
    for data in classes:
        script = _script(data.script_class, tab)
        layout = ui.ScriptLayout(script.params(script.is_img2img), owner="x", order=script.param_order)
        ctx = ui.Context()
        name = data.script_class.__name__
        legacy = recorded[name]
        assert len(legacy) == len(layout.inputs), name

        for control, node in zip(legacy, layout.inputs, strict=True):
            where = f"{name}.{node.name}"
            assert isinstance(node, KINDS[control["kind"]]), where
            assert control["label"] == node.label, where
            if callable(node.default):  # follows the UI preset (the old UI updated it on page load)
                default = node.resolve_default(ctx)
                assert default in node.values(ctx) if isinstance(node, ui.Choice) else default is not None, where
            else:
                assert control["default"] == node.resolve_default(ctx), where
            for bound in ("minimum", "maximum", "step"):
                if isinstance(node, ui.Slider):
                    assert control[bound] == getattr(node, bound), where
            if isinstance(node, ui.Choice):
                assert control["choices"] == node.values(ctx), where
            if "infotext" in control:
                assert node.infotext == control["infotext"], where
            elif "paste" in control:
                assert node.paste is not None, where
                assert [node.paste(dict(sample)) for sample in SNAPSHOT["samples"]] == control["paste"], where
            else:
                assert node.infotext is None and node.paste is None, where


def test_runner_builds_layouts_ids_and_slots_without_a_ui(webui):
    runner = webui.txt2img
    assert [script.script_id for script in runner.scripts] == ["seed", "sampler", "refiner", "comments"]
    assert all(script.layout is not None for script in runner.scripts)
    total = runner.ensure_arg_slots()
    # The A1111 slot order: sampler, accordions (Refiner), seed, then the scripts section
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
    assert seed["kind"] == "alwayson" and seed["section"] == "seed" and "legacy" not in seed
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
    """An A1111 selectable script: Gradio controls in ui(), no params()."""

    def title(self):
        return "Legacy Demo"

    def ui(self, is_img2img):
        raise AssertionError("ui() is never called")

    def run(self, p, count, word):
        raise AssertionError("a disabled script must not run")


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
    # A script that only has A1111's ui() is left out, with the reason
    assert runner.selector_choices() == ["None", "Loopback Demo"]
    assert "legacy_demo" not in runner.by_id
    assert [(script.title(), reason) for script, reason in runner.disabled_scripts] == [("Legacy Demo", "declares ui() instead of params()")]

    args = runner.script_args_from_values("Loopback Demo", {"loopback_demo": {"loops": 5}})
    ported = runner.by_id["loopback_demo"]
    assert args[0] == 1
    assert args[ported.args_from : ported.args_to] == [5, True]
    assert runner.run(None, *args) == (5, True)
    # Values may also be a list in argument order
    args = runner.script_args_from_values("Loopback Demo", {"loopback_demo": [7, False]})
    assert args[ported.args_from : ported.args_to] == [7, False]
    with pytest.raises(ui.ValidationError):
        runner.script_args_from_values("Loopback Demo", {"loopback_demo": [7, False, 1]})
    assert runner.set_named_arg(list(args), "loopback demo", "loops", 3)[ported.args_from] == 3

    entries = {entry["id"]: entry for entry in runner.scripts_schema()}
    assert entries["loopback_demo"]["visible_if"] == {"param": "core.script", "eq": "Loopback Demo"}
    assert set(entries) >= {"loopback_demo"} and "legacy_demo" not in entries


# region Unported scripts and failures


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
        raise AssertionError("ui() is never called")

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
        reasons = {script.title(): reason for script, reason in runner.disabled_scripts}
        assert reasons == {"Broken Params": "params() failed: 'dave.nope'", "Broken UI": "declares ui() instead of params()"}
        assert "broken_params" not in runner.by_id and "broken_ui" not in runner.by_id
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
