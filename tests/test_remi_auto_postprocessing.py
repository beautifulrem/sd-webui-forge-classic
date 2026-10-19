"""Extras operations in txt2img / img2img (setting
``postprocessing_enable_in_main_ui``): ported to ``params()``, never
disabled, their actions work and they run with their values by name."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from modules import scripts, scripts_postprocessing
from remi import ui
from remi.server import runtime
from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"remi_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Brighten(scripts_postprocessing.ScriptPostprocessing):
    name = "Brighten"
    calls: list = []

    def params(self):
        return ui.Section("Brighten", enable=ui.Toggle("brighten_enabled", True), children=[ui.Slider("brighten_amount", 0, 100, 10, 1), ui.Button("Max", "maximum")])

    @ui.action
    def maximum(self, ctx, values):
        return {"brighten_amount": ui.update(value=100)}

    def process_firstpass(self, pp, **args):
        Brighten.calls.append(("firstpass", args))

    def process(self, pp, brighten_enabled=True, brighten_amount=10):
        Brighten.calls.append(("process", {"brighten_enabled": brighten_enabled, "brighten_amount": brighten_amount}))
        if brighten_enabled:
            pp.image = pp.image.point(lambda v: min(255, v + brighten_amount))
            pp.info["Brighten"] = brighten_amount


class GradioOperation(scripts_postprocessing.ScriptPostprocessing):
    """An A1111 operation: Gradio controls in ui(), no params()."""

    name = "Gradio operation"

    def ui(self):
        raise AssertionError("ui() is never called")


@pytest.fixture
def operations(monkeypatch):
    with webui_state():  # the operations import modules.processing, which reads the options
        upscale, degrid = _load("postprocessing_upscale"), _load("degrid")
    data = [
        scripts.ScriptClassData(upscale.ScriptPostprocessingUpscale, str(ROOT / "scripts" / "postprocessing_upscale.py"), str(ROOT), upscale),
        scripts.ScriptClassData(degrid.ScriptPostprocessingDegrid, str(ROOT / "scripts" / "degrid.py"), str(ROOT), degrid),
        scripts.ScriptClassData(Brighten, "brighten.py", "", None),
        scripts.ScriptClassData(GradioOperation, "gradio_operation.py", "", None),
    ]
    monkeypatch.setattr(scripts, "postprocessing_scripts_data", data)
    Brighten.calls = []
    return data


def test_enabled_operations_are_ported_scripts_of_both_tabs(operations):
    with webui_state({"postprocessing_enable_in_main_ui": ["Upscale", "DeGrid", "Brighten"]}) as state:
        assert runtime.not_ported() == [], runtime.not_ported()
        for runner in (state.txt2img, state.img2img):
            for script_id, first in (("upscale", "upscale_enabled"), ("degrid", "degrid_enabled"), ("brighten", "brighten_enabled")):
                script = runner.by_id[script_id]
                assert script.alwayson and script.layout.names[0] == first

        from remi.core.tabs import tab_schema

        entries = {entry["id"]: entry for entry in tab_schema("txt2img")["scripts"]}
        assert {"upscale", "degrid", "brighten"} <= set(entries)


def test_operation_actions_work_in_the_generation_form(operations):
    with webui_state({"postprocessing_enable_in_main_ui": ["Upscale", "Brighten"]}) as state:
        upscale = state.txt2img.by_id["upscale"]
        assert {"swap_size", "refresh_upscalers", "upscaler_scale"} <= set(ui.actions_of(upscale))
        patch = ui.call_action(upscale, "swap_size", ui.ActionContext(tab="txt2img", owner="upscale"), {"upscale_to_width": 512, "upscale_to_height": 768})
        assert patch == {"upscale_to_width": {"value": 768}, "upscale_to_height": {"value": 512}}
        brighten = state.txt2img.by_id["brighten"]
        assert ui.call_action(brighten, "maximum", ui.ActionContext(tab="txt2img", owner="brighten"), {}) == {"brighten_amount": {"value": 100}}


def test_operations_run_after_composite_with_their_values_by_name(operations):
    with webui_state({"postprocessing_enable_in_main_ui": ["Brighten"]}) as state:
        runner = state.txt2img
        script = runner.by_id["brighten"]
        args = runner.script_args_from_values("None", {"brighten": {"brighten_amount": 40}})
        p = SimpleNamespace(extra_generation_params={})
        image = SimpleNamespace(image=Image.new("L", (4, 4), 100))
        script.postprocess_image_after_composite(p, image, *args[script.args_from : script.args_to])
        assert Brighten.calls == [("firstpass", {"brighten_enabled": True, "brighten_amount": 40}), ("process", {"brighten_enabled": True, "brighten_amount": 40})]
        assert image.image.getpixel((0, 0)) == 140
        assert p.extra_generation_params == {"Brighten": 40}


def test_an_operation_without_params_is_listed_not_crashing(operations):
    with webui_state({"postprocessing_enable_in_main_ui": ["Gradio operation", "Brighten"]}) as state:
        assert "gradio_operation" not in state.txt2img.by_id and "brighten" in state.txt2img.by_id
        found = [item for item in runtime.not_ported() if item["title"] == "Gradio operation"]
        assert {item["tab"] for item in found} == {"txt2img", "img2img"}
        assert all("declares ui() instead of params()" in item["reason"] for item in found)


def test_the_setting_needs_a_ui_reload():
    from modules import shared_options

    assert shared_options.options_templates["postprocessing_enable_in_main_ui"].requires_reload_ui
