"""The selectable scripts and Extras operations of ``scripts/`` on
``params()``: the legacy ``ui()`` contract (tests/fixtures/legacy_ui), the
processing arguments, and their actions. The X/Y/Z plot has its own file."""

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

SCRIPTS = ("loopback", "prompt_matrix", "prompts_from_file", "sd_upscale", "postprocessing_upscale", "degrid", "xyz_grid")


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        from modules import modelloader

        modelloader.load_upscalers()  # the snapshots list the built-in upscalers
        yield state


def script_layout(module, cls_name, tab):
    script = getattr(module, cls_name)()
    script.is_img2img = tab == "img2img"
    script.layout = ui.ScriptLayout(script.params(script.is_img2img), owner="script", order=script.param_order)
    return script


def operation_layout(module, cls_name):
    operation = getattr(module, cls_name)()
    return operation, ui.ScriptLayout(operation.params(), owner="operation")


@pytest.mark.parametrize(
    ("name", "cls_name", "tabs", "expected"),
    [
        ("loopback", "Loopback", ("img2img",), {}),
        ("prompt_matrix", "PromptMatrix", ("txt2img", "img2img"), {}),
        # The text box is the text-file widget (it also loads a file, in the browser); still a string
        ("prompts_from_file", "PromptsFromTexts", ("txt2img", "img2img"), {3: {"kind": "custom"}}),
        ("sd_upscale", "SDUpscale", ("img2img",), {}),
    ],
)
def test_scripts_keep_the_legacy_contract(webui, name, cls_name, tabs, expected):
    module = load(name)
    for tab in tabs:
        script = script_layout(module, cls_name, tab)
        assert contract.differences(script.layout, contract.load_snapshot(FIXTURES / f"{name}_{tab}.json"), expected=expected) == [], tab


def test_extras_operations_keep_the_legacy_contract(webui):
    _op, upscale = operation_layout(load("postprocessing_upscale"), "ScriptPostprocessingUpscale")
    snapshot = contract.load_snapshot(FIXTURES / "postprocessing_upscale_extras.json")
    # Extras operations get their values by name, so the order is free: compare by name
    legacy = {entry.pop("name"): entry for entry in snapshot["inputs"]}
    assert set(legacy) == set(upscale.names)
    ordered = {"inputs": [legacy[name] for name in upscale.names]}
    # The hidden number Gradio's tab events set is the choice the tab strip is bound to
    mode = upscale.names.index("upscale_mode")
    expected = {mode: {"kind": "choice", "label": "Resize mode", "choices": [0, 1], "style": "dropdown", "allow_custom": False, "returns": "value"}}
    assert contract.differences(upscale, ordered, expected=expected) == []

    _op, degrid = operation_layout(load("degrid"), "ScriptPostprocessingDegrid")
    snapshot = contract.load_snapshot(FIXTURES / "degrid_extras.json")
    assert degrid.names == [entry.pop("name") for entry in snapshot["inputs"]]
    assert contract.differences(degrid, snapshot) == []


def test_processing_arguments(webui):
    from modules import shared

    matrix = script_layout(load("prompt_matrix"), "PromptMatrix", "txt2img")
    assert matrix.layout.values_to_args({"margin_size": 32, "prompt_type": "negative"}) == [False, False, "negative", "comma", 32]

    upscale = script_layout(load("sd_upscale"), "SDUpscale", "img2img")
    names = [x.name for x in shared.sd_upscalers]
    # run() gets the upscaler's index, as with Gradio's type="index"
    assert upscale.layout.values_to_args({"upscaler_index": names[-1], "overlap": 32}) == [32, len(names) - 1, 2.0, False]
    assert upscale.layout.values_to_args({})[1] == 0

    loop = script_layout(load("loopback"), "Loopback", "img2img")
    assert loop.layout.values_to_args({"loops": 99}) == [8, 0.5, "Linear"]  # sliders clamp


def test_prompts_from_file_takes_text(webui):
    script = script_layout(load("prompts_from_file"), "PromptsFromTexts", "txt2img")
    assert script.layout.values_to_args({"prompt_txt": "a cat\na dog"}) == [False, False, "start", "a cat\na dog"]
    assert script.layout.values_to_args({"prompt_txt": None})[3] == ""
    with pytest.raises(ui.ValidationError):
        script.layout.values_to_args({"prompt_txt": ["a cat"]})


def test_upscale_tabs_actions_and_kwargs(webui, monkeypatch):
    from modules import shared

    operation, layout = operation_layout(load("postprocessing_upscale"), "ScriptPostprocessingUpscale")
    tabs = next(node for node in layout.root.walk() if isinstance(node, ui.Tabs))
    assert tabs.bind == "upscale_mode" and [tab.value for tab in tabs.items] == [0, 1]
    assert layout.values_to_args({"upscale_mode": "1"})[layout.names.index("upscale_mode")] == 1

    def call(action, values):
        return ui.call_action(operation, action, ui.ActionContext(tab="extras", owner="upscale"), values)

    assert call("swap_size", {"upscale_to_width": 640, "upscale_to_height": 1536}) == {"upscale_to_width": {"value": 1536}, "upscale_to_height": {"value": 640}}
    monkeypatch.setattr(shared.opts, "set_scale_by_when_changing_upscaler", True, raising=False)
    assert call("upscaler_scale", {"upscaler_1_name": "4x-UltraSharp"}) == {"upscale_by": {"value": 4}}
    assert call("upscaler_scale", {"upscaler_1_name": "RealESRGAN_x2plus"}) == {"upscale_by": {"value": 2}}
    assert call("upscaler_scale", {"upscaler_1_name": "Lanczos"}) == {}
    monkeypatch.setattr(shared.opts, "set_scale_by_when_changing_upscaler", False, raising=False)
    assert call("upscaler_scale", {"upscaler_1_name": "4x-UltraSharp"}) == {}
    choices = call("refresh_upscalers", {})["upscaler_1_name"]["choices"]
    assert [choice["value"] for choice in choices] == [x.name for x in shared.sd_upscalers]

    # Upscaler 2 is editable only with some visibility, as before
    assert not ui.evaluate(layout.by_name["upscaler_2_name"].enabled_if, {"upscaler_2_visibility": 0.0}.get)
    assert ui.evaluate(layout.by_name["upscaler_2_name"].enabled_if, {"upscaler_2_visibility": 0.5}.get)


def test_extras_runner_passes_named_values(webui):
    """Extras operations receive keyword arguments named after their params."""

    from modules import scripts_postprocessing

    from remi.core.extras import assign_args, postprocessing_args, scripts_schema

    runner = scripts_postprocessing.ScriptPostprocessingRunner()
    runner.scripts = []
    for name, cls_name in (("postprocessing_upscale", "ScriptPostprocessingUpscale"), ("degrid", "ScriptPostprocessingDegrid")):
        operation = getattr(load(name), cls_name)()
        operation.filename = f"{name}.py"
        runner.scripts.append(operation)
    assigned = {layout.owner: (operation, layout) for operation, layout in assign_args(runner)}
    assert set(assigned) == {"upscale", "degrid"}
    args = postprocessing_args(runner, {"upscale": {"upscale_mode": 1, "upscale_to_width": 512}, "degrid": {"degrid_threshold": 0.4}})
    upscale, _ = assigned["upscale"]
    received = dict(zip(upscale.controls, args[upscale.args_from : upscale.args_to]))
    assert received["upscale_mode"] == 1 and received["upscale_to_width"] == 512 and received["upscale_enabled"] is True
    degrid, _ = assigned["degrid"]
    assert dict(zip(degrid.controls, args[degrid.args_from : degrid.args_to])) == {"degrid_enabled": True, "degrid_threshold": 0.4}
    json.dumps(scripts_schema(runner), allow_nan=False)


def test_degrid_runs_on_the_torch_device(webui, monkeypatch):
    """DeGrid is on by default, so it must not call ``.cuda()``: every Extras job
    failed on CPU / MPS machines ("Torch not compiled with CUDA enabled")."""

    import numpy as np
    import torch
    from PIL import Image

    from modules import devices, scripts_postprocessing

    monkeypatch.setattr(devices, "device", torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda *_a, **_k: pytest.fail("DeGrid called .cuda()"))
    rng = np.random.default_rng(0)
    pp = scripts_postprocessing.PostprocessedImage(Image.fromarray(rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)))
    load("degrid").ScriptPostprocessingDegrid().process(pp)
    assert pp.image.size == (48, 64) and pp.image.mode == "RGB"


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_gradio(name):
    contract.assert_gradio_free(ROOT / "scripts" / f"{name}.py")


def test_zh_cn_covers_every_string(webui):
    """scripts/ strings are in the core catalog (localizations/zh_CN.json),
    or in the SPA's own catalog, which wins over it (so the core catalog must
    not repeat them); upscaler names are model names."""

    import re

    from modules import shared

    strings = set()
    for name, cls_name in (("loopback", "Loopback"), ("prompt_matrix", "PromptMatrix"), ("prompts_from_file", "PromptsFromTexts"), ("sd_upscale", "SDUpscale"), ("xyz_grid", "Script")):
        module = load(name)
        for tab in ("txt2img", "img2img"):
            script = script_layout(module, cls_name, tab)
            strings |= contract.schema_strings(script.layout.schema(ui.Context())) | {script.title()}
        strings |= set(re.findall(r'ui\.toast\("([^"]+)"', (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")))
    for name, cls_name in (("postprocessing_upscale", "ScriptPostprocessingUpscale"), ("degrid", "ScriptPostprocessingDegrid")):
        operation, layout = operation_layout(load(name), cls_name)
        strings |= contract.schema_strings(layout.schema(ui.Context())) | {operation.name}
    upscalers = {x.name for x in shared.sd_upscalers}
    spa = json.loads((ROOT / "webui" / "src" / "locales" / "zh-CN.json").read_text(encoding="utf-8"))
    assert contract.missing_translations(strings, ROOT / "localizations" / "zh_CN.json", ignore=lambda text: text in upscalers or text in spa) == []
    core = json.loads((ROOT / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    assert sorted(text for text in strings if text in spa and core.get(text, spa[text]) != spa[text]) == []  # shadowed, inconsistent entries
