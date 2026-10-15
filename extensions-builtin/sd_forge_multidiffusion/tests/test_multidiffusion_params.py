"""The ported panel: params() declares what the Gradio ui() did; the former
📐 button is the ``detect_size`` action, reading the img2img source image's
size from the upload index."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "forge_multidiffusion.py"
# backend.patcher.controlnet and backend.nn.cnets import each other; the WebUI's start order imports cldm first
PREIMPORT = ("backend.nn.cnets.cldm",)


@pytest.fixture(scope="module")
def multidiffusion():
    import importlib

    from remi.testing import webui_state

    for name in PREIMPORT:
        importlib.import_module(name)
    with webui_state():
        yield harness.load_script(SCRIPT, "multidiffusion_params_under_test")


def test_params_match_the_legacy_ui(multidiffusion):
    script = multidiffusion.MultiDiffusionForForge()
    assert not script.show(False) and script.show(True)
    harness.assert_matches_snapshot(harness.script_layout(script, "img2img"), harness.load_snapshot(SCRIPT, "img2img"))


def test_arguments_are_the_processing_signature(multidiffusion):
    layout = harness.script_layout(multidiffusion.MultiDiffusionForForge(), "img2img")
    hook = inspect.signature(multidiffusion.MultiDiffusionForForge.process_before_every_sampling)
    assert layout.names == [name for name in hook.parameters if name not in ("self", "p", "kwargs")]


def _context(init_img="upload-1", record=SimpleNamespace(width=1216, height=832)):
    from remi import ui

    files = SimpleNamespace(get=lambda file_id: record if file_id == "upload-1" else None)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=SimpleNamespace(files=files))))
    return ui.ActionContext(tab="img2img", owner="multidiffusion_integrated", core={"init_img": init_img}, request=request)


def test_detect_size_sets_the_tiles_to_the_source_image(multidiffusion):
    from remi import ui

    script = multidiffusion.MultiDiffusionForForge()
    patch = ui.call_action(script, "detect_size", _context(), {"tile_width": 768, "tile_height": 768})
    assert patch == {"tile_width": {"value": 1216}, "tile_height": {"value": 832}}


def test_detect_size_keeps_the_tiles_inside_the_slider_bounds(multidiffusion):
    from remi import ui

    patch = ui.call_action(multidiffusion.MultiDiffusionForForge(), "detect_size", _context(record=SimpleNamespace(width=4096, height=128)), {})
    assert patch == {"tile_width": {"value": 2048}, "tile_height": {"value": 256}}


@pytest.mark.parametrize("context", [_context(init_img=None), _context(init_img="unknown"), _context(record=SimpleNamespace(width=None, height=None))])
def test_detect_size_without_a_source_image_changes_nothing(multidiffusion, context):
    from remi import ui

    patch = ui.call_action(multidiffusion.MultiDiffusionForForge(), "detect_size", context, {})
    assert set(patch) == {"__toast__"} and patch["__toast__"]["level"] == "warning"


def test_detect_size_needs_the_server(multidiffusion):
    from remi import ui

    # Wiring errors surface instead of passing for a missing source image
    with pytest.raises(AttributeError):
        ui.call_action(multidiffusion.MultiDiffusionForForge(), "detect_size", ui.ActionContext(core={"init_img": "upload-1"}), {})


def test_schema_export(multidiffusion):
    schema = harness.script_layout(multidiffusion.MultiDiffusionForForge(), "img2img").schema()
    json.dumps(schema, allow_nan=False)
    sizes, button = schema["children"][1:3]
    assert [child["name"] for child in sizes["children"]] == ["tile_width", "tile_height"]
    assert button == {"type": "button", "label": "Use source image size", "action": "detect_size", "variant": "tool"}


def test_zh_cn_covers_the_panel(multidiffusion):
    schema = harness.script_layout(multidiffusion.MultiDiffusionForForge(), "img2img").schema()
    harness.assert_localized(SCRIPT.parents[1], [schema], extra=["Add a source image to read its size"])


def test_imports_without_gradio():
    harness.assert_gradio_free([SCRIPT], preimport=PREIMPORT)
