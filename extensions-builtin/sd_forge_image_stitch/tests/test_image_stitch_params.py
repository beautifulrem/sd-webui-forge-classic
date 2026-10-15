"""ImageStitch's ``params()``: the reference list as an ``image-list`` widget over uploads."""

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "image_stitch.py"


@pytest.fixture(scope="module")
def env(ext_kit):
    from modules import scripts
    from remi.testing import webui_state

    with webui_state():
        module = ext_kit.load_script(SCRIPT, "image_stitch_under_test")
    data = scripts.ScriptClassData(module.ImageStitch, str(SCRIPT), str(EXTENSION), module)
    with webui_state(script_classes=[data]):
        yield SimpleNamespace(kit=ext_kit, module=module)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_matches_the_gradio_panel(env, tab):
    _script, layout = env.kit.layout_of(env.module.ImageStitch, tab)
    # Intended: the interactive gallery (a JSON value of (image, caption) pairs)
    # is an image list of upload ids (resolved to PIL images) shown by the
    # `image-list` widget
    env.kit.assert_matches_snapshot(layout, env.kit.load_snapshot(SCRIPT, tab), {1: {"kind": "image_list"}})


def test_reference_uploads_reach_the_script_as_images(env):
    from remi import ui

    _script, layout = env.kit.layout_of(env.module.ImageStitch, "img2img")
    images = {"up-1": Image.new("RGB", (64, 64), "red"), "up-2": Image.new("RGB", (32, 32), "blue")}
    ctx = ui.Context(resolve_upload=lambda node, upload_id: images[upload_id])
    enable, references, max_dim = layout.values_to_args({"enable": True, "references": ["up-2", "up-1"]}, ctx)
    assert enable is True and max_dim == 1024
    assert references == [images["up-2"], images["up-1"]]
    with pytest.raises(ui.ValidationError):
        layout.values_to_args({"references": "up-1"})


def test_extract_images_reads_every_reference_form(env):
    image = Image.new("RGB", (8, 8), "green")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    extracted = env.module.ImageStitch.extract_images([image, (image, None), encoded])
    assert extracted[0] is image and extracted[1] is image
    assert extracted[2].size == (8, 8)


def test_schema_export(env):
    from remi.core.tabs import tab_schema
    from remi.ui.schema import dumps

    for tab in ("txt2img", "img2img"):
        schema = json.loads(dumps(tab_schema(tab)))
        entry = next(script for script in schema["scripts"] if script["id"] == "imagestitch_integrated")
        assert entry["args"] == ["enable", "references", "max_dim"] and entry["defaults"]["references"] == []
        references = next(node for node in entry["layout"]["children"] if node.get("name") == "references")
        # An upload type (drafts never restore its ids) with a widget hint
        assert references == {"type": "image_list", "name": "references", "label": "Reference Image(s)", "default": [], "widget": "image-list", "props": {"columns": 3}}


def test_localized(env):
    strings = set()
    for tab in ("txt2img", "img2img"):
        script, layout = env.kit.layout_of(env.module.ImageStitch, tab)
        strings |= env.kit.schema_strings(layout.schema())
    env.kit.assert_localized(EXTENSION, strings, extra={script.title()})


def test_gradio_free(ext_kit):
    ext_kit.assert_gradio_free(EXTENSION, [SCRIPT])
