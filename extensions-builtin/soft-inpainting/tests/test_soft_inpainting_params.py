"""Soft inpainting's ``params()``: the Gradio panel's contract, the schema, infotext and processing args."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "soft_inpainting.py"


@pytest.fixture(scope="module")
def env(ext_kit):
    from modules import scripts
    from remi.testing import webui_state

    with webui_state():
        module = ext_kit.load_script(SCRIPT, "soft_inpainting_under_test")
        with webui_state(script_classes=[scripts.ScriptClassData(module.Script, str(SCRIPT), str(EXTENSION), module)]):
            yield SimpleNamespace(kit=ext_kit, module=module)


def test_matches_the_legacy_panel(env):
    _script, layout = env.kit.layout_of(env.module.Script, "img2img")
    env.kit.assert_matches_snapshot(layout, env.kit.load_snapshot(SCRIPT, "img2img"))


def test_img2img_only(env):
    script, layout = env.kit.layout_of(env.module.Script, "txt2img")
    assert layout.root is None and not script.show(False)


def test_arguments_follow_the_processing_signature(env):
    _script, layout = env.kit.layout_of(env.module.Script, "img2img")
    for hook in ("process", "on_mask_blend", "post_sample", "postprocess_maskoverlay"):
        names = list(inspect.signature(getattr(env.module.Script, hook)).parameters)
        hook_args = names[3:] if hook != "process" else names[2:]
        assert len(hook_args) == len(layout.names), hook
    args = layout.values_to_args({"enabled": True, "mask_blend_power": 2, "composite_difference_contrast": 99})
    assert args == [True, 2.0, 0.5, 4.0, 0.0, 0.5, 8.0]


def test_infotext_round_trip(env):
    script, layout = env.kit.layout_of(env.module.Script, "img2img")
    p = SimpleNamespace(extra_generation_params={}, mask_round=True, image_mask=object())
    args = layout.values_to_args({"enabled": True, "mask_blend_scale": 1.25})
    script.process(p, *args)
    assert p.mask_round is False
    params = {key: str(value) for key, value in p.extra_generation_params.items()}
    values, problems = layout.paste(params)
    assert not problems
    assert layout.values_to_args(values) == args


def test_schema_export(env):
    from remi.core.tabs import tab_schema
    from remi.ui.schema import dumps

    schema = json.loads(dumps(tab_schema("img2img")))
    entry = next(script for script in schema["scripts"] if script["id"] == "soft_inpainting")
    _script, layout = env.kit.layout_of(env.module.Script, "img2img")
    assert entry["section"] == "inpaint" and entry["args"] == layout.names
    assert entry["layout"]["type"] == "section" and entry["layout"]["enable"]["name"] == "enabled"
    help_group = entry["layout"]["children"][-1]
    assert help_group["type"] == "group" and help_group["title"] == "Help" and help_group["open"] is False
    assert all(script["id"] != "soft_inpainting" for script in tab_schema("txt2img")["scripts"])


def test_localized(env):
    script, layout = env.kit.layout_of(env.module.Script, "img2img")
    env.kit.assert_localized(EXTENSION, env.kit.schema_strings(layout.schema()), extra={script.title()})


def test_standalone(ext_kit):
    ext_kit.assert_standalone(EXTENSION, [SCRIPT])
