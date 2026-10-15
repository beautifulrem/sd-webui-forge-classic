"""Lock img2img Steps: one switch that locks ``sampler.steps`` in the form."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "lock_steps.py"


@pytest.fixture(scope="module")
def env(ext_kit):
    from modules import scripts
    from remi.testing import webui_state

    with webui_state():
        module = ext_kit.load_script(SCRIPT, "lock_steps_under_test")
        with webui_state(script_classes=[scripts.ScriptClassData(module.LockImg2ImgSteps, str(SCRIPT), str(EXTENSION), module)]):
            yield SimpleNamespace(kit=ext_kit, module=module)


def test_matches_the_gradio_panel(env):
    _script, layout = env.kit.layout_of(env.module.LockImg2ImgSteps, "img2img")
    env.kit.assert_matches_snapshot(layout, env.kit.load_snapshot(SCRIPT, "img2img"))


def test_img2img_only(env):
    script, layout = env.kit.layout_of(env.module.LockImg2ImgSteps, "txt2img")
    assert layout.root is None and not script.show(False)


def test_value_is_a_boolean(env):
    from remi.ui import ValidationError

    _script, layout = env.kit.layout_of(env.module.LockImg2ImgSteps, "img2img")
    assert layout.values_to_args({"lock": True}) == [True]
    assert layout.values_to_args({}) == [False]
    with pytest.raises(ValidationError):
        layout.values_to_args({"lock": "yes"})


def test_schema_export(env):
    from remi.core.tabs import tab_schema
    from remi.ui.schema import dumps

    schema = json.loads(dumps(tab_schema("img2img")))
    entry = next(script for script in schema["scripts"] if script["id"] == "lock_img2img_steps")
    assert entry["section"] == "sampler" and entry["args"] == ["lock"]
    # The param the switch locks exists in the same tab
    sampler = next(script for script in schema["scripts"] if script["id"] == "sampler")
    assert "steps" in sampler["args"]
    assert entry["layout"] == {"type": "toggle", "name": "lock", "label": env.module.LABEL, "default": False, "locks": "sampler.steps"}


def test_localized(env):
    script, layout = env.kit.layout_of(env.module.LockImg2ImgSteps, "img2img")
    env.kit.assert_localized(EXTENSION, env.kit.schema_strings(layout.schema()), extra={script.title()})


def test_gradio_free(ext_kit):
    ext_kit.assert_gradio_free(EXTENSION, [SCRIPT])
