"""The ported panel: params() declares what the Gradio ui() did. The PiD /
Hires. fix one-of-two wiring is the exclusive group ``pid_hires``; the
checkpoint and module choices are read per request."""

import inspect
import json
from pathlib import Path

import pytest
import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pid.py"
# The legacy wire_exclusive_components(("pid", "hires")) call, now declared on the switch (txt2img only)
EXCLUSIVE = {"txt2img": {0: {"exclusive": ["pid_hires", "pid"]}}, "img2img": {}}


@pytest.fixture(scope="module")
def pid():
    from remi.testing import webui_state

    with webui_state():
        yield harness.load_script(SCRIPT, "pid_params_under_test")


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(pid, tab):
    layout = harness.script_layout(pid.PiDForForge(), tab)
    harness.assert_matches_snapshot(layout, harness.load_snapshot(SCRIPT, tab), EXCLUSIVE[tab])


def test_arguments_are_the_processing_signature(pid):
    layout = harness.script_layout(pid.PiDForForge())
    hook = inspect.signature(pid.PiDForForge.post_sample)
    assert layout.names == [name for name in hook.parameters if name not in ("self", "p", "ps")]


def test_pid_and_hires_fix_are_one_of_two_in_txt2img(pid):
    from modules.anima_feature_conflicts import EXCLUSIVE_GROUPS, PID_HIRES_GROUP

    txt2img = harness.script_layout(pid.PiDForForge(), "txt2img")
    assert txt2img.exclusive == {PID_HIRES_GROUP: {"pid": "enable"}}
    assert "pid" in EXCLUSIVE_GROUPS[PID_HIRES_GROUP]
    # img2img has no Hires. fix
    assert harness.script_layout(pid.PiDForForge(), "img2img").exclusive == {}


def test_choices_follow_the_installed_models(pid, monkeypatch):
    from remi import ui

    monkeypatch.setattr(pid, "checkpoint_tiles", lambda use_short=True: ["anima.safetensors", "PiD-4x.safetensors", "pid_v2.safetensors"])
    monkeypatch.setattr(pid, "vae_dict", {"b.vae": 1, "a.vae": 2})
    monkeypatch.setattr(pid, "module_list", {"gemma.safetensors": 1, "a.vae": 2, "clip.safetensors": 3})
    script = pid.PiDForForge()
    assert script.show(False)
    layout = harness.script_layout(script)
    ctx = ui.Context()
    assert layout.by_name["ckpt"].values(ctx) == ["PiD-4x.safetensors", "pid_v2.safetensors"]
    assert layout.by_name["vae"].values(ctx) == ["a.vae", "b.vae"]
    assert layout.by_name["tec"].values(ctx) == ["clip.safetensors", "gemma.safetensors"]
    assert layout.defaults(ctx) == {"enable": False, "prompt": "", "ckpt": "PiD-4x.safetensors", "vae": "a.vae", "tec": "clip.safetensors", "degrade_sigma": 0.0, "cc": True}


def test_infotext_pastes_back(pid, monkeypatch):
    monkeypatch.setattr(pid, "checkpoint_tiles", lambda use_short=True: ["pid.safetensors"])
    monkeypatch.setattr(pid, "vae_dict", {"x.vae": 1})
    monkeypatch.setattr(pid, "module_list", {"gemma.safetensors": 1})
    layout = harness.script_layout(pid.PiDForForge())
    values, problems = layout.paste({"pid_prompt": "a cat", "pid_ckpt": "pid.safetensors", "pid_vae": "x.vae", "pid_tec": "gemma.safetensors", "degrade_sigma": "0.35"})
    assert problems == []
    assert values == {"prompt": "a cat", "ckpt": "pid.safetensors", "vae": "x.vae", "tec": "gemma.safetensors", "degrade_sigma": 0.35}


def test_schema_export(pid):
    schema = harness.script_layout(pid.PiDForForge(), "txt2img").schema()
    json.dumps(schema, allow_nan=False)
    assert schema["enable"]["exclusive"] == {"group": "pid_hires", "member": "pid"} and schema["enable"]["preset"] == "pid.enabled"
    assert schema["children"][0] == {"type": "markdown", "text": pid.DESCRIPTION}


def test_zh_cn_covers_the_panel(pid):
    harness.assert_localized(SCRIPT.parents[1], [harness.script_layout(pid.PiDForForge(), tab).schema() for tab in harness.TABS])


def test_imports_without_gradio():
    harness.assert_gradio_free([SCRIPT])
