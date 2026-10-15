"""Prompt Anchor's ``params()``: the panel in the prompt slot, infotext and the prompt join."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "prompt_anchor.py"


@pytest.fixture(scope="module")
def env(ext_kit):
    from modules import scripts
    from remi.testing import webui_state

    with webui_state():
        module = ext_kit.load_script(SCRIPT, "prompt_anchor_under_test")
    data = scripts.ScriptClassData(module.PromptAnchorScript, str(SCRIPT), str(EXTENSION), module)
    with webui_state(script_classes=[data]) as webui:
        yield SimpleNamespace(kit=ext_kit, module=module, webui=webui)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_matches_the_legacy_panel(env, tab):
    _script, layout = env.kit.layout_of(env.module.PromptAnchorScript, tab)
    snapshot = env.kit.load_snapshot(SCRIPT, tab)
    # Intended: the anchor editor has no label (the panel title names it; Gradio's
    # record shows the param name), and the 4th argument, the Gradio collapse
    # button that process() swallowed with *_, is gone: the panel collapses itself.
    assert len(snapshot) == 4 and snapshot[3]["kind"] == "json"
    env.kit.assert_matches_snapshot(layout, snapshot[:3], {1: {"label": ""}})
    assert layout.by_name["anchor"].prompt is True


def test_settings_set_the_separator_and_the_open_state(env):
    _script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
    assert layout.defaults()["separator"] == ", " and layout.root.open is True
    env.webui.opts.data.update(prompt_anchor_default_separator=" | ", prompt_anchor_collapsed_by_default=True)
    try:
        _script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
        assert layout.defaults()["separator"] == " | " and layout.root.open is False
    finally:
        del env.webui.opts.data["prompt_anchor_default_separator"], env.webui.opts.data["prompt_anchor_collapsed_by_default"]


def test_settings_say_they_need_a_ui_reload(env, monkeypatch):
    # The open state is read when the layouts are built (at start and on Reload UI)
    added = {}
    monkeypatch.setattr(env.module.shared.opts, "add_option", lambda key, info: added.__setitem__(key, info))
    env.module._on_ui_settings()
    assert set(added) == {"prompt_anchor_default_separator", "prompt_anchor_collapsed_by_default"}
    assert getattr(added["prompt_anchor_collapsed_by_default"], "requires_reload_ui", False)
    assert not getattr(added["prompt_anchor_default_separator"], "requires_reload_ui", False)


def test_process_prepends_and_round_trips_through_infotext(env):
    script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
    anchor = 'masterpiece, "best" quality,\nartist'
    args = layout.values_to_args({"anchor": anchor})
    p = SimpleNamespace(prompt="1girl", all_prompts=["1girl", "1boy"], hr_prompt="1girl", all_hr_prompts=["1girl", "1boy"], extra_generation_params={})
    script.before_process(p, *args)
    script.process(p, *args)
    assert p.prompt == f"{anchor}, 1girl" and p.all_prompts == [f"{anchor}, 1girl", f"{anchor}, 1boy"]
    assert p.all_hr_prompts == p.all_prompts and p.hr_prompt == p.prompt
    values, problems = layout.paste(dict(p.extra_generation_params))
    assert not problems and values == {"anchor": anchor.strip()}
    # A pasted infotext without the key leaves the anchor alone
    assert layout.paste({"Prompt": "x"}) == ({}, [])


def test_disabled_or_empty_anchor_leaves_the_prompt(env):
    script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
    for values in ({"enabled": False, "anchor": "style"}, {"anchor": "  "}):
        p = SimpleNamespace(prompt="1girl", all_prompts=["1girl"], extra_generation_params={})
        script.process(p, *layout.values_to_args(values))
        assert p.prompt == "1girl" and p.extra_generation_params == {}


def test_renders_in_the_prompt_slot(env):
    from remi.core.tabs import tab_schema
    from remi.ui.schema import dumps

    for tab in ("txt2img", "img2img"):
        schema = json.loads(dumps(tab_schema(tab)))
        entry = next(script for script in schema["scripts"] if script["id"] == "prompt_anchor")
        assert entry["section"] == "prompt" and entry["args"] == ["enabled", "anchor", "separator"]
        assert entry["layout"]["type"] == "section" and entry["layout"]["enable"]["default"] is True
        # The core form's prompt group holds the slot, right after the prompts
        prompts = next(node for node in schema["core"]["layout"]["children"] if any(child.get("name") == "prompt" for child in node.get("children", [])))
        assert prompts["children"][-1] == {"type": "slot", "section": "prompt"}


def test_localized(env):
    script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
    settings = {"Prompt Anchor", "Default separator between anchor and main prompt", "Start the anchor box collapsed on page load"}
    env.kit.assert_localized(EXTENSION, env.kit.schema_strings(layout.schema()), extra={script.title(), *settings})


def test_standalone(ext_kit):
    ext_kit.assert_standalone(EXTENSION, [SCRIPT])
