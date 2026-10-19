"""Prompt Anchor's ``params()``: the panel above the main prompt, infotext and the prompt join."""

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
    # The switch and the separator are pasted too, so a pasted anchored image
    # regenerates with the anchor on and joined the same way.
    assert len(snapshot) == 4 and snapshot[3]["kind"] == "json"
    env.kit.assert_matches_snapshot(layout, snapshot[:3], {0: {"paste": True}, 1: {"label": ""}, 2: {"paste": True}})
    assert layout.by_name["anchor"].prompt is True
    # Placement: legacy prompt_anchor.js lifted the panel right above the main
    # prompt; slot "prompt_before" puts it there, and section "prompt" keeps
    # the arguments where Gradio's section placed them (test_renders_above_the_main_prompt)
    assert _script.section == "prompt" and _script.slot == "prompt_before"


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
    # The anchor comes back switched on, with the separator it was joined with
    assert not problems and values == {"enabled": True, "anchor": anchor.strip(), "separator": ", "}
    # A pasted infotext without the key leaves the anchor alone
    assert layout.paste({"Prompt": "x"}) == ({}, [])


def _generate_infotext(script, layout, values, prompt, hr_prompt=None):
    """process() on a fake p, then the infotext processing.create_infotext writes."""
    from modules.infotext_utils import quote

    args = layout.values_to_args(values)
    hr_prompt = prompt if hr_prompt is None else hr_prompt
    p = SimpleNamespace(prompt=prompt, all_prompts=[prompt], hr_prompt=hr_prompt, all_hr_prompts=[hr_prompt], extra_generation_params={})
    script.before_process(p, *args)
    script.process(p, *args)
    params = {"Steps": 20, "Sampler": "Euler", "Seed": 1, **p.extra_generation_params}
    if p.all_hr_prompts[0] != p.all_prompts[0]:
        params["Hires prompt"] = p.all_hr_prompts[0]
    return p, f"{p.all_prompts[0]}\nNegative prompt: \n" + ", ".join(f"{key}: {quote(value)}" for key, value in params.items())


@pytest.mark.parametrize(
    ("values", "prompt"),
    [
        ({"anchor": "red hair"}, "masterpiece, 1girl"),  # feat_anchor_0.png
        ({"anchor": "red hair"}, "red hair, masterpiece, 1girl"),  # the main prompt itself starts with the anchor
        ({"anchor": "masterpiece, best quality,"}, "1girl, solo"),
        ({"anchor": 'artist "x", \\(style\\)\nscore_9'}, ", 1girl"),
        ({"anchor": "style", "separator": " BREAK "}, "1girl"),
        ({"anchor": "style", "separator": "\n"}, "1girl, smile"),
        ({"anchor": "style", "separator": ","}, "1girl"),
        ({"anchor": "only anchor"}, ""),
    ],
)
def test_pasted_infotext_regenerates_the_same_prompt(env, values, prompt):
    """Paste (POST /api/v2/infotext/parse) then generate again: the same prompt and infotext."""
    from remi.core.infotext import parse_infotext

    script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
    first, text = _generate_infotext(script, layout, values, prompt)

    parsed = parse_infotext(text, "txt2img")
    assert parsed["problems"] == []
    pasted = parsed["scripts"]["prompt_anchor"]
    assert pasted["enabled"] is True and pasted["anchor"] == values["anchor"].strip()
    assert pasted["separator"] == values.get("separator", ", ")
    # The main prompt is the one the user typed (no anchor, which would be prepended twice)
    assert parsed["core"]["prompt"] == prompt.lstrip(", ")

    again, text_again = _generate_infotext(script, layout, pasted, parsed["core"]["prompt"])
    assert again.all_prompts == first.all_prompts and text_again == text


def test_pasted_custom_hires_prompt_is_not_anchored_twice(env):
    from remi.core.infotext import parse_infotext

    script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
    first, text = _generate_infotext(script, layout, {"anchor": "red hair"}, "1girl", hr_prompt="1girl, detailed")
    parsed = parse_infotext(text, "txt2img")
    again, text_again = _generate_infotext(script, layout, parsed["scripts"]["prompt_anchor"], parsed["core"]["prompt"], hr_prompt=parsed["params"]["Hires prompt"])
    assert again.all_hr_prompts == first.all_hr_prompts == ["red hair, 1girl, detailed"] and text_again == text


def test_hand_edited_prompt_is_left_alone(env):
    from remi.core.infotext import parse_infotext

    text = 'something else, red hair\nNegative prompt: \nSteps: 20, Sampler: Euler, Anchor prompt: "red hair"'
    parsed = parse_infotext(text, "txt2img")
    assert parsed["core"]["prompt"] == "something else, red hair" and parsed["scripts"]["prompt_anchor"]["anchor"] == "red hair"


def test_disabled_or_empty_anchor_leaves_the_prompt(env):
    script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
    for values in ({"enabled": False, "anchor": "style"}, {"anchor": "  "}):
        p = SimpleNamespace(prompt="1girl", all_prompts=["1girl"], extra_generation_params={})
        script.process(p, *layout.values_to_args(values))
        assert p.prompt == "1girl" and p.extra_generation_params == {}


def test_renders_above_the_main_prompt(env):
    from remi.core.tabs import tab_schema
    from remi.ui.schema import dumps

    for tab in ("txt2img", "img2img"):
        schema = json.loads(dumps(tab_schema(tab)))
        entry = next(script for script in schema["scripts"] if script["id"] == "prompt_anchor")
        assert entry["section"] == "prompt_before" and entry["args"] == ["enabled", "anchor", "separator"]
        assert entry["layout"]["type"] == "section" and entry["layout"]["enable"]["default"] is True
        # The core form's prompt group opens with the slot, before the main
        # prompt (as legacy prompt_anchor.js reparented the panel); no script
        # of section "prompt" renders below the prompts here, so no slot there
        prompts = next(node for node in schema["core"]["layout"]["children"] if any(child.get("name") == "prompt" for child in node.get("children", [])))
        assert prompts["children"][0] == {"type": "slot", "section": "prompt_before"}
        assert prompts["children"][1]["name"] == "prompt"
        assert {"type": "slot", "section": "prompt"} not in prompts["children"]


def test_arguments_stay_in_the_prompt_category(env):
    """The slot moves the panel only: no new UI category, and the positional
    arguments keep their legacy place (section "prompt" decides it, which
    positional /sdapi clients rely on)."""
    from modules import shared_items

    categories = list(shared_items.ui_reorder_categories())
    assert "prompt" in categories and "prompt_before" not in categories
    order = env.webui.txt2img.slot_order()
    anchor = next(index for index, script in enumerate(order) if script.title() == "Prompt Anchor")
    seed = next(index for index, script in enumerate(order) if script.section == "seed")
    assert categories.index("prompt") < categories.index("seed") and anchor < seed


def test_localized(env):
    script, layout = env.kit.layout_of(env.module.PromptAnchorScript, "txt2img")
    settings = {"Prompt Anchor", "Default separator between anchor and main prompt", "Start the anchor box collapsed on page load"}
    env.kit.assert_localized(EXTENSION, env.kit.schema_strings(layout.schema()), extra={script.title(), *settings})


def test_standalone(ext_kit):
    ext_kit.assert_standalone(EXTENSION, [SCRIPT])
