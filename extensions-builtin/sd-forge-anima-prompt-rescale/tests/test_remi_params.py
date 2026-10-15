"""The rescaler's params() against the legacy Gradio ui(), its schema, the
manual tool's action and the processing hook."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import remi_script_harness as harness

from remi import ui

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = harness.load_script(ROOT / "scripts" / "anima_prompt_rescale.py", "anima_prompt_rescale_under_test")


def _layout(tab):
    return harness.script_layout(SCRIPT.AnimaPromptRescaleScript(), tab)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_match_the_legacy_ui(tab):
    # The three auto-apply args are the legacy ones, unchanged; the manual
    # tool's inputs (not script args in Gradio either: a click handler read
    # them) are tool inputs (arg=False), rendered and read by its actions only.
    harness.assert_matches_legacy(_layout(tab), ROOT, tab)
    legacy_tool = [(control["legacy_control"], control["label"], control["value"]) for control in harness.legacy_snapshot(ROOT, tab)["non_arg_controls"] if control["legacy_control"] in ("Textbox", "Slider", "Checkbox")]
    layout = _layout(tab)
    assert [node.name for node in layout.tools] == list(SCRIPT.MANUAL_PARAMS) and layout.names == list(SCRIPT.AUTO_PARAMS)
    tool = [(node.type, node.label, node.resolve_default(ui.Context())) for node in (layout.by_name[name] for name in SCRIPT.MANUAL_PARAMS)]
    kinds = {"Textbox": "text", "Slider": "slider", "Checkbox": "toggle"}
    # The tool's prompt box and its result line (an output now), then the same inputs
    assert [(kinds[kind], label, "" if value is None else value) for kind, label, value in legacy_tool if label != "Rescaled prompt"] == tool


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_auto_apply_is_shown_on_img2img_only(tab):
    layout = _layout(tab)
    assert [layout.by_name[name].hidden for name in SCRIPT.AUTO_PARAMS] == [tab == "txt2img"] * 3
    texts = [node.text for node in layout.root.walk() if isinstance(node, ui.Markdown)]
    assert (SCRIPT._AUTO_INFO in texts) is (tab == "img2img")


def test_apply_preset_turns_auto_apply_off_at_its_defaults():
    harness.assert_registers_preset(_layout("img2img"), "prompt_rescale", unowned=SCRIPT.MANUAL_PARAMS)


def test_schema_export():
    schema = harness.schema_json(_layout("img2img"))
    assert schema["type"] == "section" and "enable" not in schema
    nodes = []

    def walk(node):
        nodes.append(node)
        for child in node.get("children", []):
            walk(child)

    walk(schema)
    assert {"type": "button", "label": "Rescale schedule", "action": "rescale", "variant": "primary"} in nodes
    assert {"type": "button", "label": "Use as prompt", "action": "use_as_prompt", "variant": "secondary"} in nodes
    assert {"type": "output", "name": "rescaled", "kind": "text", "label": "Rescaled prompt"} in nodes


@pytest.mark.parametrize(("fixed_steps", "expected"), [(False, "[@guweiz:@wlop:7], masterpiece"), (True, "[@guweiz:@wlop:15], masterpiece")])
def test_manual_tool_action(monkeypatch, fixed_steps, expected):
    monkeypatch.setattr(SCRIPT.shared, "opts", SimpleNamespace(img2img_fix_steps=fixed_steps))
    script = SCRIPT.AnimaPromptRescaleScript()
    values = {"manual_prompt": "[@guweiz:@wlop:15], masterpiece", "manual_source": 35, "manual_target": 35, "manual_denoise": 0.5, "manual_fraction": False, "auto_enable": True}
    patch = ui.call_action(script, "rescale", ui.ActionContext(tab="img2img"), values)
    assert patch == {"__outputs__": {"rescaled": expected}}
    # Values as the form sends them (strings, missing ones) are coerced like the form's
    patch = ui.call_action(script, "rescale", ui.ActionContext(tab="img2img"), {"manual_prompt": "[a:b:15]", "manual_denoise": "0.5"})
    assert patch["__outputs__"]["rescaled"] == ("[a:b:15]" if fixed_steps else "[a:b:7]")


def test_use_as_prompt_writes_the_result_into_the_prompt_box(monkeypatch):
    # The legacy result box had a copy button; the action patches the core prompt
    monkeypatch.setattr(SCRIPT.shared, "opts", SimpleNamespace(img2img_fix_steps=False))
    values = {"manual_prompt": "[@guweiz:@wlop:15], masterpiece", "manual_denoise": 0.5}
    patch = ui.call_action(SCRIPT.AnimaPromptRescaleScript(), "use_as_prompt", ui.ActionContext(tab="img2img"), values)
    assert patch == {"core.prompt": {"value": "[@guweiz:@wlop:7], masterpiece"}, "__outputs__": {"rescaled": "[@guweiz:@wlop:7], masterpiece"}}


def _img2img_process():
    return SimpleNamespace(
        init_images=[object()],
        steps=35,
        denoising_strength=0.5,
        prompt="[a:b:15]",
        negative_prompt="[x:y:10]",
        all_prompts=["[a:b:15]"],
        all_negative_prompts=["[x:y:10]"],
        extra_generation_params={},
    )


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_process_reads_the_auto_args_and_ignores_the_tool(monkeypatch, tab):
    monkeypatch.setattr(SCRIPT.shared, "opts", SimpleNamespace(img2img_fix_steps=False))
    script = SCRIPT.AnimaPromptRescaleScript()
    layout = harness.script_layout(script, tab)
    # A tool input sent along (by an API caller) is accepted and is not an argument
    args = layout.values_to_args({"auto_enable": True, "manual_prompt": "[c:d:3]"})
    assert len(args) == len(SCRIPT.AUTO_PARAMS)
    p = _img2img_process()
    script.process(p, *args)
    if tab == "img2img":
        assert (p.all_prompts, p.all_negative_prompts, p.prompt) == (["[a:b:7]"], ["[x:y:5]"], "[a:b:7]")
        assert p.extra_generation_params == {"Anima schedule rescale": "src35->tgt35@d0.5"}
    else:
        assert p.all_prompts == ["[a:b:15]"] and p.extra_generation_params == {}
    # Jobs queued by older Remi builds also passed the tool's five values
    p = _img2img_process()
    script.process(p, *args, "[c:d:3]", 35, 35, 0.5, False)
    assert p.all_prompts == (["[a:b:7]"] if tab == "img2img" else ["[a:b:15]"])


def test_standalone():
    harness.assert_standalone(ROOT, ("scripts/anima_prompt_rescale.py",))


def test_zh_cn_catalog_covers_the_panel():
    harness.assert_localized(ROOT, [_layout("txt2img"), _layout("img2img")])
