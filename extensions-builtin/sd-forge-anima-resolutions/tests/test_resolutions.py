"""The resolution helper: the img2img size fit, Randomize, the prompt
history and their actions, and the params() contract against the legacy
Gradio ui() (``fixtures/legacy_ui_<tab>.json``, snapshot before the port)."""

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "anima_resolutions.py"
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("remi_panel_harness", ROOT.parents[1] / "tests" / "remi_panel_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

res = harness.load_script(SCRIPT, "anima_resolutions_under_test")


@pytest.fixture(autouse=True)
def empty_history():
    res._prompt_history.clear()
    res._last_job.clear()
    yield
    res._prompt_history.clear()
    res._last_job.clear()


def _script(tab="txt2img"):
    script = res.AnimaRandomizeResolutionScript()
    harness.script_layout(script, tab)  # sets the tab attributes
    return script


def _ctx(core=None, user=None, **extra):
    from remi import ui

    return ui.ActionContext(tab="img2img", owner="anima_random_resolution", core=core or {}, user=user, extra=extra)


# region params()


def test_txt2img_params_keep_the_legacy_contract():
    legacy = harness.load_fixture(ROOT / "tests" / "fixtures" / "legacy_ui_txt2img.json")
    layout = harness.script_layout(res.AnimaRandomizeResolutionScript(), "txt2img")
    # Intentional: the preset picker (a widget) and the history pick were
    # Gradio-only controls; as params they follow the two processing arguments
    assert harness.contract_differences(legacy, layout) == {"presets": {"added": "custom"}, "history": {"added": "choice"}}
    assert layout.names == ["randomize", "randomize_highres", "presets", "history"]
    assert {node.preset_key for node in layout.inputs if node.preset_key} == {"resolution.random_standard", "resolution.random_highres"}


def test_img2img_gains_the_source_size_fit():
    legacy = harness.load_fixture(ROOT / "tests" / "fixtures" / "legacy_ui_img2img.json")
    assert legacy == {"shown": False, "inputs": []}
    layout = harness.script_layout(res.AnimaRandomizeResolutionScript(), "img2img")
    # Intentional: the Send-to auto-resize became this widget on img2img
    assert harness.contract_differences(legacy, layout) == {"fit_source": {"added": "custom"}}


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_schema_export(tab):
    layout = harness.script_layout(res.AnimaRandomizeResolutionScript(), tab)
    schema = layout.schema()
    json.dumps(schema, allow_nan=False)
    assert schema["type"] == "group" and schema["title"] == "Anima resolution helper"
    widgets = {node["name"]: node for node in _walk(schema) if node.get("type") == "custom"}
    if tab == "img2img":
        assert widgets["fit_source"]["widget"] == "source-size-fit" and widgets["fit_source"]["default"] is True
        assert widgets["fit_source"]["props"] == {"source": "core.init_img", "action": "fit_to_source", "manual_action": "fit_to_source_now"}
        return
    props = widgets["presets"]["props"]
    assert widgets["presets"]["widget"] == "resolution-presets"
    assert props["standard"] == [[w, h] for w, h in res.RESOLUTIONS]
    assert props["highres"] == [[w, h] for w, h in res.HIGH_RES_RESOLUTIONS]
    assert (props["width"], props["height"]) == ("core.width", "core.height")
    history = next(node for node in _walk(schema) if node.get("name") == "history")
    assert history["on_change"] == "use_history" and history["allow_custom"] and history["default"] == ""
    assert any(node.get("action") == "refresh_history" for node in _walk(schema))


def _walk(node):
    yield node
    for child in node.get("children", []):
        yield from _walk(child)


def test_the_processing_arguments_come_first():
    hook = inspect.signature(res.AnimaRandomizeResolutionScript.before_process)
    assert list(hook.parameters)[2:4] == ["randomize_enabled", "randomize_highres_enabled"]
    layout = harness.script_layout(res.AnimaRandomizeResolutionScript())
    assert layout.values_to_args({}) == [False, False, None, ""]
    # A stale history pick (from an earlier session) still validates
    assert layout.values_to_args({"history": "a prompt nobody logged"})[3] == "a prompt nobody logged"


def test_every_string_has_a_zh_cn_translation():
    layouts = [harness.script_layout(res.AnimaRandomizeResolutionScript(), tab) for tab in ("txt2img", "img2img")]
    # The history's choices are the user's prompts
    harness.assert_localized(layouts, ROOT / "localizations" / "zh_CN.json", skip_choices=("history",), extra=("Anima random resolution",))


def test_the_extension_imports_no_gradio():
    harness.assert_gradio_free(SCRIPT)
    source = SCRIPT.read_text(encoding="utf8")
    assert "on_after_component" not in source and "print(" not in source


# endregion

# region img2img size fit


def test_upscale_targets_match_the_legacy_adjuster():
    legacy = harness.load_fixture(ROOT / "tests" / "fixtures" / "legacy_upscale_targets.json")["targets"]
    assert len(legacy) > 250
    for width, height, target in legacy:
        assert res.upscale_target(width, height) == (None if target is None else tuple(target)), (width, height)


@pytest.mark.parametrize("width, height", [(0, 512), (512, -1), ("x", 512), (None, None)])
def test_upscale_target_rejects_bad_sizes(width, height):
    assert res.upscale_target(width, height) is None


def test_every_target_is_in_the_window():
    for width, height in res.RESOLUTIONS + res.HIGH_RES_RESOLUTIONS:
        tw, th = res.upscale_target(width, height)
        assert tw % 64 == 0 and th % 64 == 0 and tw >= width and th >= height
        assert 2560 <= tw + th <= 3072 and max(tw, th) <= 1856


def test_the_infotext_size_wins_over_the_image_size():
    assert res.parse_size_from_infotext("a cat\nSteps: 20, Size: 832x1216, Hires upscale: 2") == (832, 1216)
    assert res.parse_size_from_infotext("no parameters here") is None
    assert res.parse_size_from_infotext(None) is None
    script = _script("img2img")
    # A Hires. fix output (1664 x 2432) recorded its first pass: the fit starts from it
    patch = script.fit_to_source(_ctx(source={"width": 1664, "height": 2432, "infotext": "x\nSteps: 20, Size: 832x1216"}), {"fit_source": True})
    assert (patch["core.width"]["value"], patch["core.height"]["value"]) == res.upscale_target(832, 1216)
    assert patch["__toast__"]["level"] == "info"
    patch = script.fit_to_source(_ctx(source={"width": 1024, "height": 1024}), {})
    assert (patch["core.width"]["value"], patch["core.height"]["value"]) == res.upscale_target(1024, 1024)


def test_the_source_is_read_from_the_file_store():
    record = SimpleNamespace(is_image=True, infotext="p\nSteps: 20, Size: 1216x832", width=2432, height=1664, path="unused")
    files = SimpleNamespace(get=lambda upload_id: record if upload_id == "out-1" else None)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=SimpleNamespace(files=files))))
    from remi import ui

    ctx = ui.ActionContext(tab="img2img", owner="x", core={"init_img": "out-1"}, request=request)
    assert res.source_size(ctx) == (1216, 832)
    record.infotext = None
    record.path = str(ROOT / "missing.png")  # unreadable: the image size
    assert res.source_size(ctx) == (2432, 1664)
    ctx.core["init_img"] = "unknown"
    assert res.source_size(ctx) is None


def test_the_automatic_fit_is_quiet_and_the_button_explains():
    script = _script("img2img")
    assert script.fit_to_source(_ctx(), {}) == {}
    assert script.fit_to_source_now(_ctx(), {})["__toast__"]["level"] == "warning"
    too_big = {"width": 4096, "height": 4096}
    assert script.fit_to_source(_ctx(source=too_big), {}) == {}
    assert "unchanged" in script.fit_to_source_now(_ctx(source=too_big), {})["__toast__"]["message"]


def test_actions_are_declared_and_return_valid_patches():
    from remi import ui

    script = _script("img2img")
    assert set(ui.actions_of(script)) == {"use_history", "refresh_history", "fit_to_source", "fit_to_source_now"}
    patch = ui.call_action(script, "fit_to_source", _ctx(source={"width": 832, "height": 1216}), {})
    assert set(patch) == {"core.width", "core.height", "__toast__"}


# endregion

# region Randomize and history


def _txt2img_process(prompt="a cat"):
    return SimpleNamespace(width=1024, height=1024, prompt=prompt)


@pytest.mark.parametrize(
    "standard, highres, pool",
    [(True, False, "RESOLUTIONS"), (False, True, "HIGH_RES_RESOLUTIONS"), (True, True, "both")],
)
def test_randomize_rolls_from_its_pool(monkeypatch, standard, highres, pool):
    rolled = []
    monkeypatch.setattr(res.random, "choice", lambda items: rolled.append(items) or items[-1])
    p = _txt2img_process()
    _script().before_process(p, standard, highres, None, "")
    expected = res.RESOLUTIONS + res.HIGH_RES_RESOLUTIONS if pool == "both" else getattr(res, pool)
    assert rolled == [expected]
    assert (p.width, p.height) == expected[-1]


def test_randomize_off_and_img2img_leave_the_size():
    p = _txt2img_process()
    _script().before_process(p, False, False, None, "")
    assert (p.width, p.height) == (1024, 1024)
    p = SimpleNamespace(width=512, height=512, prompt="x", init_images=[object()])
    _script("img2img").before_process(p, True)  # img2img's only param is the size fit
    assert (p.width, p.height) == (512, 512) and res.history_entries() == ["a cat"]


def test_prompt_history_is_a_capped_most_recent_first_ring():
    script = _script()
    for index in range(res.PROMPT_HISTORY_MAX + 5):
        script.before_process(_txt2img_process(f"prompt {index}"), False, False)
    script.before_process(_txt2img_process("  prompt 50  "), False, False)
    script.before_process(_txt2img_process("   "), False, False)
    history = res.history_entries()
    assert len(history) == res.PROMPT_HISTORY_MAX
    assert history[:2] == ["prompt 50", f"prompt {res.PROMPT_HISTORY_MAX + 4}"]
    assert history.count("prompt 50") == 1 and "prompt 4" not in history


def test_prompt_history_is_per_user_and_logs_a_job_once(monkeypatch):
    script = _script()
    job = ["job-1"]
    monkeypatch.setattr(res, "_running_job", lambda: job[0])
    # An X/Y/Z or Prompt S/R job: the cells after the first add nothing
    for prompt in ("a cat", "a dog", "a fox"):
        script.before_process(SimpleNamespace(width=1024, height=1024, prompt=prompt, user="alice"), False, False)
    script.before_process(SimpleNamespace(width=1024, height=1024, prompt="a bird", user="bob"), False, False)
    job[0] = "job-2"
    script.before_process(SimpleNamespace(width=1024, height=1024, prompt="a cow", user="alice"), False, False)
    assert res.history_entries("alice") == ["a cow", "a cat"]
    assert res.history_entries("bob") == ["a bird"]
    assert res.history_entries() == []
    # Each user refreshes and picks from their own list only
    assert [value for _label, value in script.refresh_history(_ctx(user="bob"), {})["history"]["choices"]] == ["a bird"]
    assert script.use_history(_ctx(user="bob"), {"history": "a cat"}) == {"history": {"value": ""}}
    assert script.use_history(_ctx(user="alice"), {"history": "a cat"})["core.prompt"] == {"value": "a cat"}


def test_history_actions():
    script = _script()
    long_prompt = "masterpiece,\n  a cat " + "x" * 300
    res.remember_prompt(long_prompt)
    res.remember_prompt("a dog")
    patch = script.refresh_history(_ctx(), {"history": ""})
    choices = patch["history"]["choices"]
    assert [value for _label, value in choices] == ["a dog", long_prompt.strip()]
    label = choices[1][0]
    assert len(label) == res.PROMPT_PREVIEW_CHARS and label.endswith("…") and "\n" not in label
    assert patch["history"]["value"] == ""
    assert script.use_history(_ctx(), {"history": "a dog"}) == {"core.prompt": {"value": "a dog"}, "history": {"value": ""}}
    assert script.use_history(_ctx(), {"history": ""}) == {}
    # Text typed into the field is not a pick: it does not replace the prompt
    assert script.use_history(_ctx(), {"history": "typed text"}) == {"history": {"value": ""}}
    # The schema lists the history as it is when the form loads
    history = harness.script_layout(res.AnimaRandomizeResolutionScript()).by_name["history"]
    from remi import ui

    assert [value for _label, value in history.resolve_choices(ui.Context())] == ["a dog", long_prompt.strip()]


# endregion
