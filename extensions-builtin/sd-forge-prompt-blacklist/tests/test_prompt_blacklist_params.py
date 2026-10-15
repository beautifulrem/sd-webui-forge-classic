"""Prompt Blacklist's ``params()`` and actions: cleaning patches the core prompt."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "prompt_blacklist.py"


@pytest.fixture(scope="module")
def env(ext_kit):
    from modules import scripts
    from remi.testing import webui_state

    with webui_state():
        module = ext_kit.load_script(SCRIPT, "prompt_blacklist_under_test")
    data = scripts.ScriptClassData(module.PromptBlacklistScript, str(SCRIPT), str(EXTENSION), module)
    with webui_state(script_classes=[data]):
        yield SimpleNamespace(kit=ext_kit, module=module)


@pytest.fixture
def blacklist_file(env, tmp_path, monkeypatch):
    target = tmp_path / "state" / "blacklist.txt"
    monkeypatch.setattr(env.module, "STATE_DIR", str(target.parent))
    monkeypatch.setattr(env.module, "BLACKLIST_FILE", str(target))
    return target


def call(env, name, values, core=None):
    from remi import ui

    script, _layout = env.kit.layout_of(env.module.PromptBlacklistScript, "txt2img")
    return ui.call_action(script, name, ui.ActionContext(tab="txt2img", owner="prompt_blacklist", core=core or {}), values)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_matches_the_gradio_panel(env, tab):
    _script, layout = env.kit.layout_of(env.module.PromptBlacklistScript, tab)
    # Intended: the blacklist box and the duplicate switch were controls the
    # Gradio handlers read without passing them to process(); actions read the
    # owner's values, so they are params now (process() ignores its args).
    # They are action-only (submit=False, persist=False): see test_action_only_inputs.
    added = {
        0: {"kind": "text", "label": "Blacklisted tags", "default": ..., "infotext": None, "paste": False, "preset": None},
        1: {"kind": "toggle", "label": "Also remove duplicate tags", "default": False, "infotext": None, "paste": False, "preset": None},
    }
    env.kit.assert_matches_snapshot(layout, env.kit.load_snapshot(SCRIPT, tab), added)


def test_blacklist_default_is_the_saved_file(env, blacklist_file):
    env.module.save_blacklist("logo, *_username")
    _script, layout = env.kit.layout_of(env.module.PromptBlacklistScript, "txt2img")
    assert layout.defaults()["blacklist"] == "logo, *_username"


def test_clean_patches_the_core_prompt(env):
    patch = call(env, "clean", {"blacklist": "watermark, *_username", "remove_dupes": True}, {"prompt": "1girl, (watermark:1.2), foo_username, 1girl, solo"})
    assert patch["core.prompt"] == {"value": "1girl, solo"}
    assert patch["__outputs__"]["status"] == {
        "text": env.module.MSG_REMOVED_DUPES,
        "params": {"count": 3, "duplicates": 1, "tags": "(watermark:1.2), foo_username, 1girl"},
    }


def test_clean_reports_long_removals_shortened(env):
    prompt = ", ".join(f"tag{i}" for i in range(30))
    patch = call(env, "clean", {"blacklist": "tag*", "remove_dupes": False}, {"prompt": prompt})
    status = patch["__outputs__"]["status"]
    assert status["text"] == env.module.MSG_REMOVED_MORE and status["params"]["count"] == 30 and status["params"]["more"] == 5
    assert status["params"]["tags"] == ", ".join(f"tag{i}" for i in range(25))


def test_clean_without_matches_leaves_the_prompt(env):
    patch = call(env, "clean", {"blacklist": "watermark", "remove_dupes": False}, {"prompt": "1girl, solo"})
    assert "core.prompt" not in patch and patch["__outputs__"]["status"] == {"text": env.module.MSG_UNCHANGED}
    patch = call(env, "clean", {"blacklist": "", "remove_dupes": False}, {"prompt": "1girl"})
    assert "core.prompt" not in patch and patch["__outputs__"]["status"] == {"text": env.module.MSG_EMPTY}


def test_save_and_reload(env, blacklist_file):
    patch = call(env, "save", {"blacklist": " logo, signature \n"})
    assert patch == {"__outputs__": {"status": {"text": "Blacklist saved."}}}
    assert blacklist_file.read_text(encoding="utf-8") == "logo, signature\n"
    patch = call(env, "reload", {"blacklist": "edited"})
    assert patch == {"blacklist": {"value": "logo, signature"}, "__outputs__": {"status": {"text": "Blacklist reloaded from file."}}}


def test_save_failure_is_reported(env, blacklist_file):
    blacklist_file.parent.write_text("a file where the folder should be", encoding="utf-8")
    patch = call(env, "save", {"blacklist": "logo"})
    assert patch["__toast__"] == {"level": "error", "message": env.module.MSG_SAVE_FAILED_TOAST}
    status = patch["__outputs__"]["status"]
    assert status["text"] == env.module.MSG_SAVE_FAILED and status["params"]["error"]


def test_action_only_inputs(env):
    """Neither input is sent with jobs or kept in drafts; processing gets the defaults."""

    _script, layout = env.kit.layout_of(env.module.PromptBlacklistScript, "txt2img")
    for name in ("blacklist", "remove_dupes"):
        node = layout.by_name[name]
        assert not node.submit and not node.persist
    schema = layout.schema()
    flags = {node["name"]: (node.get("submit"), node.get("persist")) for node in schema["children"] if node["type"] in ("text", "toggle")}
    assert flags == {"blacklist": (False, False), "remove_dupes": (False, False)}


def test_schema_export(env):
    from remi.core.tabs import tab_schema
    from remi.ui.schema import dumps

    schema = json.loads(dumps(tab_schema("txt2img")))
    entry = next(script for script in schema["scripts"] if script["id"] == "prompt_blacklist")
    assert entry["section"] == "prompt" and entry["args"] == ["blacklist", "remove_dupes"]
    buttons = [node for node in entry["layout"]["children"] if node["type"] == "row"][0]["children"]
    assert [(button["label"], button["action"]) for button in buttons] == [("Clean Prompt", "clean"), ("Save Blacklist", "save"), ("Reload", "reload")]
    assert {"type": "output", "name": "status", "kind": "text"} in entry["layout"]["children"]


def test_localized(env, blacklist_file):
    script, layout = env.kit.layout_of(env.module.PromptBlacklistScript, "txt2img")
    env.kit.assert_localized(EXTENSION, env.kit.schema_strings(layout.schema()), extra={script.title(), *env.module.MESSAGES})

    # Every status and toast the actions produce is one of those catalog keys
    shown = []
    for name, values, core in [
        ("clean", {"blacklist": "a", "remove_dupes": True}, {"prompt": "a, b, b"}),
        ("clean", {"blacklist": "a", "remove_dupes": False}, {"prompt": "a, b"}),
        ("clean", {"blacklist": "x*", "remove_dupes": False}, {"prompt": ", ".join(f"x{i}" for i in range(30))}),
        ("clean", {"blacklist": "x*", "remove_dupes": True}, {"prompt": ", ".join(f"x{i}" for i in range(30)) + ", b, b"}),
        ("clean", {"blacklist": "zzz", "remove_dupes": False}, {"prompt": "b"}),
        ("clean", {"blacklist": "", "remove_dupes": False}, {"prompt": "b"}),
        ("save", {"blacklist": "a"}, None),
        ("reload", {}, None),
    ]:
        patch = call(env, name, values, core)
        shown.append(patch["__outputs__"]["status"]["text"])
    assert set(shown) - set(env.module.MESSAGES) == set()
    assert len(set(shown)) == len(shown) == 8


def test_gradio_free(ext_kit):
    ext_kit.assert_gradio_free(EXTENSION, [SCRIPT])
