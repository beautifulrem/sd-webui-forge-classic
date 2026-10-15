"""Extra options: the chosen settings as inputs, applied as override settings.

``fixtures/legacy_ui_<tab>.json`` record the Gradio panel with the options
below (``CLIP_stop_at_last_layers`` and ``sd_vae`` are refused by the panel).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "extra_options_section.py"

OPTIONS = {
    "extra_options_txt2img": ["CLIP_stop_at_last_layers", "randn_source", "eta_ancestral", "tiling", "eta_noise_seed_delta", "emphasis"],
    "extra_options_img2img": ["initial_noise_multiplier", "img2img_color_correction", "sd_vae", "img2img_extra_noise"],
    "extra_options_cols": 2,
}


@pytest.fixture(scope="module")
def env(ext_kit):
    from modules import scripts
    from remi.testing import webui_state

    with webui_state():
        module = ext_kit.load_script(SCRIPT, "extra_options_under_test")
    data = scripts.ScriptClassData(module.ExtraOptionsSection, str(SCRIPT), str(EXTENSION), module)
    with webui_state({key: list(value) if isinstance(value, list) else value for key, value in OPTIONS.items()}, script_classes=[data]) as webui:
        yield SimpleNamespace(kit=ext_kit, module=module, webui=webui)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_matches_the_legacy_panel(env, tab):
    _script, layout = env.kit.layout_of(env.module.ExtraOptionsSection, tab)
    # Intended: a free number has no step (Gradio's Number defaulted to 1)
    changes = {3: {"step": None}} if tab == "txt2img" else {}
    env.kit.assert_matches_snapshot(layout, env.kit.load_snapshot(SCRIPT, tab), changes)


def test_defaults_follow_the_current_settings(env):
    _script, layout = env.kit.layout_of(env.module.ExtraOptionsSection, "txt2img")
    env.webui.opts.data["eta_ancestral"] = 0.25
    try:
        assert layout.defaults()["eta_ancestral"] == 0.25
    finally:
        env.webui.opts.data["eta_ancestral"] = 1.0


def test_values_are_not_drafted(env):
    """Gradio reset each control to the setting on every page load: the
    inputs stay out of form drafts, so the current setting is the start value."""

    _script, layout = env.kit.layout_of(env.module.ExtraOptionsSection, "txt2img")
    assert all(node.persist is False and node.submit for node in layout.inputs)
    rows = layout.schema()["children"]
    assert all(node["persist"] is False and "submit" not in node for row in rows for node in row["children"])


def test_layout_follows_the_column_and_accordion_options(env):
    _script, layout = env.kit.layout_of(env.module.ExtraOptionsSection, "txt2img")
    root = layout.schema()
    assert root["type"] == "group" and [len(row["children"]) for row in root["children"]] == [2, 2, 1]
    env.webui.opts.data.update(extra_options_accordion=True, extra_options_cols=3)
    try:
        _script, layout = env.kit.layout_of(env.module.ExtraOptionsSection, "txt2img")
        root = layout.schema()
        assert root["type"] == "section" and root["title"] == "Options" and root["open"] is False
        assert [len(row["children"]) for row in root["children"]] == [3, 2]
    finally:
        env.webui.opts.data.update(extra_options_accordion=False, extra_options_cols=2)


def test_no_settings_no_panel(env):
    env.webui.opts.data["extra_options_txt2img"] = ["sd_vae", "no_such_setting"]
    try:
        script, layout = env.kit.layout_of(env.module.ExtraOptionsSection, "txt2img")
        assert layout.root is None and script.setting_names == []
    finally:
        env.webui.opts.data["extra_options_txt2img"] = list(OPTIONS["extra_options_txt2img"])


def test_values_become_override_settings(env):
    script, layout = env.kit.layout_of(env.module.ExtraOptionsSection, "txt2img")
    assert script.setting_names == ["randn_source", "eta_ancestral", "tiling", "eta_noise_seed_delta", "emphasis"]
    args = layout.values_to_args({"randn_source": "GPU", "tiling": True, "eta_noise_seed_delta": 31337})
    p = SimpleNamespace(override_settings={"emphasis": "None"})
    script.before_process(p, *args)
    # A setting the request already overrides (e.g. pasted) keeps that value
    assert p.override_settings == {"emphasis": "None", "randn_source": "GPU", "eta_ancestral": 1.0, "tiling": True, "eta_noise_seed_delta": 31337}


def test_schema_export(env):
    from remi.core.tabs import tab_schema
    from remi.ui.schema import dumps

    for tab, names in (("txt2img", 5), ("img2img", 3)):
        schema = json.loads(dumps(tab_schema(tab)))
        entry = next(script for script in schema["scripts"] if script["id"] == "extra_options")
        assert entry["section"] == "extra_options" and len(entry["args"]) == names
        # The script's own category gets its slot in the core form
        slots = []
        stack = [schema["core"]["layout"]]
        while stack:
            node = stack.pop()
            if node.get("type") == "slot":
                slots.append(node["section"])
            stack.extend(node.get("children", []))
        assert "extra_options" in slots


def test_settings_are_remi_widgets(env):
    labels = env.webui.opts.data_labels
    assert labels["extra_options_txt2img"].component.__name__ == "MultiChoice"
    assert labels["extra_options_cols"].component.__name__ == "Slider"


def test_localized(env):
    script, _layout = env.kit.layout_of(env.module.ExtraOptionsSection, "txt2img")
    labels = env.webui.opts.data_labels
    # The panel's own strings and its settings; the chosen settings' labels are the core catalog's
    own = {"Options", script.title(), "Settings in UI", labels["settings_in_ui"].default}
    for key in ("extra_options_txt2img", "extra_options_img2img", "extra_options_cols", "extra_options_accordion"):
        own.add(labels[key].label)
        own.update(str(note) for note in getattr(labels[key], "notes", ()))
    env.kit.assert_localized(EXTENSION, own)


def test_standalone(ext_kit):
    ext_kit.assert_standalone(EXTENSION, [SCRIPT], options=OPTIONS)
