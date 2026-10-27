"""The JSON Builder page (upstream ``sd_forge_json_builder``, a JavaScript-only
``on_ui_tabs`` tab): registered with ``remi.pages`` as a dedicated SPA view,
Gradio-free, with no JavaScript or CSS left in the extension, its title
translated, and the view it names present in the SPA.

The editor itself (compile / decompile, canvas, send to txt2img / img2img) is
tested by vitest: ``webui/src/pages/custom/json-builder/*.test.ts(x)``."""

import importlib.util
import json
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parents[1]
REPO = EXT.parents[1]
SCRIPT = EXT / "scripts" / "builder.py"
VIEW = REPO / "webui" / "src" / "pages" / "custom" / "json-builder"

_spec = importlib.util.spec_from_file_location("remi_page_harness", REPO / "tests" / "remi_page_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


@pytest.fixture
def module(webui):
    from remi import pages

    pages.clear()
    yield harness.load_page_script(SCRIPT, "json_builder_under_test")
    pages.clear()


def test_the_script_registers_a_widget_page(module):
    from remi import pages

    page = pages.get("json-builder")
    assert page is not None and module.PAGE_ID == "json-builder"
    assert (page.title, page.icon, page.widget) == ("JSON Builder", "braces", "json-builder")
    assert page.params() is None and page.owner == "page:json-builder"
    schema = page.schema()
    assert schema["widget"] == "json-builder" and schema["defaults"] == {}
    assert "send_to" not in schema and "on_load" not in schema
    assert [p.id for p in pages.registered()] == ["json-builder"]


def test_the_page_follows_the_tab_settings(module, webui):
    from remi import pages

    webui.opts.data["hidden_tabs"] = ["JSON Builder"]
    try:
        assert pages.registered() == [] and "JSON Builder" in pages.tab_titles()
    finally:
        webui.opts.data["hidden_tabs"] = []


def test_the_page_goes_with_its_scripts(module):
    from modules import script_callbacks
    from remi import pages

    script_callbacks.script_unloaded_callback()
    assert pages.get("json-builder") is None


def test_no_gradio_javascript_or_css_is_left():
    assert sorted(path.relative_to(EXT).as_posix() for path in EXT.rglob("*") if path.suffix in {".js", ".css", ".html"}) == []
    assert not (EXT / "javascript").exists() and not (EXT / "style.css").exists()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "gradio" not in source and "on_ui_tabs" not in source


def test_the_spa_has_the_view_and_the_icon():
    # pages/ext/index.tsx finds dedicated views at src/pages/custom/<widget>/index.tsx
    assert (VIEW / "index.tsx").is_file()
    assert "braces: Braces" in (REPO / "webui" / "src" / "app" / "icons.ts").read_text(encoding="utf-8")


def test_zh_cn_translates_the_title():
    catalog = json.loads((EXT / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    assert catalog["JSON Builder"].strip() and catalog["JSON Builder"] != "JSON Builder"
