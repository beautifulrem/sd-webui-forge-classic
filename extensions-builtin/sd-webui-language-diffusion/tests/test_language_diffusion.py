"""sd-webui-language-diffusion in Remi: the catalogs are served by
/api/v2/i18n, the language switcher has a native name for each, the
``localization`` setting is presented with autoglottonyms, and nothing
Gradio-era is left."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parents[1]
REPO = EXT.parents[1]
SCRIPT = EXT / "scripts" / "language_diffusion_setup.py"
LOCALES = sorted(path.stem for path in (EXT / "localizations").glob("*.json"))

_spec = importlib.util.spec_from_file_location("remi_page_harness", REPO / "tests" / "remi_page_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


def _load():
    spec = importlib.util.spec_from_file_location("language_diffusion_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ten_locales_ship():
    assert LOCALES == ["de_DE", "es_ES", "fr_FR", "it_IT", "ja_JP", "ko_KR", "pl_PL", "pt_BR", "ru_RU", "zh_CN"]


@pytest.mark.parametrize("locale", LOCALES)
def test_catalogs_are_flat_string_maps(locale):
    data = json.loads((EXT / "localizations" / f"{locale}.json").read_text(encoding="utf-8"))
    assert data and all(isinstance(key, str) and isinstance(value, str) for key, value in data.items())
    placeholders = lambda text: sorted(re.findall(r"\{\{(\w+)\}\}", text))  # noqa: E731
    assert [key for key, value in data.items() if placeholders(key) != placeholders(value)] == []


def test_the_switcher_names_every_locale():
    from remi.server.i18n import LOCALE_NAMES

    assert all(LOCALE_NAMES.get(locale) for locale in LOCALES)


def test_catalogs_are_served_merged_over_the_core(webui, monkeypatch, tmp_path):
    from modules import localization
    from remi.server import i18n

    core = REPO / "localizations" / "zh_CN.json"
    monkeypatch.setattr(localization, "localizations", {"zh_CN": [str(core), str(EXT / "localizations" / "zh_CN.json")], "de_DE": [str(EXT / "localizations" / "de_DE.json")]})
    assert i18n.normalize("de-DE") == "de_DE" and {"id": "de_DE", "name": "Deutsch"} in i18n.locales()
    german, _ = i18n.catalog("de_DE")
    assert german == json.loads((EXT / "localizations" / "de_DE.json").read_text(encoding="utf-8"))
    chinese, _ = i18n.catalog("zh_CN")
    core_data = json.loads(core.read_text(encoding="utf-8"))
    # the core terms win: the pack does not repeat them
    assert all(chinese[key] == value for key, value in core_data.items())


def test_the_localization_setting_is_presented_in_each_language(webui):
    info = webui.opts.data_labels["localization"]
    saved = (info.label, info.component_args, info.onchange)
    try:
        module = _load()
        assert info.label == "Language"
        # the same names as the SPA's language switcher
        from remi.server.i18n import LOCALE_NAMES

        named = {code: name for code, name in module.LOCALE_DISPLAY_NAMES.items() if code != "None"}
        assert named == {code: LOCALE_NAMES[code] for code in named}
        choices = info.component_args()["choices"]
        assert choices[0] == ("English", "None")
        module.present_localization_setting()  # idempotent
        assert info.component_args()["choices"][0] == ("English", "None")
        from remi.ui.settings import settings_schema

        items = [item for category in settings_schema(webui.opts)["categories"] for section in category["sections"] for item in section["items"] if item.get("name") == "localization"]
        assert items and items[0]["label"] == "Language" and items[0]["choices"][0] == {"value": "None", "label": "English"}
        assert webui.opts.quicksettings_list == webui.opts.data_labels["quicksettings_list"].default  # no pin any more
    finally:
        info.label, info.component_args, info.onchange = saved


def test_no_tab_code_left():
    harness.assert_page_standalone(EXT, [SCRIPT])
