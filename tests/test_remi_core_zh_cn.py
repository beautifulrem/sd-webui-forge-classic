"""The core Simplified Chinese catalog (``localizations/zh_CN.json``) covers
every core UI string: the generation forms and built-in processing scripts,
every setting (categories, sections, labels, infos, notes, links, choices) and
the server's translatable messages (tests/remi_core_strings.py)."""

import json
import re
from pathlib import Path

import pytest

from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "localizations" / "zh_CN.json"


def _load(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strings_module = _load("remi_core_strings")


@pytest.fixture(scope="module")
def strings():
    with webui_state() as state:
        yield strings_module.core_strings(state.opts)


@pytest.fixture(scope="module")
def catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _placeholders(text: str) -> list[str]:
    return sorted(re.findall(r"\{\{(\w+)\}\}", text))


def test_the_collector_sees_the_whole_ui(strings):
    # forms, settings and messages all contribute
    assert {"Hires. fix", "Denoising strength", "Stable Diffusion", "Saving Images/Grids", "This task is already running"} <= strings
    assert len(strings) > 500
    # data is left out: sampler and setting names, the choices of DATA_CHOICES
    assert "Euler a" not in strings and "samples_format" not in strings and not {"png", "veryfast", "gradio/soft"} & strings
    # lowercase choice labels elsewhere are UI text ("Masked content")
    assert {"fill", "original"} <= strings


def test_every_core_string_is_translated(strings, catalog):
    spa = strings_module.spa_catalog()
    missing = sorted(text for text in strings if not (catalog.get(text) or "").strip() and text not in spa)
    assert missing == [], f"localizations/zh_CN.json lacks {len(missing)} core strings: {missing[:40]}"


def test_entries_are_well_formed(catalog):
    assert catalog and all(isinstance(key, str) and isinstance(value, str) and value.strip() for key, value in catalog.items())
    broken = [key for key, value in catalog.items() if _placeholders(key) != _placeholders(value)]
    assert broken == []


def test_terms_follow_the_glossary(catalog):
    """Consistent terminology for the core concepts."""

    glossary = {
        "Seed": "种子",
        "Sampler": "采样器",
        "Sampling Method": "采样器",
        "Schedule type": "调度器",
        "CFG Scale": "提示词相关性 CFG",
        "Hires. fix": "高分辨率修复",
        "Denoising strength": "重绘幅度",
        "Batch Count": "生成批次",
        "Batch Size": "每批数量",
        "Extra Networks": "附加网络",
        "Infotext": "生成信息",
    }
    assert {key: catalog.get(key) for key in glossary} == glossary
    # no half-translated core terms left behind
    leftovers = [key for key, value in catalog.items() if re.search(r"Denoising 强度|负向提示词|Hires 步数|Batch 大小", value)]
    assert leftovers == []


def test_the_spa_catalog_is_not_contradicted(catalog):
    """The SPA's own catalog wins at runtime; the core one must agree with it."""

    spa = strings_module.spa_catalog()
    assert sorted(key for key, value in catalog.items() if key in spa and spa[key] != value) == []


def test_the_language_pack_does_not_shadow_the_core_catalog(catalog):
    """Extension catalogs are merged after the core one: the language pack
    keeps only texts the core catalog does not translate."""

    pack = json.loads((ROOT / "extensions-builtin" / "sd-webui-language-diffusion" / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    assert sorted(set(pack) & set(catalog)) == []


def test_extension_catalogs_agree_with_the_core_catalog(catalog):
    """Extension catalogs are merged after the core one and win: a different
    translation of a core text would rename that text everywhere."""

    conflicts = {}
    for path in sorted((ROOT / "extensions-builtin").glob("*/localizations/zh_CN.json")):
        own = json.loads(path.read_text(encoding="utf-8"))
        conflicts |= {f"{path.parts[-3]}: {key}": (value, catalog[key]) for key, value in own.items() if key in catalog and catalog[key] != value}
    assert conflicts == {}
