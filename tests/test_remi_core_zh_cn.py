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
    assert "Euler a" not in strings and "samples_format" not in strings and not {"png", "veryfast"} & strings
    # lowercase choice labels elsewhere are UI text ("Masked content")
    assert {"fill", "original"} <= strings
    # settings help markup (keyed by its exact text, as the SPA looks it up) and the Callbacks section
    assert any(text.startswith("These settings allow you to enable PyTorch profiler") for text in strings)
    assert {"Callbacks", "For categories below, callbacks added to dropdowns happen before others, in order listed."} <= strings
    assert not any(text.startswith(("prioritized_callbacks_", "base/")) for text in strings)


def test_the_callbacks_section_keeps_the_vae_explanation():
    """The Callbacks section's help once reused the key ``sd_vae_explanation``
    and replaced the VAE section's explanation."""

    from modules import shared_items, shared_options

    with webui_state():
        callbacks = shared_items.callbacks_order_settings()
    assert not set(callbacks) & set(shared_options.options_templates)


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


# Everyday UI words that have an established Chinese term (步数, 卸载, 分块,
# 批次, 种子, 提示词, 模型, 局部重绘, ...): left in English inside a Chinese
# sentence they make a mixed-language label, not a technical name. Syntax
# (``blocks=``), identifiers (``positive_prompt``), paths and `code` are exempt.
UNTRANSLATED_WORDS = frozenset(
    {
        *("step", "steps", "offload", "offloaded", "tiled", "tile", "tiles", "batch", "batches", "seed", "seeds"),
        *("prompt", "prompts", "checkpoint", "checkpoints", "sampler", "samplers", "scheduler", "schedulers"),
        *("inpaint", "inpainted", "inpainting", "sketch", "hires", "detect", "detected", "detector", "detectors"),
        *("detection", "encoder", "encoders", "outputs", "profiler", "epoch", "epochs", "workflow", "block"),
        *("blocks", "mask", "masks", "strength", "upscaler", "denoise", "padding", "noise", "multiplier"),
        *("enabled", "disabled", "image", "images", "width", "height", "infotext", "refiner"),
    }
)
_WORD = re.compile(r"(?<![\w=./\\\-])([A-Za-z][a-z]*)(?![\w=./\\\-])")
# Not labels: placeholders, URLs, `code`, markup, regular expressions (``^...$``), names of other
# products and models that contain a common word ("Prompt Queue", "Wan 2.2 High Noise"), a name
# kept with its gloss ("Tile (分块)", ControlNet's control type) and the English term given after
# its translation ("生成信息（Infotext）").
_EXEMPT = re.compile(
    r"\{\{?\w+\}?\}|https?://\S+|`[^`]*`|<[^>]+>|\^\S*\$"
    r"|Agent Scheduler|Prompt (?:Queue|Workshop)|Qwen Image|Image Browser|(?:High|Low) Noise"
    r"|\b[A-Z][a-z]+ \([一-鿿]+\)|(?<=[一-鿿])（[A-Za-z][A-Za-z .]*）"
)


def _mixed_words(value: str) -> set[str]:
    """The common English words (also capitalised: "Step", "Tile") left in a Chinese text."""

    return {word for word in _WORD.findall(_EXEMPT.sub(" ", value)) if word.lower() in UNTRANSLATED_WORDS}


def _zh_catalogs():
    yield "localizations/zh_CN.json", CATALOG
    yield from ((str(path.relative_to(ROOT)), path) for path in sorted((ROOT / "extensions-builtin").glob("*/localizations/zh_CN.json")))
    yield from ((str(path.relative_to(ROOT)), path) for path in sorted((ROOT / "webui" / "src").rglob("zh-CN.json")))


def test_no_mixed_language_labels_in_any_catalog():
    """zh-CN keeps English only for technical names (LoRA, VAE, UNet, sigma,
    bbox, ...), never for common words such as step, offload or tiled."""

    mixed = []
    for name, path in _zh_catalogs():
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if not isinstance(value, str) or not re.search(r"[一-鿿]", value):
                continue
            words = _mixed_words(value)
            if words:
                mixed.append(f"{name}: {key[:60]!r} -> {sorted(words)}")
    assert mixed == [], "\n".join(mixed)


# Kept in English on purpose: technical names that read the same in the
# Chinese UI (parameter names of a paper or library, prompt examples).
KEPT_IN_ENGLISH = frozenset({"Dense Block", "sigma noise", "retro artstyle, sketch, watercolor, flat color, 1990s (style)"})


def test_untranslated_entries_are_technical_names():
    """An entry left in English ("Block {{index}}": "Block {{index}}") is
    either a technical name listed above or contains no common word."""

    left = []
    for name, path in _zh_catalogs():
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if isinstance(value, str) and not re.search(r"[一-鿿]", value) and value not in KEPT_IN_ENGLISH and _mixed_words(value):
                left.append(f"{name}: {key[:60]!r} -> {value[:60]!r}")
    assert left == [], "\n".join(left)


# One Chinese term per concept in every catalog (core, extensions, SPA, widgets).
SHARED_TERMS = {"Hires. fix": ("高分辨率修复", "高清修复"), "Refiner": ("精炼器", "Refiner")}


def test_every_catalog_uses_the_shared_terms():
    stray = []
    for name, path in _zh_catalogs():
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if not isinstance(value, str) or not re.search(r"[一-鿿]", value):
                continue
            for _concept, (_term, wrong) in SHARED_TERMS.items():
                if wrong in _EXEMPT.sub(" ", value):
                    stray.append(f"{name}: {key[:60]!r} uses {wrong!r}")
    assert stray == [], "\n".join(stray)


def test_the_glossary_check_catches_mixed_labels():
    flagged = _mixed_words
    assert flagged("停止缓存 step") == {"step"}
    assert flagged("为 UNet 启用(始终 offload)") == {"offload"}
    assert flagged("为 VAE 启用(始终 tiled)") == {"tiled"}
    # capitalised common words are no technical names either
    assert flagged("绕过 RadialAttention 的最后几个 Step 的数量") == {"Step"}
    assert flagged("Tile Batch 大小") == {"Tile", "Batch"}
    assert flagged("将 Lora 哈希值添加到 infotext") == {"infotext"}
    assert flagged("生成信息（Infotext）是 WebUI 对包含生成参数的文本的称呼") == set()
    # syntax, identifiers, code, placeholders, regular expressions and product names are not labels
    assert flagged("每行一个预设：名称=blocks=...;modules=...") == set()
    assert flagged("格式：classname: positive_prompt | negative_prompt") == set()
    assert flagged("示例 `<lora:name:blocks=0-9=0.8>`，共 {{steps}} 步") == set()
    assert flagged("regex - 例如 ^(?!.*Hires).*$") == set()
    assert flagged("从 Agent Scheduler 或 Prompt Queue 导入…") == set()
    assert flagged("Tile (分块)") == set()


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
