"""The English strings of the core UI the server catalog must translate
(``localizations/<locale>.json``): the generation forms (core params and the
built-in processing scripts), every setting (category, section, label, info,
notes, link labels, choice labels), and the server's translatable messages
(action toasts and texts, page task messages, extension admin errors).

Not a test module: ``tests/test_remi_core_zh_cn.py`` checks the zh_CN
catalog with it, and the CLI lists what a catalog lacks::

    $PYTHON tests/remi_core_strings.py [--locale zh_CN] [--all] [--out list.json]

``$PYTHON``: an interpreter with requirements.txt (no Gradio), as ``tools/check.sh`` picks.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABS = ("txt2img", "img2img", "extras")

# Python sources whose toast / message literals are user-facing server texts
MESSAGE_SOURCES = ("remi", "modules/processing_scripts")
# Calls whose first string argument is a catalog key (ui.toast / ui.message,
# a page task's progress text, an extension admin error's template)
MESSAGE_CALLS = {"toast": 0, "message": 0, "progress": 1, "AdminError": 0, "GitError": 0}


def _text_strings(node, found: set[str]) -> None:
    """Strings of an exported layout / settings item the SPA translates
    (the choices of :data:`DATA_CHOICES` are data, not text)."""

    if isinstance(node, list):
        for item in node:
            _text_strings(item, found)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") != "html":
        for key in ("label", "info", "title", "placeholder"):
            if isinstance(node.get(key), str) and node[key].strip():
                found.add(node[key])
        if node.get("type") == "markdown" and isinstance(node.get("text"), str):
            found.add(node["text"])
    for note in node.get("notes") or ():
        if isinstance(note, str) and note.strip():
            found.add(note)
    for link in node.get("links") or ():
        if isinstance(link, dict) and isinstance(link.get("label"), str):
            found.add(link["label"])
    if (node.get("name") or node.get("key")) not in DATA_CHOICES:
        for choice in node.get("choices") or ():
            if isinstance(choice, dict) and isinstance(choice.get("label"), str) and choice["label"].strip():
                found.add(choice["label"])
    for key in ("children", "enable", "layout", "sections", "items", "categories", "scripts", "core"):
        if key in node:
            _text_strings(node[key], found)


# Params / settings whose choice labels are identifiers, listed one by one
# (a new lowercase label anywhere else is UI text and must be translated)
DATA_CHOICES = frozenset(
    {
        "samples_format",  # file formats
        "grid_format",
        "live_previews_image_format",
        "video_container",  # ffmpeg container, preset and profile names
        "video_preset",
        "video_profile",
        "svdq_attention",  # attention backend ids
        "infotext_skip_pasting",  # infotext keys
        "ui_reorder_list",  # the generation form's section ids
    }
)


def _dynamic_names() -> set[str]:
    """Names that are data, not UI text: samplers, schedulers, upscalers,
    locale ids, UI presets, model and file names offered as choices."""

    from modules import localization, sd_samplers, sd_schedulers, shared
    from modules_forge.presets import PresetArch

    names: set[str] = {"None"}
    for sampler in sd_samplers.all_samplers:
        names.add(sampler.name)
        names.update(sampler.aliases or ())
    for scheduler in sd_schedulers.schedulers:
        names.update((scheduler.name, scheduler.label))
    names.update(upscaler.name for upscaler in shared.sd_upscalers or ())
    names.update(localization.localizations)
    names.update(PresetArch.choices())
    names.update(shared.latent_upscale_modes)
    names.update(shared.opts.data_labels)  # setting keys (quick settings, …)
    names.update(face.name() for face in shared.face_restorers or ())
    return names


def form_strings() -> set[str]:
    """Labels, infos, titles, placeholders, choices and Markdown of the three
    core tab schemas (core form and the built-in processing scripts)."""

    from remi.core.tabs import tab_schema

    found: set[str] = set()
    for tab in TABS:
        schema = tab_schema(tab)
        _text_strings(schema["core"]["layout"], found)
        for script in schema["scripts"]:
            found.add(script["title"])
            _text_strings(script["layout"], found)
    return found


def settings_strings(opts) -> set[str]:
    from remi.server.settings import register_options
    from remi.ui.settings import settings_schema

    register_options(opts)
    found: set[str] = set()
    _text_strings(settings_schema(opts)["categories"], found)
    return found


def message_strings() -> set[str]:
    """String literals passed as catalog keys in the server's sources."""

    found: set[str] = set()
    for folder in MESSAGE_SOURCES:
        for path in sorted((REPO / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                index = MESSAGE_CALLS.get(name)
                if index is None:
                    continue
                args = list(node.args)
                if name == "progress":
                    args += [kw.value for kw in node.keywords if kw.arg == "text"]
                if len(args) > index and isinstance(args[index], ast.Constant) and isinstance(args[index].value, str) and args[index].value.strip():
                    found.add(args[index].value)
    return found


def core_strings(opts) -> set[str]:
    """Everything the core catalog must translate (data names left out)."""

    strings = form_strings() | settings_strings(opts) | message_strings()
    dynamic = _dynamic_names()
    return {text for text in strings if text not in dynamic and not _is_data(text)}


def _is_data(text: str) -> bool:
    """Numbers, units and symbols need no translation."""

    return bool(re.fullmatch(r"[\d\s.,:%x×+\-/()#*]+|[A-Z0-9_]{1,6}", text.strip()))


def spa_catalog(locale: str = "zh-CN") -> dict[str, str]:
    """The SPA's own catalog (it wins over the server catalog at runtime)."""

    return json.loads((REPO / "webui" / "src" / "locales" / f"{locale}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--locale", default="zh_CN")
    parser.add_argument("--all", action="store_true", help="every string, not only the missing ones")
    parser.add_argument("--out", type=Path, help="write the JSON list here (the WebUI logs to stdout)")
    options = parser.parse_args()
    sys.argv = sys.argv[:1]
    sys.path[:0] = [str(REPO / "modules_forge" / "packages"), str(REPO)]
    from remi.testing import webui_state

    with webui_state() as state:
        strings = core_strings(state.opts)
    catalog = json.loads((REPO / "localizations" / f"{options.locale}.json").read_text(encoding="utf-8"))
    spa = spa_catalog(options.locale.replace("_", "-"))
    wanted = sorted(strings if options.all else (text for text in strings if not catalog.get(text) and text not in spa))
    text = json.dumps(wanted, ensure_ascii=False, indent=1)
    if options.out:
        options.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)  # noqa: T201 - command-line output
