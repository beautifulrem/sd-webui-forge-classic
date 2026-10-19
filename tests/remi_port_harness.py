"""Harness for extension scripts ported from A1111's ``ui()`` to ``params()``.

Not a test module; extension suites load it by file path (the repo
``tests`` folder is not an importable package there, see
``extension_script_harness.py``):

- :func:`prepare_imports` / :func:`load_script` import an extension script
  as Forge does; :func:`script_layout` builds its ``ScriptLayout`` for a tab
  as the runner does.
- :func:`describe` reduces a layout to one entry per positional argument
  (kind, label, default, bounds or choices, widget, infotext key, paste
  function, Anima preset field, and the wiring: exclusive lock,
  ``visible_if`` / ``enabled_if``); :func:`assert_matches_snapshot` compares
  it with the legacy ``ui()`` record (``tests/fixtures/legacy_ui_<tab>.json``
  of each extension, recorded from the Gradio UI before it was removed),
  listing every intended difference.
- :func:`assert_imports_standalone` imports scripts in a fresh interpreter.
- :func:`ui_strings` / :func:`assert_localized` check an extension's
  ``localizations/<locale>.json`` covers every string its schema shows (core
  ``localizations/<locale>.json`` may cover some) without contradicting the
  core or language pack catalogs it is merged with.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TABS = ("txt2img", "img2img")


def prepare_imports() -> None:
    """The WebUI modules importable outside the WebUI: pytest's arguments
    kept from the WebUI's argv parser, the repo and its bundled packages on
    ``sys.path``. Idempotent."""

    sys.argv = sys.argv[:1]
    for path in (str(REPO / "modules_forge" / "packages"), str(REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_script(path: str | Path, name: str):
    """Import one extension script the way Forge does (its extension folder
    on ``sys.path``, for its ``lib_*`` packages)."""

    prepare_imports()
    path = Path(path).resolve()
    root = str(path.parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_layout(script, tab: str = "txt2img"):
    """``script``'s layout for ``tab``, built as ``ScriptRunner.build_layouts`` does."""

    from remi.ui import ScriptLayout

    script.tabname = tab
    script.is_img2img = tab == "img2img"
    script.is_txt2img = not script.is_img2img
    return ScriptLayout(script.params(script.is_img2img), owner="script", order=script.param_order)


_TEXT_KEYS = ("label", "info", "title", "placeholder")


def ui_strings(schema: Any) -> set[str]:
    """The English strings the SPA translates in an exported layout: labels,
    info texts, titles, placeholders, Markdown texts, HTML texts (keyed by
    their exact markup) and choice labels (numbers excepted)."""

    found: set[str] = set()
    if isinstance(schema, list):
        for item in schema:
            found |= ui_strings(item)
        return found
    if not isinstance(schema, dict):
        return found
    for key in _TEXT_KEYS:
        if isinstance(schema.get(key), str) and schema[key].strip():
            found.add(schema[key])
    if schema.get("type") in ("markdown", "html") and isinstance(schema.get("text"), str) and schema["text"].strip():
        found.add(schema["text"])
    for choice in schema.get("choices") or ():
        label = choice.get("label")
        if isinstance(label, str) and label.strip() and not label.replace(".", "", 1).lstrip("-").isdigit():
            found.add(label)
    for key in ("enable", "children"):
        if key in schema:
            found |= ui_strings(schema[key])
    return found


# Catalogs merged with every extension's into one flat catalog (core first,
# then extensions in load order): an extension entry for the same English
# text must agree with them, or the winner depends on load order.
SHARED_CATALOGS = (Path("localizations"), Path("extensions-builtin") / "sd-webui-language-diffusion" / "localizations")


def _catalog(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf8")) if path.is_file() else {}


def catalog_conflicts(extension_root: str | Path, locale: str = "zh_CN") -> list[tuple[str, str, str, str]]:
    """``(text, extension translation, shared catalog, its translation)`` for
    every entry of the extension's catalog a shared catalog translates differently."""

    own = _catalog(Path(extension_root) / "localizations" / f"{locale}.json")
    conflicts = []
    for folder in SHARED_CATALOGS:
        shared = _catalog(REPO / folder / f"{locale}.json")
        conflicts += [(text, value, str(folder), shared[text]) for text, value in own.items() if text in shared and shared[text] != value]
    return conflicts


def assert_localized(extension_root: str | Path, schemas: Iterable[Any], locale: str = "zh_CN", extra: Iterable[str] = ()) -> dict[str, str]:
    """``extension_root/localizations/<locale>.json`` (or the core catalog)
    translates every string of ``schemas`` (and ``extra``), each into a
    non-empty text, and contradicts no shared catalog; returns the
    extension's catalog."""

    catalog = _catalog(Path(extension_root) / "localizations" / f"{locale}.json")
    available = {**_catalog(REPO / "localizations" / f"{locale}.json"), **catalog}
    wanted: set[str] = set(extra)
    for schema in schemas:
        wanted |= ui_strings(schema)
    missing = sorted(text for text in wanted if not isinstance(available.get(text), str) or not available[text].strip())
    assert not missing, f"{locale} lacks {missing}"
    conflicts = catalog_conflicts(extension_root, locale)
    assert not conflicts, f"{locale} entries contradict shared catalogs: {conflicts}"
    return catalog


def assert_imports_standalone(scripts: Iterable[str | Path], *, preimport: Iterable[str] = (), modules: Iterable[str] = (), extra_paths: Iterable[str | Path] = ()) -> None:
    """Each script file and module of ``modules`` (e.g. a core processing
    script) imports in a fresh interpreter, default options as at a WebUI
    start, with only the WebUI's own dependencies."""

    paths = [str(REPO), str(REPO / "modules_forge" / "packages"), *(str(path) for path in extra_paths)]
    lines = ["import importlib.util, sys", "sys.argv = sys.argv[:1]"]
    lines += [f"import {name}" for name in preimport]
    lines += ["from remi.testing import webui_state", "with webui_state():", "    pass"]
    lines += [f"    import {name}" for name in modules]
    for index, script in enumerate(scripts):
        script = Path(script).resolve()
        lines += [
            f"    sys.path.insert(0, {str(script.parents[1])!r})",
            f"    spec = importlib.util.spec_from_file_location('standalone_{index}', {str(script)!r})",
            "    spec.loader.exec_module(importlib.util.module_from_spec(spec))",
        ]
    lines += ["print('imported')"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    result = subprocess.run([sys.executable, "-c", "\n".join(lines)], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and result.stdout.strip().endswith("imported"), result.stdout[-2000:] + result.stderr[-4000:]


def describe(layout, ctx=None) -> list[dict[str, Any]]:
    """One entry per argument of ``layout``, in positional order."""

    from remi import ui
    from remi.ui.schema import jsonable

    ctx = ctx or ui.Context()
    entries = []
    for node in layout.inputs:
        entry: dict[str, Any] = {"kind": node.type, "label": node.label, "default": jsonable(node.resolve_default(ctx))}
        if isinstance(node, (ui.Slider, ui.Number)):
            entry.update(minimum=node.minimum, maximum=node.maximum, step=node.step)
        if isinstance(node, (ui.Choice, ui.MultiChoice)):
            entry["choices"] = jsonable(node.values(ctx))
        if isinstance(node, ui.Custom):
            entry["widget"] = node.widget
            entry["props"] = jsonable(ctx.resolve(node, "props", node.props))
        entry["infotext"] = node.infotext
        entry["paste"] = node.paste is not None
        entry["preset"] = node.preset_key
        entry["exclusive"] = list(node.exclusive) if node.exclusive else None
        entry["visible_if"] = jsonable(node.visible_if)
        entry["enabled_if"] = jsonable(node.enabled_if)
        entries.append(entry)
    return entries


# Recorded snapshots predate these keys: absent means "none", so wiring the
# port adds must be listed among the documented changes.
WIRING_KEYS = ("exclusive", "visible_if", "enabled_if")


def fixture_path(script_path: str | Path, tab: str) -> Path:
    return Path(script_path).resolve().parents[1] / "tests" / "fixtures" / f"legacy_ui_{tab}.json"


def load_snapshot(script_path: str | Path, tab: str) -> list[dict[str, Any]]:
    return json.loads(fixture_path(script_path, tab).read_text(encoding="utf8"))


def assert_matches_snapshot(layout, snapshot: list[dict[str, Any]], changes: Mapping[int, Mapping[str, Any]] | None = None, *, ctx=None) -> None:
    """``layout`` declares the arguments of ``snapshot`` (a recorded legacy
    ``ui()``), except the documented ``changes``: ``{argument index:
    {key: new value}}`` (a key mapped to ``...`` is not compared). An
    index past the snapshot is an added argument, described in full."""

    changes = dict(changes or {})
    actual = describe(layout, ctx)
    expected = [dict(entry) for entry in snapshot]
    for index, change in sorted(changes.items()):
        if index >= len(expected):
            assert index == len(expected), f"added arguments must follow the recorded ones (index {index})"
            expected.append(dict(change))
        else:
            expected[index].update(change)
    for entry in expected:
        for key in WIRING_KEYS:
            entry.setdefault(key, None)
    assert len(actual) == len(expected), f"{len(actual)} arguments, the legacy ui() had {len(snapshot)} (+{len(expected) - len(snapshot)} added)"
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        for key in sorted(set(got) | set(want)):
            if want.get(key) is ...:
                continue
            assert got.get(key) == want.get(key), f"argument {index} ({got['label']!r}): {key} is {got.get(key)!r}, expected {want.get(key)!r}"
