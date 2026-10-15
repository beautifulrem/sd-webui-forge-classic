"""Parity of a ported ``params()`` layout with the legacy Gradio ``ui()``.

Not a test module (the extension suites load it by file path, see
``extension_script_harness.py``). The snapshots (``legacy_ui_<tab>.json``)
were taken from each script's ``ui()`` before it was ported: per input, in
argument order, its kind, label, info text (Gradio ``info=`` or Forge
``tooltip=``), default, bounds, choices, infotext key, paste function and
preset field. Snapshots recorded before ``info`` was added lack the key; their
info texts are not compared. A port keeps them, except where a test documents
the change (``expected``) or adds params after the legacy ones (``extra``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FIELDS = ("kind", "label", "info", "default", "minimum", "maximum", "step", "precision", "integer", "lines", "style", "allow_custom", "choices", "returns", "infotext", "paste", "preset")


def prepare_imports() -> None:
    """The WebUI modules importable outside the WebUI (idempotent)."""

    sys.argv = sys.argv[:1]
    for path in (REPO / "modules_forge" / "packages", REPO):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def describe(layout, ctx=None) -> list[dict[str, Any]]:
    """The contract of every input of ``layout``, in argument order."""

    from remi import ui

    ctx = ctx or ui.Context()
    entries = []
    for node in layout.inputs:
        schema = node.schema(ctx)
        entry = {"name": node.name, "kind": schema["type"], "label": schema.get("label"), "info": schema.get("info"), "default": schema.get("default")}
        for key in FIELDS[4:12]:
            if key in schema:
                entry[key] = schema[key]
        if "choices" in schema:
            entry["choices"] = [choice["value"] for choice in schema["choices"]]
        if isinstance(node, (ui.Choice, ui.MultiChoice)):
            entry["returns"] = node.returns
        entry.update(infotext=node.infotext, paste=node.paste is not None, preset=node.preset_key)
        entries.append(entry)
    return entries


def load_snapshot(path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def differences(layout, snapshot: dict[str, Any], *, expected: dict[int, dict[str, Any]] | None = None, extra: tuple[str, ...] = (), ctx=None) -> list[str]:
    """What differs from the snapshot, beyond the documented changes:
    ``expected`` maps an argument position to the fields it now has (a
    ``kind`` change drops the old kind's keys); ``extra`` names the params
    that follow the legacy ones. Empty when the port keeps the contract."""

    entries = describe(layout, ctx)
    legacy = snapshot["inputs"]
    problems = []
    names = [entry["name"] for entry in entries]
    if names[len(legacy):] != list(extra):
        problems.append(f"params after the legacy ones: {names[len(legacy):]} (expected {list(extra)})")
    if len(entries) < len(legacy):
        problems.append(f"{len(entries)} params, the legacy ui() had {len(legacy)}")
    for position, (old, new) in enumerate(zip(legacy, entries)):
        want = dict(old)
        want.pop("name", None)
        if "info" not in old:
            new = {key: value for key, value in new.items() if key != "info"}  # an older snapshot
        change = (expected or {}).get(position, {})
        if "kind" in change and change["kind"] != old["kind"]:
            want = {key: want[key] for key in ("label", "info", "default", "infotext", "paste", "preset") if key in want}
        want.update(change)
        for key in set(want) | (set(new) - {"name"}):
            if want.get(key) != new.get(key):
                problems.append(f"#{position} {new['name']}: {key} is {new.get(key)!r}, the legacy ui() had {want.get(key)!r}")
    return problems


def schema_strings(schema) -> set[str]:
    """Every string of an exported layout the SPA translates: labels, info
    texts, titles, choice labels, Markdown and button labels."""

    found: set[str] = set()

    def visit(node):
        if isinstance(node, dict):
            for key in ("label", "info", "title", "text", "placeholder"):
                if isinstance(node.get(key), str) and node.get("type") != "html" and node[key].strip():
                    found.add(node[key])
            for choice in node.get("choices") or ():
                if isinstance(choice, dict) and isinstance(choice.get("label"), str):
                    found.add(choice["label"])
            for value in node.values():
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return found


def missing_translations(strings, catalog_path, *, ignore=lambda text: False) -> list[str]:
    """The ``strings`` a locale catalog (``localizations/<id>.json``) lacks."""

    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    return sorted(text for text in strings if not catalog.get(text) and not ignore(text))


def assert_ported(path) -> None:
    """No class in ``path`` keeps an A1111 UI hook (``ui()``, component
    callbacks): its arguments come from ``params()``."""

    import ast

    hooks = {"ui", "on_before_component", "on_after_component", "before_component", "after_component"}
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    left = [f"{cls.name}.{item.name}" for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) for item in cls.body if isinstance(item, ast.FunctionDef) and item.name in hooks]
    assert left == [], f"{path}: {left}"
