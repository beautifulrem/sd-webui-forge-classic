"""Test support for the Anima LoRA panels ported to ``params()``: Artist
Scheduled Mixer, LoRA Layer Weight and LoRA Stage Scheduler.

Not a test module; their suites load it by file path (the repo ``tests``
folder is not importable from an extension folder, see
``extension_script_harness.py``).

Each extension keeps the contract of its former Gradio ``ui()``, recorded
before the port in ``tests/fixtures/legacy_ui_<tab>.json`` (one entry per
script argument: kind, legacy control class, label, default, bounds /
choices, infotext key,
whether a paste function reads it, Anima preset field and value). The ported
layout is compared with it by :func:`describe` / :func:`assert_parity`.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TABS = ("txt2img", "img2img")


def prepare_imports(extension_root: Path | str | None = None) -> None:
    """The WebUI modules importable outside the WebUI (idempotent)."""

    sys.argv = sys.argv[:1]
    for path in (REPO / "modules_forge" / "packages", REPO, extension_root):
        if path is not None and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_script(path: Path | str, name: str):
    """Import one extension script the way Forge does (its folder on sys.path)."""

    path = Path(path).resolve()
    prepare_imports(path.parents[1])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_layout(script, tab: str = "txt2img"):
    """``script``'s ``ScriptLayout`` on ``tab``, built as the runner builds it
    (owner id = slug of the title, ``param_order``)."""

    from modules.scripts import script_slug
    from remi.ui import ScriptLayout

    script.tabname = tab
    script.is_img2img = tab == "img2img"
    script.is_txt2img = not script.is_img2img
    root = script.params(script.is_img2img)
    return ScriptLayout(root, owner=script_slug(script.title()), order=script.param_order)


def schema_json(layout) -> dict:
    """The exported layout tree, through strict JSON."""

    from remi.ui import to_schema

    return json.loads(json.dumps(to_schema(layout.root), allow_nan=False, ensure_ascii=False))


# region Legacy contract


def legacy_snapshot(extension_root: Path | str, tab: str) -> dict:
    return json.loads((Path(extension_root) / "tests" / "fixtures" / f"legacy_ui_{tab}.json").read_text(encoding="utf8"))


def describe(layout) -> list[dict[str, Any]]:
    """Each input of ``layout`` in argument order, in the snapshot's terms."""

    from remi import ui
    from remi.ui.schema import jsonable

    ctx = ui.Context()
    entries = []
    for node in layout.inputs:
        entry: dict[str, Any] = {"name": node.name, "kind": node.type, "label": node.label, "default": jsonable(node.resolve_default(ctx))}
        if node.hidden:
            entry["visible"] = False
        if isinstance(node, ui.Slider):
            entry.update(minimum=node.minimum, maximum=node.maximum, step=node.step)
        if isinstance(node, (ui.Choice, ui.MultiChoice)):
            entry["choices"] = jsonable(node.values(ctx))
        if isinstance(node, ui.Custom):
            entry["widget"] = node.widget
        if node.infotext:
            entry["infotext"] = node.infotext
        if node.paste is not None:
            entry["paste"] = True
        if node.preset_key:
            entry["preset"] = node.preset_key
        entries.append(entry)
    return entries


def legacy_entry(entry: Mapping) -> dict[str, Any]:
    """A snapshot entry reduced to what :func:`describe` reports."""

    keep = {key: entry[key] for key in ("kind", "label", "default", "visible", "infotext", "paste", "preset") if key in entry}
    if entry.get("kind") == "slider":
        keep.update({key: entry[key] for key in ("minimum", "maximum", "step")})
    if "choices" in entry:
        keep["choices"] = [choice[1] if isinstance(choice, list) else choice for choice in entry["choices"]]
    return keep


def assert_parity(layout, extension_root: Path | str, tab: str, changes: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """``layout``'s inputs are the snapshot's, position by position, except
    the documented ``changes``: ``{param name: {key: new value}}`` (``...``
    as the value: not compared)."""

    legacy = legacy_snapshot(extension_root, tab)["inputs"]
    ported = describe(layout)
    assert len(ported) == len(legacy), (len(ported), len(legacy))
    assert_entries(ported, legacy, changes)


def assert_entries(ported: list[dict], legacy: list[dict], changes: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """Each :func:`describe` entry of ``ported`` equals the snapshot entry at
    the same position of ``legacy``, but for the documented ``changes``."""

    changes = {name: dict(change) for name, change in (changes or {}).items()}
    for got, recorded in zip(ported, legacy, strict=True):
        got = dict(got)
        name = got.pop("name")
        want = legacy_entry(recorded)
        for key, value in changes.pop(name, {}).items():
            want[key] = value
        for key in sorted(set(got) | set(want)):
            if want.get(key) is ...:
                continue
            assert got.get(key) == want.get(key), f"{name}: {key} is {got.get(key)!r}, the legacy ui() had {want.get(key)!r}"
    assert not changes, f"changes for unknown params: {sorted(changes)}"


def preset_bindings(layout) -> dict[str, tuple[str, Any]]:
    """Anima preset field -> (param, the value the preset gives it)."""

    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET

    return {field: (name, SAFE_BASE_AESTHETIC_PRESET[field] if own is _no_value() else own) for field, (name, own) in layout.presets.items()}


def _no_value():
    from remi.ui.nodes import _NO_VALUE

    return _NO_VALUE


# endregion

# region Infotext


def _forge_source(path: str, names: set[str], namespace: dict) -> dict:
    tree = ast.parse((REPO / path).read_text(encoding="utf8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names) or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names)
    ]
    exec(compile(ast.Module(nodes, []), path, "exec"), namespace)
    return namespace


def through_infotext(params: Mapping) -> dict:
    """``params`` written into a parameters line as processing writes it and
    read back as ``parse_generation_parameters`` reads the pairs."""

    code = _forge_source("modules/infotext_utils.py", {"re_param_code", "quote", "unquote"}, {"json": json, "re": re})
    line = "Steps: 20, Sampler: Euler, " + ", ".join(f"{key}: {code['quote'](value)}" for key, value in params.items() if value is not None)
    return {key: code["unquote"](value) for key, value in re.findall(code["re_param_code"], line)}


def pasted(layout, params: Mapping, *, callbacks=()) -> dict:
    """Named values the layout reads from a run's infotext ``params`` (after
    the ``on_infotext_pasted`` ``callbacks``), as the SPA paste,
    ``/infotext/parse`` and the API replay read them."""

    parsed = through_infotext(params)
    for callback in callbacks:
        callback("", parsed)
    values, problems = layout.paste(parsed)
    assert not problems, [str(problem) for problem in problems]
    return values


def spa_paste(script, layout, params: Mapping, start: Mapping | None = None, *, core: Mapping | None = None, tab: str = "txt2img") -> dict:
    """The form values after the SPA pastes ``params`` into a form at
    ``start`` (default: the defaults): the pasted values, then each distinct
    ``on_change`` action of the pasted params with its patch applied, as
    ``applyPaste`` does (``on_input`` actions are user edits only)."""

    from remi import ui

    values = {**layout.defaults(), **dict(start or {})}
    taken = pasted(layout, params)
    values.update(taken)
    actions = dict.fromkeys(layout.by_name[name].on_change for name in taken if layout.by_name[name].on_change)
    for action in actions:
        patch = ui.call_action(script, action, ui.ActionContext(tab=tab, core=dict(core or {})), values)
        values.update({name: entry["value"] for name, entry in patch.items() if not name.startswith("__") and "value" in entry})
    return values


def replayed_args(layout, params: Mapping, start: Mapping | None = None, *, callbacks=()) -> list:
    """The script arguments after pasting ``params`` into a form at ``start``
    (default: the defaults)."""

    values = {**layout.defaults(), **dict(start or {}), **pasted(layout, params, callbacks=callbacks)}
    return layout.values_to_args(values)


# endregion

# region Standalone


def assert_loads_standalone(script_path: Path | str) -> None:
    """The script loads and builds its layouts on both tabs in a fresh interpreter."""

    code = f"""
import sys
sys.path.insert(0, {str(REPO / "tests")!r})
import remi_anima_lora_harness as harness
module = harness.load_script({str(script_path)!r}, "standalone_check")
for tab in harness.TABS:
    layout = harness.script_layout(module.Script(), tab)
    harness.schema_json(layout)
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr[-4000:]


# endregion
