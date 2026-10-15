"""Shared harness for built-in panels ported to ``params()`` (``remi.ui``).

Not a test module. Extension suites load it by file path (see
``extension_script_harness.py`` for why the repo ``tests`` folder is not an
importable package there). A ported panel is tested through its ``ScriptLayout``, which is what the SPA, the HTTP API, the
infotext paste (UI and ``/sdapi``) and the Anima presets all read.

- :func:`load_script` imports an extension script as Forge does.
- :func:`script_layout` builds a script's layout for one tab.
- :func:`layout_contract` / :func:`contract_differences` compare a layout
  with the legacy ``ui()`` snapshot an extension keeps in
  ``tests/fixtures/legacy_ui_<tab>.json`` (kind, label, default, bounds,
  choices, infotext key and preset field per input, in argument order).
- :func:`pasted_args` / :func:`assert_infotext_round_trip` replay a run's
  infotext through ``ScriptLayout.paste`` into script arguments.
- :func:`assert_preset_fields` checks the Anima preset bindings.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def prepare_imports() -> None:
    """The WebUI modules importable outside the WebUI (idempotent): the repo
    and Forge's bundled packages on ``sys.path`` (as pyproject's pytest
    ``pythonpath``), and pytest's own arguments kept from the WebUI's argv
    parser."""

    sys.argv = sys.argv[:1]
    for path in (REPO / "modules_forge" / "packages", REPO):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


@functools.cache
def _legacy():
    """The shared fakes (``make_process``, ``Patcher``) and ``forge_source``."""

    spec = importlib.util.spec_from_file_location("extension_script_harness", REPO / "tests" / "extension_script_harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_process(blocks=28):
    """A processing object on an Anima model, and its UNet patcher."""

    return _legacy().make_process(blocks)


def load_script(path, name):
    """Import one extension script the way Forge does (its folder on
    ``sys.path`` for its ``lib_*`` packages), without the full WebUI."""

    prepare_imports()
    basedir = str(Path(path).resolve().parents[1])
    if basedir not in sys.path:
        sys.path.insert(0, basedir)
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_ported(script_path) -> None:
    """No class of the extension script keeps an A1111 UI hook (``ui()``,
    component callbacks): its arguments come from ``params()``."""

    import ast

    hooks = {"ui", "on_before_component", "on_after_component", "before_component", "after_component"}
    tree = ast.parse(Path(script_path).read_text(encoding="utf8"))
    left = [f"{cls.name}.{item.name}" for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) for item in cls.body if isinstance(item, ast.FunctionDef) and item.name in hooks]
    assert left == [], (script_path, left)


# region Layouts


def script_layout(script, tab="txt2img"):
    """``script``'s ``ScriptLayout`` on ``tab``, as the runner builds it."""

    from remi.ui import ScriptLayout

    is_img2img = tab == "img2img"
    script.tabname = tab
    script.is_img2img = is_img2img
    script.is_txt2img = not is_img2img
    return ScriptLayout(script.params(is_img2img), owner="panel", order=script.param_order)


def layout_contract(layout) -> list[dict]:
    """The legacy snapshot's fields for each input of ``layout``, in argument order."""

    from remi.ui import Context

    ctx = Context()
    entries = []
    for node in layout.inputs:
        entry = {"name": node.name, "kind": node.type, "label": node.label, "default": node.resolve_default(ctx)}
        for key in ("minimum", "maximum", "step"):
            if getattr(node, key, None) is not None:
                entry[key] = getattr(node, key)
        if hasattr(node, "resolve_choices"):
            entry["choices"] = [value for _label, value in node.resolve_choices(ctx)]
        if node.type == "text":
            entry["lines"] = node.lines
        entry["infotext"] = node.infotext
        entry["preset"] = node.preset_key
        entries.append(entry)
    return entries


def load_fixture(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))


def contract_differences(legacy: dict, layout) -> dict:
    """``{param name: {field: (legacy, now)}}`` for every field that differs
    between the legacy snapshot and ``layout``, position by position; params
    beyond the legacy inputs as ``{name: {"added": kind}}``."""

    now = layout_contract(layout) if layout is not None else []
    differences: dict = {}
    for index, entry in enumerate(now):
        if index >= len(legacy["inputs"]):
            differences[entry["name"]] = {"added": entry["kind"]}
            continue
        old = legacy["inputs"][index]
        changed = {key: (old.get(key), entry.get(key)) for key in sorted(set(old) | set(entry) - {"name"}) if _differs(old.get(key), entry.get(key))}
        if changed:
            differences[entry["name"]] = changed
    for old in legacy["inputs"][len(now) :]:
        differences.setdefault("<removed>", []).append(old["label"])
    return differences


def _differs(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) > 1e-9
    return a != b


# endregion

# region Infotext


@functools.cache
def _infotext_code():
    """Forge's ``re_param_code``, ``quote`` and ``unquote`` read from
    modules/infotext_utils.py (the module needs the full WebUI to import)."""

    return _legacy().forge_source("modules/infotext_utils.py", {"re_param_code", "quote", "unquote"}, {"json": json})


def through_infotext(params: dict) -> dict:
    """``params`` written to a parameters line as processing.create_infotext
    writes it, then read back as parse_generation_parameters reads it."""

    code = _infotext_code()
    line = "Steps: 20, Sampler: Euler, " + ", ".join(f"{key}: {code['quote'](value)}" for key, value in params.items() if value is not None)
    return {key: code["unquote"](value) for key, value in re.findall(code["re_param_code"], line)}


def pasted_values(layout, params: dict) -> dict:
    """Named values ``ScriptLayout.paste`` reads from the infotext ``params``
    (the SPA paste, ``/infotext/parse`` and the ``/sdapi`` infotext replay)."""

    values, problems = layout.paste(through_infotext(params))
    assert problems == [], problems
    return values


def pasted_args(layout, params: dict, start=None) -> list:
    """The script args after pasting ``params`` into a panel whose values
    are ``start`` (named; default: the layout's defaults)."""

    values = dict(layout.defaults() if start is None else start)
    values.update(pasted_values(layout, params))
    return layout.values_to_args(values)


def assert_infotext_round_trip(layout, args, record, expected=None) -> dict:
    """A run with ``args`` records ``record(args)``; pasting it into a fresh
    panel gives the script args ``expected`` (default ``args``), and a run
    with those records the same infotext. Returns the recorded infotext."""

    params = record(list(args))
    expected = list(args) if expected is None else list(expected)
    replayed = pasted_args(layout, params)
    assert replayed == expected
    assert record(replayed) == params
    return params


def assert_infotext_restores_params(layout, args, written: dict) -> None:
    """Each infotext value a run with ``args`` wrote is keyed by the param it
    was read from, in argument order."""

    by_key = {node.infotext: index for index, node in enumerate(layout.inputs) if node.infotext}
    indexes = [by_key[key] for key in written]
    assert indexes == sorted(indexes)
    for key, value in written.items():
        assert value == args[by_key[key]], key


# endregion

# region Localization


def layout_strings(layout, *, skip_choices=()) -> set[str]:
    """The English strings of ``layout``'s schema the SPA translates
    (labels, info texts, titles, Markdown, placeholders, button labels and
    choice labels; ``skip_choices`` names params whose choices are user
    content)."""

    found: set[str] = set()

    def visit(node):
        for key in ("label", "info", "title", "text", "placeholder"):
            value = node.get(key)
            if isinstance(value, str) and value and not (node["type"] == "html" and key == "text"):
                found.add(value)
        if node.get("name") not in skip_choices:
            found.update(choice["label"] for choice in node.get("choices", ()) if choice["label"])
        if node.get("enable"):
            visit(node["enable"])
        for child in node.get("children", ()):
            visit(child)

    schema = layout.schema()
    if schema is not None:
        visit(schema)
    return found


def assert_localized(layouts, catalog_path, *, skip_choices=(), extra=()) -> None:
    """Every string of ``layouts`` (and ``extra``, e.g. a script title the
    layout does not show) has a non-empty translation in the extension's
    catalog (``localizations/<locale>.json``), and the catalog holds nothing
    else."""

    catalog = json.loads(Path(catalog_path).read_text(encoding="utf8"))
    strings = set(extra).union(*(layout_strings(layout, skip_choices=skip_choices) for layout in layouts))
    missing = sorted(text for text in strings if not catalog.get(text, "").strip())
    assert missing == [], missing
    assert sorted(set(catalog) - strings) == []


# endregion

# region Presets


def assert_preset_fields(layout, prefix: str, unowned=()) -> None:
    """Every ``prefix.*`` field of the Anima preset table is bound to one of
    the panel's params (in table order), at the param's default and in its
    bounds; ``unowned`` names the params the preset leaves alone (user
    content); the switch (the first param) is off in the preset."""

    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET
    from remi.ui import Context

    ctx = Context()
    expected = [name for name in SAFE_BASE_AESTHETIC_PRESET if name.startswith(f"{prefix}.")]
    bound = [node.preset_key for node in layout.inputs if node.preset_key]
    assert sorted(bound) == sorted(expected)
    assert [node.name for node in layout.inputs if not node.preset_key] == list(unowned)
    for node in layout.inputs:
        if not node.preset_key:
            continue
        value = SAFE_BASE_AESTHETIC_PRESET[node.preset_key]
        default = node.resolve_default(ctx)
        assert value == default and type(value) is type(default), node.name
        assert node.coerce(value, ctx) == value, node.name
    assert SAFE_BASE_AESTHETIC_PRESET[layout.inputs[0].preset_key] is False


# endregion
