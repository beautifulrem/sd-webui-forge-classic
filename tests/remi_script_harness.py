"""Shared helpers for the tests of extension scripts ported to ``params()``.

Not a test module: extension suites load it by file path (see
tests/extension_script_harness.py for why). Nothing here imports Gradio.

- :func:`load_script` imports a script file as Forge does.
- :func:`script_layout` builds a script's ``ScriptLayout`` for a tab, as the
  runner does (``params()`` then ``param_order``).
- :func:`contract` / :func:`legacy_contract` give the comparable per-input
  description of a layout and of the legacy Gradio snapshot
  (``tests/fixtures/legacy_ui_<tab>.json``, written before the port).
- :func:`replayed_args` replays a run's infotext into a layout: the paste
  (onto the values the form had) and the API's infotext replay (onto the
  defaults) both read the infotext through the layout.
- :func:`assert_gradio_free` checks an extension's sources and imports.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Keys of a legacy snapshot entry compared with the ported input
CONTRACT_KEYS = ("type", "label", "info", "default", "minimum", "maximum", "step", "integer", "choices", "style", "infotext", "preset")


def prepare_imports(extension_root: Path | str | None = None) -> None:
    """The WebUI modules importable outside the WebUI: pytest's arguments
    kept from its argv parser, the extension folder on sys.path (as Forge
    puts it there for its ``lib_*`` packages). Idempotent."""

    sys.argv = sys.argv[:1]
    for path in (str(REPO), str(extension_root) if extension_root is not None else None):
        if path is not None and path not in sys.path:
            sys.path.insert(0, path)


def load_script(path: Path | str, name: str):
    """Import one extension script the way Forge does."""

    path = Path(path).resolve()
    prepare_imports(path.parents[1])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_layout(script, tab: str = "txt2img"):
    """``script``'s layout on ``tab`` (None when it is not shown there), built
    like ``ScriptRunner.build_layouts``: owner id from the title."""

    from modules.scripts import script_slug
    from remi.ui import ScriptLayout

    is_img2img = tab == "img2img"
    script.tabname = tab
    script.is_img2img = is_img2img
    if not script.show(is_img2img):
        return None
    root = script.params(is_img2img)
    return ScriptLayout(root, owner=script_slug(script.title()), order=script.param_order)


def schema_json(layout) -> dict:
    """The exported layout tree, through JSON (it must serialize as is)."""

    from remi.ui import to_schema

    return json.loads(json.dumps(to_schema(layout.root), allow_nan=False))


def _entry(schema: Mapping) -> dict:
    data = {key: schema[key] for key in CONTRACT_KEYS if key in schema}
    if "choices" in data:
        data["choices"] = [choice["value"] for choice in data["choices"]]
    data.setdefault("info", None)
    data.setdefault("infotext", None)
    data.setdefault("preset", None)
    return data


def contract(layout) -> list[dict]:
    """Every input of ``layout`` in positional order: kind, label, info,
    default, bounds / choices, infotext key and preset field."""

    from remi.ui import Context

    ctx = Context()
    return [{"name": node.name, **_entry(node.schema(ctx))} for node in layout.inputs]


def legacy_snapshot(extension_root: Path | str, tab: str) -> dict:
    return json.loads((Path(extension_root) / "tests" / "fixtures" / f"legacy_ui_{tab}.json").read_text(encoding="utf-8"))


def legacy_contract(extension_root: Path | str, tab: str) -> list[dict]:
    """The legacy ``ui()`` inputs of the snapshot, in :func:`contract`'s shape."""

    return [{key: item.get(key) for key in CONTRACT_KEYS if key in item} for item in legacy_snapshot(extension_root, tab)["inputs"]]


def assert_matches_legacy(layout, extension_root, tab, *, changed: Mapping[tuple[str, str], object] | None = None, extra: tuple[str, ...] = ()) -> None:
    """``layout``'s inputs equal the legacy snapshot's, in order, except the
    documented ``changed`` entries (``(name, key) -> new value``) and the
    trailing ``extra`` params the legacy panel did not pass to the script."""

    changed = dict(changed or {})
    ported = contract(layout)
    names = [item.pop("name") for item in ported]
    if extra:
        assert names[-len(extra) :] == list(extra), names
        names, ported = names[: -len(extra)], ported[: -len(extra)]
    legacy = legacy_contract(extension_root, tab)
    assert len(ported) == len(legacy), (names, len(legacy))
    for name, now, before in zip(names, ported, legacy):
        expected = {key: changed.get((name, key), value) for key, value in before.items()}
        actual = {key: now.get(key) for key in expected}
        assert actual == expected, name
    unused = [key for key in changed if key[0] not in names]
    assert not unused, unused


def infotext_round_trip(params: Mapping) -> dict:
    """``params`` written into a parameters line as processing writes it
    (``infotext_utils.quote``) and read back as
    ``parse_generation_parameters`` reads the pairs (``re_param_code``,
    ``unquote``), from Forge's own source."""

    namespace = _forge_source("modules/infotext_utils.py", {"re_param_code", "quote", "unquote"}, {"json": json, "re": re})
    line = "Steps: 20, Sampler: Euler, " + ", ".join(f"{key}: {namespace['quote'](value)}" for key, value in params.items() if value is not None)
    return {key: namespace["unquote"](value) for key, value in re.findall(namespace["re_param_code"], line)}


def _forge_source(path: str, names: set[str], namespace: dict) -> dict:
    tree = ast.parse((REPO / path).read_text(encoding="utf8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names) or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names)
    ]
    exec(compile(ast.Module(nodes, []), path, "exec"), namespace)  # noqa: S102 - the WebUI's own source
    return namespace


def replayed_args(layout, params: Mapping, start: Mapping | None = None) -> list:
    """The positional script args after pasting a run's infotext ``params``
    into a form whose values are ``start`` (default: the defaults, which is
    the API's infotext replay). Params the infotext does not set keep their
    ``start`` value."""

    from remi.ui import Context

    ctx = Context()
    pasted, problems = layout.paste(infotext_round_trip(params), ctx)
    assert not problems, [str(problem) for problem in problems]
    values = {**layout.defaults(ctx), **dict(start or {}), **pasted}
    return layout.values_to_args(values, ctx)


def preset_fields(layout) -> dict[str, str]:
    """Anima preset field -> param name of ``layout``."""

    return {field: name for field, (name, _own) in layout.presets.items()}


def assert_registers_preset(layout, prefix: str, *, unowned: tuple[str, ...] = (), preset_order: bool = True) -> None:
    """Every ``prefix.*`` preset field is bound to one of ``layout``'s params
    (all but the ``unowned`` user-content params), in positional order, and
    its preset value is the param's default, of its type and in range; the
    first (the enable switch) is off."""

    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET
    from remi.ui import Context

    ctx = Context()
    bound = preset_fields(layout)
    expected = [field for field in SAFE_BASE_AESTHETIC_PRESET if field.startswith(f"{prefix}.")]
    owned = [node.name for node in layout.inputs if node.name not in unowned]
    assert sorted(bound) == sorted(expected)
    by_param = {name: field for field, name in bound.items()}
    assert sorted(by_param) == sorted(owned)
    if preset_order:
        assert [by_param[name] for name in owned] == expected
    for name in owned:
        node, value = layout.by_name[name], SAFE_BASE_AESTHETIC_PRESET[by_param[name]]
        default = node.resolve_default(ctx)
        assert value == default and type(value) is type(default), name
        assert node.coerce(value, ctx) == value, name
    assert SAFE_BASE_AESTHETIC_PRESET[by_param[owned[0]]] is False


def schema_strings(schema: Mapping) -> list[str]:
    """The strings the SPA translates in an exported layout (titles, labels,
    info texts, choice labels, markdown), in tree order."""

    found: list[str] = []

    def walk(node: Mapping) -> None:
        texts = [node.get(key) for key in ("title", "label", "info")]
        if node.get("type") == "markdown":
            texts.append(node.get("text"))
        texts += [choice["label"] for choice in node.get("choices") or ()]
        found.extend(text for text in texts if isinstance(text, str) and text and text not in found)
        for child in ([node["enable"]] if "enable" in node else []) + list(node.get("children") or ()):
            walk(child)

    walk(schema)
    return found


def assert_localized(extension_root: Path | str, layouts, locale: str = "zh_CN") -> None:
    """The extension's ``localizations/<locale>.json`` translates every
    string of ``layouts`` and has no entry they do not use."""

    catalog = json.loads((Path(extension_root) / "localizations" / f"{locale}.json").read_text(encoding="utf-8"))
    strings: list[str] = []
    for layout in layouts:
        strings += [text for text in schema_strings(schema_json(layout)) if text not in strings]
    assert [text for text in strings if not str(catalog.get(text, "")).strip()] == []
    assert sorted(set(catalog) - set(strings)) == []


def gradio_imports(extension_root: Path | str) -> list[str]:
    """``file:line`` of every Gradio import in the extension's Python files."""

    found = []
    for path in sorted(Path(extension_root).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(name == "gradio" or name.startswith(("gradio.", "gradio_")) for name in names):
                found.append(f"{path.relative_to(extension_root)}:{node.lineno}")
    return found


def assert_gradio_free(extension_root: Path | str, scripts: tuple[str, ...]) -> None:
    """No Gradio import in the extension's sources, and its ``scripts``
    (paths relative to it) load and build their layouts with Gradio blocked."""

    extension_root = Path(extension_root).resolve()
    assert gradio_imports(extension_root) == []
    code = f"""
import sys
sys.modules["gradio"] = None
sys.path.insert(0, {str(REPO / "tests")!r})
import remi_script_harness as harness
for index, path in enumerate({[str(extension_root / script) for script in scripts]!r}):
    module = harness.load_script(path, f"gradio_free_{{index}}")
    from modules import scripts
    for value in list(vars(module).values()):
        if isinstance(value, type) and issubclass(value, scripts.Script) and value.__module__ == module.__name__:
            for tab in ("txt2img", "img2img"):
                harness.script_layout(value(), tab)
assert "gradio" not in [name for name, mod in sys.modules.items() if mod is not None and name.split(".")[0] == "gradio"]
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=600)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr[-4000:]
