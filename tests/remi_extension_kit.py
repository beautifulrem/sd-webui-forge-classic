"""Test kit for the small built-in extensions ported to ``params()``: the
port harness (``tests/remi_port_harness.py``) plus what these suites add.

Not a test module; extension suites load it by file path (the repo
``tests`` folder is not an importable package there).

From the port harness (re-exported): :func:`prepare_imports`,
:func:`load_script`, :func:`describe`, :func:`load_snapshot` (by script
path) and :func:`assert_matches_snapshot`, against the legacy ``ui()``
records (``<extension>/tests/fixtures/legacy_ui_<tab>.json``).

Added here:

- :func:`layout_of` builds a script's ``ScriptLayout`` for a tab with the
  runner's owner id (``script_slug(title)``).
- :func:`schema_strings` / :func:`assert_localized`: the port harness's
  strings plus widget ``props`` texts, checked against the extension's
  ``localizations/zh_CN.json``.
- :func:`assert_standalone`: no JS/CSS assets left, and the scripts import
  and build both tabs' layouts in a fresh interpreter.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _port_harness():
    # One module instance whichever suite loads it first
    module = sys.modules.get("remi_port_harness")
    if module is None or not hasattr(module, "describe"):
        spec = importlib.util.spec_from_file_location("remi_port_harness", REPO / "tests" / "remi_port_harness.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["remi_port_harness"] = module
        spec.loader.exec_module(module)
    return module


port = _port_harness()
load_script = port.load_script
describe = port.describe
load_snapshot = port.load_snapshot
assert_matches_snapshot = port.assert_matches_snapshot

def prepare_imports() -> None:
    """The port harness's import setup."""

    port.prepare_imports()


def layout_of(script_class, tab: str = "txt2img", *, filename: str = "script.py"):
    """``(script, ScriptLayout)`` for ``tab``, built as ``ScriptRunner.build_layouts`` does."""

    from modules.scripts import script_slug
    from remi.ui import ScriptLayout

    script = script_class()
    script.filename = filename
    script.is_img2img = tab == "img2img"
    script.is_txt2img = not script.is_img2img
    script.tabname = tab
    layout = ScriptLayout(script.params(script.is_img2img), owner=script_slug(script.title()), order=script.param_order)
    return script, layout


# region Strings and localization


def _widget_strings(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, list):
        for item in node:
            found |= _widget_strings(item)
        return found
    if not isinstance(node, dict):
        return found
    props = node.get("props") or {}
    if isinstance(props.get("text"), str):
        found.add(props["text"])
    found |= {choice for choice in props.get("choices") or () if isinstance(choice, str)}
    for key in ("children", "enable"):
        if key in node:
            found |= _widget_strings(node[key])
    return found


def schema_strings(node: Any) -> set[str]:
    """Every English string the SPA translates in an exported layout: the
    port harness's (labels, info texts, titles, placeholders, choice labels,
    Markdown) and the ``text`` / ``choices`` props of widgets."""

    return port.ui_strings(node) | _widget_strings(node)


def assert_localized(extension: str | Path, strings: set[str], *, extra: set[str] = frozenset()) -> None:
    """``localizations/zh_CN.json`` translates every string (and ``extra``)."""

    port.assert_localized(extension, [], extra=set(strings) | set(extra))


# endregion

# region Standalone


def assert_standalone(extension: str | Path, scripts: list[str | Path], *, options: dict | None = None) -> None:
    """No JS/CSS asset left in the extension, and its scripts import and
    build their layouts for both tabs in a fresh interpreter."""

    extension = Path(extension)
    assert not list(extension.glob("javascript/*.js")) and not (extension / "style.css").exists(), "JS/CSS assets left"

    code = textwrap.dedent(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("kit", {str(Path(__file__))!r})
        kit = importlib.util.module_from_spec(spec); spec.loader.exec_module(kit)
        kit.prepare_imports()
        from modules import scripts
        from remi.testing import webui_state
        with webui_state({options or {}!r}):
            for index, path in enumerate({[str(Path(script).resolve()) for script in scripts]!r}):
                module = kit.load_script(path, f"standalone_{{index}}")
                for value in vars(module).values():
                    if isinstance(value, type) and issubclass(value, scripts.Script) and value.__module__ == module.__name__:
                        for tab in ("txt2img", "img2img"):
                            kit.layout_of(value, tab)
        print("ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr[-4000:]


# endregion
