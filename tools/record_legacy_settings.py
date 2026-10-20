"""Record the settings the built-in extensions and scripts registered before the rewrite.

    python tools/record_legacy_settings.py           # writes tests/fixtures/legacy_ui/extension_settings.json
    python tools/record_legacy_settings.py --check   # exit 1 when the file differs from a new recording

Every setting an extension added to Settings before the rewrite (``LEGACY_REV``,
the Forge Neo merge the Remi branch starts from) must still be a setting, or be
listed with its reason in the "Not carried over" table of the extension's README
(``tests/test_remi_extension_settings.py``). It is not ``tools/record_legacy_ui.py``'s
revision (the last one with the Gradio UI): by then the rewrite had already moved
tagcomplete's, Agent Scheduler's, Prompt All-in-One's and the Anima panels' settings,
so recording there would hide what was dropped. This tool reads that revision's
sources with ``git show`` (nothing is imported or run, so no Gradio is needed)
and collects, per extension folder (``extensions-builtin/<name>``) or script file
(``scripts/*.py``, ``modules/processing_scripts/*.py``), the keys of:

- ``shared.opts.add_option("key", ...)`` and the Anima panels'
  ``register_language_option(KEY, ...)`` (a module-level string constant of the
  same extension),
- ``{"key": shared.OptionInfo(...)}`` dictionaries (``options_section``, the
  tables tagcomplete registers in a loop).
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_REV = "d5a71cad"
SOURCES = ("extensions-builtin", "scripts", "modules/processing_scripts")
OUT = ROOT / "tests" / "fixtures" / "legacy_ui" / "extension_settings.json"
REGISTRATION_CALLS = {"add_option", "register_language_option"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, encoding="utf-8").stdout


def owner(path: str) -> str:
    """The extension folder of ``path``, or the script file itself."""

    parts = path.split("/")
    return "/".join(parts[:2]) if parts[0] == "extensions-builtin" else path


def python_sources(revision: str) -> dict[str, str]:
    files = [path for path in git("ls-tree", "-r", "--name-only", revision, *SOURCES).splitlines() if path.endswith(".py") and "/tests/" not in path]
    # One `git cat-file --batch` for all the files: "<sha> blob <size>\n<content>\n" each
    request = "".join(f"{revision}:{path}\n" for path in files).encode("utf-8")
    output = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, input=request, capture_output=True, check=True).stdout
    sources, position = {}, 0
    for path in files:
        end = output.index(b"\n", position)
        size = int(output[position:end].split()[2])
        sources[path] = output[end + 1 : end + 1 + size].decode("utf-8")
        position = end + 1 + size + 1
    return sources


def _string_constants(tree: ast.AST) -> dict[str, str]:
    return {target.id: node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) for target in node.targets if isinstance(target, ast.Name)}


def _is_option_info(node: ast.AST) -> bool:
    """``OptionInfo(...)``, possibly followed by ``.info(...)`` / ``.needs_reload_ui()`` ..."""

    while isinstance(node, ast.Call):
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", None)
        if name == "OptionInfo":
            return True
        if not isinstance(function, ast.Attribute):
            return False
        node = function.value
    return False


def setting_keys(trees: list[ast.AST]) -> set[str]:
    """The setting keys ``trees`` (the files of one owner) register."""

    constants: dict[str, str] = {}
    for tree in trees:
        constants.update(_string_constants(tree))
    keys: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args:
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", None)
                if name in REGISTRATION_CALLS:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        keys.add(first.value)
                    elif isinstance(first, ast.Name) and first.id in constants:
                        keys.add(constants[first.id])
            elif isinstance(node, ast.Dict):
                keys.update(key.value for key, value in zip(node.keys, node.values, strict=True) if isinstance(key, ast.Constant) and isinstance(key.value, str) and _is_option_info(value))
    return keys


def record(revision: str = LEGACY_REV) -> dict:
    grouped: dict[str, list[str]] = {}
    for path, source in python_sources(revision).items():
        grouped.setdefault(owner(path), []).append(source)
    trees: dict[str, list[ast.AST]] = {}
    for name, sources in grouped.items():
        # only the owners that register anything (the parse is the slow part)
        if not any(token in source for source in sources for token in ("OptionInfo", *REGISTRATION_CALLS)):
            continue
        for source in sources:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)  # invalid escapes in the legacy sources
                    tree = ast.parse(source)
            except SyntaxError:
                continue
            trees.setdefault(name, []).append(tree)
    settings = {name: sorted(keys) for name, files in sorted(trees.items()) if (keys := setting_keys(files))}
    return {"revision": revision, "settings": settings}


def render(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the committed file differs from a new recording")
    parser.add_argument("--revision", default=LEGACY_REV, help=f"the legacy revision (default {LEGACY_REV})")
    args = parser.parse_args()
    text = render(record(args.revision))
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is out of date: run tools/record_legacy_settings.py")
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    count = sum(len(keys) for keys in json.loads(text)["settings"].values())
    print(f"wrote {OUT.relative_to(ROOT)} ({count} settings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
