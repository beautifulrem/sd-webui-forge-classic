"""Every third-party module the WebUI imports at module level comes from
requirements.txt (or what those packages depend on): the WebUI starts in an
environment with nothing else installed. Imports inside functions and
``try`` blocks are optional features and are not checked."""

from __future__ import annotations

import ast
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ("webui.py", "launch.py", "modules", "modules_forge", "remi", "scripts", "extensions-builtin")
SKIPPED = ("modules_forge/packages", "extensions-builtin/forge_legacy_preprocessors", "/tests/", "/install.py", "/preload.py")
# Installed by the launcher itself (modules/launch_utils.py), not by requirements.txt
LAUNCHER = {"torch", "torchvision"}
# Modules imported only on a path that needs them, each with the reason
OPTIONAL = {
    "ngrok": "modules/ngrok.py is imported only with --ngrok, whose launcher step installs it",
    "ldm_patched": "lib_negpip's fallback for the classic Forge backend (IS_NEO is false only there)",
}


def _name(requirement: str) -> str:
    return re.split(r"[\s;<>=!~\[]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _closure(names: set[str]) -> set[str]:
    """``names`` and every distribution they require, as installed here."""

    seen: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            requires = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        pending += [_name(item) for item in requires if "extra ==" not in item]
    return seen


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    found: set[str] = set()

    def visit(body):
        for node in body:
            if isinstance(node, ast.Import):
                found.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.partition(".")[0])
            elif isinstance(node, ast.If) and not (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"):
                visit(node.body)
                visit(node.orelse)

    visit(tree.body)
    return found


def _local_names() -> set[str]:
    names = {path.stem for path in ROOT.iterdir()}
    names |= {path.name for path in (ROOT / "modules_forge" / "packages").iterdir()}
    for extension in (ROOT / "extensions-builtin").iterdir():
        for folder in [extension, *extension.glob("*")]:
            if folder.is_dir():
                names |= {path.stem for path in folder.iterdir()}
    return names


def test_module_level_imports_are_declared_dependencies():
    requirements = {_name(line) for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    allowed = _closure(requirements | LAUNCHER)
    distributions = metadata.packages_distributions()
    local = _local_names()
    undeclared: dict[str, list[str]] = {}
    for entry in CODE:
        paths = [ROOT / entry] if entry.endswith(".py") else sorted((ROOT / entry).rglob("*.py"))
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            if any(part in f"/{relative}" for part in SKIPPED):
                continue
            for module in _top_level_imports(path):
                if module in sys.stdlib_module_names or module in local or module in OPTIONAL or module == "__future__":
                    continue
                owners = {_name(owner) for owner in distributions.get(module, ())}
                if not owners & allowed:
                    undeclared.setdefault(module, []).append(relative)
    assert undeclared == {}, undeclared
