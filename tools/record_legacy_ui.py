"""Record a script's Gradio ``ui()`` as a legacy snapshot for its ``params()`` port.

A port of an A1111 / Forge script keeps the positional arguments of its
``ui()`` (API payloads, infotext and queued tasks depend on them). The
extension's tests compare the port with a record of the Gradio ``ui()``
(``tests/fixtures/legacy_ui_<tab>.json``, read by
``tests/remi_port_harness.py``: ``load_snapshot`` / ``assert_matches_snapshot``),
in the format of the harness's ``describe()``.

Remi has no Gradio, so the recording runs in a copy of the last revision that
still had the Gradio UI (``LEGACY_REV``, exported with ``git archive`` to a
temporary folder, or ``--legacy-tree``) with an interpreter that has Gradio 4.40.
That interpreter gets its own environment from ``tools/legacy-ui-requirements.txt``
(``LEGACY_REV``'s requirements with Gradio and the Pillow 10 it needs; the current
``requirements.txt`` cannot be installed next to Gradio 4.40)::

    python3.13 -m venv ../remi-gradio-env      # outside the checkout (and outside /tmp, which macOS purges)
    ../remi-gradio-env/bin/pip install -r tools/legacy-ui-requirements.txt

    # a new upstream extension, taken from upstream/neo:
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py \\
        extensions-builtin/<ext>/scripts/<script>.py --script-rev upstream/neo
    # a script file on disk (an upstream checkout, a third-party extension):
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py /path/to/ext/scripts/x.py --out /path/to/ext/tests/fixtures
    # compare a recording with the committed fixtures instead of writing them:
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py <script> --script-rev <rev> --check
    # check the whole setup end to end: record a known script, compare with its fixtures
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py --self-test

``tools/legacy_ui_env.sh`` builds that environment (network needed once), and
``tools/check.sh`` runs ``--self-test`` as its ``legacy-ui`` step when
``REMI_LEGACY_PYTHON`` names the environment's interpreter.

It writes ``legacy_ui_<tab>.json`` for each tab the script shows on (or the
``--tabs`` given) into ``--out``, by default
``extensions-builtin/<ext>/tests/fixtures`` of this checkout. Only this tool
needs a Gradio interpreter, and the Gradio code it runs (``gr.Blocks``,
``remi.ui.legacy``, the scripts' ``ui()``) is the exported legacy revision's:
this file imports no Gradio module itself, and nothing imports this file.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TABS = ("txt2img", "img2img")
LEGACY_REV = "9c96c45ee561487d2c4baa6900a885d3a71affe5"
"""The last revision with the Gradio UI (``modules/ui*.py``, ``remi/ui/legacy.py``):
the parent of 2b45c9b9, which removed it."""
SELF_TEST = ("extensions-builtin/sd_forge_spectrum/scripts/spectrum.py", "0cf62dd0e8431e3a48827e5d6f739aa96d5601ba")
"""``--self-test``: a script and the revision before its port (the last commit of
its Gradio ``ui()``), whose recording must equal its committed fixtures."""
WIRING_KEYS = ("exclusive", "visible_if", "enabled_if")
"""Keys ``describe()`` added after the fixtures were recorded: absent means none
(as ``tests/remi_port_harness.assert_matches_snapshot`` reads them)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", type=Path, nargs="?", help="the script to record (repo-relative with --script-rev)")
    parser.add_argument("--class", dest="class_name", help="the Script class, when the file defines several")
    parser.add_argument("--tabs", nargs="+", choices=TABS, help="record these tabs whatever show() says (e.g. a panel gated on an optional package)")
    parser.add_argument("--preimport", nargs="+", default=[], help="modules to import first (e.g. to settle a circular import the WebUI's start order avoids)")
    parser.add_argument("--options", type=json.loads, default={}, help='settings for the recording, as JSON (e.g. \'{"extra_options_txt2img": ["sd_vae"]}\')')
    parser.add_argument("--script-rev", help="take the script's extension folder (or the file) from this git revision, e.g. upstream/neo")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--rev", default=LEGACY_REV, help="revision with the Gradio UI to record in (default: %(default).10s)")
    source.add_argument("--legacy-tree", type=Path, help="an existing checkout of that revision instead of a temporary export")
    parser.add_argument("--out", type=Path, help="folder for legacy_ui_<tab>.json (default: extensions-builtin/<ext>/tests/fixtures)")
    parser.add_argument("--check", action="store_true", help="compare the recording with the legacy_ui_<tab>.json in --out (or the default folder) instead of writing; exit 1 when they differ")
    parser.add_argument("--self-test", action="store_true", help=f"record {SELF_TEST[0]} at {SELF_TEST[1][:10]} (before its port) and --check it against its committed fixtures")
    options = parser.parse_args(argv)
    if options.self_test:
        if options.script is not None or options.script_rev or options.out or options.tabs or options.class_name or options.options or options.preimport:
            parser.error("--self-test takes no script and no other option but --rev / --legacy-tree")
        options.script, options.script_rev, options.check = Path(SELF_TEST[0]), SELF_TEST[1], True
    if options.script is None:
        parser.error("the script to record is required (or --self-test)")
    if options.script_rev and options.script.is_absolute():
        parser.error("--script-rev takes a repository-relative script path")
    if options.out is None and extension_folder(options.script) is None:
        parser.error("--out is required for a script outside extensions-builtin/<ext>/scripts")
    return options


def extension_folder(script: Path) -> Path | None:
    """``extensions-builtin/<ext>`` for a script of a built-in extension (path relative to the repo)."""

    parts = (script.resolve().relative_to(REPO) if script.is_absolute() and script.resolve().is_relative_to(REPO) else script).parts
    if len(parts) >= 4 and parts[0] == "extensions-builtin" and parts[2] == "scripts":
        return Path(*parts[:2])
    return None


def output_folder(options: argparse.Namespace) -> Path:
    if options.out is not None:
        return options.out.resolve()
    return REPO / extension_folder(options.script) / "tests" / "fixtures"


def export(rev: str, target: Path, *paths: str) -> None:
    """``git archive`` of ``rev`` (or of ``paths`` in it) extracted into ``target``."""

    archive = subprocess.run(["git", "archive", "--format=tar", rev, *paths], cwd=REPO, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")


def require_gradio() -> None:
    if importlib.util.find_spec("gradio") is None:
        raise SystemExit(f"{sys.executable} cannot import Gradio: run this tool with an interpreter set up from tools/legacy-ui-requirements.txt (docs/remi-webui/extension-guide.md §13)")


def require_legacy_tree(tree: Path) -> None:
    if not (tree / "remi" / "ui" / "legacy.py").is_file() or not (tree / "modules" / "ui_components.py").is_file():
        raise SystemExit(f"{tree} is not a revision with the Gradio UI (remi/ui/legacy.py, modules/ui_components.py): use --rev {LEGACY_REV[:10]}")


def normalized(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**dict.fromkeys(WIRING_KEYS), **entry} for entry in entries]


def compare(recorded: Path, committed: Path) -> list[str]:
    """Differences between a new recording and a committed fixture (empty when they match)."""

    if not committed.is_file():
        return [f"{committed} does not exist"]
    new = normalized(json.loads(recorded.read_text(encoding="utf8")))
    old = normalized(json.loads(committed.read_text(encoding="utf8")))
    problems = [f"{committed.name}: {len(new)} arguments recorded, the fixture has {len(old)}"] if len(new) != len(old) else []
    for index, (got, want) in enumerate(zip(new, old, strict=False)):
        for key in sorted(set(got) | set(want)):
            if got.get(key) != want.get(key):
                problems.append(f"{committed.name}: argument {index} ({want.get('label')!r}) {key} is {got.get(key)!r}, the fixture has {want.get(key)!r}")
    return problems


def load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_snapshot(script, tab: str, describe, build_ui) -> list[dict[str, Any]]:
    """Build ``script``'s Gradio ``ui()`` for ``tab`` and describe it as ``describe`` does.

    ``build_ui`` is the legacy tree's ``tests/extension_script_harness.build_ui``: it
    resets the tab's preset registry and builds the panel in a ``gr.Blocks``, so the
    Gradio code that runs is the legacy revision's own, never this checkout's.
    """

    from modules.anima_presets import preset_controls
    from remi.ui.legacy import legacy_layout  # the legacy tree's Gradio bridge (sys.path)

    script.is_img2img = tab == "img2img"
    script.is_txt2img = not script.is_img2img
    script.infotext_fields = []
    controls = list(build_ui(script, tab) or [])
    layout = legacy_layout("legacy", controls, script.infotext_fields)
    names, components, _values = preset_controls(tab)
    presets = {id(component): name for name, component in zip(names, components, strict=True)}
    entries = describe(layout)
    for entry, control in zip(entries, controls, strict=False):
        entry["preset"] = presets.get(id(control))
    return entries


def record(options: argparse.Namespace, tree: Path, scratch: Path, out: Path | None = None) -> list[Path]:
    """Record in the Gradio-era ``tree`` into ``out`` (default :func:`output_folder`);
    ``scratch`` receives a ``--script-rev`` export."""

    require_legacy_tree(tree)
    out = out or output_folder(options)
    if options.script_rev:
        folder = extension_folder(options.script)
        export(options.script_rev, scratch, folder.as_posix() if folder else options.script.as_posix())
        script_path = scratch / options.script
    else:
        script_path = options.script.resolve()
    if not script_path.is_file():
        raise SystemExit(f"no script at {script_path}")

    # The legacy revision's modules (and its remi.ui) come first; describe() is
    # this checkout's, so the record matches what the port's tests compare.
    describe = load_file(REPO / "tests" / "remi_port_harness.py", "remi_port_harness_current").describe
    sys.argv = sys.argv[:1]
    for path in (str(tree / "modules_forge" / "packages"), str(tree)):
        sys.path.insert(0, path)
    for name in options.preimport:
        importlib.import_module(name)

    from remi.testing import webui_state

    with webui_state():  # the WebUI's import order first, so modules.infotext_utils imports for real
        pass
    harness = load_file(tree / "tests" / "extension_script_harness.py", "extension_script_harness")
    harness.prepare_webui_imports()

    from modules import scripts

    written = []
    with webui_state(options.options):  # default options (as a fresh WebUI start) plus --options
        module = harness.load_script(script_path, f"legacy_snapshot_{script_path.stem}")
        classes = [value for value in vars(module).values() if isinstance(value, type) and issubclass(value, scripts.Script) and value.__module__ == module.__name__]
        if options.class_name:
            classes = [cls for cls in classes if cls.__name__ == options.class_name]
        if len(classes) != 1:
            raise SystemExit(f"expected one script class in {script_path}, found {[cls.__name__ for cls in classes]} (use --class)")
        if classes[0].ui is scripts.Script.ui:
            where = f"at {options.script_rev}" if options.script_rev else "in this file"
            raise SystemExit(f"{classes[0].__name__} has no Gradio ui() {where} (already ported to params()?): record the revision before its port, e.g. --script-rev $(git log -1 --format=%h <port commit>^ -- {options.script})")
        for tab in options.tabs or TABS:
            script = classes[0]()
            script.filename = str(script_path)
            script.is_img2img = tab == "img2img"
            if not options.tabs and not script.show(script.is_img2img):
                continue
            target = out / f"legacy_ui_{tab}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(legacy_snapshot(script, tab, describe, harness.build_ui), indent=1, ensure_ascii=False) + "\n", encoding="utf8")
            written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    require_gradio()
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="remi-legacy-") as folder:
        scratch = Path(folder) / "script"
        if options.legacy_tree is not None:
            tree = options.legacy_tree.resolve()
        else:
            tree = Path(folder) / "tree"
            export(options.rev, tree)
        written = record(options, tree, scratch, Path(folder) / "recorded" if options.check else None)
        if options.check:
            for path in written:
                problems += compare(path, output_folder(options) / path.name)
            written = [output_folder(options) / path.name for path in written]
    if not written:
        print("the script shows on no tab: pass --tabs", file=sys.stderr)
        return 1
    if options.check:
        for problem in problems:
            print(problem, file=sys.stderr)
        status = "differs from" if problems else "matches"
        print(f"the recording {status} " + ", ".join(str(path) for path in written))
        return 1 if problems else 0
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
