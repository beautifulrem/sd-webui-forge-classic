"""Record a script's Gradio ``ui()`` as a legacy snapshot for its ``params()`` port.

A port of an A1111 / Forge script keeps the positional arguments of its
``ui()`` (API payloads, infotext and queued tasks depend on them). The
extension's tests compare the port with a record of the Gradio ``ui()``
(``tests/fixtures/legacy_ui_<tab>.json``, read by
``tests/remi_port_harness.py``: ``load_snapshot`` / ``assert_matches_snapshot``),
in the format of the harness's ``describe()``.

Remi has no Gradio, so the recording runs in a copy of the last revision that
still had the Gradio UI (``LEGACY_REV``, exported with ``git archive`` to a
temporary folder, or ``--legacy-tree``) with an interpreter that has Gradio 4.40::

    python -m venv /tmp/remi-gradio-env
    /tmp/remi-gradio-env/bin/pip install -r requirements.txt "gradio==4.40.0" "gradio_rangeslider==0.0.8"

    # a new upstream extension, taken from upstream/neo:
    /tmp/remi-gradio-env/bin/python tools/record_legacy_ui.py \\
        extensions-builtin/<ext>/scripts/<script>.py --script-rev upstream/neo
    # a script file on disk (an upstream checkout, a third-party extension):
    /tmp/remi-gradio-env/bin/python tools/record_legacy_ui.py /path/to/ext/scripts/x.py --out /path/to/ext/tests/fixtures

It writes ``legacy_ui_<tab>.json`` for each tab the script shows on (or the
``--tabs`` given) into ``--out``, by default
``extensions-builtin/<ext>/tests/fixtures`` of this checkout. Only this tool
needs Gradio; nothing imports it.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", type=Path, help="the script to record (repo-relative with --script-rev)")
    parser.add_argument("--class", dest="class_name", help="the Script class, when the file defines several")
    parser.add_argument("--tabs", nargs="+", choices=TABS, help="record these tabs whatever show() says (e.g. a panel gated on an optional package)")
    parser.add_argument("--preimport", nargs="+", default=[], help="modules to import first (e.g. to settle a circular import the WebUI's start order avoids)")
    parser.add_argument("--options", type=json.loads, default={}, help='settings for the recording, as JSON (e.g. \'{"extra_options_txt2img": ["sd_vae"]}\')')
    parser.add_argument("--script-rev", help="take the script's extension folder (or the file) from this git revision, e.g. upstream/neo")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--rev", default=LEGACY_REV, help="revision with the Gradio UI to record in (default: %(default).10s)")
    source.add_argument("--legacy-tree", type=Path, help="an existing checkout of that revision instead of a temporary export")
    parser.add_argument("--out", type=Path, help="folder for legacy_ui_<tab>.json (default: extensions-builtin/<ext>/tests/fixtures)")
    options = parser.parse_args(argv)
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
        raise SystemExit(f"{sys.executable} cannot import Gradio: run this tool with an interpreter that has gradio==4.40.0 (see --help)")


def require_legacy_tree(tree: Path) -> None:
    if not (tree / "remi" / "ui" / "legacy.py").is_file() or not (tree / "modules" / "ui_components.py").is_file():
        raise SystemExit(f"{tree} is not a revision with the Gradio UI (remi/ui/legacy.py, modules/ui_components.py): use --rev {LEGACY_REV[:10]}")


def load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_snapshot(script, tab: str, describe) -> list[dict[str, Any]]:
    """Build ``script``'s Gradio ``ui()`` for ``tab`` and describe it as ``describe`` does."""

    import gradio as gr

    from modules.anima_presets import preset_controls, reset_preset_controls
    from remi.ui.legacy import legacy_layout

    reset_preset_controls(tab)
    script.tabname = tab
    script.is_img2img = tab == "img2img"
    script.is_txt2img = not script.is_img2img
    script.infotext_fields = []
    with gr.Blocks():
        controls = list(script.ui(script.is_img2img) or [])
    layout = legacy_layout("legacy", controls, script.infotext_fields)
    names, components, _values = preset_controls(tab)
    presets = {id(component): name for name, component in zip(names, components, strict=True)}
    entries = describe(layout)
    for entry, control in zip(entries, controls, strict=False):
        entry["preset"] = presets.get(id(control))
    return entries


def record(options: argparse.Namespace, tree: Path, scratch: Path) -> list[Path]:
    """Record in the Gradio-era ``tree``; ``scratch`` receives a ``--script-rev`` export."""

    require_legacy_tree(tree)
    out = output_folder(options)
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
        for tab in options.tabs or TABS:
            script = classes[0]()
            script.filename = str(script_path)
            script.is_img2img = tab == "img2img"
            if not options.tabs and not script.show(script.is_img2img):
                continue
            target = out / f"legacy_ui_{tab}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(legacy_snapshot(script, tab, describe), indent=1, ensure_ascii=False) + "\n", encoding="utf8")
            written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    require_gradio()
    with tempfile.TemporaryDirectory(prefix="remi-legacy-") as folder:
        scratch = Path(folder) / "script"
        if options.legacy_tree is not None:
            tree = options.legacy_tree.resolve()
        else:
            tree = Path(folder) / "tree"
            export(options.rev, tree)
        written = record(options, tree, scratch)
    for path in written:
        print(path)
    if not written:
        print("the script shows on no tab: pass --tabs", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
