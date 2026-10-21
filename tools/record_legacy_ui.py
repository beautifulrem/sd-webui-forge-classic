"""Record a script's Gradio ``ui()`` as a legacy snapshot for its ``params()`` port.

A port of an A1111 / Forge script keeps the positional arguments of its
``ui()`` (API payloads, infotext and queued tasks depend on them). The
extension's tests compare the port with a record of the Gradio ``ui()``
(``tests/fixtures/legacy_ui_<tab>.json``, read by
``tests/remi_port_harness.py``: ``load_snapshot`` / ``assert_matches_snapshot``),
in the format of the harness's ``describe()``.

Remi has no Gradio, so the recording runs in a copy of the last revision that
still had the Gradio UI (``LEGACY_REV``, exported with ``git archive`` to a
temporary folder, or ``--legacy-tree``) with an interpreter that has the Gradio
of that era. That interpreter gets an environment of its own, outside the
checkout, whose requirements ``--requirements`` derives from the git history:
``LEGACY_REV``'s ``requirements.txt`` plus the Gradio packages the last
Gradio-era launcher installed (``GRADIO_LAUNCHER_REV``), with the Pillow 10 that
Gradio needs (the current ``requirements.txt`` cannot be installed next to it).
No file of the checkout lists Gradio as a dependency::

    tools/legacy_ui_env.sh                     # builds ../remi-gradio-env from --requirements, runs --self-test
    # or by hand, outside the checkout (and outside /tmp, which macOS purges):
    python3 tools/record_legacy_ui.py --requirements > ../remi-gradio-requirements.txt
    python3.13 -m venv ../remi-gradio-env
    ../remi-gradio-env/bin/pip install -r ../remi-gradio-requirements.txt

    # a new upstream extension, taken from upstream/neo:
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py \\
        extensions-builtin/<ext>/scripts/<script>.py --script-rev upstream/neo
    # a script file on disk (an upstream checkout, a third-party extension):
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py /path/to/ext/scripts/x.py --out /path/to/ext/tests/fixtures
    # compare a recording with the committed fixtures instead of writing them:
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py <script> --script-rev <rev> --check
    # check the whole setup end to end: record a known script, compare with its fixtures
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py --self-test
    # record every committed fixture again and compare (or rewrite) them, in their own formats
    ../remi-gradio-env/bin/python tools/record_legacy_ui.py --fixtures [NAME ...] --check

``tools/legacy_ui_recordings.py`` lists every committed ``legacy_ui_*.json``
with the script and revision it was recorded from, its format (several older
ports use the dict formats of their own harnesses) and the WebUI context its
``ui()`` needs; a script listed there records in that format and context when
given directly too. Every ``ui()`` is built as the WebUI built it: the core
components the scripts wire to (``on_after_component``) first, then the panel.

``tools/legacy_ui_env.sh`` builds that environment (network needed once), and
``tools/check.sh`` runs ``--self-test`` as its ``legacy-ui`` step with
``REMI_LEGACY_PYTHON``, else ``../remi-gradio-env/bin/python`` (a ``skip`` with
the reason when neither exists). ``--report <file>.json`` keeps a ``--check`` /
``--self-test`` result as dated evidence (``docs/research/legacy-ui/``).

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
import importlib.metadata
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TABS = ("txt2img", "img2img")
LEGACY_REV = "9c96c45ee561487d2c4baa6900a885d3a71affe5"
"""The last revision with the Gradio UI (``modules/ui*.py``, ``remi/ui/legacy.py``):
the parent of 2b45c9b9, which removed it."""
GRADIO_LAUNCHER_REV = "fa02febfe42a0df93ffb15e1d52781ce9fa1928e"
"""The last revision whose launcher installed Gradio (``modules/launch_utils.py``,
``GRADIO_PACKAGE``): the parent of 80cfa274, which stopped installing it. The
recorder's environment takes its Gradio pins from there."""
LEGACY_ENV_PINS = {"pillow": "Pillow==10.4.0", "pillow-heif": "pillow-heif==0.22.0", "pillow-jxl-plugin": "pillow-jxl-plugin==1.3.8"}
"""Replace ``LEGACY_REV``'s Pillow pins (12.x, which that Gradio cannot be installed
next to: it requires ``pillow<11``): the last Pillow 10 and the last releases of
the Pillow plugins the legacy ``modules/images.py`` imports that support it."""
LEGACY_ENV_EXTRA = ("socksio==1.0.0",)
"""Added: httpx (Gradio's client) needs socksio when ALL_PROXY / HTTPS_PROXY is a socks5:// proxy."""
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
    parser.add_argument("--report", type=Path, help="with --check / --self-test: also write the result as JSON evidence (date, revision, interpreter, compared files, differences), e.g. docs/research/legacy-ui/<date>.json")
    parser.add_argument("--requirements", action="store_true", help="print the requirements of this tool's own interpreter, derived from the git history (any Python 3; tools/legacy_ui_env.sh installs them)")
    parser.add_argument("--fixtures", nargs="*", metavar="NAME", help="record the committed fixtures again (tools/legacy_ui_recordings.py; every one when no NAME is given), each with its format and context: rewrite them, or compare with --check")
    parser.add_argument("--jobs", type=int, default=4, help="--fixtures: recordings run at once, each in its own interpreter (default: %(default)s)")
    parser.add_argument("--record-fixture", metavar="NAME", help=argparse.SUPPRESS)  # one --fixtures recording (a subprocess)
    options = parser.parse_args(argv)
    if options.fixtures is not None or options.record_fixture:
        return check_fixture_options(parser, options)
    if options.requirements:
        if options.script is not None or options.self_test or options.script_rev or options.out or options.tabs or options.class_name or options.options or options.preimport or options.check or options.report is not None or options.legacy_tree is not None:
            parser.error("--requirements takes no script and no other option but --rev")
        return options
    if options.self_test:
        if options.script is not None or options.script_rev or options.out or options.tabs or options.class_name or options.options or options.preimport:
            parser.error("--self-test takes no script and no other option but --rev / --legacy-tree / --report")
        options.script, options.script_rev, options.check = Path(SELF_TEST[0]), SELF_TEST[1], True
    if options.report is not None and not options.check:
        parser.error("--report records a comparison: use it with --check or --self-test")
    if options.script is None:
        parser.error("the script to record is required (or --self-test)")
    if options.script_rev and options.script.is_absolute():
        parser.error("--script-rev takes a repository-relative script path")
    if options.out is None and extension_folder(options.script) is None:
        parser.error("--out is required for a script outside extensions-builtin/<ext>/scripts")
    return options


def check_fixture_options(parser: argparse.ArgumentParser, options: argparse.Namespace) -> argparse.Namespace:
    """``--fixtures``: names of :data:`legacy_ui_recordings.RECORDINGS`, no script or per-script option."""

    if options.script is not None or options.self_test or options.requirements or options.script_rev or options.tabs or options.class_name or options.options or options.preimport:
        parser.error("--fixtures takes fixture names (RECORDINGS in tools/legacy_ui_recordings.py), no script and no per-script option")
    names = recordings().BY_NAME
    unknown = [name for name in [*(options.fixtures or ()), *([options.record_fixture] if options.record_fixture else [])] if name not in names]
    if unknown:
        parser.error(f"unknown fixtures {unknown}: one of {sorted(names)}")
    if options.report is not None and not options.check:
        parser.error("--report records a comparison: use it with --check or --self-test")
    if options.record_fixture and (options.legacy_tree is None or options.out is None):
        parser.error("--record-fixture needs --legacy-tree and --out")
    if options.jobs < 1:
        parser.error("--jobs must be at least 1")
    return options


def recordings():
    """``tools/legacy_ui_recordings.py``: every committed fixture, its format and context (loaded by path)."""

    module = sys.modules.get("legacy_ui_recordings")
    if module is None:
        spec = importlib.util.spec_from_file_location("legacy_ui_recordings", REPO / "tools" / "legacy_ui_recordings.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # its dataclasses resolve their module
        spec.loader.exec_module(module)
    return module


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


def git_show(rev: str, path: str) -> str:
    """``path`` at ``rev`` (``git show``), or SystemExit when the clone does not have it."""

    shown = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO, capture_output=True, text=True, encoding="utf8")
    if shown.returncode != 0:
        raise SystemExit(f"cannot read {path} at {rev[:10]} (a shallow clone? fetch the full history): {shown.stderr.strip()}")
    return shown.stdout


def requirement_name(line: str) -> str:
    """The normalized project name of a requirements line (``Pillow==12.3.0`` is ``pillow``)."""

    return re.sub(r"[-_.]+", "-", re.split(r"[\s=<>!~;\[]", line, maxsplit=1)[0]).lower()


def launcher_gradio_packages(source: str) -> list[str]:
    """The ``GRADIO_PACKAGE`` default of a Gradio-era ``modules/launch_utils.py``."""

    match = re.search(r"""os\.environ\.get\(\s*["']GRADIO_PACKAGE["']\s*,\s*["']([^"']+)["']""", source)
    if match is None:
        raise SystemExit("the launcher at GRADIO_LAUNCHER_REV has no GRADIO_PACKAGE default")
    return match.group(1).split()


def legacy_requirements(rev: str = LEGACY_REV, launcher_rev: str = GRADIO_LAUNCHER_REV) -> str:
    """The requirements of the recorder's interpreter: ``rev``'s ``requirements.txt``
    with :data:`LEGACY_ENV_PINS` for its Pillow pins, the Gradio packages the
    launcher at ``launcher_rev`` installed, and :data:`LEGACY_ENV_EXTRA`. Derived
    from the history, so no file of the checkout names Gradio as a dependency."""

    lines = [
        "# The interpreter of tools/record_legacy_ui.py only (never the WebUI's): generated by its --requirements",
        f"# from {rev[:10]}'s requirements.txt and the Gradio packages of {launcher_rev[:10]}'s launcher.",
        *launcher_gradio_packages(git_show(launcher_rev, "modules/launch_utils.py")),
        *LEGACY_ENV_PINS.values(),
        *LEGACY_ENV_EXTRA,
    ]
    for raw in git_show(rev, "requirements.txt").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and requirement_name(line) not in LEGACY_ENV_PINS:
            lines.append(line)
    return "\n".join(lines) + "\n"


def require_gradio() -> None:
    if importlib.util.find_spec("gradio") is None:
        raise SystemExit(f"{sys.executable} cannot import Gradio: run this tool with the interpreter tools/legacy_ui_env.sh builds (docs/remi-webui/extension-guide.md §13)")


def require_legacy_tree(tree: Path) -> None:
    if not (tree / "remi" / "ui" / "legacy.py").is_file() or not (tree / "modules" / "ui_components.py").is_file():
        raise SystemExit(f"{tree} is not a revision with the Gradio UI (remi/ui/legacy.py, modules/ui_components.py): use --rev {LEGACY_REV[:10]}")


def normalized(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**dict.fromkeys(WIRING_KEYS), **entry} for entry in entries]


def compare(recorded: Path, committed: Path, fmt: str | None = None) -> list[str]:
    """Differences between a new recording and a committed fixture (empty when they match).

    A bare-list fixture (``remi_port_harness``) is compared argument by argument;
    a fixture of another format (``fmt``, else read from its shape:
    ``legacy_ui_recordings.format_of``) value by value, every key of both."""

    if not committed.is_file():
        return [f"{committed} does not exist"]
    got = json.loads(recorded.read_text(encoding="utf8"))
    want = json.loads(committed.read_text(encoding="utf8"))
    if not (isinstance(got, list) and isinstance(want, list)):
        formats = recordings()
        if formats.format_of(got) != formats.format_of(want):
            return [f"{committed.name}: the fixture is in the {formats.format_of(want)!r} format, the recording in {formats.format_of(got)!r} (record it with --fixtures)"]
        return formats.compare_data(got, want, fmt or formats.format_of(want), committed.name)
    new, old = normalized(got), normalized(want)
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
    formats = recordings()
    known = formats.recording_of(options.script)
    if known is not None:  # a committed fixture's script: its format and context (tools/legacy_ui_recordings.py)
        return formats.record(formats.with_options(known, options), tree, scratch, out, export, script_path=script_path)

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
            # built as the WebUI did: the core components first (formats.build), so a script that
            # wires itself to them through on_after_component finds them
            build_ui = lambda script, tab: formats.build(script, tab, check_shown=False).controls  # noqa: E731
            target.write_text(json.dumps(legacy_snapshot(script, tab, describe, build_ui), indent=1, ensure_ascii=False) + "\n", encoding="utf8")
            written.append(target)
    return written


def report(options: argparse.Namespace, compared: list[Path], problems: list[str]) -> dict[str, Any]:
    """A ``--check`` run as committable evidence (``--report``): what was recorded where, and the outcome."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def shown(path: Path) -> str:
        return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)

    return {
        "tool": "tools/record_legacy_ui.py",
        "mode": "self-test" if options.self_test else "fixtures" if options.fixtures is not None else "check",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "revision": git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "gradio": version("gradio"),
        "legacy_rev": str(options.legacy_tree) if options.legacy_tree is not None else options.rev,
        "script": options.script.as_posix() if options.script is not None else None,
        "script_rev": options.script_rev,
        "fixtures": [entry.name for entry in fixture_recordings(options)] if options.fixtures is not None else None,
        "compared": [shown(path) for path in compared],
        "differences": problems,
        "passed": not problems,
    }


def write_report(options: argparse.Namespace, compared: list[Path], problems: list[str]) -> None:
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report(options, compared, problems), indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"report: {options.report}")


def fixture_recordings(options: argparse.Namespace) -> list[Any]:
    """The ``--fixtures`` named (all of :data:`legacy_ui_recordings.RECORDINGS` when none is)."""

    formats = recordings()
    return [formats.BY_NAME[name] for name in options.fixtures] if options.fixtures else list(formats.RECORDINGS)


def record_fixture(options: argparse.Namespace) -> int:
    """``--record-fixture``: one recording of ``--fixtures``, in this interpreter, into ``--out``."""

    with tempfile.TemporaryDirectory(prefix="remi-legacy-") as folder:
        tree = options.legacy_tree.resolve()
        require_legacy_tree(tree)
        for path in recordings().record(recordings().BY_NAME[options.record_fixture], tree, Path(folder) / "script", options.out.resolve(), export):
            print(path)
    return 0


def run_fixtures(options: argparse.Namespace) -> int:
    """``--fixtures``: record each committed fixture again in the legacy revision (one
    interpreter per recording, ``--jobs`` at once), then compare it with the committed
    file (``--check``) or write it over it."""

    entries = fixture_recordings(options)
    problems: list[str] = []
    compared: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="remi-legacy-") as folder:
        if options.legacy_tree is not None:
            tree = options.legacy_tree.resolve()
        else:
            tree = Path(folder) / "tree"
            export(options.rev, tree)
        require_legacy_tree(tree)

        def run(entry) -> tuple[Any, Path, subprocess.CompletedProcess]:
            out = Path(folder) / "recorded" / entry.name.replace("/", "__")
            command = [sys.executable, str(Path(__file__).resolve()), "--record-fixture", entry.name, "--legacy-tree", str(tree), "--out", str(out)]
            return entry, out, subprocess.run(command, cwd=REPO, capture_output=True, text=True, encoding="utf8", errors="replace")

        with ThreadPoolExecutor(max_workers=options.jobs) as pool:
            for entry, out, result in pool.map(run, entries):
                if result.returncode != 0:
                    tail = "\n    ".join((result.stderr or result.stdout).strip().splitlines()[-12:])
                    problems.append(f"{entry.name}: the recording failed (exit {result.returncode}):\n    {tail}")
                    print(f"{entry.name}: the recording failed", file=sys.stderr)
                    continue
                found = []
                for path in entry.files.values():
                    recorded, committed = out / Path(path).name, REPO / path
                    if options.check:
                        found += compare(recorded, committed, entry.format)
                        compared.append(committed)
                    else:
                        committed.parent.mkdir(parents=True, exist_ok=True)
                        committed.write_text(recorded.read_text(encoding="utf8"), encoding="utf8")
                        print(committed.relative_to(REPO))
                problems += [f"{entry.name}: {problem}" for problem in found]
                if options.check:
                    print(f"{entry.name}: {'differs' if found else 'matches'} ({', '.join(entry.files.values())})")
    for problem in problems:
        print(problem, file=sys.stderr)
    if options.check:
        print(f"{len(entries) - len({problem.split(':', 1)[0] for problem in problems})} of {len(entries)} recordings match their committed fixtures")
        if options.report is not None:
            write_report(options, compared, problems)
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if options.requirements:
        sys.stdout.write(legacy_requirements(options.rev))
        return 0
    require_gradio()
    if options.record_fixture:
        return record_fixture(options)
    if options.fixtures is not None:
        return run_fixtures(options)
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
        if options.report is not None:
            write_report(options, written, problems)
        return 1 if problems else 0
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
