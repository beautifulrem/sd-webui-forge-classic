"""Coverage floors of tools/check.sh: ``remi/`` as a whole, and every Remi file outside it.

Remi's own Python outside ``remi/`` (``modules/redact.py``, the ControlNet unit
editor, the extensions' ``params()`` libraries, ...) is listed in
``pyproject.toml``'s strict ruff list (the negated ``per-file-ignores`` glob that
switches the strict rules on). Those files are tested by different suites -
``tests/`` or their extension's own ``tests/`` - so every pytest step of
``tools/check.sh`` measures them (``sources``) into a data file of its own, and
``strict`` combines those files and holds each Remi file to the floor on its own
(a small file cannot hide behind a large one)::

    tools/check_coverage.py sources                        # the --cov options for a pytest run
    tools/check_coverage.py files                          # the Remi files outside remi/ under the floor
    tools/check_coverage.py remi --data .coverage.tests --floor 92
    tools/check_coverage.py strict --data .coverage.* --floor 90

A file no pytest step imported counts as 0 %. New Remi files join the strict list
(pyproject.toml says so), and with it this floor.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMI_SCOPES = ("remi/", "tools/", "webui/tools/", "docs/")
"""Remi's own folders: ``remi/`` has its own floor, ``tools/``, ``webui/tools/`` and ``docs/`` hold
command-line scripts (tested where they have logic, not held to a floor)."""


def strict_patterns(pyproject: Path = REPO / "pyproject.toml") -> list[str]:
    """The globs of pyproject.toml's strict list (the negated per-file-ignores entry)."""

    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]
    negated = [key for key in ignores if key.startswith("!{") and key.endswith("}")]
    if len(negated) != 1:
        raise SystemExit(f"expected one negated per-file-ignores entry (the strict list) in {pyproject}, found {len(negated)}")
    return [pattern.strip() for pattern in negated[0][2:-1].split(",") if pattern.strip()]


def is_test_file(path: str) -> bool:
    return path.startswith("tests/") or "/tests/" in path


def strict_files(root: Path = REPO, patterns: list[str] | None = None) -> list[str]:
    """Repository-relative paths of the Remi runtime files outside Remi's own folders (no tests)."""

    found = set()
    for pattern in strict_patterns(root / "pyproject.toml") if patterns is None else patterns:
        if pattern.startswith(REMI_SCOPES) or is_test_file(pattern):
            continue
        matches = [path for path in root.glob(pattern) if path.is_file() and path.suffix == ".py"]
        if not matches:
            raise SystemExit(f"the strict list in pyproject.toml names {pattern}, which does not exist")
        found.update(path.relative_to(root).as_posix() for path in matches)
    return sorted(found)


def cov_sources(files: list[str]) -> list[str]:
    """``--cov`` sources measuring ``files``: a module name for an importable core module, else its folder."""

    sources = []
    for file in files:
        parts = Path(file).with_suffix("").parts
        source = ".".join(parts) if parts[0] in {"modules", "modules_forge", "backend"} else Path(file).parent.as_posix()
        if source not in sources:
            sources.append(source)
    return sources


def load(data_files: list[Path], combined: Path):
    """A Coverage object with the data of every ``data_files`` (kept), saved to ``combined``."""

    import coverage

    present = [str(path) for path in data_files if path.is_file() and path != combined]
    combined.unlink(missing_ok=True)  # combine() adds to an existing file
    cov = coverage.Coverage(data_file=str(combined))
    cov.combine(data_paths=present, keep=True)
    cov.save()
    cov.load()
    return cov, present


def file_coverage(cov, root: Path, file: str) -> tuple[float, int, list[int]]:
    """``(percent, statements, missing lines)`` of a repository file (0 % when no run imported it)."""

    path = str((root / file).resolve())
    measured = {str(Path(name).resolve()) for name in cov.get_data().measured_files()}
    _name, statements, _excluded, missing, _text = cov.analysis2(path)
    if path not in measured:
        missing = statements
    percent = 100.0 if not statements else 100.0 * (len(statements) - len(missing)) / len(statements)
    return percent, len(statements), list(missing)


def ranges(lines: list[int]) -> str:
    """``1-3, 7`` for ``[1, 2, 3, 7]``."""

    spans: list[list[int]] = []
    for line in sorted(lines):
        if spans and line == spans[-1][1] + 1:
            spans[-1][1] = line
        else:
            spans.append([line, line])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def check_strict(data_files: list[Path], floor: float, root: Path = REPO, combined: Path | None = None) -> int:
    files = strict_files(root)
    cov, present = load(data_files, combined or data_files[0].with_name(".coverage.strict"))
    if not present:
        print("no coverage data: run the pytest steps first (tools/check.sh --only pytest)", file=sys.stderr)
        return 2
    failed = 0
    print(f"{'file':<80} {'stmts':>6} {'cover':>7}  missing")
    for file in files:
        percent, statements, missing = file_coverage(cov, root, file)
        low = percent < floor
        failed += low
        print(f"{file:<80} {statements:>6} {percent:>6.1f}%  {ranges(missing)}{'   < ' + str(floor) + ' %' if low else ''}")
    print(f"{len(files)} Remi files outside remi/, floor {floor} % each, data from {len(present)} pytest runs")
    if failed:
        print(f"{failed} below the floor: test the missing lines", file=sys.stderr)
        return 1
    return 0


def check_remi(data_file: Path, floor: float, root: Path = REPO) -> int:
    import coverage

    if not data_file.is_file():
        print(f"no coverage data at {data_file}", file=sys.stderr)
        return 2
    cov = coverage.Coverage(data_file=str(data_file))
    cov.load()
    total = cov.report(include=[str(root / "remi" / "*")], skip_covered=True, show_missing=False)
    if total < floor:
        print(f"remi/ line coverage {total:.1f} % is below the floor of {floor} %", file=sys.stderr)
        return 1
    print(f"remi/ line coverage {total:.1f} % (floor {floor} %)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sources", help="print the --cov options that measure the Remi files outside remi/")
    commands.add_parser("files", help="list the Remi files outside remi/ held to the floor")
    for name, text in (("remi", "fail when remi/ as a whole is below --floor"), ("strict", "fail when any Remi file outside remi/ is below --floor")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--data", type=Path, nargs="+", required=True, help="coverage data files (pytest-cov's COVERAGE_FILE)")
        command.add_argument("--floor", type=float, required=True, help="minimum line coverage in percent")
    options = parser.parse_args(argv)
    if options.command == "sources":
        print(" ".join(f"--cov={source}" for source in cov_sources(strict_files())))
        return 0
    if options.command == "files":
        print("\n".join(strict_files()))
        return 0
    if options.command == "remi":
        return check_remi(options.data[0], options.floor)
    return check_strict(options.data, options.floor)


if __name__ == "__main__":
    sys.exit(main())
