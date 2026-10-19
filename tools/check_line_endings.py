"""Keep upstream's line endings on the files Remi shares with Forge Neo.

Forge Neo commits about a hundred files with CRLF line endings
(``modules/scripts.py``, ``modules/processing.py``, …). When an editor or a
tool saves such a file with LF, every line differs from upstream: a Git merge
of the next Neo update conflicts on the whole file and ``git diff`` against
upstream hides the real change. So files that came from upstream keep
upstream's line endings byte for byte (there is no ``.gitattributes``
conversion: the repository stores exactly what upstream stores).

``tools/upstream_crlf.txt`` lists the files that are CRLF in the last absorbed
upstream commit and still exist in Remi.

    $PYTHON tools/check_line_endings.py            # exit 1 when a listed file has a bare LF
    $PYTHON tools/check_line_endings.py --fix      # rewrite the listed files with CRLF
    $PYTHON tools/check_line_endings.py --update   # regenerate the list from the merge base
                                                   # with upstream/neo (after a sync, §3 of
                                                   # docs/remi-webui/upstream-absorption.md)
    $PYTHON tools/check_line_endings.py --update --base 46af880c

``$PYTHON``: any Python 3.10+ (standard library only). ``tools/check.sh`` runs
the check as ``contract:line-endings``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "upstream_crlf.txt")
HEADER = (
    "# Files that are CRLF in the last absorbed Forge Neo commit and still exist in Remi.\n"
    "# They must keep CRLF line endings: tools/check_line_endings.py (--update regenerates this list).\n"
)


def read_manifest(path: str = MANIFEST) -> list[str]:
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


def write_manifest(files: list[str], path: str = MANIFEST) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(HEADER + "".join(f"{name}\n" for name in sorted(files)))


def bare_lf_count(data: bytes) -> int:
    """Line feeds that are not part of a CRLF pair."""
    return data.count(b"\n") - data.count(b"\r\n")


def to_crlf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def problems(files: list[str], root: str = ROOT) -> list[str]:
    found = []
    for name in files:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            found.append(f"{name}: listed but missing (deleted? run --update)")
            continue
        with open(path, "rb") as file:
            count = bare_lf_count(file.read())
        if count:
            found.append(f"{name}: {count} LF line ending(s), upstream uses CRLF")
    return found


def fix(files: list[str], root: str = ROOT) -> list[str]:
    changed = []
    for name in files:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as file:
            data = file.read()
        converted = to_crlf(data)
        if converted != data:
            with open(path, "wb") as file:
                file.write(converted)
            changed.append(name)
    return changed


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def upstream_crlf_files(base: str) -> list[str]:
    """Text files that are CRLF at ``base`` and exist in the working tree."""
    files = []
    for name in git("ls-tree", "-r", "-z", "--name-only", base).decode().split("\0"):
        if not name or not os.path.isfile(os.path.join(ROOT, name)):
            continue
        data = git("cat-file", "blob", f"{base}:{name}")
        if b"\0" in data[:8000]:  # binary
            continue
        if b"\r\n" in data:
            files.append(name)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fix", action="store_true", help="rewrite the listed files with CRLF line endings")
    mode.add_argument("--update", action="store_true", help="regenerate tools/upstream_crlf.txt from git")
    parser.add_argument("--base", help="upstream commit for --update (default: merge base of HEAD and upstream/neo)")
    args = parser.parse_args(argv)

    if args.base and not args.update:
        parser.error("--base only applies to --update")
    if args.update:
        base = args.base or git("merge-base", "HEAD", "upstream/neo").decode().strip()
        files = upstream_crlf_files(base)
        write_manifest(files)
        print(f"wrote {os.path.relpath(MANIFEST, ROOT)}: {len(files)} CRLF files at {base[:10]}")
        return 0

    files = read_manifest()
    if args.fix:
        for name in fix(files):
            print(f"converted to CRLF: {name}")
        return 0
    found = problems(files)
    for line in found:
        print(line)
    if found:
        print("Fix: python tools/check_line_endings.py --fix (see docs/remi-webui/upstream-absorption.md §3)")
        return 1
    print(f"{len(files)} upstream CRLF files keep their line endings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
