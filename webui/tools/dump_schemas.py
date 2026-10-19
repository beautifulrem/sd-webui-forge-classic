"""Dump the real tab schemas (``remi.core.tabs.tab_schema``) as contract fixtures.

The frontend's contract tests (``src/ui/contract.test.tsx``) render these
files, so a backend schema change that the renderer does not understand fails
``pnpm test``. Re-run after changing ``remi/ui`` or ``remi/core``:

    $PYTHON webui/tools/dump_schemas.py          # writes webui/src/api/mock/contract/{tab}.json
    $PYTHON webui/tools/dump_schemas.py --check  # exit 1 when a fixture is out of date

``$PYTHON``: an interpreter with requirements.txt (no Gradio), as ``tools/check.sh``
picks; its ``contract:schemas`` step runs the check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "webui", "src", "api", "mock", "contract")
TABS = ("txt2img", "img2img", "extras")


def render(schema: dict) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def schema_files(out: str) -> dict[str, str]:
    """Path → content of every fixture, rendered from the live schemas."""

    from remi.core.tabs import tab_schema
    from remi.testing import webui_state

    with webui_state():
        return {os.path.join(out, f"{tab}.json"): render(tab_schema(tab)) for tab in TABS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail when a fixture is out of date instead of writing it")
    parser.add_argument("--out", default=OUT, help="output directory")
    options = parser.parse_args()

    sys.argv = sys.argv[:1]  # modules.shared parses argv on import
    os.chdir(ROOT)
    sys.path[:0] = [ROOT, os.path.join(ROOT, "modules_forge", "packages")]

    files = schema_files(os.path.abspath(options.out))
    if options.check:
        stale = []
        for path, text in files.items():
            try:
                with open(path, encoding="utf-8") as file:
                    current = file.read()
            except OSError:
                current = ""
            if current != text:
                stale.append(os.path.relpath(path, ROOT))
        for path in stale:
            print(f"{path} is out of date; run webui/tools/dump_schemas.py")
        return 1 if stale else 0
    os.makedirs(options.out, exist_ok=True)
    for path, text in files.items():
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        print("wrote", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
