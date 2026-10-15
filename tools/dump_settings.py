"""Dump the real settings schema (``GET /api/v2/options``) for the SPA's
tests and mock mode.

    /tmp/remi-testenv/bin/python tools/dump_settings.py          # writes webui/src/api/mock/contract/settings.json
    /tmp/remi-testenv/bin/python tools/dump_settings.py --check  # exit 1 when it is out of date

Built from the default options (no ``config.json``), without models or UI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "webui", "src", "api", "mock", "contract", "settings.json")


def settings_document() -> dict:
    from remi.server.settings import register_options
    from remi.testing import webui_state
    from remi.ui.settings import settings_schema

    with webui_state() as state:
        register_options(state.opts)  # the server adds its own options at startup
        return settings_schema(state.opts)


def render(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the file is out of date")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # the WebUI parses the command line when imported
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "modules_forge", "packages"))
    text = render(settings_document())
    if args.check:
        try:
            with open(OUT, encoding="utf-8") as file:
                current = file.read()
        except OSError:
            current = ""
        if current != text:
            print(f"{os.path.relpath(OUT, ROOT)} is out of date: run tools/dump_settings.py")
            return 1
        return 0
    with open(OUT, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
