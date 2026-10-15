"""Export the Remi server's OpenAPI document for the generated TS client.

    /tmp/remi-testenv/bin/python tools/export_openapi.py          # writes webui/src/api/openapi.json
    /tmp/remi-testenv/bin/python tools/export_openapi.py --check  # exit 1 when it is out of date

Covers ``/api/v2`` and ``/internal`` (``/sdapi/v1`` is the A1111 API and is
left out). No model or UI is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "webui", "src", "api", "openapi.json")


def openapi_document() -> dict:
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    with tempfile.TemporaryDirectory() as directory:
        services = Services.create(directory, dispatch=run_inline)
        try:
            app = create_app(services=services, auth=Auth(), sdapi=False, start_services=False)
            return app.openapi()
        finally:
            services.files.close()
            services.store.close()


def render(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail when the file is out of date instead of writing it")
    parser.add_argument("--out", default=OUT, help="output file")
    options = parser.parse_args()

    sys.argv = sys.argv[:1]  # modules.shared parses argv on import
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.chdir(ROOT)
    sys.path[:0] = [ROOT, os.path.join(ROOT, "modules_forge", "packages")]

    text = render(openapi_document())
    if options.check:
        try:
            with open(options.out, encoding="utf-8") as file:
                current = file.read()
        except OSError:
            current = ""
        if current != text:
            print(f"{os.path.relpath(options.out, ROOT)} is out of date; run tools/export_openapi.py")
            return 1
        return 0
    os.makedirs(os.path.dirname(options.out), exist_ok=True)
    with open(options.out, "w", encoding="utf-8") as file:
        file.write(text)
    print("wrote", os.path.relpath(options.out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
