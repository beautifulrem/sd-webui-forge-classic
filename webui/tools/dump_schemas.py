"""Dump the real tab schemas (``remi.core.tabs.tab_schema``) as contract fixtures.

The frontend's contract tests (``src/ui/contract.test.tsx``) render these
files, so a backend schema change that the renderer does not understand fails
``pnpm test``. Re-run after changing ``remi/ui`` or ``remi/core``:

    /tmp/remi-testenv/bin/python webui/tools/dump_schemas.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "webui", "src", "api", "mock", "contract")

sys.argv = sys.argv[:1]  # modules.shared parses argv on import
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "modules_forge", "packages")]

from remi.core.tabs import tab_schema  # noqa: E402
from remi.testing import webui_state  # noqa: E402

os.makedirs(OUT, exist_ok=True)
with webui_state():
    for tab in ("txt2img", "img2img", "extras"):
        path = os.path.join(OUT, f"{tab}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(tab_schema(tab), fh, ensure_ascii=False, indent=1, allow_nan=False)
            fh.write("\n")
        print("wrote", os.path.relpath(path, ROOT))
