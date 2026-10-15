"""Dump the real settings schema (``GET /api/v2/options``) as a contract fixture.

The settings page tests (``src/pages/settings/contract.test.tsx``) index,
search and render this file, and mock mode serves it, so a settings-schema
change the page does not understand fails ``pnpm test``. Re-run after
changing ``remi/ui/settings.py`` or the registered options:

    /tmp/remi-testenv/bin/python webui/tools/dump_settings.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "webui", "src", "api", "mock", "contract", "settings.json")

sys.argv = sys.argv[:1]  # modules.shared parses argv on import
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "modules_forge", "packages")]

from remi.testing import webui_state  # noqa: E402

with webui_state() as state:
    from remi.server.settings import register_options
    from remi.ui.settings import settings_schema

    register_options(state.opts)
    schema = settings_schema(state.opts)
    # Machine-specific paths and lists are not part of the contract.
    for key, value in list(schema["values"].items()):
        if isinstance(value, str) and (value.startswith("/") or ":\\" in value):
            schema["values"][key] = ""
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")
    print("wrote", os.path.relpath(OUT, ROOT), len(schema["values"]), "settings")
