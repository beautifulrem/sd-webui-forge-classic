"""JSON Builder: a visual editor for MOSS-Image's structured ("Layout")
prompts, as a page of the SPA (``#/ext/json-builder``).

Upstream (Forge Neo ``be6fa5d4``) registers an empty Gradio tab here and
builds everything in ``javascript/main.js``. In Remi the page is a dedicated
frontend view, ``webui/src/pages/custom/json-builder``: the canvas, the
prompt compiler and decompiler (TypeScript ports of upstream's
``compile.js`` / ``decompile.js``) and "Send to txt2img / img2img", which
fills the prompt, width and height of that form. The server only registers
the page, so it follows "UI Tab Order", "Hide UI Tabs" and the extension's
enabled state like any page.
"""

from remi import pages

PAGE_ID = "json-builder"

pages.register(pages.Page(PAGE_ID, "JSON Builder", "braces", widget="json-builder"))
