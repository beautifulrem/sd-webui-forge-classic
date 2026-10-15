"""Infotext -> named values (``POST /api/v2/infotext/parse``, schema-format.md §7).

Uses Forge's ``parse_generation_parameters`` and the ``on_infotext_pasted``
callbacks, then every layout's infotext keys and paste functions; no UI is
involved. Settings the infotext differs in (and that no param reads) become
the core ``override_settings`` value, as the old paste button did.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from remi.ui import Context, jsonable


def last_generation_text() -> str:
    """The infotext of the last generation (``params.txt``), as the old paste
    button read it for an empty prompt; ``""`` when it is not available."""

    from modules import shared
    from modules.paths_internal import data_path

    if shared.cmd_opts.hide_ui_dir_config or shared.cmd_opts.no_prompt_history:
        return ""
    try:
        with open(os.path.join(data_path, "params.txt"), encoding="utf8") as file:
            return file.read()
    except OSError:
        return ""


def detect_tab(params: Mapping[str, Any]) -> str:
    """``img2img`` when the infotext has a key only the img2img form reads
    (mask/inpaint settings, Image CFG scale) or a denoising strength without
    Hires. fix keys; ``txt2img`` otherwise."""

    from remi.core import forms
    from remi.core.tabs import runner_for

    txt2img = set(forms.core_layout("txt2img", runner_for("txt2img")).infotext)
    img2img = set(forms.core_layout("img2img", runner_for("img2img")).infotext)
    if any(key in params for key in img2img - txt2img):
        return "img2img"
    # parse_generation_parameters always adds "Hires resize-1" (0 when absent)
    hires = "Hires upscale" in params or "Hires upscaler" in params or str(params.get("Hires resize-1", 0)) not in ("0", "")
    if "Denoising strength" in params and not hires:
        return "img2img"
    return "txt2img"


def parse_infotext(text: str, tab: str | None = None, *, ctx: Context | None = None, skip_fields: list[str] | None = None) -> dict[str, Any]:
    """``{"tab", "core", "scripts", "params", "problems"}`` for ``text``.

    ``tab=None`` detects the tab (:func:`detect_tab`). An empty ``text``
    reads the last generation's infotext. ``params`` is the parsed infotext
    dict (what PNG Info shows); ``problems`` lists values that did not fit
    their param (skipped)."""

    from modules import infotext_utils, script_callbacks
    from remi.core import forms
    from remi.core.tabs import runner_for

    ctx = ctx or Context()
    text = text if isinstance(text, str) else ""
    if not text.strip():
        text = last_generation_text()
    params = infotext_utils.parse_generation_parameters(text, skip_fields)
    script_callbacks.infotext_pasted_callback(text, params)
    tab = tab or detect_tab(params)

    runner = runner_for(tab)
    layout = forms.core_layout(tab, runner)
    core, problems = layout.paste(params, ctx)
    handled = set(layout.infotext)
    scripts: dict[str, dict[str, Any]] = {}
    if runner is not None:
        scripts, failed = runner.paste_values(params, ctx)
        problems += failed
        for script in runner.scripts:
            handled |= set(runner.layout_for(script).infotext)

        overrides = infotext_utils.get_override_settings(params, skip_fields=sorted(handled))
        pairs = [f"{key}: {value}" for key, _setting, value in overrides]
        if pairs:
            core["override_settings"] = pairs

    return {
        "tab": tab,
        "core": jsonable(core),
        "scripts": jsonable(scripts),
        "params": jsonable(params),
        "problems": [problem.to_json() for problem in problems],
    }
