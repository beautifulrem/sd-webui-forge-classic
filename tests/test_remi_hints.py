"""Tooltips: ``hint=`` on inputs and ``hints=`` on choices (schema-format.md
§1.2), and the core hint map of the Gradio UI's ``javascript/hints.js``
declared on the core params (``remi/core/hints.py``), translated."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from remi import ui
from remi.core import hints
from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]


def test_hints_are_exported_and_validated():
    ctx = ui.Context()
    assert ui.Slider("cfg", 1, 30, 7, 0.5, hint="How strongly").schema(ctx)["hint"] == "How strongly"
    assert "hint" not in ui.Slider("cfg", 1, 30, 7, 0.5).schema(ctx)
    choice = ui.Choice("mode", ["fill", "original"], "fill", style="radio", hints={"fill": "Fill it"}).schema(ctx)
    assert choice["choice_hints"] == {"fill": "Fill it"} and choice["choices"] == [{"value": "fill", "label": "fill"}, {"value": "original", "label": "original"}]
    assert "choice_hints" not in ui.Choice("mode", ["a"], "a").schema(ctx)
    with pytest.raises(ui.SchemaError):
        ui.Text("t", hint=3)
    with pytest.raises(ui.SchemaError):
        ui.Choice("mode", ["a"], "a", hints={"a": 1})
    assert ui.update(hint="new") == {"hint": "new"}  # patches may change it with the label


def _walk(node):
    if isinstance(node, dict):
        yield node
        for key in ("children", "enable", "layout"):
            if key in node:
                yield from _walk(node[key])
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


def test_the_core_params_carry_the_legacy_hint_map(webui):
    from remi.core.tabs import tab_schema

    by_label: dict[str, str] = {}
    choice_hints: dict[str, str] = {}
    for tab in ("txt2img", "img2img"):
        schema = tab_schema(tab)
        layouts = [schema["core"]["layout"], *(script["layout"] for script in schema["scripts"])]
        for node in _walk(layouts):
            if node.get("hint"):
                by_label[node["label"]] = node["hint"]
            choice_hints.update(node.get("choice_hints") or {})
    # Rescale CFG and MaHiRo show only with their settings on: their params directly
    from modules.processing_scripts.mahiro import ScriptMahiro
    from modules.processing_scripts.rescale_cfg import ScriptRescaleCFG

    for script in (ScriptRescaleCFG(), ScriptMahiro()):
        node = script.params(False)
        by_label[node.label] = node.hint
    # hints.js: label -> tooltip (the labels the Gradio UI matched)
    expected = {
        "Sampling Method": hints.SAMPLER,
        "Schedule Type": hints.SCHEDULER,
        "Sampling Steps": hints.STEPS,
        "Batch Count": hints.BATCH_COUNT,
        "Batch Size": hints.BATCH_SIZE,
        "CFG Scale": hints.CFG_SCALE,
        "Rescale CFG": hints.RESCALE_CFG,
        "MaHiRo": hints.MAHIRO,
        "Seed": hints.SEED,
        "Mask blur": hints.MASK_BLUR,
        "Denoising Strength": hints.DENOISING_STRENGTH,
        "Hires. fix": hints.HIRES_FIX,
        "Hires steps": hints.HIRES_STEPS,
        "Upscale by": hints.UPSCALE_BY,
        "Resize width to": hints.RESIZE_WIDTH_TO,
        "Resize height to": hints.RESIZE_HEIGHT_TO,
    }
    assert {label: by_label.get(label) for label in expected} == expected
    assert choice_hints == {**hints.RESIZE_MODES, **hints.MASKED_CONTENT}
    # "Frames" (the batch size of video presets) is a preset patch: test_remi_ui_core


def test_every_hint_is_translated():
    catalog = json.loads((ROOT / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    texts = hints.all_hints()
    assert len(texts) == 26
    assert [text for text in texts if not catalog.get(text)] == []
