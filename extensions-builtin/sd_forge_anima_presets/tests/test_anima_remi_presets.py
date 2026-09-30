"""The whole-stack preset panel, loaded on its own like Forge loads it."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

# Building gr.Blocks otherwise waits on Gradio's online version check
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_remi_presets.py"
BUILTIN = Path(__file__).parents[2]


def _load_script():
    sys.argv = sys.argv[:1]
    spec = importlib.util.spec_from_file_location("anima_remi_presets_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_MODULE = _load_script()


def test_preset_warnings_are_visible():
    # Forge's root logger is at ERROR: without its own handler the preset
    # script's Turbo advice would never be shown.
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        assert SCRIPT_MODULE.logger.isEnabledFor(logging.WARNING)
        assert SCRIPT_MODULE.logger.handlers and not SCRIPT_MODULE.logger.propagate
    finally:
        root.setLevel(level)


def test_panel_builds_after_every_other_builtin_panel():
    # ui() collects the controls other panels registered in their own ui(),
    # so it must sort after all of them, in whichever extension they live.
    priority = SCRIPT_MODULE.AnimaRemiPresetScript.sorting_priority
    others = []
    for script in BUILTIN.glob("*/scripts/*.py"):
        if script.resolve() == SCRIPT.resolve():
            continue
        for match in re.finditer(r"^\s*(?:self\.)?sorting_priority\s*=\s*(\d+)\s*$", script.read_text(encoding="utf-8"), re.M):
            others.append((int(match.group(1)), script.relative_to(BUILTIN).as_posix()))
    assert others
    assert all(value < priority for value, _ in others), max(others)


# Extensions may register a localized value in place of the base one; a preset
# that does not override the field must apply the registered value.
LOCALIZED = {"guidance.mode": "Standard / preserve existing (localized)", "guidance.nag_scale": 2.5}


def _build_apply_handler():
    """Build the txt2img panel over one fake control per preset field, the way
    other extensions register theirs, and return the Apply button's handler."""

    import gradio as gr
    from modules import anima_presets

    saved = anima_presets._CONTROLS.get("txt2img")
    try:
        with gr.Blocks() as demo:
            anima_presets.reset_preset_controls("txt2img")
            components = []
            for name in anima_presets.SAFE_BASE_AESTHETIC_PRESET:
                component = gr.Textbox(elem_id=f"fake_{name}")
                components.append(component)
                if name in LOCALIZED:
                    anima_presets.register_preset_control("txt2img", name, component, LOCALIZED[name])
                else:
                    anima_presets.register_preset_control("txt2img", name, component)
            assert SCRIPT_MODULE.AnimaRemiPresetScript().ui(False) == []
    finally:
        if saved is None:
            anima_presets._CONTROLS.pop("txt2img", None)
        else:
            anima_presets._CONTROLS["txt2img"] = saved

    apply = [b for b in demo.blocks.values() if isinstance(b, gr.Button) and b.value == "Apply preset"]
    assert len(apply) == 1
    handlers = [dep for dep in demo.fns.values() if (apply[0]._id, "click") in dep.targets]
    assert len(handlers) == 1
    handler = handlers[0]
    # One update per registered control, in registration order, then the status
    assert handler.outputs[:-1] == components
    assert isinstance(handler.inputs[0], gr.Dropdown) and len(handler.inputs) == 1
    names = tuple(anima_presets.SAFE_BASE_AESTHETIC_PRESET)
    return handler.fn, names


def _applied(fn, names, label):
    *updates, status = fn(label)
    assert len(updates) == len(names)
    return dict(zip(names, updates)), status


def test_apply_sets_every_control_to_the_chosen_preset():
    from modules import anima_presets

    fn, names = _build_apply_handler()
    updates, status = _applied(fn, names, anima_presets.TURBO_NAG_LABEL)

    values = {name: update["value"] for name, update in updates.items()}
    # Turbo + NAG overrides
    assert values["native.sampler"] == "Euler"
    assert values["native.steps"] == 10
    assert values["native.cfg"] == 1.0
    assert values["guidance.enabled"] is True
    assert values["guidance.nag_enabled"] is True
    # Not overridden: the registered (localized) value, else the base value
    assert values["guidance.mode"] == LOCALIZED["guidance.mode"]
    assert values["guidance.nag_scale"] == 2.5
    assert values["native.scheduler"] == "Automatic"
    assert values["native.hires"] is False
    overridden = set(anima_presets.TURBO_NAG_OVERRIDES) | set(LOCALIZED)
    for name in set(names) - overridden:
        assert values[name] == anima_presets.SAFE_BASE_AESTHETIC_PRESET[name], name

    assert status == {
        "__type__": "update",
        "value": f"**Applied:** {anima_presets.TURBO_NAG_LABEL} ({len(names)} controls)",
    }


def test_apply_falls_back_to_the_default_preset_for_unknown_labels():
    from modules import anima_presets

    fn, names = _build_apply_handler()
    default = fn(anima_presets.SAFE_BASE_AESTHETIC_LABEL)
    assert default[-1]["value"] == f"**Applied:** {anima_presets.SAFE_BASE_AESTHETIC_LABEL} ({len(names)} controls)"
    for label in ("", None, "Removed preset"):
        assert fn(label) == default, label
    updates, _ = _applied(fn, names, "Removed preset")
    assert updates["native.sampler"]["value"] == "ER SDE"
    assert updates["native.steps"]["value"] == 36
    assert updates["native.cfg"]["value"] == 4.5
    assert updates["guidance.mode"]["value"] == LOCALIZED["guidance.mode"]


def test_apply_unlocks_exactly_the_conflict_controls():
    from modules import anima_presets

    fn, names = _build_apply_handler()
    for label in anima_presets.PRESETS:
        updates, _ = _applied(fn, names, label)
        unlocked = {name for name, update in updates.items() if update.get("interactive") is True}
        assert unlocked == anima_presets.PRESET_CONFLICT_CONTROLS, label
        # Every other control keeps its extension-defined interactivity
        for name, update in updates.items():
            if name not in anima_presets.PRESET_CONFLICT_CONTROLS:
                assert set(update) == {"__type__", "value"}, name
