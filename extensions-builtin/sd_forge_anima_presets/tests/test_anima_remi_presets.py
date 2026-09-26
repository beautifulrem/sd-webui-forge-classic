"""The whole-stack preset panel, loaded on its own like Forge loads it."""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path

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
