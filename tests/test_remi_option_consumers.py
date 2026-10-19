"""No dead settings: every core option (``modules/shared_options.py`` and the
UI presets' options of ``modules_forge/presets.py``) is read by something —
processing, the server, a script or the SPA. A setting the Gradio UI used
and nothing reads any more must be implemented or removed, not left in the
settings page doing nothing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]

# Where readers live (tests, generated API types and mock data do not count)
SOURCES = ("modules", "modules_forge", "backend", "remi", "scripts", "extensions-builtin", "webui/src", "launch.py", "webui.py")
SKIP_PARTS = ("/tests/", "/node_modules/", "/webui/src/api/gen/", "/webui/src/api/mock/")
DEFINITION = "modules/shared_options.py"

# Options kept without a reader, each with its reason (none today).
KEPT: dict[str, str] = {}


def _words() -> set[str]:
    words: set[str] = set()
    for source in SOURCES:
        base = ROOT / source
        paths = [base] if base.is_file() else [*base.rglob("*.py"), *base.rglob("*.ts"), *base.rglob("*.tsx")]
        for path in paths:
            relative = "/" + path.relative_to(ROOT).as_posix()
            if relative == f"/{DEFINITION}" or any(part in relative for part in SKIP_PARTS) or ".test." in path.name:
                continue
            words.update(re.findall(r"\w+", path.read_text(encoding="utf-8", errors="replace")))
    return words


@pytest.fixture(scope="module")
def options():
    with webui_state():
        from modules import options as options_module
        from modules import shared_options
        from modules_forge import presets

        preset_options: dict = {}
        presets.register(preset_options)
        static = {key: info for key, info in shared_options.options_templates.items() if key not in preset_options}
        yield options_module, static, preset_options, presets


def test_every_setting_has_a_reader(options):
    options_module, static, _, _ = options
    settings = [key for key, info in static.items() if type(info) is options_module.OptionInfo]
    assert len(settings) > 200  # the scan itself works
    words = _words()
    unread = sorted(key for key in settings if key not in words and key not in KEPT)
    assert unread == [], f"settings nothing reads (implement them or remove them from {DEFINITION}): {unread}"
    assert not set(KEPT) - set(settings), "KEPT lists settings that do not exist"


class _Recorder:
    """An ``opts`` that records the names read (every one falls back to its default)."""

    def __init__(self) -> None:
        self.read: set[str] = set()

    def __getattr__(self, name: str):
        self.read.add(name)
        raise AttributeError(name)


def test_every_ui_preset_setting_is_read_for_its_preset(options):
    options_module, _, preset_options, presets = options
    from remi.core.presets import forge_preset_patch, forge_preset_settings

    for arch in presets.PresetArch:
        name = arch.name
        own = {key for key, info in preset_options.items() if type(info) is options_module.OptionInfo and (key.startswith(f"{name}_") or (key.endswith(f"_{name}") and key.startswith("forge_")))}
        assert own, name
        recorder = _Recorder()
        for tab in ("txt2img", "img2img"):
            forge_preset_patch(tab, name, recorder)
        forge_preset_settings(name, recorder)
        assert own - recorder.read == set(), f"UI preset {name}: settings nothing reads"
