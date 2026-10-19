"""The recorder's own environment, ``tools/legacy-ui-requirements.txt``.

``tools/record_legacy_ui.py`` runs a script's Gradio ``ui()`` inside an export
of ``LEGACY_REV``, so its interpreter needs that revision's requirements plus
the Gradio its launcher installed. Gradio 4.40 requires ``pillow<11``, which
the WebUI's own Pillow pin (12 and later) contradicts, so the recorder gets a
requirements file of its own instead of ``requirements.txt`` plus Gradio (an
install that cannot be resolved). These tests keep that file consistent with
the legacy revision and with the documented setup; the install itself (pip,
Python 3.13) was checked when the file was written and needs the network.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "tools" / "legacy-ui-requirements.txt"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)\s*(==\s*[^;\s]+)?\s*(;.*)?$")
PILLOW_FAMILY = {"pillow", "pillow-heif", "pillow-jxl-plugin"}


def _recorder():
    spec = importlib.util.spec_from_file_location("tool_record_legacy_ui_env", REPO / "tools" / "record_legacy_ui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(text: str) -> dict[str, str]:
    """``{normalized name: the rest of the line}`` (version and marker) of a requirements file."""

    found = {}
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        match = PIN.match(line)
        assert match, f"unexpected requirement line {raw!r}"
        found[_normalize(match.group(1))] = (match.group(2) or "").replace(" ", "") + (match.group(3) or "")
    return found


def test_pins_the_gradio_the_legacy_launcher_installed():
    pins = requirements(REQUIREMENTS.read_text(encoding="utf-8"))
    assert pins["gradio"] == "==4.40.0"
    assert pins["gradio-rangeslider"] == "==0.0.8"
    # gradio 4.40.0 requires pillow>=8.0,<11.0
    major = int(pins["pillow"].removeprefix("==").split(".")[0])
    assert 8 <= major < 11


def test_matches_the_legacy_revision_requirements():
    recorder = _recorder()
    shown = subprocess.run(["git", "show", f"{recorder.LEGACY_REV}:requirements.txt"], cwd=REPO, capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip(f"{recorder.LEGACY_REV[:10]} is not in this clone")
    legacy = requirements(shown.stdout)
    pins = requirements(REQUIREMENTS.read_text(encoding="utf-8"))
    differs = {name: (pin, pins.get(name)) for name, pin in legacy.items() if name not in PILLOW_FAMILY and pins.get(name) != pin}
    assert differs == {}
    # the Pillow plugins that the legacy modules/images.py imports are there, for Pillow 10
    assert pins.keys() >= PILLOW_FAMILY


def test_documented_setup_uses_the_file():
    stale = re.compile(r"-r requirements\.txt\s+\"?gradio==")
    for path in (REPO / "tools" / "record_legacy_ui.py", REPO / "docs" / "remi-webui" / "extension-guide.md", REPO / "docs" / "remi-webui" / "upstream-absorption.md"):
        text = path.read_text(encoding="utf-8")
        assert "tools/legacy-ui-requirements.txt" in text, path
        assert not stale.search(text), path
