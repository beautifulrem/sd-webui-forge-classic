"""The recorder's own environment: ``tools/record_legacy_ui.py --requirements``.

``tools/record_legacy_ui.py`` runs a script's Gradio ``ui()`` inside an export
of ``LEGACY_REV``, so its interpreter needs that revision's requirements plus
the Gradio its launcher installed. Gradio 4.40 requires ``pillow<11``, which
the WebUI's own Pillow pin (12 and later) contradicts, so the recorder gets an
environment of its own instead of ``requirements.txt`` plus Gradio (an install
that cannot be resolved). No tracked file lists those requirements (the WebUI
has no Gradio dependency anywhere in the checkout): ``--requirements`` derives
them from the git history, and ``tools/legacy_ui_env.sh`` installs them outside
the checkout. These tests keep the derivation consistent with the legacy
revision and with the documented setup; the install itself (pip, Python 3.13)
needs the network and runs in ``tools/legacy_ui_env.sh``.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.-]+)\s*(==\s*[^;\s]+)?\s*(;.*)?$")
PILLOW_FAMILY = {"pillow", "pillow-heif", "pillow-jxl-plugin"}


def _recorder():
    spec = importlib.util.spec_from_file_location("tool_record_legacy_ui_env", REPO / "tools" / "record_legacy_ui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recorder = _recorder()


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


def _show(rev: str, path: str) -> str:
    shown = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO, capture_output=True, text=True, encoding="utf8")
    if shown.returncode != 0:
        pytest.skip(f"{rev[:10]} is not in this clone")
    return shown.stdout


@pytest.fixture(scope="module")
def derived() -> dict[str, str]:
    _show(recorder.LEGACY_REV, "requirements.txt")
    _show(recorder.GRADIO_LAUNCHER_REV, "modules/launch_utils.py")
    return requirements(recorder.legacy_requirements())


def test_pins_the_gradio_the_legacy_launcher_installed(derived):
    assert derived["gradio"] == "==4.40.0"
    assert derived["gradio-rangeslider"] == "==0.0.8"
    # gradio 4.40.0 requires pillow>=8.0,<11.0
    major = int(derived["pillow"].removeprefix("==").split(".")[0])
    assert 8 <= major < 11
    # the pins come from the last launcher that installed Gradio; the next revision dropped them
    assert 'GRADIO_PACKAGE", "gradio==4.40.0 gradio_rangeslider==0.0.8"' in _show(recorder.GRADIO_LAUNCHER_REV, "modules/launch_utils.py")
    child = subprocess.run(["git", "rev-list", "--parents", "-n", "1", "80cfa274"], cwd=REPO, capture_output=True, text=True)
    if child.returncode == 0:
        assert recorder.GRADIO_LAUNCHER_REV in child.stdout.split()[1:]
        assert "GRADIO_PACKAGE" not in _show("80cfa274", "modules/launch_utils.py")


def test_matches_the_legacy_revision_requirements(derived):
    legacy = requirements(_show(recorder.LEGACY_REV, "requirements.txt"))
    differs = {name: (pin, derived.get(name)) for name, pin in legacy.items() if name not in PILLOW_FAMILY and derived.get(name) != pin}
    assert differs == {}
    # the Pillow plugins that the legacy modules/images.py imports are there, for Pillow 10
    assert derived.keys() >= PILLOW_FAMILY
    assert set(derived) - set(legacy) == {"gradio", "gradio-rangeslider", "socksio"}


def test_the_cli_prints_them_without_gradio():
    # the WebUI's (Gradio-free) interpreter can derive them: stdlib and git only
    result = subprocess.run([sys.executable, str(REPO / "tools" / "record_legacy_ui.py"), "--requirements"], cwd=REPO, capture_output=True, text=True, encoding="utf8")
    if result.returncode != 0 and "cannot read" in result.stderr:
        pytest.skip(result.stderr)
    assert result.returncode == 0, result.stderr
    assert result.stdout == recorder.legacy_requirements()
    with pytest.raises(SystemExit):
        recorder.parse_args(["--requirements", "--self-test"])


def test_the_gradio_package_default_is_parsed():
    source = 'gradio_package = os.environ.get("GRADIO_PACKAGE", "gradio==1.0 gradio_rangeslider==0.1")'
    assert recorder.launcher_gradio_packages(source) == ["gradio==1.0", "gradio_rangeslider==0.1"]
    with pytest.raises(SystemExit):
        recorder.launcher_gradio_packages("nothing here")
    assert recorder.requirement_name("Pillow==12.3.0") == "pillow"
    assert recorder.requirement_name('audioop-lts==0.2.2;python_version>="3.13"') == "audioop-lts"
    assert recorder.requirement_name("pillow_heif>=1") == "pillow-heif"


def test_no_tracked_file_lists_gradio_as_a_dependency():
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    # a requirement line or a quoted dependency entry ("gradio>=4", "gradio",); not ruff's banned-api table ("gradio".msg)
    dependency = re.compile(r"""^\s*["']?gradio(?:[-_]client|[-_]rangeslider)?\s*(?:[=<>~!]=|["']\s*[,\]]|$)""", re.IGNORECASE | re.MULTILINE)
    offenders = []
    for name in tracked.stdout.split("\0"):
        path = REPO / name
        if not name or not path.is_file() or not re.search(r"(requirements[^/]*\.txt|\.toml|\.cfg|setup\.py|environment[^/]*\.ya?ml|Dockerfile)$", name):
            continue
        if dependency.search(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(name)
    assert offenders == []
    # the pattern finds what it must
    assert dependency.search("Pillow==10\ngradio==4.40.0\n") and dependency.search('deps = [\n  "gradio_client",\n]') and not dependency.search('"gradio".msg = "removed"')


def test_documented_setup_derives_them():
    stale = re.compile(r"-r requirements\.txt\s+\"?gradio==|tools/legacy-ui-requirements\.txt")
    for path in (REPO / "tools" / "record_legacy_ui.py", REPO / "tools" / "legacy_ui_env.sh", REPO / "docs" / "remi-webui" / "extension-guide.md", REPO / "docs" / "remi-webui" / "upstream-absorption.md", REPO / "docs" / "remi-webui" / "architecture.md"):
        text = path.read_text(encoding="utf-8")
        assert "--requirements" in text, path
        assert not stale.search(text), path
