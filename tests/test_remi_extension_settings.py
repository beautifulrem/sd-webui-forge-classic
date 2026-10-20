"""Every setting a built-in extension or script registered before the rewrite
is still a setting, or its extension's README explains why it is gone.

The legacy settings are recorded from the last revision with the Gradio UI
(``tests/fixtures/legacy_ui/extension_settings.json``, written by
``tools/record_legacy_settings.py``). The settings registered now come from a
real startup (every built-in extension's ``on_ui_settings``) in a fresh
interpreter. A setting that is gone must be a row of the "Not carried over"
table of the extension's ``README.md``, with its reason, and be named in the
Chinese README when there is one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "legacy_ui" / "extension_settings.json"
ROW = re.compile(r"^\| `([^`]+)` \| (.+) \|$")
HEADING = re.compile(r"^(#+) (.+?)\s*$")


def legacy() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def recorder():
    spec = importlib.util.spec_from_file_location("record_legacy_settings", ROOT / "tools" / "record_legacy_settings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registered(tmp_path_factory) -> set[str]:
    """The setting keys after the server's startup with every built-in extension."""

    data = tmp_path_factory.mktemp("data")
    code = f"""
import json, sys
sys.argv = ["launch.py", "--skip-prepare-environment", "--skip-version-check", "--skip-torch-cuda-test", "--data-dir", {str(data)!r}]
import webui
from modules import shared
from remi.server import runtime
runtime.prepare(models=False)
print("SETTINGS " + json.dumps(sorted(shared.opts.data_labels)))
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "modules_forge" / "packages")])}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    line = next((line for line in result.stdout.splitlines() if line.startswith("SETTINGS ")), None)
    assert result.returncode == 0 and line, result.stdout[-3000:] + result.stderr[-6000:]
    return set(json.loads(line.removeprefix("SETTINGS ")))


def not_carried_over(readme: Path) -> dict[str, str]:
    """``{key: reason}`` of the rows of the README's "Not carried over" sections."""

    rows: dict[str, str] = {}
    level = None
    for line in readme.read_text(encoding="utf-8").splitlines():
        heading = HEADING.match(line)
        if heading:
            depth = len(heading.group(1))
            if heading.group(2) == "Not carried over":
                level = depth
            elif level is not None and depth <= level:
                level = None
        elif level is not None and (row := ROW.match(line)):
            rows[row.group(1)] = row.group(2)
    return rows


def test_the_fixture_is_the_record_of_the_legacy_revision():
    revision = legacy()["revision"]
    if subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=ROOT, capture_output=True).returncode != 0:
        pytest.skip(f"the legacy revision {revision} is not in this checkout")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        assert recorder().record(revision) == legacy()
    # tagcomplete 58, ADetailer 12, agent-scheduler 9, ControlNet 7, ... (docs/remi-webui/inventory/extensions.md)
    assert sum(len(keys) for keys in legacy()["settings"].values()) == 105


def test_not_carried_over_tables_are_read():
    readme = ROOT / "extensions-builtin" / "agent-scheduler" / "README.md"
    rows = not_carried_over(readme)
    assert "queue_paused" in rows and "queue_history_retention_days" not in rows  # the kept table is not read


def test_every_old_extension_setting_is_kept_or_explained(registered):
    missing, stale, untranslated = [], [], []
    for owner, keys in legacy()["settings"].items():
        folder = ROOT / owner
        readme = folder / "README.md"
        explained = not_carried_over(readme) if folder.is_dir() and readme.is_file() else {}
        zh = folder / "README_zh.md"
        zh_text = zh.read_text(encoding="utf-8") if folder.is_dir() and zh.is_file() else None
        for key in keys:
            # tagcomplete's dotted keys are stored with "_" (a settings name has no ".")
            kept = key in registered or key.replace(".", "_") in registered
            if kept and key in explained:
                stale.append(f"{owner}: {key} is a setting but listed as not carried over")
            elif not kept and key not in explained:
                missing.append(f"{owner}: {key}")
            elif not kept and zh_text is not None and f"`{key}`" not in zh_text:
                untranslated.append(f"{owner}: {key}")
        stale += [f"{owner}: {key} is no setting of the old extension" for key in explained if key not in keys]
        stale += [f"{owner}: {key} has no reason" for key, reason in explained.items() if len(reason) <= 30]
    assert missing == [], "old settings neither registered nor explained in README.md's 'Not carried over' table"
    assert stale == []
    assert untranslated == [], "settings explained in README.md but not in README_zh.md"


def test_the_anima_panels_follow_the_webui_language(registered):
    # The three panels' own zh/en radios were replaced by the WebUI's language
    # (their localizations/zh_CN.json); no README may still point to them.
    for name in ("anima-artist-scheduled-mixer", "anima-lora-layer-weight", "anima-lora-stage-scheduler"):
        key = name.replace("-", "_") + "_language"
        assert key not in registered
        folder = ROOT / "extensions-builtin" / name
        assert (folder / "localizations" / "zh_CN.json").is_file()
        for readme in ("README.md", "README_zh.md"):
            text = (folder / readme).read_text(encoding="utf-8")
            assert "Language Setting" not in text and "语言设置" not in text and "Settings-page UI language" not in text, (name, readme)
