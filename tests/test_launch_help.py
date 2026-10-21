"""`python launch.py --help` lists every flag, including the ones extensions add.

README.md / README.zh-CN.md ("Launch Options") promise that `--help` lists all
launch options. Bundled extensions register theirs in ``preload.py``, which runs
after the launcher's own early parse, so this guards against `--help` exiting
before the preloads have run.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]

# flags registered by extensions-builtin/*/preload.py
EXTENSION_FLAGS = ("--agent-scheduler-sqlite-file", "--controlnet-loglevel", "--lora-dir", "--ad-no-huggingface")


def run_launch(*argv: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "COMMANDLINE_ARGS"}
    with tempfile.TemporaryDirectory() as data_dir:
        return subprocess.run(
            [sys.executable, "launch.py", "--data-dir", data_dir, *argv],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )


@pytest.mark.parametrize("flag", ["--help", "-h", "--hel"])
def test_help_lists_core_and_extension_flags(flag):
    result = run_launch(flag)

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: launch.py")
    for option in ("--port", "--listen", "--api", "--data-dir", *EXTENSION_FLAGS):
        assert f"  {option}" in result.stdout, option
    # the help text of an extension flag is shown, not just its name
    assert "Agent Scheduler database to import into the Queue" in " ".join(result.stdout.split())


def test_help_with_other_flags_still_exits_cleanly():
    result = run_launch("--port", "7001", "--help")

    assert result.returncode == 0, result.stderr
    assert "--controlnet-loglevel" in result.stdout
