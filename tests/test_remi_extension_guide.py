"""The extension guide's running example works as written.

docs/remi-webui/extension-guide.md marks the files of its Prompt Suffix
example with ``<!-- guide-example: <path> -->`` before the fenced block. This
test writes them into a scratch tree shaped like the WebUI (``extensions/``
next to ``tests/remi_port_harness.py``) and runs the example's own test file
with pytest, so the guide cannot drift from the harness and ``remi.ui``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "remi-webui" / "extension-guide.md"
MARKER = re.compile(r"^(?P<indent>[ \t]*)<!-- guide-example: (?P<path>\S+) -->\n(?P=indent)```[a-z]*\n(?P<body>.*?)^(?P=indent)```$", re.MULTILINE | re.DOTALL)
EXPECTED = {
    "extensions/prompt-suffix/scripts/prompt_suffix.py",
    "extensions/prompt-suffix/localizations/zh_CN.json",
    "extensions/prompt-suffix/tests/test_prompt_suffix.py",
}


def guide_examples(text: str) -> dict[str, str]:
    return {match["path"]: textwrap.dedent(match["body"]) for match in MARKER.finditer(text)}


def test_marker_parsing_handles_indented_blocks():
    text = "intro\n\n  <!-- guide-example: a/b.json -->\n  ```json\n  {\n   \"x\": 1\n  }\n  ```\n\n<!-- guide-example: c.py -->\n```python\nprint(1)\n```\n"
    assert guide_examples(text) == {"a/b.json": '{\n "x": 1\n}\n', "c.py": "print(1)\n"}


def test_guide_example_extension_passes_its_own_tests(tmp_path):
    examples = guide_examples(GUIDE.read_text(encoding="utf-8"))
    assert set(examples) == EXPECTED
    for relative, body in examples.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "remi_port_harness.py").symlink_to(REPO / "tests" / "remi_port_harness.py")

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "extensions/prompt-suffix/tests"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert re.search(r"\b6 passed\b", result.stdout), result.stdout[-2000:]
