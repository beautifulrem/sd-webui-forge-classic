"""``tools/upstream_forks.py``: a Forge Neo merge keeps Remi's forked files.

A forked file that Remi left as it was at the merge base merges without a
conflict, so upstream's version would replace Remi's silently (Remi's
``lib_spectrum/__init__.py`` against upstream's Spectrum rewrite). ``restore``
puts every forked path the merge touched back to HEAD, conflicted or not.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "remi-webui" / "upstream-absorption.md"

spec = importlib.util.spec_from_file_location("tool_upstream_forks", REPO / "tools" / "upstream_forks.py")
forks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forks)


def git(repo: Path, *args: str) -> str:
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false", *args]
    return subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False).stdout


def write(repo: Path, files: dict[str, str | None]) -> None:
    for name, text in files.items():
        path = repo / name
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def merging_repo(tmp_path: Path) -> Path:
    """A repository in the middle of ``git merge --no-commit theirs``."""

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, {"fork/init.py": "logger only\n", "fork/core.py": "base\n", "fork/old_ui.py": "ui\n", "runtime.py": "base\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "switch", "-q", "-c", "theirs")
    write(repo, {"fork/init.py": "from .core import Spectrum\n", "fork/core.py": "upstream rewrite\n", "fork/old_ui.py": "ui v2\n", "fork/added.py": "new upstream file\n", "runtime.py": "upstream fix\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "upstream")
    git(repo, "switch", "-q", "main")
    write(repo, {"fork/core.py": "remi fork\n", "fork/old_ui.py": None, "fork/remi_only.py": "remi\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "remi")
    git(repo, "merge", "--no-ff", "--no-commit", "theirs")
    return repo


def test_restore_keeps_remis_version_of_every_forked_file_the_merge_touched(tmp_path):
    repo = merging_repo(tmp_path)
    assert forks.merge_head(repo) is not None
    # what the merge did: a clean merge of init.py, a conflict in core.py, modify/delete of old_ui.py, a new file
    assert (repo / "fork" / "init.py").read_text() == "from .core import Spectrum\n"
    assert set(forks.changed_paths(("fork",), str(repo))) == {"fork/init.py", "fork/core.py", "fork/old_ui.py", "fork/added.py"}

    planned = forks.restore(("fork",), str(repo), dry_run=True)
    assert (repo / "fork" / "init.py").read_text() == "from .core import Spectrum\n"  # a dry run changes nothing
    assert dict(planned) == {
        "fork/init.py": "kept Remi's version",
        "fork/core.py": "kept Remi's version",
        "fork/old_ui.py": "removed (not in Remi)",
        "fork/added.py": "removed (not in Remi)",
    }

    assert forks.restore(("fork",), str(repo)) == planned
    assert (repo / "fork" / "init.py").read_text() == "logger only\n"
    assert (repo / "fork" / "core.py").read_text() == "remi fork\n"
    assert (repo / "fork" / "remi_only.py").read_text() == "remi\n"
    assert not (repo / "fork" / "old_ui.py").exists() and not (repo / "fork" / "added.py").exists()
    assert forks.changed_paths(("fork",), str(repo)) == []
    assert git(repo, "ls-files", "--unmerged") == ""
    # files outside the forked paths keep the merge's result
    assert (repo / "runtime.py").read_text() == "upstream fix\n"
    assert forks.restore(("fork",), str(repo)) == []


def test_no_merge_no_merge_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, {"a.py": "a\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a")
    assert forks.merge_head(str(repo)) is None


def documented_forks() -> set[str]:
    """The ``extensions-builtin/…`` paths of the first column of upstream-absorption.md §1.1."""

    section = DOC.read_text(encoding="utf-8").split("### 1.1 ", 1)[1].split("\n## 2.", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    found = set()
    for row in rows:
        first = row.split(" | ", 1)[0]
        found |= {re.sub(r"/(\*\*?)?$", "", path) for path in re.findall(r"`(extensions-builtin/[^`]+)`", first)}
    return found


def test_the_tool_and_the_doc_list_the_same_forked_paths():
    assert documented_forks() == set(forks.FORKED)
    for path in forks.FORKED:
        assert (REPO / path).exists() or path.endswith("controlnet_ui"), path  # controlnet_ui: deleted by Remi, kept deleted
    text = DOC.read_text(encoding="utf-8")
    assert "tools/upstream_forks.py restore" in text and "tools/upstream_forks.py review" in text
