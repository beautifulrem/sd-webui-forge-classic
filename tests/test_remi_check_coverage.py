"""``tools/check_coverage.py``: the coverage floors of ``tools/check.sh``.

``remi/`` is held to a floor as a whole; every other Remi file (the strict list
of ``pyproject.toml`` outside Remi's own folders and the tests) is held to a
floor on its own, over the data of every pytest step (they are tested by
different suites). The data here is written directly (``CoverageData``), so
nothing is traced inside this test run.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest
from coverage import CoverageData

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tool_check_coverage", REPO / "tools" / "check_coverage.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

MODULE = textwrap.dedent(
    """\
    def double(x):
        return 2 * x


    def halve(x):
        return x / 2
    """
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A checkout with two Remi files outside remi/ (one in an extension), a remi/ file and a test."""

    root = tmp_path.resolve()
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [tool.ruff.lint.per-file-ignores]
            "!{remi/**,tools/**,tests/test_remi_x.py,extensions-builtin/ext/tests/test_ext.py,extensions-builtin/ext/lib_ext/params.py,modules/remi_thing.py}" = ["E"]
            "{tests/**}" = ["S102"]
            """
        ),
        encoding="utf-8",
    )
    for file in ("modules/remi_thing.py", "extensions-builtin/ext/lib_ext/params.py", "remi/core.py", "tests/test_remi_x.py", "extensions-builtin/ext/tests/test_ext.py"):
        (root / file).parent.mkdir(parents=True, exist_ok=True)
        (root / file).write_text(MODULE, encoding="utf-8")
    return root


def write_data(path: Path, lines: dict[Path, list[int]]) -> Path:
    data = CoverageData(basename=str(path))
    data.add_lines({str(file): numbers for file, numbers in lines.items()})
    data.write()
    return path


def test_the_strict_files_of_this_repository():
    files = check.strict_files()
    assert {"modules/redact.py", "modules/extra_networks_pages.py", "extensions-builtin/sd_forge_controlnet/lib_controlnet/unit_param.py", "extensions-builtin/sd_forge_controlnet/lib_controlnet/openpose.py", "extensions-builtin/anima-lora-stage-scheduler/lib_stage_scheduler/params.py"} <= set(files)
    assert not [file for file in files if file.startswith(("remi/", "tools/", "docs/")) or check.is_test_file(file)]
    sources = check.cov_sources(files)
    assert "modules.redact" in sources and "extensions-builtin/sd_forge_controlnet/lib_controlnet" in sources
    assert len(sources) == len(set(sources))


def test_strict_files_skip_remi_folders_and_tests(repo: Path):
    assert check.strict_files(repo) == ["extensions-builtin/ext/lib_ext/params.py", "modules/remi_thing.py"]
    assert check.cov_sources(check.strict_files(repo)) == ["extensions-builtin/ext/lib_ext", "modules.remi_thing"]


def test_a_stale_strict_list_entry_fails(repo: Path):
    with pytest.raises(SystemExit, match="does not exist"):
        check.strict_files(repo, ["modules/gone.py"])
    (repo / "pyproject.toml").write_text('[tool.ruff.lint.per-file-ignores]\n"{tests/**}" = ["S102"]\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="expected one negated"):
        check.strict_files(repo)


def test_ranges():
    assert check.ranges([7, 1, 2, 3, 9, 10]) == "1-3, 7, 9-10"
    assert check.ranges([]) == ""


def test_each_file_is_held_to_the_floor_over_every_run(repo: Path, capsys):
    thing, params = repo / "modules" / "remi_thing.py", repo / "extensions-builtin" / "ext" / "lib_ext" / "params.py"
    # tests/ covered modules/remi_thing.py fully, the extension's own suite covered its params.py
    runs = [write_data(repo / ".coverage.pytest_tests", {thing: [1, 2, 5, 6]}), write_data(repo / ".coverage.pytest_ext", {params: [1, 2, 5, 6]})]
    assert check.check_strict(runs, 90, root=repo) == 0
    out = capsys.readouterr().out
    assert "modules/remi_thing.py" in out and "100.0%" in out and "data from 2 pytest runs" in out

    # one function untested in one file: 75 % fails that file, whatever the others reach
    write_data(runs[1], {params: [1, 2, 5]})
    assert check.check_strict(runs, 90, root=repo) == 1
    captured = capsys.readouterr()
    assert "extensions-builtin/ext/lib_ext/params.py" in captured.out and "75.0%" in captured.out and "6   < 90" in captured.out
    assert "1 below the floor" in captured.err


def test_a_file_no_run_imported_counts_as_zero(repo: Path, capsys):
    runs = [write_data(repo / ".coverage.pytest_tests", {repo / "modules" / "remi_thing.py": [1, 2, 5, 6]})]
    assert check.check_strict(runs, 90, root=repo) == 1
    assert "0.0%" in capsys.readouterr().out
    assert check.check_strict([repo / ".coverage.missing"], 90, root=repo) == 2


def test_remi_floor(repo: Path, capsys):
    data = write_data(repo / ".coverage.pytest_tests", {repo / "remi" / "core.py": [1, 2, 5], repo / "modules" / "remi_thing.py": [1, 2, 5, 6]})
    assert check.check_remi(data, 70, root=repo) == 0
    assert "remi/ line coverage 75.0 %" in capsys.readouterr().out  # only remi/ counts
    assert check.check_remi(data, 80, root=repo) == 1
    assert check.check_remi(repo / "nothing", 80, root=repo) == 2


def test_cli_lists_files_and_sources(capsys):
    assert check.main(["files"]) == 0
    assert "modules/redact.py" in capsys.readouterr().out.split()
    assert check.main(["sources"]) == 0
    assert "--cov=modules.redact" in capsys.readouterr().out.split()
