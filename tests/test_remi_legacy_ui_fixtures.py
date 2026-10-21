"""Every committed legacy ``ui()`` fixture can be recorded again and compared.

``tools/legacy_ui_recordings.py`` lists each ``legacy_ui_*.json`` fixture with
the script and revision it was recorded from, its format and the WebUI context
its ``ui()`` needs; ``tools/record_legacy_ui.py --fixtures --check`` records
them all again in the last Gradio revision and compares every value (it needs
the recorder's Gradio 4.40 interpreter: ``tools/check.sh``'s ``legacy-ui``
step). These tests run without Gradio: the list covers every fixture of the
tree, each entry names a script that still had its Gradio ``ui()`` at its
revision, and the comparison works on every fixture format (a bare list, the
contract dicts, the Anima panels' records, ControlNet's, the pages', the
processing scripts')."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


recorder = _load("record_legacy_ui_for_fixtures", REPO / "tools" / "record_legacy_ui.py")
recordings = recorder.recordings()
FIXTURES = [(recording, tab, path) for recording in recordings.RECORDINGS for tab, path in recording.files.items()]


def _git_show(rev: str, path: str) -> str | None:
    shown = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO, capture_output=True, text=True, encoding="utf8")
    return shown.stdout if shown.returncode == 0 else None


def test_every_committed_fixture_is_recorded_or_explained():
    listed = {path for _recording, _tab, path in FIXTURES}
    assert len(listed) == len(FIXTURES)  # one recording per file
    assert sorted(listed | set(recordings.NOT_RECORDINGS)) == recordings.fixture_files()
    assert not listed & set(recordings.NOT_RECORDINGS)


def test_recordings_are_well_formed():
    assert len({recording.name for recording in recordings.RECORDINGS}) == len(recordings.RECORDINGS)
    for recording in recordings.RECORDINGS:
        assert recording.format in recordings.FORMATS, recording.name
        assert set(recording.contexts) <= set(recordings.CONTEXTS), recording.name
        assert recording.files, recording.name
        tabs = set(recording.files)
        assert tabs <= {"txt2img", "img2img"} or tabs in ({"extras"}, {"page"}, {"all"}), recording.name
        assert (REPO / recording.script).exists(), recording.name  # the port, in this checkout


@pytest.mark.parametrize("recording", recordings.RECORDINGS, ids=lambda recording: recording.name)
def test_each_recording_names_a_gradio_ui_at_its_revision(recording):
    """The script still had its Gradio ui() (or on_ui_tabs page) where it is recorded from."""

    if subprocess.run(["git", "cat-file", "-e", f"{recording.rev}^{{commit}}"], cwd=REPO, capture_output=True).returncode != 0:
        pytest.skip(f"{recording.rev[:10]} is not in this clone")
    if recording.format == "processing_scripts":
        listed = subprocess.run(["git", "ls-tree", "--name-only", recording.rev, f"{recording.script}/"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
        sources = {Path(path).stem: _git_show(recording.rev, path) for path in listed if path.endswith(".py")}
        # ScriptComments has no controls (the fixture records it with no inputs)
        assert sorted(name for name, source in sources.items() if "def ui(self" not in source) == ["comments"]
        return
    source = _git_show(recording.rev, recording.script)
    assert source is not None, f"{recording.script} is not at {recording.rev[:10]}"
    assert ("on_ui_tabs" in source) if recording.format == "page" else ("def ui(self" in source)


@pytest.mark.parametrize(("recording", "tab", "path"), FIXTURES, ids=lambda value: value if isinstance(value, str) else None)
def test_compare_reads_every_fixture_format(recording, tab, path):
    """record_legacy_ui.compare() on each committed fixture: its format is recognised and
    it matches itself (as a new recording identical to the committed one does)."""

    committed = REPO / path
    data = json.loads(committed.read_text(encoding="utf8"))
    assert recordings.format_of(data) == recording.format
    assert recorder.compare(committed, committed) == []
    assert recorder.compare(committed, committed, recording.format) == []


def _first_scalar(data, path=()):
    """The path of the first number, string or bool in ``data`` (depth first, inputs first)."""

    if isinstance(data, dict):
        keys = sorted(data, key=lambda key: key not in ("inputs", "args", "txt2img"))
        for key in keys:
            found = _first_scalar(data[key], (*path, key))
            if found is not None:
                return found
    elif isinstance(data, list):
        for index, item in enumerate(data):
            found = _first_scalar(item, (*path, index))
            if found is not None:
                return found
    elif isinstance(data, (str, int, float)):
        return path
    return None


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return value + " (changed)"


FORMAT_SAMPLES = {}
for _recording, _tab, _path in FIXTURES:
    FORMAT_SAMPLES.setdefault(_recording.format, _path)


@pytest.mark.parametrize(("fmt", "path"), sorted(FORMAT_SAMPLES.items()))
def test_compare_reports_each_difference_in_every_format(fmt, path, tmp_path):
    committed = REPO / path
    data = json.loads(committed.read_text(encoding="utf8"))
    recorded = tmp_path / Path(path).name

    where = _first_scalar(data)
    assert where is not None
    changed = copy.deepcopy(data)
    target = changed
    for part in where[:-1]:
        target = target[part]
    target[where[-1]] = _changed(target[where[-1]])
    recorded.write_text(json.dumps(changed), encoding="utf8")
    problems = recorder.compare(recorded, committed, fmt)
    assert len(problems) == 1 and problems[0].startswith(f"{committed.name}: ")

    # a key the recording lacks, and one the fixture lacks
    entry_path = where[:-1]
    changed = copy.deepcopy(data)
    target = changed
    for part in entry_path:
        target = target[part]
    if isinstance(target, dict):
        target.pop(where[-1])
        target["recorded_only"] = 1
        recorded.write_text(json.dumps(changed), encoding="utf8")
        problems = recorder.compare(recorded, committed, fmt)
        assert len(problems) == 2, problems
        if fmt == "port":  # argument by argument, as remi_port_harness.assert_matches_snapshot reads them
            assert any(f" {where[-1]} is None, the fixture has" in problem for problem in problems)
            assert any(" recorded_only is 1, the fixture has None" in problem for problem in problems)
        else:
            assert any("the recording has no such key" in problem for problem in problems)
            assert any("the fixture has no such key" in problem for problem in problems)

    # an argument too many
    if isinstance(data, list) and data:
        recorded.write_text(json.dumps([*data, data[0]]), encoding="utf8")
        assert "1 arguments recorded" not in " ".join(recorder.compare(recorded, committed, fmt))
        assert any("arguments recorded, the fixture has" in problem for problem in recorder.compare(recorded, committed, fmt))


def test_a_recording_in_another_format_is_reported_not_a_crash(tmp_path):
    """The default (bare list) record of a script whose fixture is a dict names the mismatch."""

    committed = REPO / "extensions-builtin" / "sd-webui-detail-daemon" / "tests" / "fixtures" / "legacy_ui_txt2img.json"
    recorded = tmp_path / "legacy_ui_txt2img.json"
    recorded.write_text(json.dumps(json.loads(committed.read_text(encoding="utf8"))["inputs"]), encoding="utf8")
    problems = recorder.compare(recorded, committed)
    assert problems == ["legacy_ui_txt2img.json: the fixture is in the 'contract' format, the recording in 'port' (record it with --fixtures)"]


def test_numbers_compare_by_value_bools_by_type():
    assert recordings.differences({"a": 1.0}, {"a": 1}) == []
    assert recordings.differences({"a": True}, {"a": 1}) == ["a: recorded True, the fixture has 1"]
    assert recordings.differences([1, 2], [1]) == ["the record: 2 items recorded, the fixture has 1"]


def test_a_script_of_a_committed_fixture_records_in_its_format():
    """record_legacy_ui.py <script> uses the fixture's recording (format, context) for its script."""

    recording = recordings.recording_of("extensions-builtin/sd-webui-detail-daemon/scripts/detail_daemon.py")
    assert recording is not None and recording.format == "contract"
    assert recordings.recording_of(REPO / "extensions-builtin" / "sd_forge_spectrum" / "scripts" / "spectrum.py").format == "port"
    assert recordings.recording_of("extensions-builtin/not-there/scripts/x.py") is None
    options = recorder.parse_args(["extensions-builtin/sd_forge_multidiffusion/scripts/forge_multidiffusion.py", "--tabs", "txt2img", "--options", '{"a": 1}', "--preimport", "x"])
    changed = recordings.with_options(recordings.recording_of(options.script), options)
    assert list(changed.files) == ["txt2img"] and changed.shown_without == "--tabs"
    assert changed.options == {"a": 1} and changed.preimport == ("backend.nn.cnets.cldm", "x")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--fixtures", "nope"], "unknown fixtures ['nope']"),
        (["--fixtures", "--self-test"], "--fixtures takes fixture names"),
        (["--fixtures", "extensions-builtin/a/scripts/x.py"], "unknown fixtures"),
        (["--fixtures", "--tabs", "img2img"], "--fixtures takes fixture names"),
        (["--fixtures", "--report", "/tmp/r.json"], "--report records a comparison"),
        (["--fixtures", "--jobs", "0"], "--jobs must be at least 1"),
        (["--record-fixture", "sd_forge_spectrum"], "--record-fixture needs --legacy-tree and --out"),
    ],
)
def test_fixtures_options(argv, message, capsys):
    with pytest.raises(SystemExit):
        recorder.parse_args(argv)
    assert message in capsys.readouterr().err


def test_fixtures_options_select_recordings():
    options = recorder.parse_args(["--fixtures", "--check"])
    assert recorder.fixture_recordings(options) == list(recordings.RECORDINGS)
    options = recorder.parse_args(["--fixtures", "sd_forge_controlnet", "scripts/xyz_grid", "--check", "--jobs", "2", "--report", "/tmp/r.json"])
    assert [recording.name for recording in recorder.fixture_recordings(options)] == ["sd_forge_controlnet", "scripts/xyz_grid"] and options.jobs == 2
    report = recorder.report(options, [REPO / "tests" / "fixtures" / "legacy_ui" / "xyz_grid_txt2img.json"], [])
    assert report["mode"] == "fixtures" and report["script"] is None and report["fixtures"] == ["sd_forge_controlnet", "scripts/xyz_grid"]
    assert report["compared"] == ["tests/fixtures/legacy_ui/xyz_grid_txt2img.json"] and report["passed"] is True


def test_the_recorder_imports_no_gradio():
    """The recordings run in the legacy revision and use its Gradio (legacy_gradio()): no import here."""

    import ast

    for path in (REPO / "tools" / "legacy_ui_recordings.py", REPO / "tools" / "record_legacy_ui.py"):
        tree = ast.parse(path.read_text(encoding="utf8"))
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names] + [node.module or "" for node in imports if isinstance(node, ast.ImportFrom)]
        assert not [name for name in names if name.split(".")[0] in ("gradio", "gradio_client", "gradio_rangeslider")], path
