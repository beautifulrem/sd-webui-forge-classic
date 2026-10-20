"""The upstream-absorption tools: line-ending guard and legacy ui() recorder.

``tools/check_line_endings.py`` keeps the files shared with Forge Neo on
upstream's CRLF line endings; ``tools/record_legacy_ui.py`` records a Gradio
``ui()`` for a port's snapshot test (it needs Gradio to run, so only its
command line and its Gradio-free parts are tested here).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"tool_{name}", REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eol = _load("check_line_endings")
recorder = _load("record_legacy_ui")


# region Line endings


def test_shared_upstream_files_keep_crlf():
    files = eol.read_manifest()
    assert "modules/scripts.py" in files and "modules/processing.py" in files
    assert eol.problems(files) == []


def test_bare_lf_detection_and_conversion():
    assert eol.bare_lf_count(b"a\r\nb\r\n") == 0
    assert eol.bare_lf_count(b"a\r\nb\nc\n") == 2
    assert eol.to_crlf(b"a\r\nb\nc") == b"a\r\nb\r\nc"
    assert eol.to_crlf(b"a\r\nb\r\n") == b"a\r\nb\r\n"


def test_check_and_fix_on_a_tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "ok.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (tmp_path / "pkg" / "lf.py").write_bytes(b"x = 1\r\ny = 2\nz = 3\n")
    files = ["pkg/ok.py", "pkg/lf.py", "pkg/gone.py"]
    found = eol.problems(files, root=str(tmp_path))
    assert found == ["pkg/lf.py: 2 LF line ending(s), upstream uses CRLF", "pkg/gone.py: listed but missing (deleted? run --update)"]
    assert eol.fix(files, root=str(tmp_path)) == ["pkg/lf.py"]
    assert (tmp_path / "pkg" / "lf.py").read_bytes() == b"x = 1\r\ny = 2\r\nz = 3\r\n"
    assert eol.problems(files[:2], root=str(tmp_path)) == []


def test_manifest_round_trip(tmp_path):
    path = str(tmp_path / "list.txt")
    eol.write_manifest(["b.py", "a.py"], path)
    text = Path(path).read_text(encoding="utf-8")
    assert text.startswith("# ") and text.endswith("a.py\nb.py\n")
    assert eol.read_manifest(path) == ["a.py", "b.py"]


def test_line_ending_cli_rejects_base_without_update(capsys):
    with pytest.raises(SystemExit):
        eol.main(["--base", "HEAD"])
    assert "--base only applies to --update" in capsys.readouterr().err


# endregion

# region Legacy ui() recorder


def test_recorder_defaults():
    options = recorder.parse_args(["extensions-builtin/sd_forge_spectrum/scripts/spectrum.py"])
    assert options.rev == recorder.LEGACY_REV and options.legacy_tree is None and options.script_rev is None
    assert options.tabs is None and options.preimport == [] and options.options == {} and options.class_name is None
    assert recorder.output_folder(options) == REPO / "extensions-builtin" / "sd_forge_spectrum" / "tests" / "fixtures"


def test_recorder_options():
    options = recorder.parse_args(
        [
            "extensions-builtin/extra-options-section/scripts/extra_options_section.py",
            "--script-rev",
            "upstream/neo",
            "--tabs",
            "img2img",
            "--class",
            "ExtraOptionsSection",
            "--options",
            json.dumps({"extra_options_img2img": ["sd_vae"]}),
            "--preimport",
            "modules.processing",
            "--legacy-tree",
            "/tmp/legacy",
            "--out",
            "/tmp/out",
        ]
    )
    assert options.script_rev == "upstream/neo" and options.tabs == ["img2img"] and options.class_name == "ExtraOptionsSection"
    assert options.options == {"extra_options_img2img": ["sd_vae"]} and options.preimport == ["modules.processing"]
    assert options.legacy_tree == Path("/tmp/legacy") and recorder.output_folder(options) == Path("/tmp/out").resolve()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["/abs/ext/scripts/x.py", "--script-rev", "upstream/neo"], "--script-rev takes a repository-relative script path"),
        (["scripts/xyz_grid.py"], "--out is required"),
        (["extensions-builtin/a/scripts/x.py", "--rev", "HEAD", "--legacy-tree", "/tmp/t"], "not allowed with argument"),
        (["extensions-builtin/a/scripts/x.py", "--tabs", "extras"], "invalid choice"),
        (["extensions-builtin/a/scripts/x.py", "--options", "{not json"], "invalid loads value"),
        ([], "the script to record is required"),
        (["--self-test", "extensions-builtin/a/scripts/x.py"], "--self-test takes no script"),
        (["--self-test", "--tabs", "img2img"], "--self-test takes no script"),
    ],
)
def test_recorder_rejects_bad_arguments(argv, message, capsys):
    with pytest.raises(SystemExit):
        recorder.parse_args(argv)
    assert message in capsys.readouterr().err


def test_recorder_self_test_checks_a_known_recording():
    """--self-test records Spectrum at its last Gradio revision and compares with the committed fixtures."""

    options = recorder.parse_args(["--self-test"])
    assert options.check and options.script == Path(recorder.SELF_TEST[0]) and options.script_rev == recorder.SELF_TEST[1]
    assert options.rev == recorder.LEGACY_REV
    folder = recorder.output_folder(options)
    assert (folder / "legacy_ui_txt2img.json").is_file() and (folder / "legacy_ui_img2img.json").is_file()
    shown = subprocess.run(["git", "show", f"{recorder.SELF_TEST[1]}:{recorder.SELF_TEST[0]}"], cwd=REPO, capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip("the self-test revision is not in this clone")
    assert "def ui(self" in shown.stdout  # the revision still has the Gradio ui() to record


def test_recorder_check_compares_like_the_snapshot_harness(tmp_path):
    entry = {"kind": "slider", "label": "Steps", "default": 20, "infotext": "s", "paste": False, "preset": None}
    committed = tmp_path / "legacy_ui_txt2img.json"
    committed.write_text(json.dumps([entry]))
    recorded = tmp_path / "recorded.json"
    # describe() now adds the wiring keys; a fixture without them means "none"
    recorded.write_text(json.dumps([{**entry, "exclusive": None, "visible_if": None, "enabled_if": None}]))
    assert recorder.compare(recorded, committed) == []
    recorded.write_text(json.dumps([{**entry, "default": 30}, entry]))
    assert recorder.compare(recorded, committed) == [
        "legacy_ui_txt2img.json: 2 arguments recorded, the fixture has 1",
        "legacy_ui_txt2img.json: argument 0 ('Steps') default is 30, the fixture has 20",
    ]
    assert recorder.compare(recorded, tmp_path / "missing.json") == [f"{tmp_path / 'missing.json'} does not exist"]


def test_recorder_extension_folder():
    assert recorder.extension_folder(Path("extensions-builtin/ext/scripts/x.py")) == Path("extensions-builtin/ext")
    assert recorder.extension_folder(REPO / "extensions-builtin" / "ext" / "scripts" / "x.py") == Path("extensions-builtin/ext")
    assert recorder.extension_folder(Path("scripts/xyz_grid.py")) is None
    assert recorder.extension_folder(Path("/elsewhere/ext/scripts/x.py")) is None


def test_recorder_refuses_without_gradio_or_legacy_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit, match="cannot import Gradio"):
        recorder.main(["extensions-builtin/a/scripts/x.py"])  # before exporting anything
    with pytest.raises(SystemExit, match="not a revision with the Gradio UI"):
        recorder.require_legacy_tree(tmp_path)
    (tmp_path / "remi" / "ui").mkdir(parents=True)
    (tmp_path / "remi" / "ui" / "legacy.py").write_text("")
    (tmp_path / "modules").mkdir()
    (tmp_path / "modules" / "ui_components.py").write_text("")
    recorder.require_legacy_tree(tmp_path)


def test_recorder_builds_the_ui_with_the_legacy_trees_gradio(monkeypatch):
    """legacy_snapshot runs the ui() through the legacy tree's build_ui and remi.ui.legacy (this file has no Gradio import)."""

    calls = []
    controls = [object(), object()]

    def build_ui(script, tab):
        calls.append((tab, script.is_img2img, script.is_txt2img, script.infotext_fields))
        return controls

    presets = types.ModuleType("modules.anima_presets")
    presets.preset_controls = lambda tab: (("Steps",), (controls[1],), (20,))
    legacy = types.ModuleType("remi.ui.legacy")
    legacy.legacy_layout = lambda owner, given, infotext: ("layout", owner, tuple(given))
    monkeypatch.setitem(sys.modules, "modules.anima_presets", presets)
    monkeypatch.setitem(sys.modules, "remi.ui.legacy", legacy)

    def describe(layout):
        assert layout == ("layout", "legacy", tuple(controls))
        return [{"name": "a"}, {"name": "b"}]

    entries = recorder.legacy_snapshot(types.SimpleNamespace(), "img2img", describe, build_ui)
    assert calls == [("img2img", True, False, [])]
    assert entries == [{"name": "a", "preset": None}, {"name": "b", "preset": "Steps"}]
    source = (REPO / "tools" / "record_legacy_ui.py").read_text(encoding="utf-8")
    assert "import gradio" not in source and "from gradio" not in source


def test_recorder_uses_the_harness_snapshot_format():
    """Records are written in describe()'s format, read back by load_snapshot."""

    spec = importlib.util.spec_from_file_location("remi_port_harness_for_tools", REPO / "tests" / "remi_port_harness.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    script = REPO / "extensions-builtin" / "sd_forge_spectrum" / "scripts" / "spectrum.py"
    options = recorder.parse_args([str(script.relative_to(REPO))])
    assert recorder.output_folder(options) / "legacy_ui_txt2img.json" == harness.fixture_path(script, "txt2img")


# endregion
