"""The gallery's Save / Save zip / Open folder helpers (remi/server/gallery.py)
on their own: log.csv upgrades, non-image files, zip names, free file
names, the output folder options and the file manager per platform."""

from __future__ import annotations

import csv
import time
import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image

from remi.files import FileRecord
from remi.server import gallery
from remi.server.gallery import LOG_FIELDS, GalleryError, SavedParams


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


def record_for(path, *, mime="image/png", infotext=None, width=None, height=None) -> FileRecord:
    return FileRecord("f" * 12, "output", str(path), path.name, mime, path.stat().st_size, width, height, time.time(), infotext=infotext)


# region Saving


def test_saved_params_read_the_infotext_and_answer_none_for_the_rest(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(path)
    p = SavedParams({"Prompt": "a fox", "Seed": 7, "Steps": 20, "Shift": 3.0, "Size-1": 640, "Size-2": 480}, record_for(path, width=64))
    assert (p.prompt, p.seed, p.all_seeds, p.steps, p.distilled_cfg_scale) == ("a fox", 7, [7], 20, 3.0)
    assert (p.width, p.height) == (64, 480)  # the file's own size wins over the infotext's
    assert p.negative_prompt == "" and p.batch_size == p.n_iter == 1 and p.styles == []
    assert p.anything_the_filename_pattern_asks_for is None


def test_an_old_log_gains_the_new_columns(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", encoding="utf8", newline="") as file:
        csv.writer(file).writerows([LOG_FIELDS[:9], ["a fox", "1", "64", "64", "Euler", "7", "20", "a.png", ""]])
    gallery.update_logfile(str(path), LOG_FIELDS)
    with open(path, encoding="utf8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == LOG_FIELDS and rows[1] == ["a fox", "1", "64", "64", "Euler", "7", "20", "a.png", "", "", ""]


def test_a_current_or_empty_log_is_left_alone(tmp_path):
    current = tmp_path / "log.csv"
    current.write_text(",".join(LOG_FIELDS) + "\r\n", encoding="utf8")
    before = current.read_bytes()
    gallery.update_logfile(str(current), LOG_FIELDS)
    assert current.read_bytes() == before
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf8")
    gallery.update_logfile(str(empty), LOG_FIELDS)
    assert empty.read_text(encoding="utf8") == ""


def test_free_names_count_up_and_keep_the_extension(tmp_path):
    assert gallery._free_name(str(tmp_path), "a.png") == str(tmp_path / "a.png")
    (tmp_path / "a.png").touch()
    (tmp_path / "a-1.png").touch()
    assert gallery._free_name(str(tmp_path), "sub/a.png") == str(tmp_path / "a-2.png")  # never a sub folder
    assert gallery._free_name(str(tmp_path), "") == str(tmp_path / "file")


def test_save_copies_other_files_and_updates_an_old_log(webui, tmp_path, monkeypatch):
    from modules import shared

    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    with open(save_dir / "log.csv", "w", encoding="utf8", newline="") as file:
        csv.writer(file).writerows([LOG_FIELDS[:9], ["older", "1", "8", "8", "Euler", "7", "20", "old.png", ""]])
    monkeypatch.setattr(shared.opts, "outdir_save", str(save_dir), raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)
    monkeypatch.setattr(shared.opts, "save_write_log_csv", True, raising=False)
    image = tmp_path / "gen.png"
    Image.new("RGB", (16, 12), "red").save(image)
    notes = tmp_path / "notes.txt"
    notes.write_text("prompt ideas", encoding="utf8")
    (save_dir / "notes.txt").write_text("taken", encoding="utf8")

    result = gallery.save_outputs([record_for(notes, mime="text/plain"), record_for(image, infotext="a red square\nSteps: 5, Seed: 9, Size: 16x12", width=16, height=12)])

    assert result.saved[0] == "notes-1.txt" and (save_dir / "notes-1.txt").read_text(encoding="utf8") == "prompt ideas"
    with Image.open(save_dir / result.saved[1]) as saved:
        assert saved.size == (16, 12)
    with open(save_dir / "log.csv", encoding="utf8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == LOG_FIELDS and len(rows[1]) == len(LOG_FIELDS)  # the old row padded
    assert rows[2][:4] == ["a red square", "9", "16", "12"] and len(rows) == 3  # only images are logged
    assert result.zip_path is None


def test_save_without_the_log_setting_writes_no_log(webui, tmp_path, monkeypatch):
    from modules import shared

    save_dir = tmp_path / "saved"
    monkeypatch.setattr(shared.opts, "outdir_save", str(save_dir), raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)
    monkeypatch.setattr(shared.opts, "save_write_log_csv", False, raising=False)
    image = tmp_path / "gen.png"
    Image.new("RGB", (8, 8)).save(image)
    result = gallery.save_outputs([record_for(image, infotext="x\nSteps: 5, Seed: 1, Size: 8x8")])
    assert len(result.saved) == 1 and not (save_dir / "log.csv").exists()


def test_a_zip_is_named_by_the_pattern_or_images_when_that_fails(webui, tmp_path, monkeypatch):
    from modules import images, shared

    save_dir = tmp_path / "saved"
    monkeypatch.setattr(shared.opts, "outdir_save", str(save_dir), raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)
    monkeypatch.setattr(shared.opts, "grid_zip_filename_pattern", "pack-[seed]", raising=False)
    image = tmp_path / "gen.png"
    Image.new("RGB", (8, 8)).save(image)
    record = record_for(image, infotext="x\nSteps: 5, Seed: 31, Size: 8x8")  # (a parameters line has 3 or more)

    named = gallery.save_outputs([record], make_zip=True)
    assert named.zip_path is not None and named.zip_path.endswith("pack-31.zip")
    with zipfile.ZipFile(named.zip_path) as archive:
        assert archive.namelist() == named.saved

    class BrokenZipNames(images.FilenameGenerator):
        """Saving the image still works; naming the zip (``zip=True``) fails."""

        def __init__(self, p, seed, prompt, image, zip=False, basename=""):
            if zip:
                raise RuntimeError("bad pattern")
            super().__init__(p, seed, prompt, image, zip, basename)

    monkeypatch.setattr(images, "FilenameGenerator", BrokenZipNames)
    fallback = gallery.save_outputs([record], make_zip=True)
    assert fallback.zip_path is not None and fallback.zip_path.endswith("images.zip")


# endregion

# region Folders


def test_output_folder_takes_the_first_set_option(monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared, "opts", SimpleNamespace(outdir_txt2img_samples="", outdir_samples="/all", outdir_txt2img_grids="/grids", outdir_grids="", outdir_save=""))
    assert gallery.output_folder("txt2img") == "/all"
    assert gallery.output_folder("grids") == "/grids"
    with pytest.raises(GalleryError) as unset:
        gallery.output_folder("save")
    assert unset.value.status == 404 and "not set" in str(unset.value)
    with pytest.raises(GalleryError) as unknown:
        gallery.output_folder("../etc")
    assert unknown.value.status == 404


def test_open_folder_refuses_a_missing_folder(tmp_path):
    with pytest.raises(GalleryError) as missing:
        gallery.open_folder(str(tmp_path / "not-yet"))
    assert missing.value.status == 404


@pytest.mark.parametrize(
    ("system", "release", "expected"),
    [
        ("Linux", "6.8.0-generic", ["xdg-open"]),
        ("Linux", "5.15.153.1-microsoft-standard-WSL2", ["explorer.exe", "C:\\out"]),
    ],
)
def test_open_folder_uses_the_platforms_file_manager(tmp_path, monkeypatch, system, release, expected):
    calls = []
    monkeypatch.setattr(gallery.platform, "system", lambda: system)
    monkeypatch.setattr(gallery.platform, "uname", lambda: SimpleNamespace(release=release))
    monkeypatch.setattr(gallery.subprocess, "Popen", calls.append)
    monkeypatch.setattr(gallery.subprocess, "check_output", lambda command, text: "C:\\out\n" if command[0] == "wslpath" else pytest.fail(command))
    gallery.open_folder(str(tmp_path) + "/")
    target = [str(tmp_path)] if expected == ["xdg-open"] else []
    assert calls == [expected + target]


def test_open_folder_on_windows_starts_the_folder(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(gallery.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gallery.os, "startfile", started.append, raising=False)
    gallery.open_folder(str(tmp_path))
    assert started == [str(tmp_path)]


# endregion
