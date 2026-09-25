import os

import pytest

from modules import anima_user_data, paths_internal
from modules.anima_user_data import UnreadableDataFile, data_file, read_json_object, write_json_atomic


def test_data_file_lives_in_the_data_dir_and_is_seeded_once(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path / "data"))
    seed = tmp_path / "bundled.json"
    seed.write_text('{"default": 1}', encoding="utf-8")

    path = data_file("some-extension", "templates.json", seed=seed)
    assert path == tmp_path / "data" / "extension-data" / "some-extension" / "templates.json"
    assert read_json_object(path) == {"default": 1}

    write_json_atomic(path, {"mine": 2})
    assert read_json_object(data_file("some-extension", "templates.json", seed=seed)) == {"mine": 2}  # not re-seeded


def test_unreadable_files_raise_instead_of_reading_as_empty(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"half": ', encoding="utf-8")
    listing = tmp_path / "list.json"
    listing.write_text("[1, 2]", encoding="utf-8")

    for path in (broken, listing):
        with pytest.raises(UnreadableDataFile):
            read_json_object(path)
    assert read_json_object(tmp_path / "missing.json") == {}


def test_atomic_write_keeps_permissions_and_leaves_no_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    write_json_atomic(path, {"a": 1})
    assert os.stat(path).st_mode & 0o777 == 0o644
    os.chmod(path, 0o600)
    write_json_atomic(path, {"a": 2})
    assert os.stat(path).st_mode & 0o777 == 0o600 and read_json_object(path) == {"a": 2}

    monkeypatch.setattr(anima_user_data.json, "dump", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        write_json_atomic(path, {"a": 3})
    assert read_json_object(path) == {"a": 2}
    assert [entry.name for entry in tmp_path.iterdir()] == ["state.json"]
