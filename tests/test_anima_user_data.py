import os
import threading
import time

import pytest

from modules import anima_user_data, paths_internal
from modules.anima_user_data import TemplateStore, UnreadableDataFile, data_file, read_json_object, write_json_atomic


def test_data_file_lives_in_the_data_dir_and_is_seeded_once(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path / "data"))
    seed = tmp_path / "bundled.json"
    seed.write_text('{"default": 1}', encoding="utf-8")

    path = data_file("some-extension", "templates.json", seed=seed)
    assert path == tmp_path / "data" / "extension-data" / "some-extension" / "templates.json"
    assert read_json_object(path) == {"default": 1}

    write_json_atomic(path, {"mine": 2})
    assert read_json_object(data_file("some-extension", "templates.json", seed=seed)) == {"mine": 2}  # not re-seeded


def test_seeding_copies_under_the_edit_lock_and_never_leaves_a_partial_file(tmp_path, monkeypatch):
    """A seed copy racing an edit could overwrite it, and a copy cut short would
    leave half a JSON file that every later read refuses (UnreadableDataFile)."""

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path / "data"))
    seed = tmp_path / "bundled.json"
    seed.write_text('{"default": 1}', encoding="utf-8")
    target = tmp_path / "data" / "extension-data" / "some-extension" / "templates.json"
    copy = anima_user_data.shutil.copyfile

    def interrupted_copy(source, destination):
        assert anima_user_data._edit_lock(target).locked()
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write('{"def')
        raise OSError("disk full")

    monkeypatch.setattr(anima_user_data.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError):
        data_file("some-extension", "templates.json", seed=seed)
    assert not target.exists() and os.listdir(target.parent) == []

    monkeypatch.setattr(anima_user_data.shutil, "copyfile", copy)
    assert read_json_object(data_file("some-extension", "templates.json", seed=seed)) == {"default": 1}
    assert os.stat(target).st_mode & 0o777 == 0o644


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


def test_template_store_edits_in_place_and_never_replaces_an_unreadable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    store = TemplateStore("some-extension", "templates.json")
    assert store.load() == {}

    assert store.edit(lambda data: data.setdefault("mine", {"a": 1}) and "saved") == "saved"
    assert store.load() == {"mine": {"a": 1}}
    writes = []
    monkeypatch.setattr(anima_user_data, "write_json_atomic", lambda path, data: writes.append(data))
    store.edit(lambda data: data.get("missing"))  # nothing changed: nothing written
    assert writes == []
    monkeypatch.undo()

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    store.path().write_text('{"mine": ', encoding="utf-8")
    assert store.load() == {}
    with pytest.raises(UnreadableDataFile):
        store.edit(lambda data: data.update(new=1))
    assert store.path().read_text(encoding="utf-8") == '{"mine": '


def test_concurrent_template_edits_all_land(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    stores = [TemplateStore("some-extension", "templates.json") for _ in range(8)]
    start = threading.Barrier(len(stores))

    def save(index):
        def change(data):
            time.sleep(0.01)  # widen the read-modify-write window
            data[f"template {index}"] = {"index": index}

        start.wait()
        stores[index].edit(change)

    threads = [threading.Thread(target=save, args=(index,)) for index in range(len(stores))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(stores[0].load()) == sorted(f"template {index}" for index in range(len(stores)))
