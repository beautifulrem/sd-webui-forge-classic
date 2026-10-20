"""The queue importers (remi/jobs/importers.py) piece by piece: the restricted unpickler, value
descriptions and images, the argument mappers (Gradio handler names, A1111 API fields, script
slots), the source readers and the import failures. Whole imports through the API are tested in
test_remi_server_queue_tools.py."""

import base64
import collections
import io
import json
import os
import pickle
import sqlite3
import zlib

import numpy as np
import pytest
from PIL import Image

from remi.jobs import importers
from remi.jobs.importers import ImportFailed, ImportItem, ImportReport, Mapper, note, note_text
from remi.server.testing import server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def server(webui, tmp_path):
    with server_app(tmp_path) as server:
        yield server


def mapper(tab="txt2img"):
    stored = []

    def store(image):
        stored.append(image)
        return f"upload-{len(stored)}"

    instance = Mapper(tab, store)
    instance.stored = stored
    return instance


def png_data_url(size=(3, 2)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (9, 9, 9)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def texts(notes):
    return [note_text(entry) for entry in notes]


# region notes and reports


def test_notes_render_as_english_text():
    assert note_text("already text") == "already text"
    assert note_text(note("Only the first of {{count}} source images was kept.", count=3)) == "Only the first of 3 source images was kept."
    assert note_text({"message": "no values"}) == "no values"


def test_the_report_counts_everything_but_lists_a_bounded_number_of_items(monkeypatch):
    monkeypatch.setattr(importers, "MAX_REPORT_ITEMS", 3)
    report = ImportReport("prompt-queue", "/q.json")
    for index in range(5):
        report.add(ImportItem(str(index), "t", "queued"))
    report.add(ImportItem("s", "t", "skipped"))
    report.add(ImportItem("e", "t", "error"))
    assert (report.queued, report.skipped, report.errors) == (5, 1, 1)
    assert [item.source_id for item in report.items] == ["0", "1", "2"]
    assert report.json()["items"][0] == {"source_id": "0", "title": "t", "outcome": "queued", "job_id": None, "notes": []}


# endregion

# region unpickling and stored script arguments


def test_the_unpickler_accepts_plain_containers_and_numpy_data():
    value = [{1, 2}, frozenset({3}), complex(1, 2), bytearray(b"x"), collections.OrderedDict(a=1), np.arange(6, dtype=np.uint8).reshape(2, 3), np.float32(1.5)]
    loaded = importers.safe_loads(pickle.dumps(value))
    assert loaded[:5] == value[:5] and loaded[5].tolist() == [[0, 1, 2], [3, 4, 5]] and loaded[6] == 1.5


@pytest.mark.parametrize("value", [print, collections.Counter("ab"), Image.Resampling.LANCZOS])
def test_the_unpickler_refuses_other_globals(value):
    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        importers.safe_loads(pickle.dumps(value))


def test_script_arguments_that_are_empty_or_not_a_list():
    assert importers.read_script_params(None, None) == ([], [])
    assert importers.read_script_params(b"", b"k" * 32) == ([], [])
    values, notes = importers.read_script_params(zlib.compress(pickle.dumps({"a": 1})), None)
    assert values is None
    assert texts(notes) == ["The script arguments were saved unsigned; they were read with a restricted loader.", "The script arguments are not a list."]
    values, notes = importers.read_script_params(b"not zlib", None)
    assert values is None and "could not be read" in texts(notes)[-1]


# endregion

# region values


def test_opaque_values_are_described_briefly():
    assert importers.describe(Image.new("RGB", (4, 3))) == "<image 4×3>"
    assert importers.describe({"cls": "ndarray", "data": "..."}) == "<ndarray>"
    assert importers.describe({str(i): i for i in range(8)}) == "{0, 1, 2, 3, 4, 5, …}"
    assert importers.describe({"a": 1}) == "{a}"
    assert importers.describe((1, 2, 3)) == "[3 items]"
    assert importers.describe("x" * 100) == "'" + "x" * 76 + "…"
    assert importers._jsonable({"ok": [1, 2]}) == {"ok": [1, 2]}
    assert importers._jsonable(float("nan")) == "nan"
    assert importers._jsonable({1, 2}) == "{1, 2}"


def test_stored_images_of_every_kind_become_pil_images():
    image = Image.new("RGB", (2, 2))
    assert importers.deserialize_image(image) is image
    array = np.full((2, 3, 3), 200, dtype=np.uint8)
    as_dict = {"cls": "ndarray", "dtype": "uint8", "shape": [2, 3, 3], "data": base64.b64encode(zlib.compress(array.tobytes())).decode()}
    restored = importers.deserialize_image(as_dict)
    assert restored.size == (3, 2) and restored.getpixel((0, 0)) == (200, 200, 200)
    assert importers.deserialize_image(png_data_url()).size == (3, 2)
    assert importers.deserialize_image(np.zeros((4, 5), dtype=np.float32)).size == (5, 4)
    # anything else is returned as it is
    for value in ("a path.png", 7, None, np.zeros(3), {"cls": "Tensor"}):
        assert importers.deserialize_image(value) is value


# endregion

# region mapping a task's arguments


def test_core_values_that_do_not_fit_are_kept_for_validation_to_report(webui):
    m = mapper("img2img")
    assert m.core_value("batch_upload", ["a.png"]) == []
    assert m.core_value("init_img", None) is None
    assert m.core_value("init_img", "upload-id") == "upload-id"
    assert m.core_value("init_img", png_data_url()) == "upload-1" and m.stored[0].size == (3, 2)
    width = m.layout.by_name["width"]
    original = width.coerce

    def refuse(value, ctx):
        raise ValueError("no")

    try:
        width.coerce = refuse
        assert m.core_value("width", {"weird": float("inf")}) == "{weird}"
    finally:
        width.coerce = original


def test_gradio_img2img_arguments_take_each_modes_layers(webui):
    sketch = {"image": png_data_url((5, 5)), "mask": png_data_url((5, 5))}
    core = mapper("img2img").legacy_core({"mode": 2, "init_img_with_mask": sketch, "prompt": "inpaint", "n_iter": 3})
    assert core["init_img"] == "upload-1" and core["mask"] == "upload-2" and core["mask_upload"] is None and core["batch_count"] == 3

    m = mapper("img2img")
    core = m.legacy_core({"mode": 4, "init_img_inpaint": png_data_url(), "init_mask_inpaint": png_data_url(), "mask_alpha": 0.5})
    assert (core["init_img"], core["mask"], core["mask_upload"]) == ("upload-1", None, "upload-2")
    # an unknown mode index reads no images; a mode that is not an index is the first mode
    assert mapper("img2img").legacy_core({"mode": 9, "init_img": png_data_url()})["init_img"] is None
    assert mapper("img2img").legacy_core({"mode": "sketch?", "init_img": png_data_url()})["init_img"] == "upload-1"


def test_api_fields_map_to_core_params_scripts_and_setting_overrides(webui):
    m = mapper("img2img")
    args = {
        "prompt": "api",
        "init_images": [png_data_url(), png_data_url()],
        "mask": png_data_url(),
        "override_settings": {"CLIP_stop_at_last_layers": 2, "no_such_setting": 1},
        "sampler_index": "Euler",
        "sampler_name": "DPM++ 2M",
        "scheduler": None,
        "unknown_zero": 0,
        "unknown_set": "x",
    }
    core, paste = m.api_core(args)
    assert core["prompt"] == "api" and core["init_img"] == "upload-1" and core["mask_upload"] == "upload-2"
    assert core["mode"] == "Inpaint upload" and core["override_settings"] == ["Clip skip: 2"]
    assert paste == {"Sampler": "DPM++ 2M"}
    assert texts(m.notes) == [
        "Setting overrides without an infotext name were dropped: no_such_setting.",
        "Only the first of 2 source images was kept.",
        "Not mapped (no such parameter now): unknown_set.",
    ]
    _core, paste = mapper().api_core({"sampler_index": "Euler", "steps": None})
    assert paste == {"Sampler": "Euler"}


def test_the_mode_index_falls_back_to_the_first_mode(webui):
    m = mapper("img2img")
    assert m._mode_index("Inpaint upload") == 4
    assert m._mode_index("no such mode") == 0
    assert mapper("txt2img")._mode_index("img2img") == 0


def test_script_slots_short_long_and_forms_without_scripts(webui):
    m = mapper()
    slots = m.runner.ensure_arg_slots()
    selected, scripts = m.script_values([0])
    assert selected == "None" and "seed" in scripts
    assert texts(m.notes) == [f"Script arguments were missing; their defaults are used: {slots - 1}."]
    # trailing values that are all None are not opaque
    m = mapper()
    m.script_values([0] * slots + [None, None])
    assert m.opaque == [] and m.notes == []
    # serialized images in the slots are uploaded
    assert m._upload(Image.new("RGB", (1, 1))) == "upload-1"
    assert mapper("extras").script_values([1, 2]) == (None, {})
    assert mapper("extras").api_script_values({}, {}) == (None, {})


def test_api_scripts_by_name_with_named_or_positional_arguments(webui):
    m = mapper()
    seed_layout = m.runner.layout_for(m.runner.by_id["seed"])
    count = len(seed_layout.names)
    args = {
        "alwayson_scripts": {"Sampler": {"args": {"steps": 12, "not_a_param": 1}}, "Agent Scheduler": {"args": []}},
        "script_name": "seed",
        "script_args": [77, *([None] * (count - 1)), "extra 1", "extra 2"],
    }
    selected, values = m.api_script_values(args, {"Seed": 5})
    assert selected == "Seed"
    assert values["sampler"]["steps"] == 12 and "not_a_param" not in values["sampler"]
    assert values["seed"]["seed"] == 77
    assert m.opaque == ["extra 1", "extra 2"]
    assert texts(m.notes) == ["Seed: arguments past its parameters could not be mapped: 2."]
    # positional args as a bare list, and a selected script that is not installed
    m = mapper()
    _selected, values = m.api_script_values({"alwayson_scripts": {"sampler": [None, None, None]}, "script_name": "X/Y/Z plot"}, {})
    assert "sampler" in values and texts(m.notes) == ["The script “X/Y/Z plot” is not installed; its arguments were dropped."]


# endregion

# region sources on disk


def test_source_paths_and_counts(webui, tmp_path, monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "agent_scheduler_sqlite_file", "custom/tasks.sqlite3", raising=False)
    paths = importers.agent_scheduler_paths()
    assert paths[0].endswith(os.path.join("custom", "tasks.sqlite3")) and len(paths) == len(set(paths))
    monkeypatch.setattr(shared.cmd_opts, "agent_scheduler_sqlite_file", str(tmp_path / "abs.sqlite3"), raising=False)
    assert importers.agent_scheduler_paths()[0] == str(tmp_path / "abs.sqlite3")
    assert importers.find_source("elsewhere") is None

    broken = tmp_path / "broken.sqlite3"
    broken.write_bytes(b"not a database")
    assert importers.count_agent_scheduler(str(broken)) == 0
    queue = tmp_path / "queue.json"
    for content in ("{", "[]", json.dumps({"items": [{"status": "pending"}, {"status": "done"}, "junk", {"status": "running"}]})):
        queue.write_text(content, encoding="utf-8")
        assert importers.count_prompt_queue(str(queue)) == (2 if content.startswith('{"') else 0)
    assert importers.count_prompt_queue(str(tmp_path / "missing.json")) == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (1700000000, 1700000000.0), (1.5, 1.5), ("2026-01-02T03:04:05Z", 1767323045.0), ("2026-01-02 03:04:05", 1767323045.0), ("2026-01-02T05:04:05+02:00", 1767323045.0), ("yesterday", None)],
)
def test_timestamps_of_every_stored_form(value, expected):
    assert importers._timestamp(value) == expected


def test_the_signing_key_and_image_paths(webui, tmp_path, monkeypatch):
    db = tmp_path / "task_scheduler.sqlite3"
    assert importers._signing_key(str(db)) is None
    (tmp_path / "agent_scheduler_signing.key").write_bytes(b"short")
    assert importers._signing_key(str(db)) is None

    from modules import paths_internal

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    image = outputs / "a.png"
    Image.new("RGB", (2, 2)).save(image)
    roots = [os.path.realpath(outputs)]
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    assert importers._resolve_image_path("outputs/a.png", roots) == os.path.realpath(image)
    for bad in (None, "", 5, "outputs/a.png\x00x", "outputs/missing.png", str(tmp_path / "outputs")):
        assert importers._resolve_image_path(bad, roots) is None
    assert importers._is_image(str(image)) and not importers._is_image(str(tmp_path / "missing.png"))


# endregion

# region import failures and edge cases


def agent_database(path, rows, schema=None):
    db = sqlite3.connect(path)
    db.executescript(schema or "CREATE TABLE task (id TEXT PRIMARY KEY, name TEXT, type TEXT, params TEXT, script_params BLOB, priority INTEGER, status TEXT, result TEXT, bookmarked BOOLEAN, created_at TEXT, updated_at TEXT);")
    for row in rows:
        db.execute(f"INSERT INTO task ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    db.commit()
    db.close()
    return str(path)


def api_task(task_id, status, args=None, result=None, **params):
    return {"id": task_id, "type": "txt2img", "params": json.dumps({"args": args or {"prompt": task_id}, "is_ui": False, **params}), "script_params": b"", "priority": 0, "status": status, "result": result}


def test_a_database_without_the_task_table_or_unreadable_is_refused(server, tmp_path):
    importer = importers.Importer(server.services.queue)
    with pytest.raises(ImportFailed) as missing:
        importer.agent_scheduler(str(tmp_path / "none.sqlite3"))
    assert missing.value.status == 404
    other = agent_database(tmp_path / "other.sqlite3", [], schema="CREATE TABLE task (id TEXT, name TEXT);")
    with pytest.raises(ImportFailed, match="no Agent Scheduler task table") as wrong:
        importer.agent_scheduler(other)
    assert wrong.value.status == 422
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"SQLite format 3\x00" + b"\xff" * 200)
    with pytest.raises(ImportFailed, match="cannot read the database"):
        importer.agent_scheduler(str(corrupt))


def test_task_by_task_outcomes(server, webui, tmp_path, monkeypatch):
    outputs = tmp_path / "out"
    outputs.mkdir()
    monkeypatch.setattr(importers, "output_roots", lambda: [os.path.realpath(outputs)])
    rows = [
        {"id": "broken", "type": "txt2img", "params": "{not json", "script_params": b"", "priority": 0, "status": "pending"},
        api_task("vae", "done", result=json.dumps({"images": [str(outputs / "gone.png")]}), checkpoint="System", vae="kl-f8.safetensors"),
        api_task("not-json-result", "done", result="saved somewhere"),
        api_task("bad-size", "pending", args={"prompt": "too wide", "width": "very"}),
        api_task("interrupted", "interrupted", result="Interrupted by the user"),
    ]
    path = agent_database(tmp_path / "tasks.sqlite3", rows)
    report = importers.Importer(server.services.queue).agent_scheduler(path)
    items = {item.source_id: item for item in report.items}
    assert items["broken"].outcome == "error" and "JSONDecodeError" in texts(items["broken"].notes)[0]
    assert (report.errors, report.history, report.invalid) == (1, 3, 1)

    vae = server.services.store.get(items["vae"].job_id)
    assert vae.core["override_settings"] == ["VAE: kl-f8.safetensors"]
    assert "The images of this task are no longer on disk." in texts(items["vae"].notes)
    assert server.services.store.get(items["not-json-result"].job_id).outputs == []
    bad = server.services.store.get(items["bad-size"].job_id)
    assert bad.status == "cancelled" and bad.errors and items["bad-size"].outcome == "invalid"
    assert "Some values do not fit the current parameters; edit the job to queue it." in texts(items["bad-size"].notes)
    assert server.services.store.get(items["interrupted"].job_id).error == "Interrupted by the user"


def test_outputs_that_cannot_be_registered_are_left_out(server, tmp_path, monkeypatch):
    outputs = tmp_path / "out"
    outputs.mkdir()
    Image.new("RGB", (2, 2)).save(outputs / "a.png")
    monkeypatch.setattr(importers, "output_roots", lambda: [os.path.realpath(outputs)])
    importer = importers.Importer(server.services.queue)

    def fail(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(importer.files, "register_output", fail)
    result = json.dumps({"images": [str(outputs / "a.png")], "geninfo": {"infotexts": ["a\nSteps: 1", 5]}})
    assert importer._agent_outputs(result, "txt2img") == ([], "a\nSteps: 1")
    assert importer._agent_outputs(json.dumps(["a list"]), "txt2img") == ([], None)


def test_prompt_queue_files_that_are_missing_or_unreadable(server, tmp_path, monkeypatch):
    importer = importers.Importer(server.services.queue)
    monkeypatch.setattr(importers, "prompt_queue_paths", lambda: [str(tmp_path / "none.json")])
    with pytest.raises(ImportFailed, match="no Prompt Queue file") as missing:
        importer.prompt_queue()
    assert missing.value.status == 404
    bad = tmp_path / "queue.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ImportFailed, match="cannot read the queue file") as unreadable:
        importer.prompt_queue(str(bad))
    assert unreadable.value.status == 422
    # an item without an id gets a stable one; an anchor with an odd separator falls back to ", "
    bad.write_text(json.dumps({"items": [{"prompt": "cat", "status": "pending", "anchor": "best", "anchor_enabled": True, "anchor_separator": 3}]}), encoding="utf-8")
    first = importer.prompt_queue(str(bad))
    assert first.queued == 1 and server.services.store.get(first.items[0].job_id).core["prompt"] == "best, cat"
    assert importer.prompt_queue(str(bad)).skipped == 1


def test_an_unknown_source_leaves_the_queue_running(server):
    queue = server.services.queue
    queue.set_paused(False)
    with pytest.raises(ImportFailed, match="unknown source") as unknown:
        importers.Importer(queue).run("comfy")
    assert unknown.value.status == 422 and not queue.paused


def test_a_failed_import_resumes_the_queue_it_paused(server, tmp_path, monkeypatch):
    queue = server.services.queue
    queue.set_paused(False)
    monkeypatch.setattr(importers, "agent_scheduler_paths", lambda: [str(tmp_path / "none.sqlite3")])
    with pytest.raises(ImportFailed):
        importers.Importer(queue).run(importers.AGENT_SCHEDULER)
    assert not queue.paused
    # an import that queued nothing does not leave the queue paused either
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"items": [{"id": "x", "prompt": "old", "status": "done"}]}), encoding="utf-8")
    monkeypatch.setattr(importers, "prompt_queue_paths", lambda: [str(path)])
    report = importers.Importer(queue).run(importers.PROMPT_QUEUE)
    assert report.skipped == 1 and report.paused is False and not queue.paused


# endregion


def test_images_are_indexed_only_from_the_configured_output_folders_and_jobs_survive_a_layout_hash_failure(server, monkeypatch):
    roots = importers.output_roots()
    assert roots and all(root == os.path.realpath(root) for root in roots)
    importer = importers.Importer(server.services.queue)

    def fail(tab):
        raise RuntimeError("schema not built")

    monkeypatch.setattr(server.services.queue, "hash_layout", fail)
    assert importer._layout_hash("txt2img") == ""
