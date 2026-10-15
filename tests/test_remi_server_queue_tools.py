"""Queue page tools: duplicate, edit, star, search, clear history, and the
importers of the Agent Scheduler and Prompt Queue data (no model)."""

import base64
import hashlib
import hmac
import io
import json
import os
import pickle
import sqlite3
import zlib

import pytest
from PIL import Image

from remi.jobs import importers
from remi.jobs.importers import SafeUnpickler, note_text, read_script_params
from remi.jobs.store import CANCELLED, COMPLETED, QUEUED, Job, JobStore
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


def submit(server, prompt="a cat", **extra):
    response = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": prompt}, "scripts": {}, **extra})
    assert response.status_code == 201, response.text
    return response.json()["job_id"]


def queued_ids(server):
    return [job["id"] for job in server.client.get("/api/v2/queue/items").json()]


# region Queue tools


def test_duplicating_a_queued_job_puts_the_copy_right_after_it(server):
    server.services.queue.set_paused(True)
    first, second = submit(server, "first"), submit(server, "second")
    response = server.client.post(f"/api/v2/jobs/{first}/duplicate")
    assert response.status_code == 201
    copy = response.json()["job_id"]
    assert queued_ids(server) == [first, copy, second]
    detail = server.client.get(f"/api/v2/jobs/{copy}").json()
    assert detail["core"] == {"prompt": "first"} and detail["retry_of"] == first and detail["status"] == QUEUED


def test_a_finished_job_is_duplicated_at_the_end(server):
    done = submit(server, "done")
    assert server.services.queue.wait_idle()
    server.services.queue.set_paused(True)
    later = submit(server, "later")
    copy = server.client.post(f"/api/v2/jobs/{done}/duplicate").json()["job_id"]
    assert queued_ids(server) == [later, copy]


def test_a_queued_jobs_parameters_can_be_edited_and_are_validated(server):
    server.services.queue.set_paused(True)
    job = submit(server, "old prompt")
    response = server.client.put(f"/api/v2/jobs/{job}/params", json={"core": {"prompt": "new prompt", "width": 640}, "scripts": {"seed": {"seed": 7}}})
    assert response.status_code == 200, response.text
    detail = response.json()
    assert detail["core"] == {"prompt": "new prompt", "width": 640} and detail["scripts"] == {"seed": {"seed": 7}} and detail["summary"] == "new prompt"

    rejected = server.client.put(f"/api/v2/jobs/{job}/params", json={"core": {"width": "wide"}})
    assert rejected.status_code == 422 and rejected.json()["errors"][0]["param"] == "width"
    assert server.client.get(f"/api/v2/jobs/{job}").json()["core"]["prompt"] == "new prompt"


def test_only_queued_jobs_can_be_edited(server):
    job = submit(server)
    assert server.services.queue.wait_idle()
    response = server.client.put(f"/api/v2/jobs/{job}/params", json={"core": {"prompt": "x"}})
    assert response.status_code == 409
    assert server.client.put("/api/v2/jobs/nope/params", json={"core": {}}).status_code == 404


def test_starred_jobs_survive_clearing_the_history_and_can_be_filtered(server):
    keep, drop = submit(server, "keep me"), submit(server, "drop me")
    assert server.services.queue.wait_idle()
    assert server.client.post("/api/v2/jobs/star", json={"ids": [keep], "starred": True}).json() == {"count": 1}
    starred = server.client.get("/api/v2/jobs", params={"starred": True}).json()
    assert [job["id"] for job in starred["items"]] == [keep] and starred["items"][0]["starred"] is True

    assert server.client.post("/api/v2/jobs/clear-history", json={}).json() == {"count": 1}
    remaining = [job["id"] for job in server.client.get("/api/v2/jobs").json()["items"]]
    assert remaining == [keep] and drop not in remaining
    assert server.client.post("/api/v2/jobs/clear-history", json={"keep_starred": False}).json() == {"count": 1}


def test_history_search_matches_every_word_in_prompt_or_infotext(server):
    submit(server, "a red fox in snow")
    submit(server, "a blue whale")
    assert server.services.queue.wait_idle()
    found = server.client.get("/api/v2/jobs", params={"q": "fox snow"}).json()
    assert [job["summary"] for job in found["items"]] == ["a red fox in snow"] and found["total"] == 1
    assert server.client.get("/api/v2/jobs", params={"q": "100%_"}).json()["total"] == 0
    assert server.client.get("/api/v2/jobs", params={"q": "Steps: 1", "tab": "txt2img"}).json()["total"] == 2
    assert server.client.get("/api/v2/jobs", params={"tab": "img2img"}).json()["total"] == 0


def test_a_job_that_is_not_replayable_is_never_retried_or_duplicated(server):
    job = server.services.store.add_record(
        Job(id="", seq=0, tab="txt2img", core={"prompt": "x"}, scripts={}, status=CANCELLED, replayable=False, meta={"opaque": {"count": 2, "preview": ["1", "2"]}})
    )
    for action in ("retry", "duplicate"):
        response = server.client.post(f"/api/v2/jobs/{job.id}/{action}")
        assert response.status_code == 409 and "do not map" in response.json()["detail"]
    detail = server.client.get(f"/api/v2/jobs/{job.id}").json()
    assert detail["replayable"] is False and detail["opaque"] == {"count": 2, "preview": ["1", "2"]}


def test_the_worker_never_claims_a_job_that_is_not_replayable(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    try:
        store.add_record(Job(id="", seq=0, tab="txt2img", core={}, scripts={}, status=QUEUED, replayable=False))
        runnable = store.add("txt2img", {"prompt": "ok"}, {})
        assert store.claim_next().id == runnable.id
        assert store.claim_next() is None
    finally:
        store.close()


def test_the_worker_cancels_a_queued_job_it_cannot_run_instead_of_spinning(server, monkeypatch):
    store, queue = server.services.store, server.services.queue
    queue.set_paused(True)
    legacy = store.add("txt2img", {"prompt": "legacy"}, {})
    store._db.execute("UPDATE jobs SET replayable = 0 WHERE id = ?", (legacy.id,))  # written by an older version
    claims = []
    original = store.claim_next
    monkeypatch.setattr(store, "claim_next", lambda: claims.append(1) or original())
    queue.set_paused(False)
    assert queue.wait_idle()
    import time

    time.sleep(0.3)
    assert len(claims) <= 2  # woke once for it, then waits
    assert store.get(legacy.id).status == CANCELLED and store.count(QUEUED) == 0 and queue.status()["size"] == 0
    assert server.executor.calls == []


def test_a_job_that_is_not_replayable_is_never_stored_queued(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    try:
        job = store.add_record(Job(id="", seq=0, tab="txt2img", core={}, scripts={}, status=QUEUED, replayable=False))
        assert job.status == CANCELLED and job.finished is not None and store.count(QUEUED) == 0
    finally:
        store.close()


def test_a_queued_job_moves_next_to_a_neighbour_in_a_long_queue(server):
    server.services.queue.set_paused(True)
    store = server.services.store
    jobs = [store.add("txt2img", {"prompt": f"job {index}"}, {}) for index in range(600)]
    first, last = jobs[0].id, jobs[-1].id
    response = server.client.post("/api/v2/queue/move", json={"id": last, "before": first})
    assert response.status_code == 200, response.text
    order = queued_ids(server)
    assert order[:2] == [last, first] and len(order) == 600
    assert server.client.post("/api/v2/queue/move", json={"id": last, "after": jobs[299].id}).status_code == 200
    order = queued_ids(server)
    assert order.index(last) == order.index(jobs[299].id) + 1

    # into another priority group, next to a neighbour there
    urgent = store.add("txt2img", {"prompt": "urgent"}, {}, priority=10)
    assert server.client.post("/api/v2/queue/move", json={"id": jobs[10].id, "after": urgent.id, "priority": 10}).status_code == 200
    assert queued_ids(server)[:2] == [urgent.id, jobs[10].id] and store.get(jobs[10].id).priority == 10

    # a neighbour that started meanwhile is ignored; the other one places it
    store.cancel_queued(jobs[1].id)
    assert server.client.post("/api/v2/queue/move", json={"id": jobs[50].id, "after": jobs[1].id, "before": jobs[2].id}).status_code == 200
    order = queued_ids(server)
    assert order.index(jobs[50].id) + 1 == order.index(jobs[2].id)

    assert server.client.post("/api/v2/queue/move", json={"id": jobs[1].id, "before": first}).status_code == 409
    assert server.client.post("/api/v2/queue/move", json={"id": "nope", "before": first}).status_code == 404


def test_moving_many_times_between_the_same_neighbours_keeps_the_order(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    try:
        a, b, c = (store.add("txt2img", {}, {}) for _ in range(3))
        for _ in range(80):  # bisecting the same gap runs out of floats: the queue is renumbered
            assert store.move(c.id, after=a.id)
            assert [job.id for job in store.queued()] == [a.id, c.id, b.id]
            assert store.move(b.id, after=a.id)
            assert [job.id for job in store.queued()] == [a.id, b.id, c.id]
    finally:
        store.close()


def test_the_store_adds_the_new_columns_to_an_old_database(tmp_path):
    path = tmp_path / "jobs.db"
    db = sqlite3.connect(path)
    db.executescript(
        """CREATE TABLE jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, tab TEXT NOT NULL, core TEXT NOT NULL,
        scripts TEXT NOT NULL, layout_hash TEXT NOT NULL DEFAULT '', schema_version INTEGER NOT NULL DEFAULT 1, priority INTEGER NOT NULL DEFAULT 0,
        position REAL NOT NULL DEFAULT 0, status TEXT NOT NULL, created REAL NOT NULL, started REAL, finished REAL, outputs TEXT NOT NULL DEFAULT '[]',
        infotext TEXT, error TEXT, errors TEXT, summary TEXT NOT NULL DEFAULT '', forever INTEGER NOT NULL DEFAULT 0, user TEXT, retry_of TEXT);
        INSERT INTO jobs (id, tab, core, scripts, status, created) VALUES ('old', 'txt2img', '{}', '{}', 'completed', 1.0);"""
    )
    db.close()
    store = JobStore(path)
    job = store.get("old")
    assert job.starred is False and job.replayable is True and job.meta == {}
    store.close()


# endregion

# region Safe loading


class Boom:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_the_unpickler_refuses_code_and_accepts_data():
    data = pickle.dumps([1, "a", {"x": (1.5, None)}, {1, 2}])
    assert SafeUnpickler(io.BytesIO(data)).load() == [1, "a", {"x": (1.5, None)}, {1, 2}]
    with pytest.raises(pickle.UnpicklingError):
        SafeUnpickler(io.BytesIO(pickle.dumps([Boom()]))).load()
    image = Image.new("RGB", (3, 2), (1, 2, 3))
    restored = SafeUnpickler(io.BytesIO(pickle.dumps(image))).load()
    assert restored.size == (3, 2) and restored.getpixel((0, 0)) == (1, 2, 3)


def test_script_params_are_read_signed_or_not():
    key = b"k" * 32
    payload = zlib.compress(pickle.dumps([0, True, 5]))
    signed = b"ASSIG1" + hmac.new(key, payload, hashlib.sha256).digest() + payload
    assert read_script_params(signed, key) == ([0, True, 5], [])
    values, notes = read_script_params(signed, b"x" * 32)
    assert values == [0, True, 5] and "not signed by this installation" in note_text(notes[0])
    values, notes = read_script_params(zlib.compress(pickle.dumps([Boom()])), key)
    assert values is None and any("could not be read" in note_text(note) for note in notes)


# endregion

# region Agent Scheduler import

AS_SCHEMA = """
CREATE TABLE task (
    id VARCHAR(64) NOT NULL PRIMARY KEY, api_task_id VARCHAR(64), api_task_callback VARCHAR(255), name VARCHAR(255),
    type VARCHAR(20) NOT NULL, params TEXT NOT NULL, script_params BLOB NOT NULL, priority INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL, result TEXT, bookmarked BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT (datetime('now')) NOT NULL, updated_at DATETIME DEFAULT (datetime('now')) NOT NULL
);
CREATE TABLE app_state (key VARCHAR(64) PRIMARY KEY, value VARCHAR(255));
"""
KEY = b"s" * 32


def sign(values):
    payload = zlib.compress(pickle.dumps(values))
    return b"ASSIG1" + hmac.new(KEY, payload, hashlib.sha256).digest() + payload


def txt2img_args(prompt, **extra):
    args = {
        "id_task": "task(x)", "request": {"username": None}, "prompt": prompt, "negative_prompt": "bad", "prompt_styles": [], "n_iter": 2,
        "batch_size": 1, "cfg_scale": 6.5, "distilled_cfg_scale": 3.5, "height": 768, "width": 512, "enable_hr": False, "denoising_strength": 0.7,
        "hr_scale": 2, "hr_upscaler": "Latent", "hr_second_pass_steps": 0, "hr_resize_x": 0, "hr_resize_y": 0, "hr_checkpoint_name": "Use same checkpoint",
        "hr_additional_modules": ["Use same choices"], "hr_sampler_name": "Use same sampler", "hr_scheduler": "Use same scheduler", "hr_prompt": "",
        "hr_negative_prompt": "", "hr_cfg": 1.0, "hr_distilled_cfg": 3.5, "override_settings_texts": [], "restore_faces": False,
    }
    args.update(extra)
    return args


def default_script_args(runner, seed=1234):
    args = runner.script_args_from_values("None", {})
    seed_script = runner.by_id["seed"]
    layout = runner.layout_for(seed_script)
    args[seed_script.args_from + layout.names.index("seed")] = seed
    return args


def serialized_image(color):
    image = Image.new("RGB", (8, 4), color)
    return {"size": list(image.size), "mode": image.mode, "data": base64.b64encode(zlib.compress(image.tobytes())).decode(), "cls": "Image"}


@pytest.fixture
def agent_db(webui, tmp_path, monkeypatch):
    folder = tmp_path / "agent"
    folder.mkdir()
    (folder / "agent_scheduler_signing.key").write_bytes(KEY)
    path = folder / "task_scheduler.sqlite3"
    image_path = tmp_path / "outputs" / "done.png"
    image_path.parent.mkdir()
    Image.new("RGB", (20, 10), (200, 0, 0)).save(image_path)
    runner = webui.txt2img
    rows = [
        ("pending-1", "Night fox", "txt2img", {"args": txt2img_args("a fox at night"), "checkpoint": "anima.safetensors", "is_ui": True}, sign(default_script_args(runner)), 1, "pending", None, False),
        ("trailing", None, "txt2img", {"args": txt2img_args("trailing args"), "is_ui": True}, sign([*default_script_args(runner), "extra", {"is_cnet": True}]), 2, "pending", None, False),
        ("evil", None, "txt2img", {"args": txt2img_args("evil"), "is_ui": True}, sign([Boom()]), 3, "pending", None, False),
        ("done", None, "txt2img", {"args": txt2img_args("finished"), "is_ui": True}, sign(default_script_args(runner)), 0, "done", json.dumps({"images": [str(image_path), str(tmp_path / "gone.png")], "geninfo": {"infotexts": ["finished\nSteps: 20, Seed: 1"]}}), True),
        ("api", None, "txt2img", {"args": {"prompt": "api prompt", "steps": 33, "seed": 99, "n_iter": 1, "width": 640, "height": 640, "unknown_field": "x", "alwayson_scripts": {"Agent Scheduler": {"args": []}, "missing script": {"args": [1]}}}, "is_ui": False}, sign([]), 4, "failed", "CUDA out of memory", False),
        ("img", None, "img2img", {"args": {"mode": 0, "prompt": "repaint", "init_img": serialized_image((0, 0, 255)), "denoising_strength": 0.4, "n_iter": 1}, "is_ui": True, "is_img2img": True}, sign(default_script_args(webui.img2img)), 5, "pending", None, False),
    ]
    db = sqlite3.connect(path)
    db.executescript(AS_SCHEMA)
    db.executemany(
        "INSERT INTO task (id, name, type, params, script_params, priority, status, result, bookmarked, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-01-02 03:04:05', '2026-01-02 03:05:00')",
        [(task_id, name, kind, json.dumps(params), blob, priority, status, result, bookmarked) for task_id, name, kind, params, blob, priority, status, result, bookmarked in rows],
    )
    db.commit()
    db.close()
    monkeypatch.setattr(importers, "agent_scheduler_paths", lambda: [str(path)])
    monkeypatch.setattr(importers, "output_roots", lambda: [os.path.realpath(tmp_path / "outputs")])
    return path


def test_agent_scheduler_sources_are_listed(server, agent_db):
    sources = {source["id"]: source for source in server.client.get("/api/v2/queue/import").json()}
    assert sources["agent-scheduler"]["available"] is True and sources["agent-scheduler"]["count"] == 6
    assert sources["prompt-queue"]["available"] in (True, False)


def test_agent_scheduler_tasks_become_jobs(server, agent_db):
    server.services.queue.set_paused(False)
    response = server.client.post("/api/v2/queue/import/agent-scheduler")
    assert response.status_code == 200, response.text
    report = response.json()
    outcomes = {item["source_id"]: item for item in report["items"]}
    assert report["total"] == 6 and report["paused"] is True and server.services.queue.paused
    assert outcomes["pending-1"]["outcome"] == "queued" and outcomes["img"]["outcome"] == "queued"
    assert outcomes["trailing"]["outcome"] == "refused" and outcomes["evil"]["outcome"] == "refused"
    assert outcomes["done"]["outcome"] == "history" and outcomes["api"]["outcome"] == "history"
    assert (report["queued"], report["refused"], report["history"], report["errors"]) == (2, 2, 2, 0)

    def detail(source_id):
        return server.client.get(f"/api/v2/jobs/{outcomes[source_id]['job_id']}").json()

    fox = detail("pending-1")
    assert fox["status"] == QUEUED and fox["name"] == "Night fox" and fox["source"] == "agent-scheduler" and fox["replayable"]
    assert fox["core"]["prompt"] == "a fox at night" and fox["core"]["batch_count"] == 2 and fox["core"]["cfg_scale"] == 6.5
    assert fox["core"]["override_settings"] == ["Model hash: anima.safetensors"]
    assert fox["scripts"]["seed"]["seed"] == 1234 and fox["core"]["script"] == "None"
    assert any("restore_faces" in note_text(note) for note in fox["notes"])

    trailing = detail("trailing")
    assert trailing["status"] == CANCELLED and trailing["replayable"] is False
    assert trailing["opaque"]["count"] == 2 and trailing["opaque"]["preview"][0] == "'extra'"
    assert detail("evil")["replayable"] is False and any("could not be read" in note_text(note) for note in detail("evil")["notes"])

    done = detail("done")
    assert done["status"] == COMPLETED and done["starred"] is True and len(done["outputs"]) == 1
    assert done["infotext"].startswith("finished") and done["created_at"].startswith("2026-01-02T03:04:05")
    assert server.client.get(f"/api/v2/files/{done['outputs'][0]['id']}").status_code == 200

    api = detail("api")
    assert api["status"] == "failed" and api["error"] == "CUDA out of memory"
    assert api["core"]["width"] == 640 and api["scripts"]["sampler"]["steps"] == 33 and api["scripts"]["seed"]["seed"] == 99
    assert any("unknown_field" in note_text(note) for note in api["notes"]) and any("missing script" in note_text(note) for note in api["notes"])

    img = detail("img")
    upload = server.services.files.get(img["core"]["init_img"])
    assert upload is not None and (upload.width, upload.height) == (8, 4) and img["core"]["mode"] == "img2img"

    again = server.client.post("/api/v2/queue/import/agent-scheduler").json()
    assert again["skipped"] == 6 and again["queued"] == 0 and server.services.store.count(QUEUED) == 2


def test_imported_history_sorts_by_when_it_was_made(server, agent_db):
    recent = submit(server, "made today")
    assert server.services.queue.wait_idle()
    server.client.post("/api/v2/queue/import/agent-scheduler")
    history = server.client.get("/api/v2/jobs", params={"status": "completed,failed,cancelled,interrupted"}).json()["items"]
    assert history[0]["id"] == recent and all(item["created_at"].startswith("2026-01-02") for item in history[1:])
    paged, cursor = [], None
    while True:
        page = server.client.get("/api/v2/jobs", params={"limit": 2, **({"cursor": cursor} if cursor else {})}).json()
        paged += [item["id"] for item in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(paged) == len(set(paged)) == server.client.get("/api/v2/jobs").json()["total"]


def test_the_importer_only_indexes_images_in_the_output_folders_not_removed(server, webui, tmp_path, monkeypatch):
    outputs = tmp_path / "saved"  # not the store's own outputs/ (unsaved images, deleted when removed)
    outputs.mkdir()
    kept, removed = outputs / "kept.png", outputs / "removed.png"
    for path in (kept, removed):
        Image.new("RGB", (4, 4)).save(path)
    fake = outputs / "secrets.png"
    fake.write_text("not an image")
    outside = tmp_path / "elsewhere.png"
    Image.new("RGB", (4, 4)).save(outside)
    record = server.services.files.register_output(removed)
    server.client.post("/api/v2/gallery/delete", json={"ids": [record.id]})
    assert server.services.files.is_removed(removed)

    path = tmp_path / "task_scheduler.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(AS_SCHEMA)
    images = [str(kept), str(removed), str(fake), str(outside), "../" * 12 + "etc/hosts"]
    db.execute(
        "INSERT INTO task (id, type, params, script_params, priority, status, result) VALUES ('t', 'txt2img', ?, ?, 0, 'done', ?)",
        (json.dumps({"args": {"prompt": "x"}, "is_ui": False}), sign([]), json.dumps({"images": images})),
    )
    db.commit()
    db.close()
    monkeypatch.setattr(importers, "agent_scheduler_paths", lambda: [str(path)])
    monkeypatch.setattr(importers, "output_roots", lambda: [os.path.realpath(outputs)])
    report = server.client.post("/api/v2/queue/import/agent-scheduler").json()
    job = server.client.get(f"/api/v2/jobs/{report['items'][0]['job_id']}").json()
    assert [server.services.files.get(output["id"]).path for output in job["outputs"]] == [os.path.realpath(kept)]
    assert server.services.files.is_removed(removed)
    assert any("left out: 2" in note_text(note) for note in job["notes"])
    registered = {row[0] for row in server.services.files._db.execute("SELECT path FROM files WHERE kind = 'output'").fetchall()}
    assert os.path.realpath(fake) not in registered and os.path.realpath(outside) not in registered


def test_import_paths_are_hidden_with_hide_ui_dir_config(server, agent_db, monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    sources = server.client.get("/api/v2/queue/import").json()
    assert all(source["path"] is None for source in sources)
    assert server.client.post("/api/v2/queue/import/agent-scheduler").json()["path"] is None


def test_imported_queued_jobs_run_once_resumed(server, agent_db):
    server.client.post("/api/v2/queue/import/agent-scheduler")
    server.client.post("/api/v2/queue/resume")
    assert server.services.queue.wait_idle()
    prompts = sorted(job.core["prompt"] for job, _ctx in server.executor.calls)
    assert prompts == ["a fox at night", "repaint"]


def test_a_missing_database_is_a_404(server, monkeypatch, tmp_path):
    monkeypatch.setattr(importers, "agent_scheduler_paths", lambda: [str(tmp_path / "none.sqlite3")])
    response = server.client.post("/api/v2/queue/import/agent-scheduler")
    assert response.status_code == 404 and "no Agent Scheduler database" in response.json()["detail"]
    assert server.client.post("/api/v2/queue/import/other").status_code == 422


# endregion

# region Prompt Queue import


def test_prompt_queue_pending_prompts_are_queued_with_defaults(server, monkeypatch, tmp_path):
    path = tmp_path / "queue.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "a", "tab": "txt2img", "prompt": "castle", "negative": "blurry", "status": "pending", "created": 1700000000},
                    {"id": "b", "tab": "img2img", "prompt": "needs a source", "negative": "", "status": "running"},
                    {"id": "c", "tab": "txt2img", "prompt": "anchored", "status": "pending", "anchor": "masterpiece", "anchor_enabled": True, "anchor_separator": ", "},
                    {"id": "d", "tab": "txt2img", "prompt": "old", "status": "done"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(importers, "prompt_queue_paths", lambda: [str(path)])
    report = server.client.post("/api/v2/queue/import/prompt-queue").json()
    outcomes = {item["source_id"]: item["outcome"] for item in report["items"]}
    assert outcomes == {"a": "queued", "b": "invalid", "c": "queued", "d": "skipped"}
    assert report["paused"] is True
    jobs = {job["summary"]: job for job in server.client.get("/api/v2/queue/items").json()}
    assert set(jobs) == {"castle", "masterpiece, anchored"}
    castle = server.client.get(f"/api/v2/jobs/{jobs['castle']['id']}").json()
    assert castle["core"] == {"prompt": "castle", "negative_prompt": "blurry"} and castle["source"] == "prompt-queue"
    assert server.client.post("/api/v2/queue/import/prompt-queue").json()["skipped"] == 4


# endregion


def test_every_import_note_is_translated_in_the_spa():
    """Notes are translated by the SPA from its zh-CN catalog (``t(message, values)``)."""

    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "remi/jobs/importers.py").read_text(encoding="utf-8"))
    messages = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "note" and node.args and isinstance(node.args[0], ast.Constant)
    }
    assert len(messages) >= 15
    catalog = json.loads((root / "webui/src/locales/zh-CN.json").read_text(encoding="utf-8"))
    assert sorted(message for message in messages if message not in catalog) == []
