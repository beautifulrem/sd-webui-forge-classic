"""The persistent job queue and its routes (stub executor, no model)."""

import threading
import time

import pytest

from remi.jobs import JobStore
from remi.jobs.store import CANCELLED, COMPLETED, FAILED, INTERRUPTED, QUEUED, RUNNING
from remi.server.testing import FakeState, StubExecutor, server_app
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
    return response.json()


def wait_until(predicate, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def events_of(services):
    seen = []

    class Recorder:
        def send_event(self, event):
            seen.append(event)

        def send_frame(self, frame):
            seen.append(frame)

    services.bus.subscribe(Recorder())
    return seen


# region Lifecycle


def test_a_job_runs_and_its_outputs_are_served(server):
    events = events_of(server.services)
    created = submit(server, "a red fox")
    assert server.services.queue.wait_idle()

    job = server.client.get(f"/api/v2/jobs/{created['job_id']}").json()
    assert job["status"] == "completed"
    assert job["core"] == {"prompt": "a red fox"} and job["layout_hash"] and job["schema_version"] == 1
    assert job["summary"] == "a red fox" and job["infotext"].startswith("a red fox")
    [output] = job["outputs"]
    assert output["kind"] == "image" and (output["width"], output["height"]) == (16, 8)

    image = server.client.get(f"/api/v2/files/{output['id']}")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    gallery = server.client.get("/api/v2/outputs").json()
    assert [item["id"] for item in gallery["items"]] == [output["id"]] and gallery["items"][0]["job_id"] == created["job_id"]

    types = [event["type"] for event in events if isinstance(event, dict)]
    assert types.index("job.queued") < types.index("job.started") < types.index("job.completed")
    completed = next(event for event in events if isinstance(event, dict) and event["type"] == "job.completed")
    assert completed["job_id"] == created["job_id"] and completed["seq"] == created["seq"] and completed["outputs"][0]["id"] == output["id"]


def test_the_executor_gets_the_named_values_and_an_upload_resolver(server):
    submit(server, "dog", priority=3)
    assert server.services.queue.wait_idle()
    job, ctx = server.executor.calls[-1]
    assert job.tab == "txt2img" and job.core == {"prompt": "dog"} and job.priority == 3
    assert ctx.id_task == f"task(remi-{job.id})" and ctx.resolve_upload == server.services.files.resolve_upload


def test_invalid_values_are_rejected_before_queueing(server):
    response = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"width": "wide", "cfg_scale": "nan"}, "scripts": {"nope": {}}})
    assert response.status_code == 422
    errors = {(error["owner"], error["param"]) for error in response.json()["errors"]}
    assert ("core", "width") in errors and ("nope", None) in errors
    assert server.services.store.count(QUEUED) == 0 and server.client.get("/api/v2/jobs").json()["total"] == 0


def test_img2img_needs_its_source_image_and_known_uploads(server):
    response = server.client.post("/api/v2/jobs", json={"tab": "img2img", "core": {"init_img": "unknownupload1234"}})
    assert response.status_code == 422
    assert any(error["param"] == "init_img" for error in response.json()["errors"])


def test_a_failing_job_reports_its_error(webui, tmp_path):
    state = FakeState()
    executor = StubExecutor(state, error=RuntimeError("boom"))
    with server_app(tmp_path, state=state, executor=executor) as server:
        events = events_of(server.services)
        created = submit(server)
        assert server.services.queue.wait_idle()
        job = server.client.get(f"/api/v2/jobs/{created['job_id']}").json()
        assert job["status"] == FAILED and job["error"] == "RuntimeError: boom" and job["outputs"] == []
        failed = [event for event in events if isinstance(event, dict) and event["type"] == "job.failed"]
        assert failed and failed[0]["error"] == "RuntimeError: boom"


def test_repeat_queues_copies(server):
    server.services.queue.set_paused(True)
    created = submit(server, repeat=3)
    assert len(created["job_ids"]) == 3 and created["job_id"] == created["job_ids"][0]
    assert server.client.get("/api/v2/queue").json() == {"size": 3, "paused": True, "running": None}
    server.client.post("/api/v2/queue/resume")
    assert server.services.queue.wait_idle()
    assert server.services.store.count(COMPLETED) == 3


# endregion

# region Ordering, pause, priority


def test_priority_then_order_decide_what_runs_next(server):
    queue = server.services.queue
    queue.set_paused(True)
    first = submit(server, "first")["job_id"]
    second = submit(server, "second")["job_id"]
    urgent = submit(server, "urgent", priority=5)["job_id"]
    third = submit(server, "third")["job_id"]
    assert [job.id for job in server.services.store.queued()] == [urgent, first, second, third]

    assert server.client.post("/api/v2/queue/reorder", json={"order": [third, first]}).status_code == 200
    assert [item["id"] for item in server.client.get("/api/v2/queue/items").json()] == [urgent, third, second, first]

    server.client.patch(f"/api/v2/jobs/{second}", json={"priority": 9})
    assert server.services.store.queued()[0].id == second

    queue.set_paused(False)
    assert queue.wait_idle()
    ran = [job.core["prompt"] for job, _ctx in server.executor.calls]
    assert ran == ["second", "urgent", "third", "first"]


def test_pause_is_persisted(webui, tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    store.set_meta("paused", "1")
    store.close()
    with server_app(tmp_path) as server:
        assert server.client.get("/api/v2/queue").json()["paused"] is True
        submit(server)
        time.sleep(0.1)
        assert server.services.store.count(QUEUED) == 1


def test_history_is_paged_newest_first(server):
    server.services.queue.set_paused(True)
    ids = [submit(server, f"p{index}")["job_id"] for index in range(5)]
    page = server.client.get("/api/v2/jobs", params={"limit": 2}).json()
    assert [item["id"] for item in page["items"]] == ids[:-3:-1] and page["total"] == 5 and page["next_cursor"]
    rest = server.client.get("/api/v2/jobs", params={"limit": 10, "cursor": page["next_cursor"]}).json()
    assert [item["id"] for item in rest["items"]] == ids[2::-1] and rest["next_cursor"] is None
    assert server.client.get("/api/v2/jobs", params={"status": "completed"}).json()["total"] == 0


# endregion

# region Interrupt, skip, cancel, delete, retry


def blocked_server(tmp_path):
    state = FakeState()
    executor = StubExecutor(state, gate=threading.Event())
    return server_app(tmp_path, state=state, executor=executor)


def test_interrupting_the_running_job_keeps_its_results(webui, tmp_path):
    with blocked_server(tmp_path) as server:
        job_id = submit(server)["job_id"]
        assert server.executor.started.wait(5)
        assert server.client.get("/api/v2/queue").json()["running"] == job_id
        assert server.client.post(f"/api/v2/jobs/{job_id}/interrupt").status_code == 200
        assert server.state.interrupted
        assert server.services.queue.wait_idle()
        job = server.client.get(f"/api/v2/jobs/{job_id}").json()
        assert job["status"] == INTERRUPTED and len(job["outputs"]) == 1


def test_skip_flips_the_skip_flag_of_the_running_job_only(webui, tmp_path):
    with blocked_server(tmp_path) as server:
        job_id = submit(server)["job_id"]
        assert server.executor.started.wait(5)
        assert server.client.post(f"/api/v2/jobs/{job_id}/skip").status_code == 204
        assert server.state.skipped
        server.executor.gate.set()
        assert server.services.queue.wait_idle()
        assert server.client.post(f"/api/v2/jobs/{job_id}/skip").status_code == 409


def test_deleting_the_running_job_cancels_it(webui, tmp_path):
    with blocked_server(tmp_path) as server:
        job_id = submit(server)["job_id"]
        assert server.executor.started.wait(5)
        assert server.client.delete(f"/api/v2/jobs/{job_id}").status_code == 204
        assert server.services.queue.wait_idle()
        assert server.services.store.get(job_id).status == CANCELLED


def test_a_job_waiting_for_the_generation_lock_is_not_current_and_its_interrupt_waits(webui, tmp_path):
    # Someone else (/sdapi, a GPU action) generates: the claimed job must not
    # flip shared.state (that would interrupt them) nor claim their progress.
    lock = threading.Lock()
    state = FakeState()
    with server_app(tmp_path, state=state, executor=StubExecutor(state), lock=lock) as server:
        events = events_of(server.services)
        with lock:
            job_id = submit(server)["job_id"]
            assert wait_until(lambda: server.services.queue.busy)
            assert server.services.queue.current_ref() is None
            assert server.client.post(f"/api/v2/jobs/{job_id}/interrupt").status_code == 200
            assert not state.interrupted
        assert server.services.queue.wait_idle()
        assert server.services.store.get(job_id).status == INTERRUPTED
        assert server.executor.calls == []
        types = [event["type"] for event in events if isinstance(event, dict)]
        assert "job.started" not in types and "job.completed" in types


def test_a_skip_that_arrives_before_the_job_begins_is_applied_after_begin(webui, tmp_path):
    lock = threading.Lock()
    state = FakeState()
    executor = StubExecutor(state, gate=threading.Event())
    with server_app(tmp_path, state=state, executor=executor, lock=lock) as server:
        with lock:
            job_id = submit(server)["job_id"]
            assert wait_until(lambda: server.services.queue.busy)
            assert server.client.post(f"/api/v2/jobs/{job_id}/skip").status_code == 204
            assert not state.skipped
        assert executor.started.wait(5)
        assert state.skipped  # begin() reset it, on_begin re-applied it
        executor.gate.set()
        assert server.services.queue.wait_idle()
        assert server.services.store.get(job_id).status == COMPLETED


def test_interrupt_all_while_a_job_waits_interrupts_the_other_caller(webui, tmp_path):
    lock = threading.Lock()
    state = FakeState()
    with server_app(tmp_path, state=state, executor=StubExecutor(state), lock=lock) as server:
        with lock:
            job_id = submit(server)["job_id"]
            assert wait_until(lambda: server.services.queue.busy)
            assert server.client.post("/api/v2/interrupt").status_code == 200
            assert state.interrupted  # what holds the lock is generating
        assert server.services.queue.wait_idle()
        assert server.services.store.get(job_id).status == INTERRUPTED


def test_queued_jobs_can_be_cancelled_deleted_and_retried(server):
    server.services.queue.set_paused(True)
    cancel = submit(server, "cancel me")["job_id"]
    remove = submit(server, "remove me")["job_id"]
    assert server.client.post(f"/api/v2/jobs/{cancel}/interrupt").json()["status"] == CANCELLED
    assert server.client.delete(f"/api/v2/jobs/{remove}").status_code == 204
    assert server.client.get(f"/api/v2/jobs/{remove}").status_code == 404
    assert server.client.post(f"/api/v2/jobs/{remove}/retry").status_code == 404

    retried = server.client.post(f"/api/v2/jobs/{cancel}/retry")
    assert retried.status_code == 201
    new_id = retried.json()["job_id"]
    assert server.services.store.get(new_id).retry_of == cancel
    assert server.client.post(f"/api/v2/jobs/{new_id}/retry").status_code == 409  # not finished
    server.services.queue.set_paused(False)
    assert server.services.queue.wait_idle()
    assert server.services.store.get(new_id).status == COMPLETED


def test_interrupt_all_can_clear_the_queue(webui, tmp_path):
    with blocked_server(tmp_path) as server:
        running = submit(server)["job_id"]
        assert server.executor.started.wait(5)
        waiting = submit(server)["job_id"]
        assert server.client.post("/api/v2/interrupt", json={"clear_queue": True}).status_code == 200
        assert server.services.queue.wait_idle()
        assert server.services.store.get(running).status == INTERRUPTED
        assert server.services.store.get(waiting).status == CANCELLED


def test_generate_forever_requeues_until_stopped(server):
    job_id = submit(server, "loop", forever=True)["job_id"]
    assert wait_until(lambda: len(server.executor.calls) >= 3)
    assert server.client.post("/api/v2/queue/stop-forever").json()["paused"] is False
    assert server.services.queue.wait_idle()
    jobs = server.client.get("/api/v2/jobs", params={"limit": 200}).json()["items"]
    assert all(not item["forever"] for item in jobs) and any(item["retry_of"] == job_id for item in jobs)
    count = len(server.executor.calls)
    time.sleep(0.1)
    assert len(server.executor.calls) == count


def test_pausing_keeps_generate_forever_going_after_resume(webui, tmp_path):
    with blocked_server(tmp_path) as server:
        first = submit(server, "loop", forever=True)["job_id"]
        assert server.executor.started.wait(5)
        server.services.queue.set_paused(True)
        server.executor.gate.set()
        assert server.services.queue.wait_idle()
        assert server.services.store.get(first).status == COMPLETED
        [again] = server.services.store.queued()
        assert again.forever and again.retry_of == first
        server.services.queue.stop_forever()
        server.services.queue.set_paused(False)
        assert server.services.queue.wait_idle()
        assert server.services.store.get(again.id).status == COMPLETED


# endregion

# region Persistence


def test_queued_jobs_survive_a_restart_and_running_ones_become_interrupted(webui, tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    left_running = store.add("txt2img", {"prompt": "was running"}, {})
    store.claim_next()
    waiting = store.add("txt2img", {"prompt": "still queued"}, {}, priority=1)
    store.close()

    with server_app(tmp_path) as server:
        assert server.services.queue.wait_idle()
        assert server.services.store.get(left_running.id).status == INTERRUPTED
        assert server.services.store.get(left_running.id).error
        assert server.services.store.get(waiting.id).status == COMPLETED
        assert [job.core["prompt"] for job, _ctx in server.executor.calls] == ["still queued"]


def test_store_round_trips_a_job(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = store.add("img2img", {"prompt": "x", "init_img": "abc"}, {"seed": {"seed": 5}}, priority=2, layout_hash="h", summary="x", forever=True, user="ann")
    assert store.get(job.id) == job and job.status == QUEUED and job.forever and job.user == "ann"
    claimed = store.claim_next()
    assert claimed.id == job.id and claimed.status == RUNNING and claimed.started
    done = store.finish(job.id, COMPLETED, outputs=[{"id": "f1"}], infotext="info")
    assert done.outputs == [{"id": "f1"}] and done.infotext == "info" and done.finished
    with pytest.raises(ValueError):
        store.finish(job.id, RUNNING)
    assert "abc" not in store.referenced_strings()
    store.close()


# endregion


def test_numpy_extra_images_become_pil_outputs():
    """ControlNet adds its detected maps to ``extra_images`` as numpy arrays."""

    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    from remi.jobs.runner import _from_processed

    image = Image.new("RGB", (8, 8))
    detected = np.zeros((4, 6, 3), dtype=np.uint8)
    result = _from_processed(SimpleNamespace(images=[image], extra_images=[detected], infotexts=["info"], info="info"))
    assert result.images[0] is image and isinstance(result.images[1], Image.Image) and result.images[1].size == (6, 4)
    assert result.infotexts == ["info", "info"]
