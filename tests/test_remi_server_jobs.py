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


def test_script_errors_during_a_job_become_its_warnings(webui, tmp_path):
    """A script hook that raises is reported by the script runner and skipped; the job
    completes and says so (history and the job.completed event) instead of failing silently."""

    from modules import errors

    state = FakeState()
    stub = StubExecutor(state)

    def executor(job, ctx):
        if stub.calls:  # only the first job hits the error
            return stub(job, ctx)
        try:
            raise ModuleNotFoundError("No module named 'ultralytics'")
        except ModuleNotFoundError:
            errors.report("Error running postprocess_image: /x/extensions/ADetailer/scripts/adetailer.py", exc_info=True)
            errors.report("Error running postprocess_image: /x/extensions/ADetailer/scripts/adetailer.py", exc_info=True)  # once per note
        return stub(job, ctx)

    with server_app(tmp_path, state=state, executor=executor) as server:
        events = events_of(server.services)
        created = submit(server)
        assert server.services.queue.wait_idle()
        job = server.client.get(f"/api/v2/jobs/{created['job_id']}").json()
        note = {"message": "{{script}} failed ({{hook}}): {{error}}", "values": {"script": "adetailer.py", "hook": "postprocess_image", "error": "ModuleNotFoundError: No module named 'ultralytics'"}}
        assert job["status"] == COMPLETED and job["outputs"] and job["warnings"] == [note]
        completed = [event for event in events if isinstance(event, dict) and event["type"] == "job.completed"]
        assert completed[0]["warnings"] == [note]
        assert errors.report.__name__ == "report" and errors.report.__module__ == "modules.errors"  # restored
        # a clean job has none
        clean = submit(server)
        assert server.services.queue.wait_idle()
        assert server.client.get(f"/api/v2/jobs/{clean['job_id']}").json()["warnings"] == []


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


def pending_tasks():
    from modules import progress

    return progress.get_pending_tasks().tasks


def test_the_progress_registry_only_lists_jobs_of_a_running_queue(webui, tmp_path):
    """A failed job and a stopped queue leave nothing "pending" in /internal/pending-tasks
    (the registry is process-wide; the next start registers the queued jobs again)."""

    state = FakeState()
    with server_app(tmp_path, state=state, executor=StubExecutor(state, error=RuntimeError("boom"))) as server:
        failed = submit(server)
        assert server.services.queue.wait_idle()
        assert f"task(remi-{failed['job_id']})" not in pending_tasks()
        server.services.queue.set_paused(True)
        queued = submit(server)
        assert f"task(remi-{queued['job_id']})" in pending_tasks()
    assert f"task(remi-{queued['job_id']})" not in pending_tasks()
    with server_app(tmp_path) as server:
        assert f"task(remi-{queued['job_id']})" in pending_tasks()
        server.client.delete(f"/api/v2/jobs/{queued['job_id']}")
    assert pending_tasks() == []


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


def test_interrupt_after_current_lets_the_current_image_finish_first(webui, tmp_path, monkeypatch):
    # The setting "Don't Interrupt in the middle": the first interrupt of a
    # job making several images stops after the current one; a second stops at once
    from modules import shared

    monkeypatch.setitem(shared.opts.data, "interrupt_after_current", True)
    with blocked_server(tmp_path) as server:
        job_id = submit(server)["job_id"]
        assert server.executor.started.wait(5)
        server.state.job_count = 3
        assert server.client.post(f"/api/v2/jobs/{job_id}/interrupt").status_code == 200
        assert server.state.stopping_generation and not server.state.interrupted
        assert server.client.post("/api/v2/interrupt", json={}).status_code in (200, 204)
        assert server.state.interrupted
        assert server.services.queue.wait_idle()
        assert server.client.get(f"/api/v2/jobs/{job_id}").json()["status"] == INTERRUPTED

    monkeypatch.setitem(shared.opts.data, "interrupt_after_current", False)
    with blocked_server(tmp_path / "off") as server:
        job_id = submit(server)["job_id"]
        assert server.executor.started.wait(5)
        server.state.job_count = 3
        assert server.client.post(f"/api/v2/jobs/{job_id}/interrupt").status_code == 200
        assert server.state.interrupted and not server.state.stopping_generation
        assert server.services.queue.wait_idle()


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


# region Queue settings (the Agent Scheduler's keys)


def settings(queue, **values):
    """Point the queue at these settings; returns what it changed and ran."""

    changed, ran = {}, []
    queue.option = lambda name, default: values.get(name, default)

    def set_option(name, value):
        values[name] = value
        changed[name] = value

    queue.set_option = set_option
    queue.when_done = ran.append
    return changed, ran


def test_failed_jobs_are_queued_again_up_to_three_attempts(webui, tmp_path):
    state = FakeState()
    executor = StubExecutor(state, error=RuntimeError("boom"))
    with server_app(tmp_path, state=state, executor=executor) as server:
        settings(server.services.queue, queue_automatic_requeue_failed_task=True)
        first = submit(server, "flaky")
        assert wait_until(lambda: len(executor.calls) == 3 and server.services.store.count(QUEUED) == 0)
        assert server.services.queue.wait_idle()
        jobs = server.client.get("/api/v2/jobs", params={"status": "failed"}).json()["items"]
        assert len(jobs) == 3 and all(job["summary"] == "flaky" for job in jobs)
        chain = {job["id"]: job.get("retry_of") for job in jobs}
        assert chain[first["job_id"]] is None and len([value for value in chain.values() if value]) == 2

    # off (the default): failed once, stays failed
    executor = StubExecutor(state, error=RuntimeError("boom"))
    with server_app(tmp_path / "off", state=state, executor=executor) as server:
        settings(server.services.queue)
        submit(server)
        assert server.services.queue.wait_idle() and len(executor.calls) == 1


def test_history_retention_deletes_old_unstarred_jobs(server):
    queue, store = server.services.queue, server.services.store
    ids = [submit(server, f"p{index}")["job_id"] for index in range(3)]
    assert queue.wait_idle()
    old = time.time() - 10 * 86400
    with store._lock:
        store._db.execute("UPDATE jobs SET finished = ? WHERE id IN (?, ?)", (old, ids[0], ids[1]))
    store.set_starred([ids[1]], True)
    settings(queue)
    assert queue.apply_retention(force=True) == 0  # "Keep forever" by default
    settings(queue, queue_history_retention_days="7 days")
    assert queue.apply_retention(force=True) == 1
    assert store.get(ids[0]) is None and store.get(ids[1]) is not None and store.get(ids[2]) is not None
    assert queue.apply_retention() == 0  # at most once an hour


def test_the_completion_action_runs_once_when_the_queue_is_done(webui, tmp_path):
    state = FakeState()
    gate = threading.Event()
    executor = StubExecutor(state, gate=gate)
    with server_app(tmp_path, state=state, executor=executor) as server:
        changed, ran = settings(server.services.queue, queue_completion_action="Shut down")
        submit(server, "one")
        submit(server, "two")
        assert wait_until(lambda: len(executor.calls) == 1)
        assert ran == []
        gate.set()
        assert server.services.queue.wait_idle() and wait_until(lambda: ran == ["Shut down"])
        assert changed == {"queue_completion_action": "Do nothing"} and len(executor.calls) == 2
        submit(server, "three")
        assert server.services.queue.wait_idle() and ran == ["Shut down"]

    # a run stopped by hand does not end in the action
    executor = StubExecutor(state, gate=threading.Event())
    with server_app(tmp_path / "interrupted", state=state, executor=executor) as server:
        changed, ran = settings(server.services.queue, queue_completion_action="Stop webui")
        created = submit(server)
        assert wait_until(lambda: len(executor.calls) == 1)
        server.client.post(f"/api/v2/jobs/{created['job_id']}/interrupt")
        executor.gate.set()
        assert server.services.queue.wait_idle() and ran == [] and changed == {}


def test_completion_actions_run_the_systems_commands():
    from remi.jobs import completion

    calls, stops = [], []

    def perform(action, platform):
        return completion.perform(action, platform=platform, run=calls.append, stop_server=lambda: stops.append(action))

    assert not perform("Do nothing", "linux") and not perform("Format the disk", "linux")
    assert perform("Stop webui", "win32") and calls == [] and stops == ["Stop webui"]
    assert perform("Shut down", "linux") and calls[-1] == ["systemctl", "poweroff"] and stops[-1] == "Shut down"
    assert perform("Restart", "win32") and calls[-1] == ["shutdown", "/r", "/t", "0"]
    assert perform("Sleep", "darwin") and calls[-1] == ["pmset", "sleepnow"] and stops[-1] == "Restart"  # sleeping keeps the server
    assert perform("Hibernate", "linux") and calls[-1] == ["systemctl", "hibernate"]
    assert set(completion.ACTIONS) == {"Do nothing", "Stop webui", "Sleep", "Hibernate", "Restart", "Shut down"}


def test_queue_settings_keep_the_agent_scheduler_keys(webui):
    from modules import shared, shared_options

    from remi.jobs import completion
    from remi.jobs.queue import RETENTION_DAYS

    labels = shared.opts.data_labels
    assert labels["queue_history_retention_days"].default == "Keep forever"
    assert list(labels["queue_history_retention_days"].component_args["choices"]) == list(RETENTION_DAYS)
    assert [value for _label, value in labels["queue_completion_action"].component_args["choices"]] == list(completion.ACTIONS)
    assert labels["queue_completion_action"].restrict_api and "queue_completion_action" in shared_options.restricted_opts
    assert labels["queue_automatic_requeue_failed_task"].default is False


# endregion


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
