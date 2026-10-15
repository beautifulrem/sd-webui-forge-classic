"""The generation queue: submission, ordering, and the single worker.

One worker thread takes the next queued job, holds the generation lock
(``modules.call_queue.queue_lock``, shared with ``/sdapi`` and GPU actions)
and runs the job on the WebUI main thread (``modules_forge.main_thread``)
through the named-value entry points. Events go to the :class:`EventBus`.

A claimed job only becomes *current* (``job.started``, progress, previews,
``shared.state`` interrupts) once the worker holds the lock: until then
someone else may be generating. Interrupts and skips that arrive before the
job's ``shared.state.begin()`` are recorded and applied right after it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from remi.core import JobContext
from remi.files import FileStore
from remi.jobs.runner import Executor, JobResult, run_job, run_on_main_thread
from remi.jobs.store import CANCELLED, COMPLETED, FAILED, INTERRUPTED, QUEUED, RUNNING, Job, JobStore
from remi.realtime import EventBus, JobRef

log = logging.getLogger(__name__)

TABS = ("txt2img", "img2img", "extras")
MAX_REPEAT = 100

Dispatch = Callable[..., Any]
Validator = Callable[[str, Any, Any], None]


class JobError(ValueError):
    """A queue operation that cannot be done (``status``: HTTP status)."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


@dataclass
class Submission:
    tab: str
    core: dict[str, Any]
    scripts: dict[str, Any]
    priority: int = 0
    repeat: int = 1
    forever: bool = False


def _state() -> Any:
    from modules import shared

    return shared.state


def _lock() -> Any:
    from modules.call_queue import queue_lock

    return queue_lock


def layout_hash(tab: str) -> str:
    """A fingerprint of the tab's parameter layout (owners and argument
    names), recorded with each job."""

    from remi.core import forms
    from remi.core.tabs import runner_for

    runner = runner_for(tab)
    parts: list[Any] = [forms.core_layout(tab, runner).names]
    if runner is not None:
        parts += [(script.script_id, runner.layout_for(script).names) for script in runner.scripts]
    else:
        from modules import scripts

        from remi.core.extras import assign_args

        parts += [(layout.owner, layout.names) for _script, layout in assign_args(scripts.scripts_postproc)]
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def validate_values(files: FileStore) -> Validator:
    """Checks a submission the way the entry point will (every bad value at
    once, as ``remi.ui.ValidationError``), uploads only by id."""

    def validate(tab: str, core: Any, scripts: Any) -> None:
        from remi.core.tabs import processing_args
        from remi.ui import Context

        processing_args(tab, core, scripts, Context(resolve_upload=files.check_upload))

    return validate


def _summary(core: dict[str, Any], tab: str) -> str:
    prompt = core.get("prompt") if isinstance(core, dict) else None
    text = prompt if isinstance(prompt, str) and prompt.strip() else tab
    return " ".join(text.split())[:200]


class JobQueue:
    def __init__(
        self,
        store: JobStore,
        files: FileStore,
        bus: EventBus,
        *,
        executor: Executor = run_job,
        dispatch: Dispatch = run_on_main_thread,
        lock: Any = None,
        validate: Validator | None = None,
        state: Callable[[], Any] = _state,
        hash_layout: Callable[[str], str] = layout_hash,
    ) -> None:
        self.store, self.files, self.bus = store, files, bus
        self.executor, self.dispatch = executor, dispatch
        self._lock = lock
        self.validate = validate if validate is not None else validate_values(files)
        self.state = state
        self.hash_layout = hash_layout
        self._cv = threading.Condition()
        self._claimed: Job | None = None
        """taken from the queue (maybe still waiting for the lock)"""
        self._current: Job | None = None
        """holding the lock, generating"""
        self._begun = False
        """``shared.state.begin()`` ran for the current job"""
        self._interrupted: set[str] = set()
        self._cancelled: set[str] = set()
        self._skips: set[str] = set()
        self._holds = 0
        self._paused = store.get_meta("paused") == "1"
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._loading_model: str | None = None

    # region Lifecycle

    @property
    def lock(self) -> Any:
        return self._lock if self._lock is not None else _lock()

    def start(self) -> None:
        with self._cv:
            if self._thread is not None and self._thread.is_alive():
                return
            recovered = self.store.recover()
            if recovered:
                log.info("Marked %d job(s) interrupted by the last shutdown", recovered)
            self._stopping = False
            self._register_pending()
            self._thread = threading.Thread(target=self._loop, name="remi-jobs", daemon=True)
            self._thread.start()

    def stop(self, timeout: float | None = 5, *, interrupt: bool = False) -> bool:
        """Stop the worker after the running job (``interrupt``: interrupt
        that job first); returns whether the worker ended in time. Queued
        jobs stay queued (persisted) for the next start."""

        with self._cv:
            self._stopping = True
            self._cv.notify_all()
            job = self._claimed
        if interrupt and job is not None:
            self._signal(job.id, self._interrupted)
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        """No job starts inside this block (UI reload); waits for the
        running one to finish first."""

        with self._cv:
            self._holds += 1
            while self._claimed is not None:
                self._cv.wait()
        try:
            yield
        finally:
            with self._cv:
                self._holds -= 1
                self._cv.notify_all()

    def wait_idle(self, timeout: float = 10) -> bool:
        """Until nothing runs and nothing runnable is queued (tests)."""

        with self._cv:
            return self._cv.wait_for(lambda: self._claimed is None and (self._paused or self._holds or self.store.count(QUEUED) == 0), timeout)

    # endregion

    # region Queries

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def current(self) -> Job | None:
        return self._current

    @property
    def busy(self) -> bool:
        """A job is running or waiting for the generation lock."""

        return self._claimed is not None

    def current_ref(self) -> JobRef | None:
        job = self._current
        return JobRef(job.id, job.seq, job.tab) if job is not None else None

    def status(self) -> dict[str, Any]:
        job = self._current
        return {"size": self.store.count(QUEUED), "paused": self._paused, "running": job.id if job else None}

    # endregion

    # region Submission and control

    def submit(self, submission: Submission, *, user: str | None = None) -> list[Job]:
        """Validate and queue (``repeat`` copies); raises
        ``remi.ui.ValidationError`` or :class:`JobError`."""

        from remi.ui import ParamError, ValidationError
        from remi.ui.schema import SCHEMA_VERSION

        if submission.tab not in TABS:
            raise ValidationError([ParamError(f"tab is one of {', '.join(TABS)}")])
        if not 1 <= submission.repeat <= MAX_REPEAT:
            raise ValidationError([ParamError(f"repeat is between 1 and {MAX_REPEAT}")])
        self.validate(submission.tab, submission.core, submission.scripts)
        layout = self.hash_layout(submission.tab)
        jobs = [self.store.add(submission.tab, submission.core or {}, submission.scripts or {}, priority=submission.priority, layout_hash=layout, schema_version=SCHEMA_VERSION, summary=_summary(submission.core or {}, submission.tab), forever=submission.forever, user=user) for _ in range(submission.repeat)]
        for job in jobs:
            self._enqueued(job)
        self._changed()
        return jobs

    def retry(self, job_id: str, *, user: str | None = None) -> Job:
        job = self.get(job_id)
        if job.status in (QUEUED, RUNNING):
            raise JobError("the job has not finished")
        return self._copy(job, user=user)

    def duplicate(self, job_id: str, *, user: str | None = None) -> Job:
        """Queue a copy of any job: right after it while it is queued, else at the end."""

        job = self.get(job_id)
        return self._copy(job, user=user, after=job.id if job.status == QUEUED else None)

    def _copy(self, job: Job, *, user: str | None, after: str | None = None) -> Job:
        if not job.replayable:
            raise JobError("this job has arguments that do not map to the current parameters; open it in its form to queue it again")
        self.validate(job.tab, job.core, job.scripts)  # the layout or its uploads may have changed
        meta = {key: job.meta[key] for key in ("name",) if job.meta.get(key)}
        new = self.store.add(job.tab, job.core, job.scripts, priority=job.priority, layout_hash=self.hash_layout(job.tab), schema_version=job.schema_version, summary=job.summary, user=user, retry_of=job.id, after=after, meta=meta or None)
        self._enqueued(new)
        self._changed()
        return new

    def update_params(self, job_id: str, core: dict[str, Any], scripts: dict[str, Any]) -> Job:
        """Replace a queued job's parameters (validated like a submission)."""

        from remi.ui.schema import SCHEMA_VERSION

        job = self.get(job_id)
        if job.status != QUEUED:
            raise JobError("only queued jobs can be edited")
        self.validate(job.tab, core, scripts)
        updated = self.store.update_params(job_id, core or {}, scripts or {}, layout_hash=self.hash_layout(job.tab), schema_version=SCHEMA_VERSION, summary=_summary(core or {}, job.tab))
        if updated is None:
            raise JobError("the job started meanwhile")
        self._changed()
        return updated

    def set_starred(self, ids: list[str], starred: bool) -> int:
        count = self.store.set_starred(ids, starred)
        self._changed()
        return count

    def clear_history(self, *, keep_starred: bool = True) -> int:
        count = self.store.clear_finished(keep_starred=keep_starred)
        self._changed()
        return count

    def add_imported(self, job: Job, *, source: str, source_id: str) -> Job:
        """Store an imported job (queued ones are announced like submissions)."""

        added = self.store.add_record(job)
        self.store.record_import(source, source_id, added.id)
        if added.status == QUEUED:
            self._enqueued(added)
        return added

    def changed(self) -> None:
        """Announce the queue state (after a batch of changes)."""

        self._changed()

    def interrupt(self, job_id: str) -> Job:
        """Stop the running job (keeping what it made) or cancel a queued one."""

        job = self.get(job_id)
        if job.status == RUNNING:
            self._signal(job_id, self._interrupted)
        elif job.status == QUEUED:
            self._cancel_queued(job)
        return self.get(job_id)

    def skip(self, job_id: str) -> None:
        if not self._signal(job_id, self._skips):
            raise JobError("the job is not running")

    def delete(self, job_id: str) -> None:
        """Cancel a running job, or remove a queued or finished one."""

        job = self.get(job_id)
        if job.status == RUNNING and self._signal(job_id, self._cancelled):
            return
        was_queued = job.status == QUEUED
        if self.store.delete(job_id) and was_queued:
            self._pending_done(job)
            self._changed()

    def update(self, job_id: str, *, priority: int | None = None, forever: bool | None = None) -> Job:
        self.get(job_id)
        job = self.store.update(job_id, priority=priority, forever=forever)
        self._changed()
        return job

    def reorder(self, ids: list[str]) -> None:
        self.store.reorder(ids)
        self._changed()

    def move(self, job_id: str, *, after: str | None = None, before: str | None = None, priority: int | None = None) -> None:
        """Move a queued job next to a neighbour (see :meth:`JobStore.move`)."""

        self.get(job_id)
        if not self.store.move(job_id, after=after, before=before, priority=priority):
            raise JobError("the job is not queued any more")
        self._changed()

    def interrupt_all(self, *, clear_queue: bool = False, stop_forever: bool = True) -> None:
        """Interrupt whatever generates (also ``/sdapi`` requests)."""

        if stop_forever:
            self.store.clear_forever()
        if clear_queue:
            for job in self.store.queued():
                self._cancel_queued(job, notify=False)
        job = self._claimed
        if job is not None:
            self._signal(job.id, self._interrupted)
        if job is None or self._current is None:
            self.state().interrupt()  # an /sdapi request or a GPU action
        self._changed()

    def set_paused(self, paused: bool) -> None:
        with self._cv:
            self._paused = paused
            self.store.set_meta("paused", "1" if paused else "0")
            self._cv.notify_all()
        self._changed()

    def stop_forever(self) -> int:
        count = self.store.clear_forever()
        self._changed()
        return count

    # endregion

    # region Internals

    def get(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise JobError("no such job", 404)
        return job

    def _signal(self, job_id: str, bucket: set[str]) -> bool:
        """Record an interrupt/cancel (or skip, ``bucket`` ``_skips``) for
        the claimed job; flip ``shared.state`` now if the job has begun,
        else :meth:`_on_begin` does. False when ``job_id`` is not claimed."""

        with self._cv:
            job = self._claimed
            if job is None or job.id != job_id:
                return False
            bucket.add(job_id)
            if self._begun and self._current is job:
                self._apply_signals(job_id)
            return True

    def _apply_signals(self, job_id: str) -> None:
        state = self.state()
        if job_id in self._interrupted or job_id in self._cancelled:
            state.interrupt()
        elif job_id in self._skips:
            self._skips.discard(job_id)
            state.skip()

    def _on_begin(self, job_id: str) -> None:
        """``JobContext.on_begin``: after ``shared.state.begin()`` reset the
        flags, apply what arrived before."""

        with self._cv:
            self._begun = True
            self._apply_signals(job_id)

    def _enqueued(self, job: Job) -> None:
        self._progress_pending(job)
        self.bus.publish("job.queued", job_id=job.id, seq=job.seq, tab=job.tab, position=self.store.count(QUEUED))
        with self._cv:
            self._cv.notify_all()

    def _cancel_queued(self, job: Job, *, notify: bool = True) -> None:
        if self.store.cancel_queued(job.id):
            self._pending_done(job)
            self.bus.publish("job.failed", job_id=job.id, seq=job.seq, tab=job.tab, status=CANCELLED, error="Cancelled")
            if notify:
                self._changed()

    def _changed(self) -> None:
        self.bus.publish("queue.changed", **self.status())

    def _register_pending(self) -> None:
        for job in self.store.queued():
            self._progress_pending(job)

    @staticmethod
    def _progress_pending(job: Job) -> None:
        progress = sys.modules.get("modules.progress")
        if progress is not None:
            progress.add_task_to_queue(job.id_task)

    @staticmethod
    def _pending_done(job: Job) -> None:
        progress = sys.modules.get("modules.progress")
        if progress is not None:
            progress.pending_tasks.pop(job.id_task, None)

    def _runnable(self) -> bool:
        return self._stopping or (not self._paused and not self._holds and self.store.count(QUEUED) > 0)

    def _loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(self._runnable)
                if self._stopping:
                    return
                job = self.store.claim_next()
                if job is not None:
                    self._claimed = job
            if job is None:  # it cancelled queued jobs it cannot run: nothing is queued now
                self._changed()
                continue
            try:
                with self.lock:  # behind /sdapi requests and GPU actions
                    self._run(job)
            except Exception:
                log.exception("job %s", job.id)
            finally:
                with self._cv:
                    self._claimed = self._current = None
                    self._begun = False
                    self._interrupted.discard(job.id)
                    self._cancelled.discard(job.id)
                    self._skips.discard(job.id)
                    self._cv.notify_all()
                self._changed()

    def _run(self, job: Job) -> None:
        """Run ``job``; the caller holds the generation lock."""

        from remi.ui import ValidationError

        ref = JobRef(job.id, job.seq, job.tab)
        with self._cv:
            stopped = job.id in self._interrupted or job.id in self._cancelled
            if not stopped:
                self._current = job
        if stopped:  # interrupted while waiting for the lock: nothing made
            status = CANCELLED if job.id in self._cancelled else INTERRUPTED
            self.store.finish(job.id, status)
            self._pending_done(job)
            self.bus.publish("job.completed", **ref.fields(), status=status, outputs=[], infotext=None)
            return
        self.bus.publish("job.started", **ref.fields())
        self._changed()
        ctx = JobContext(id_task=job.id_task, user=job.user, resolve_upload=self.files.resolve_upload, on_begin=lambda: self._on_begin(job.id))
        self._note_model_loading()
        try:
            result = self.dispatch(self.executor, job, ctx)
        except ValidationError as error:
            self._fail(job, str(error), errors=error.to_json()["errors"])
            return
        except Exception as error:
            self._fail(job, self._error_message(error))
            return
        finally:
            self.poll()
        status = CANCELLED if job.id in self._cancelled else INTERRUPTED if job.id in self._interrupted else COMPLETED
        outputs = self._register(job, result or JobResult())
        info = (result.info if result else None) or next((output.get("infotext") for output in outputs if output.get("infotext")), None)
        self.store.finish(job.id, status, outputs=outputs, infotext=info)
        self.bus.publish("job.completed", **ref.fields(), status=status, outputs=outputs, infotext=info)
        if job.forever and status == COMPLETED:  # paused: queued for the resume
            again = self.store.requeue_forever(job.id)
            if again is not None:
                self._enqueued(again)

    def _fail(self, job: Job, message: str, *, errors: list[dict] | None = None) -> None:
        status = CANCELLED if job.id in self._cancelled else FAILED
        self.store.finish(job.id, status, error=message, errors=errors)
        self.bus.publish("job.failed", job_id=job.id, seq=job.seq, tab=job.tab, status=status, error=message, errors=errors)

    @staticmethod
    def _error_message(error: Exception) -> str:
        try:
            from backend.memory_management import is_oom
        except Exception:  # no backend (tests)
            is_oom = None
        if is_oom is not None and is_oom(error):
            try:
                from modules import sd_models

                sd_models.unload_model_weights()
            except Exception:
                log.exception("unloading the model after running out of memory")
            return "Out of memory: the model was unloaded"
        log.error("job failed", exc_info=error)
        return f"{type(error).__name__}: {error}"

    def _register(self, job: Job, result: JobResult) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for index, image in enumerate(result.images):
            infotext = result.infotexts[index] if index < len(result.infotexts) else result.info
            try:
                saved = getattr(image, "already_saved_as", None)
                if saved and os.path.isfile(saved):
                    record = self.files.register_output(saved, job_id=job.id, tab=job.tab, infotext=infotext, width=image.width, height=image.height)
                else:
                    record = self.files.save_output_image(image, infotext, job_id=job.id, tab=job.tab)
            except Exception:
                log.exception("registering an output of job %s", job.id)
                continue
            outputs.append({"id": record.id, "kind": "image", "width": record.width, "height": record.height, "infotext": infotext})
        for path in result.files:
            try:
                record = self.files.register_output(path, job_id=job.id, tab=job.tab, infotext=result.info)
            except OSError:
                continue
            outputs.append({"id": record.id, "kind": "video" if record.mime.startswith("video/") else "file", "infotext": result.info})
        return outputs

    # endregion

    # region Model loading events

    def _note_model_loading(self) -> None:
        model = pending_model()
        if model is not None:
            self._loading_model = model
            self.bus.publish("model.loading", stage="loading", model=model)

    def poll(self) -> None:
        """Announce the end of a model load started by a job (called by the
        progress monitor and after each job)."""

        model = self._loading_model
        if model is None:
            return
        if pending_model() is None:
            self._loading_model = None
            self.bus.publish("model.loaded", model=model)
        elif self._current is None:
            self._loading_model = None
            self.bus.publish("model.loaded", model=model, error="the model was not loaded")

    # endregion


def pending_model() -> str | None:
    """The checkpoint the next generation will load, if it differs from the
    loaded one."""

    sd_models = sys.modules.get("modules.sd_models")
    data = getattr(sd_models, "model_data", None)
    params = getattr(data, "forge_loading_parameters", None)
    if not params or str(params) == getattr(data, "forge_hash", None):
        return None
    info = params.get("checkpoint_info") if isinstance(params, dict) else None
    return str(getattr(info, "title", None) or getattr(info, "name", None) or "model")
