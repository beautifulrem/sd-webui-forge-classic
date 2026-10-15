"""Process-wide services of the Remi server: the event bus, the file index,
the job queue and its worker, the network index and the realtime threads.

They outlive one app instance ("Reload UI" builds a new app, the queue keeps
running). Data lives under ``<data path>/remi-data``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from remi.files import DEFAULT_MAX_UPLOAD, FileStore
from remi.jobs import JobQueue, JobStore, run_job, run_on_main_thread
from remi.networks import NetworkIndex
from remi.realtime import EventBus, PreviewEncoder, ProgressMonitor

log = logging.getLogger(__name__)

OUTPUT_DIR_OPTIONS = ("outdir_samples", "outdir_txt2img_samples", "outdir_img2img_samples", "outdir_extras_samples", "outdir_hires_samples", "outdir_grids", "outdir_txt2img_grids", "outdir_img2img_grids", "outdir_save")


@dataclass
class Services:
    root: Path
    bus: EventBus
    files: FileStore
    store: JobStore
    queue: JobQueue
    networks: NetworkIndex
    state: Callable[[], Any] | None = None
    """``shared.state`` getter for the realtime threads (tests pass a fake)"""
    realtime: bool = True
    housekeeping_enabled: bool = True
    _threads: list[Any] = field(default_factory=list)
    _started: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)

    HOUSEKEEPING_INTERVAL: ClassVar[float] = 3600

    @classmethod
    def create(cls, root: str | Path, *, max_upload_bytes: int | Callable[[], int] = DEFAULT_MAX_UPLOAD, realtime: bool = True, housekeeping: bool = True, **queue_options: Any) -> Services:
        """``queue_options`` go to :class:`JobQueue` (``executor``,
        ``dispatch``, ``lock``, ``validate``, ``state``, ``hash_layout``);
        ``realtime``/``housekeeping``: start those threads with the queue."""

        root = Path(root)
        bus = EventBus()
        files = FileStore(root, max_upload_bytes=max_upload_bytes)
        store = JobStore(root / "jobs.db")
        queue_options.setdefault("executor", run_job)
        queue_options.setdefault("dispatch", run_on_main_thread)
        queue = JobQueue(store, files, bus, **queue_options)
        return cls(root, bus, files, store, queue, NetworkIndex(files, on_change=lambda kind: bus.publish("networks.changed", **({"kind": kind} if kind else {}))), state=queue_options.get("state"), realtime=realtime, housekeeping_enabled=housekeeping)

    def start(self) -> None:
        """Start the worker (and the progress/preview threads); idempotent."""

        if self._started:
            return
        self._started = True
        self._stop.clear()
        self.queue.start()
        if self.realtime:
            state = {"state": self.state} if self.state is not None else {}
            self._threads = [ProgressMonitor(self.bus, self.queue.current_ref, on_tick=self.queue.poll, **state), PreviewEncoder(self.bus, self.queue.current_ref, **state)]
            for thread in self._threads:
                thread.start()
        if self.housekeeping_enabled:
            threading.Thread(target=self._housekeeping_loop, name="remi-housekeeping", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.stop()
        self.queue.stop()
        self._started = False

    def _housekeeping_loop(self) -> None:
        stop = self._stop
        while True:
            self.housekeeping()
            if stop.wait(self.HOUSEKEEPING_INTERVAL):
                return

    def close(self) -> None:
        """Stop everything and close the databases."""

        self.stop()
        self.files.close()
        self.store.close()

    def housekeeping(self) -> None:
        """Prune old uploads, unsaved outputs and gallery zips, and index
        outputs saved outside the queue (at start, then hourly)."""

        from remi.server.settings import option

        try:
            days = float(option("remi_upload_retention_days", 7))
            removed = self.files.prune_uploads(days * 86400, keep=self.store.referenced_strings())
            if removed:
                log.info("Removed %d old upload(s)", removed)
            removed = self.files.prune_unsaved(days * 86400)
            if removed:
                log.info("Removed %d old unsaved output(s)", removed)
        except Exception:
            log.exception("pruning uploads")
        try:
            from remi.library import gallery_for

            gallery_for(self.files).prune_exports()
        except Exception:
            log.exception("pruning gallery exports")
        try:
            added = self.files.scan_outputs(output_dirs())
            if added:
                log.info("Indexed %d output image(s)", added)
        except Exception:
            log.exception("indexing outputs")


def output_dirs() -> list[str]:
    from modules import shared

    return [path for name in OUTPUT_DIR_OPTIONS if (path := getattr(shared.opts, name, ""))]


_default: Services | None = None
_default_lock = threading.Lock()


def default_services(factory: Callable[[Path], Services] | None = None) -> Services:
    """The services of this process (created on first use)."""

    global _default
    with _default_lock:
        if _default is None:
            from modules.paths_internal import data_path
            from remi.server.settings import option

            # Not "remi": the data path defaults to the repository root, where remi/ is the package
            root = Path(data_path) / "remi-data"
            _default = factory(root) if factory else Services.create(root, max_upload_bytes=lambda: int(float(option("remi_upload_max_mb", 64)) * 1024 * 1024))
        return _default
