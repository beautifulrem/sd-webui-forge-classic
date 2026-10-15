"""Realtime events between the generation side and the WebSocket hubs.

The generation side (the jobs worker, :class:`ProgressMonitor`,
:class:`PreviewEncoder`) runs on plain threads and publishes to an
:class:`EventBus`; each running server's WebSocket hub subscribes to it and
fans events out to its clients from its own event loop. Nothing here imports
FastAPI or Gradio.

Binary preview frames (architecture.md, "WebSocket"): a little-endian header
``u8 type=1, u8 fmt (1=jpeg, 2=webp), u32 job_seq, u32 step`` followed by the
encoded image.
"""

from __future__ import annotations

import io
import logging
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)

FRAME_PREVIEW = 1
FORMAT_JPEG = 1
FORMAT_WEBP = 2
FORMAT_NAMES = {FORMAT_JPEG: "jpeg", FORMAT_WEBP: "webp"}
_HEADER = struct.Struct("<BBII")
HEADER_SIZE = _HEADER.size

DEFAULT_PREVIEW_FPS = 6.0


@dataclass(frozen=True)
class PreviewFrame:
    fmt: int
    job_seq: int
    step: int
    data: bytes

    @property
    def format_name(self) -> str:
        return FORMAT_NAMES[self.fmt]

    def pack(self) -> bytes:
        return pack_frame(self.fmt, self.job_seq, self.step, self.data)


def pack_frame(fmt: int, job_seq: int, step: int, data: bytes) -> bytes:
    return _HEADER.pack(FRAME_PREVIEW, fmt, job_seq & 0xFFFFFFFF, max(0, step) & 0xFFFFFFFF) + data


def unpack_frame(frame: bytes) -> PreviewFrame:
    if len(frame) < HEADER_SIZE:
        raise ValueError("frame too short")
    kind, fmt, job_seq, step = _HEADER.unpack_from(frame)
    if kind != FRAME_PREVIEW or fmt not in FORMAT_NAMES:
        raise ValueError(f"unknown frame type {kind}/{fmt}")
    return PreviewFrame(fmt, job_seq, step, frame[HEADER_SIZE:])


class Subscriber(Protocol):
    """Receives events from any thread; must not block."""

    def send_event(self, event: dict[str, Any]) -> None: ...

    def send_frame(self, frame: PreviewFrame) -> None: ...


class EventBus:
    """Thread-safe fan-out of JSON events (``{"type": ..., **fields}``) and
    preview frames to the subscribed hubs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)

        return unsubscribe

    def _targets(self) -> list[Subscriber]:
        with self._lock:
            return list(self._subscribers)

    def publish(self, type: str, **fields: Any) -> None:
        event = {"type": type, **fields}
        for subscriber in self._targets():
            try:
                subscriber.send_event(event)
            except Exception:
                log.exception("event subscriber failed")

    def wants_frames(self) -> bool:
        """Whether a subscriber has someone to send previews to (a hub
        without clients says no through its ``wants_frames()``)."""

        return any(getattr(subscriber, "wants_frames", lambda: True)() for subscriber in self._targets())

    def publish_frame(self, frame: PreviewFrame) -> None:
        for subscriber in self._targets():
            try:
                subscriber.send_frame(frame)
            except Exception:
                log.exception("frame subscriber failed")


@dataclass(frozen=True)
class JobRef:
    """The running job as the realtime side sees it."""

    job_id: str
    seq: int
    tab: str

    def fields(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "seq": self.seq, "tab": self.tab}


CurrentJob = Callable[[], "JobRef | None"]


def _shared_state() -> Any:
    from modules import shared

    return shared.state


def _shared_opts() -> Any:
    from modules import shared

    return shared.opts


def progress_snapshot(state: Any, now: float | None = None) -> dict[str, Any]:
    """Step, ETA and job counters of ``shared.state`` (as ``/internal/progress``
    computes them; ``job_no`` is 0-based)."""

    job_count, job_no = max(int(state.job_count or 0), 0), int(state.job_no or 0)
    steps, step = int(state.sampling_steps or 0), int(state.sampling_step or 0)
    progress = 0.0
    if job_count > 0:
        progress += job_no / job_count
        if steps > 0:
            progress += step / steps / job_count
    progress = min(progress, 1.0)
    eta = None
    if progress > 0 and state.time_start:
        elapsed = (now or time.time()) - state.time_start
        eta = max(elapsed / progress - elapsed, 0.0)
    return {
        "step": step,
        "steps": steps,
        "job_no": job_no,
        "job_count": job_count,
        "progress": round(progress, 4),
        "eta": round(eta, 2) if eta is not None else None,
        "text": state.textinfo if isinstance(state.textinfo, str) else None,
    }


class _Loop(threading.Thread):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def wait(self, seconds: float) -> bool:
        """Sleep; True once stopped."""

        return self._stop_event.wait(seconds)


class ProgressMonitor(_Loop):
    """Publishes ``job.progress`` whenever the running job's counters change."""

    def __init__(self, bus: EventBus, current_job: CurrentJob, *, interval: float = 0.2, state: Callable[[], Any] = _shared_state, on_tick: Callable[[], None] | None = None) -> None:
        super().__init__("remi-progress")
        self.bus, self.current_job, self.interval, self.state, self.on_tick = bus, current_job, interval, state, on_tick

    def run(self) -> None:
        last = None
        while not self.wait(self.interval):
            try:
                last = self.tick(last)
            except Exception:
                log.exception("progress monitor")

    def tick(self, last: Any) -> Any:
        if self.on_tick is not None:
            self.on_tick()
        ref = self.current_job()
        if ref is None:
            return None
        snapshot = progress_snapshot(self.state())
        key = (ref.job_id, snapshot["step"], snapshot["steps"], snapshot["job_no"], snapshot["job_count"], snapshot["text"])
        if key != last:
            self.bus.publish("job.progress", **ref.fields(), **snapshot)
        return key


def preview_format(opts: Any) -> int:
    """JPEG when the live preview format is JPEG, WebP otherwise (PNG previews
    are sent as WebP)."""

    return FORMAT_JPEG if getattr(opts, "live_previews_image_format", "jpeg") == "jpeg" else FORMAT_WEBP


def encode_preview(image: Any, fmt: int, quality: int = 80) -> bytes:
    """One preview frame's image bytes (an animated preview's first frame)."""

    if getattr(image, "is_animated", False):
        image.seek(0)
    buffer = io.BytesIO()
    if fmt == FORMAT_JPEG:
        image = image if image.mode == "RGB" else image.convert("RGB")
        image.save(buffer, format="JPEG", quality=quality)
    else:
        image = image if image.mode in ("RGB", "RGBA") else image.convert("RGB")
        image.save(buffer, format="WEBP", quality=quality, method=0)
    return buffer.getvalue()


class PreviewEncoder(_Loop):
    """Encodes only the newest live preview (``shared.state.current_image``),
    at most ``live_preview_fps`` frames per second, while a job runs."""

    def __init__(self, bus: EventBus, current_job: CurrentJob, *, state: Callable[[], Any] = _shared_state, opts: Callable[[], Any] = _shared_opts) -> None:
        super().__init__("remi-preview")
        self.bus, self.current_job, self.state, self.opts = bus, current_job, state, opts

    def interval(self) -> float:
        try:
            fps = float(getattr(self.opts(), "live_preview_fps", DEFAULT_PREVIEW_FPS) or DEFAULT_PREVIEW_FPS)
        except (TypeError, ValueError):
            fps = DEFAULT_PREVIEW_FPS
        return 1.0 / min(max(fps, 0.5), 30.0)

    def run(self) -> None:
        last = None
        while not self.wait(self.interval()):
            try:
                last = self.tick(last)
            except Exception:
                log.exception("preview encoder")

    def tick(self, last: Any) -> Any:
        ref = self.current_job()
        opts = self.opts()
        if ref is None or not getattr(opts, "live_previews_enable", True):
            return None
        if not self.bus.wants_frames():  # nobody watching: do not encode
            return last
        state = self.state()
        key = (ref.job_id, state.id_live_preview)
        image = state.current_image
        if key == last or image is None:
            return last
        fmt = preview_format(opts)
        data = encode_preview(image, fmt)
        step = int(state.current_image_sampling_step or state.sampling_step or 0)
        self.bus.publish_frame(PreviewFrame(fmt, ref.seq, step, data))
        return key
