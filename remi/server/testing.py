"""Test support for the server: a fake ``shared.state``, a stub job
executor (no model) and a running app around them.

    with server_app(tmp_path) as server:
        server.client.post("/api/v2/jobs", json={...})
        server.services.queue.wait_idle()
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from remi.core import JobContext
from remi.jobs import Job, JobResult, run_inline
from remi.server.auth import Auth

BASE_URL = "http://127.0.0.1:7860"
WS_URL = "ws://127.0.0.1:7860/api/v2/ws"
"""TestClient.websocket_connect ignores ``base_url``: pass full URLs."""


class FakeState:
    """The parts of ``shared.state`` the server reads and flips."""

    def __init__(self) -> None:
        self.interrupted = self.skipped = False
        self.job_count = self.job_no = 0
        self.sampling_step = self.sampling_steps = 0
        self.time_start: float | None = None
        self.textinfo: str | None = None
        self.id_live_preview = 0
        self.current_image: Any = None
        self.current_image_sampling_step = 0
        self.live_preview_listeners: list[Callable[[], None]] = []

    def assign_current_image(self, image: Any) -> None:
        """As ``shared.state.assign_current_image``: the newest preview, listeners woken."""

        self.current_image = image
        self.id_live_preview += 1
        for listener in tuple(self.live_preview_listeners):
            listener()

    def interrupt(self) -> None:
        self.interrupted = True

    def skip(self) -> None:
        self.skipped = True


@dataclass
class StubExecutor:
    """Runs a "job" without a model: optional steps with live previews,
    then ``count`` solid images. ``gate`` (an Event) holds the job until set;
    ``error`` is raised instead of producing images."""

    state: FakeState
    count: int = 1
    steps: int = 0
    step_time: float = 0.02
    gate: threading.Event | None = None
    error: Exception | None = None
    calls: list[tuple[Job, JobContext]] = field(default_factory=list)
    started: threading.Event = field(default_factory=threading.Event)

    def __call__(self, job: Job, ctx: JobContext) -> JobResult:
        self.calls.append((job, ctx))
        state = self.state
        state.interrupted = state.skipped = False  # as shared.state.begin()
        if ctx.on_begin is not None:
            ctx.on_begin()
        state.time_start, state.job_count, state.job_no = time.time(), 1, 0
        state.sampling_steps = self.steps
        self.started.set()
        for step in range(1, self.steps + 1):
            if state.interrupted:
                break
            state.sampling_step = step
            state.assign_current_image(Image.new("RGB", (32, 32), (step * 20 % 256, 0, 0)))
            time.sleep(self.step_time)
        if self.gate is not None:
            deadline = time.time() + 10
            while not self.gate.is_set() and not state.interrupted and time.time() < deadline:
                time.sleep(0.01)
        if self.error is not None:
            raise self.error
        prompt = job.core.get("prompt", "")
        images = [Image.new("RGB", (16, 8), (index * 40 % 256, 128, 64)) for index in range(self.count)]
        infotexts = [f"{prompt}\nSteps: 1, Seed: {index}" for index in range(self.count)]
        state.job_count = 0
        return JobResult(images=images, infotexts=infotexts, info=infotexts[0] if infotexts else None)


@contextmanager
def server_app(
    root: Path,
    *,
    auth: Auth | None = None,
    executor: Callable[..., JobResult] | None = None,
    state: FakeState | None = None,
    spa_dir: Path | None = None,
    validate: Callable[..., None] | None = None,
    realtime: bool = False,
    base_url: str = BASE_URL,
    lock: Any = None,
) -> Iterator[SimpleNamespace]:
    """A running app on fresh services under ``root`` (the WebUI itself must
    be set up, e.g. with ``remi.testing.webui_state``)."""

    from fastapi.testclient import TestClient

    from remi.server import Services, create_app

    state = state or FakeState()
    executor = executor or StubExecutor(state)
    options: dict[str, Any] = {"executor": executor, "dispatch": run_inline, "lock": lock or threading.Lock(), "state": lambda: state}
    if validate is not None:
        options["validate"] = validate
    services = Services.create(root, realtime=realtime, housekeeping=False, **options)
    app = create_app(services=services, auth=auth or Auth(), sdapi=False, spa_dir=spa_dir or root / "no-dist")
    try:
        with TestClient(app, base_url=base_url) as client:
            yield SimpleNamespace(client=client, app=app, services=services, state=state, executor=executor)
    finally:
        services.close()
