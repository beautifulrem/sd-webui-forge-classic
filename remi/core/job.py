"""What a generation entry point needs to know about the job running it."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from remi.ui.context import Context, UploadResolver


@dataclass
class JobContext:
    """Passed to ``run_txt2img`` / ``run_img2img`` / ``run_extras_named``.

    ``resolve_upload`` turns upload ids into PIL images / file paths (the
    jobs layer's upload store; batch inputs should carry their original file
    name as ``image.filename``); ``extra`` carries anything the caller wants
    the processing object to see (set as attributes on ``p``)."""

    id_task: str = ""
    user: str | None = None
    resolve_upload: UploadResolver | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def context(self) -> Context:
        return Context(resolve_upload=self.resolve_upload)


@contextmanager
def job_state(job: JobContext) -> Iterator[None]:
    """``shared.state`` and the progress registry around one job, as
    ``call_queue.wrap_gradio_gpu_call`` set them up: a fresh state (so an
    interrupt or skip of the previous job does not leak into this one), the
    task marked current while it runs, and the flags cleared afterwards.
    Serialising jobs (one at a time, on the main thread) is the caller's."""

    from modules import progress, shared

    id_task = job.id_task or None
    shared.state.begin(job=job.id_task or "(remi)")
    progress.start_task(id_task)
    try:
        yield
    finally:
        progress.finish_task(id_task)
        shared.state.end()
        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.stopping_generation = False
        shared.state.job_count = 0
        shared.state.job = ""
