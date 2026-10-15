"""Running one job: the named-value entry points of the generation tabs.

:func:`run_job` runs on the main thread (``modules_forge.main_thread``) under
the generation lock; the worker owns both. Results are saved by the normal
save path inside processing (``on_image_saved`` callbacks, PNG infotext);
the worker only indexes the files for ``/api/v2/files/{id}``.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from remi.core import JobContext
from remi.jobs.store import Job


@dataclass
class JobResult:
    images: list[Any] = field(default_factory=list)
    """PIL images in output order (a grid first, as the gallery showed it)"""
    infotexts: list[str] = field(default_factory=list)
    """the infotext of each image (when known)"""
    info: str | None = None
    """the job's infotext (the first image's)"""
    files: list[str] = field(default_factory=list)
    """other output files (videos) by path"""


Executor = Callable[[Job, JobContext], JobResult]

_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(text: str | None) -> str:
    return html.unescape(_TAG_RE.sub("", text or "")).strip()


def run_job(job: Job, ctx: JobContext) -> JobResult:
    """Generate ``job`` with the WebUI's processing (txt2img, img2img or
    extras); raises ``remi.ui.ValidationError`` for bad values."""

    if job.tab == "txt2img":
        from modules.txt2img import run_txt2img

        return _from_processed(run_txt2img(job.core, job.scripts, ctx))
    if job.tab == "img2img":
        from modules.img2img import run_img2img

        return _from_processed(run_img2img(job.core, job.scripts, ctx))
    if job.tab == "extras":
        from modules.postprocessing import run_extras_named

        images, info_html = run_extras_named(job.core, job.scripts, ctx)
        info = html_to_text(info_html)
        result = JobResult(info=info or None)
        for item in images or []:
            if isinstance(item, str):
                result.files.append(item)
            elif item is not None:
                if not hasattr(item, "save"):  # a video's last frame (numpy)
                    from PIL import Image

                    item = Image.fromarray(item)
                result.images.append(item)
                result.infotexts.append(_image_infotext(item) or info)
        return result
    raise ValueError(f"unknown tab {job.tab!r}")


def _image_infotext(image: Any) -> str | None:
    value = (getattr(image, "info", None) or {}).get("parameters")
    return value if isinstance(value, str) else None


def _from_processed(processed: Any) -> JobResult:
    images = [image for image in list(getattr(processed, "images", None) or []) + list(getattr(processed, "extra_images", None) or []) if image is not None]
    infotexts = [text for text in getattr(processed, "infotexts", None) or [] if isinstance(text, str)]
    info = infotexts[0] if infotexts else getattr(processed, "info", None)
    texts = [infotexts[index] if index < len(infotexts) else (_image_infotext(image) or info) for index, image in enumerate(images)]
    return JobResult(images=images, infotexts=texts, info=info if isinstance(info, str) else None)


def run_on_main_thread(func: Callable[..., Any], *args: Any) -> Any:
    """Run ``func`` on the WebUI main thread and return its result or raise
    its exception (``main_thread`` itself swallows exceptions)."""

    from modules_forge import main_thread

    box: dict[str, Any] = {}

    def call() -> None:
        try:
            box["result"] = func(*args)
        except BaseException as error:  # re-raised on the worker
            box["error"] = error

    main_thread.run_and_wait_result(call)
    if "error" in box:
        raise box["error"]
    return box.get("result")


def run_inline(func: Callable[..., Any], *args: Any) -> Any:
    """Dispatch on the calling thread (tests, API-only tools)."""

    return func(*args)
