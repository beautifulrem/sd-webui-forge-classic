"""Running one job: the named-value entry points of the generation tabs.

:func:`run_job` runs on the main thread (``modules_forge.main_thread``) under
the generation lock; the worker owns both. Results are saved by the normal
save path inside processing (``on_image_saved`` callbacks, PNG infotext);
the worker only indexes the files for ``/api/v2/files/{id}``.
"""

from __future__ import annotations

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
    roles: list[str | None] = field(default_factory=list)
    """per image: ``"grid"``, ``"extra"`` (e.g. ControlNet's detected maps) or None (an image of the batch)"""
    warnings: list[dict[str, Any]] = field(default_factory=list)
    """notes the processing left (``p.comment()``, e.g. a feature skipped on
    this device), as job warnings ``{"message", "values"}``"""


Executor = Callable[[Job, JobContext], JobResult]

def run_job(job: Job, ctx: JobContext) -> JobResult:
    """Generate ``job`` with the WebUI's processing (txt2img, img2img or
    extras); raises ``remi.ui.ValidationError`` for bad values."""

    if job.tab == "txt2img":
        if source := job.meta.get("upscale_from"):
            from modules.txt2img import run_txt2img_upscale

            image = upscale_source(ctx, source)
            return _from_processed(run_txt2img_upscale(job.core, job.scripts, ctx, image, _source_infotext(image)))
        from modules.txt2img import run_txt2img

        return _from_processed(run_txt2img(job.core, job.scripts, ctx))
    if job.tab == "img2img":
        from modules.img2img import run_img2img

        return _from_processed(run_img2img(job.core, job.scripts, ctx))
    if job.tab == "extras":
        from modules.postprocessing import run_extras_named

        images, info = run_extras_named(job.core, job.scripts, ctx)
        info = (info or "").strip()
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


def upscale_source(ctx: JobContext, file_id: str) -> Any:
    """The image a Hires. fix-only run (``upscale_from``) starts from."""

    from remi.ui import Image, ParamError, ValidationError

    if ctx.resolve_upload is None:
        raise ValidationError([ParamError("no file store to read the image from", param="upscale_from")])
    try:
        return ctx.resolve_upload(Image("upscale_from"), file_id)
    except ParamError as error:
        raise ValidationError([ParamError(str(error), param="upscale_from")]) from None


def _source_infotext(image: Any) -> str | None:
    """The generation parameters stored in an image (PNG text, EXIF, ...)."""

    from modules.images import read_info_from_image

    try:
        infotext, _ = read_info_from_image(image)
    except Exception:
        return _image_infotext(image)
    return infotext if isinstance(infotext, str) else None


def _image_infotext(image: Any) -> str | None:
    value = (getattr(image, "info", None) or {}).get("parameters")
    return value if isinstance(value, str) else None


def _as_image(image: Any) -> Any:
    """A PIL image; extra images may be numpy arrays (ControlNet's detected maps)."""

    if image is not None and not hasattr(image, "save"):
        from PIL import Image

        return Image.fromarray(image)
    return image


def _from_processed(processed: Any) -> JobResult:
    main = list(getattr(processed, "images", None) or [])
    first = getattr(processed, "index_of_first_image", 0)
    first = first if isinstance(first, int) else 0
    pairs = [(image, "grid" if index < first else None) for index, image in enumerate(main)]
    pairs += [(image, "extra") for image in getattr(processed, "extra_images", None) or []]
    pairs = [(_as_image(image), role) for image, role in pairs if image is not None]
    images = [image for image, _ in pairs]
    infotexts = [text for text in getattr(processed, "infotexts", None) or [] if isinstance(text, str)]
    info = infotexts[0] if infotexts else getattr(processed, "info", None)
    texts = [infotexts[index] if index < len(infotexts) else (_image_infotext(image) or info) for index, image in enumerate(images)]
    video = getattr(processed, "video_path", None)
    files = [video] if isinstance(video, str) and video else []
    return JobResult(images=images, infotexts=texts, info=info if isinstance(info, str) else None, files=files, roles=[role for _, role in pairs], warnings=comment_notes(getattr(processed, "comments", None)))


def comment_notes(comments: Any) -> list[dict[str, Any]]:
    """``Processed.comments`` (one ``p.comment()`` per line; the old UI showed
    them under the image) as job warnings."""

    lines = comments.splitlines() if isinstance(comments, str) else list(comments or [])
    notes: list[dict[str, Any]] = []
    for line in lines:
        text = str(line).strip()
        note = {"message": "{{note}}", "values": {"note": text}}
        if text and note not in notes:
            notes.append(note)
    return notes


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
