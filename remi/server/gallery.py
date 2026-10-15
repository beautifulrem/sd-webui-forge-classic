"""The gallery's "Save", "Save zip" and "Open folder" (A1111's ``ui_common``
behaviour): copies of outputs go to ``opts.outdir_save``
through the normal save path (filename pattern, ``on_image_saved``,
``log.csv``)."""

from __future__ import annotations

import csv
import os
import platform
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from typing import Any

from remi.files import FileRecord

LOG_FIELDS = ["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename", "negative_prompt", "sd_model_name", "sd_model_hash"]

OUTPUT_FOLDERS = {
    "txt2img": ("outdir_txt2img_samples", "outdir_samples"),
    "img2img": ("outdir_img2img_samples", "outdir_samples"),
    "extras": ("outdir_extras_samples", "outdir_samples"),
    "grids": ("outdir_txt2img_grids", "outdir_grids"),
    "save": ("outdir_save",),
}


class GalleryError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class SavedParams:
    """What ``FilenameGenerator`` reads from ``p``, from an infotext (as the
    Gradio UI rebuilt it from its JSON); unknown attributes are None."""

    def __init__(self, params: dict[str, Any], record: FileRecord) -> None:
        self.prompt = params.get("Prompt", "")
        self.negative_prompt = params.get("Negative prompt", "")
        self.seed = params.get("Seed")
        self.all_seeds = [self.seed]
        self.all_prompts = [self.prompt]
        self.steps = params.get("Steps")
        self.cfg_scale = params.get("CFG scale")
        self.distilled_cfg_scale = params.get("Distilled CFG Scale") or params.get("Shift")
        self.sampler_name = params.get("Sampler")
        self.scheduler = params.get("Schedule type")
        self.sd_model_name = params.get("Model")
        self.sd_model_hash = params.get("Model hash")
        self.denoising_strength = params.get("Denoising strength")
        self.width = record.width or params.get("Size-1")
        self.height = record.height or params.get("Size-2")
        self.styles: list[str] = []
        self.override_settings: dict[str, Any] = {}
        self.batch_size = self.n_iter = 1
        self.batch_index = self.iteration = 0
        self.user = None

    def __getattr__(self, name: str) -> Any:
        return None


@dataclass
class SaveResult:
    saved: list[str] = field(default_factory=list)
    """file names relative to ``outdir_save``"""
    paths: list[str] = field(default_factory=list)
    zip_path: str | None = None


def update_logfile(path: str, fields: list[str]) -> None:
    """Pad an old ``log.csv`` to the current columns (``ui_common``)."""

    with open(path, encoding="utf8", newline="") as file:
        rows = list(csv.reader(file))
    if not rows or len(rows[0]) == len(fields):
        return
    rows[0] = fields
    for row in rows[1:]:
        row.extend([""] * (len(fields) - len(row)))
    with open(path, "w", encoding="utf8", newline="") as file:
        csv.writer(file).writerows(rows)


def save_outputs(records: list[FileRecord], *, make_zip: bool = False) -> SaveResult:
    """Save copies of ``records`` to ``opts.outdir_save`` (images through
    ``images.save_image``, other files copied), log them to ``log.csv``
    (``save_write_log_csv``) and optionally zip them."""

    from PIL import Image

    from modules import images, infotext_utils, shared

    opts = shared.opts
    folder = opts.outdir_save
    os.makedirs(folder, exist_ok=True)
    result = SaveResult()
    rows: list[list[Any]] = []
    first: tuple[SavedParams, Any] | None = None
    for record in records:
        if not record.is_image:
            target = _free_name(folder, record.name)
            shutil.copy2(record.path, target)
            result.paths.append(target)
            result.saved.append(os.path.relpath(target, folder))
            continue
        params = infotext_utils.parse_generation_parameters(record.infotext or "", [])
        p = SavedParams(params, record)
        with Image.open(record.path) as opened:
            opened.load()
            image = opened.copy()
        fullfn, txt_fullfn = images.save_image(image, folder, "", seed=p.seed, prompt=p.prompt, extension=opts.samples_format, info=record.infotext, p=p, save_to_dirs=opts.use_save_to_dirs_for_ui)
        name = os.path.relpath(fullfn, folder)
        result.paths.append(fullfn)
        result.saved.append(name)
        if txt_fullfn:
            result.paths.append(txt_fullfn)
            result.saved.append(os.path.relpath(txt_fullfn, folder))
        rows.append([p.prompt, p.seed, record.width, record.height, p.sampler_name, p.cfg_scale, p.steps, name, p.negative_prompt, p.sd_model_name, p.sd_model_hash])
        first = first or (p, image)
    if rows and opts.save_write_log_csv:
        log_path = os.path.join(folder, "log.csv")
        if os.path.exists(log_path):
            update_logfile(log_path, LOG_FIELDS)
        with open(log_path, "a", encoding="utf8", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(LOG_FIELDS)
            writer.writerows(rows)
    if make_zip and result.paths:
        result.zip_path = _zip(folder, result, first)
    return result


def _zip(folder: str, result: SaveResult, first: tuple[SavedParams, Any] | None) -> str:
    from modules import images, shared

    name = "images"
    if first is not None:
        p, image = first
        try:
            namegen = images.FilenameGenerator(p, p.seed, p.prompt, image, True)
            name = namegen.apply(shared.opts.grid_zip_filename_pattern or "[datetime]_[[model_name]]_[seed]-[seed_last]") or name
        except Exception:
            pass
    path = _free_name(folder, f"{name}.zip")
    with zipfile.ZipFile(path, "w") as archive:
        for full, relative in zip(result.paths, result.saved, strict=True):
            archive.write(full, relative)
    return path


def _free_name(folder: str, name: str) -> str:
    base, ext = os.path.splitext(os.path.basename(name) or "file")
    candidate = os.path.join(folder, base + ext)
    index = 1
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{base}-{index}{ext}")
        index += 1
    return candidate


def output_folder(kind: str) -> str:
    """The output folder of ``kind`` (a key of :data:`OUTPUT_FOLDERS`)."""

    from modules import shared

    names = OUTPUT_FOLDERS.get(kind)
    if names is None:
        raise GalleryError(f"unknown folder {kind!r}", 404)
    for name in names:
        value = getattr(shared.opts, name, "")
        if value:
            return value
    raise GalleryError("the folder is not set", 404)


def open_folder(path: str) -> None:
    """Open ``path`` in the file manager of the machine running the WebUI
    (also behind ``util.open_folder``)."""

    if not os.path.isdir(path):
        raise GalleryError("the folder does not exist yet (it is created when an image is saved)", 404)
    path = os.path.normpath(path)
    system = platform.system()
    if system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        subprocess.Popen(["explorer.exe", subprocess.check_output(["wslpath", "-w", path], text=True).strip()])
    else:
        subprocess.Popen(["xdg-open", path])
