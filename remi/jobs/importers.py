"""Bring the queues of the extensions the Remi Queue replaces into it.

* **Agent Scheduler** (``task_scheduler.sqlite3``): every task becomes a
  job. Pending tasks are queued, finished ones go to the history (with the
  images they saved, when those files still exist), bookmarks become stars.
  A task's core arguments were stored by name (the Gradio handler's argument
  names, or the A1111 API fields); they are renamed to the core form's
  params. Script arguments were stored positionally (a pickled list, slot 0
  the selected script): they are named through the current script layouts
  (``args_from``/``args_to``). Images are indexed only when they are image
  files inside the configured output folders and were not removed from the
  gallery. Arguments past the last slot cannot be named:
  they are kept as opaque values and the job is not replayable (it is never
  run or retried as it is; opening it in its form and queueing that works).
  Pickles are read with an allow-list unpickler (plain data, numpy arrays,
  PIL images), whatever their signature, so a crafted database cannot run
  code.
* **Prompt Queue** (``queue.json``): pending prompts are queued with the
  form's default parameters and that prompt / negative prompt.

Importing twice is a no-op for records already imported. Queued jobs are
added with the queue paused, so they can be reviewed first.

Notes on what could not be imported exactly are :func:`note` dicts: an
English message with ``{{name}}`` placeholders (the SPA translates it like
its own strings, so the messages are in its zh-CN catalog; a test checks)
and the values.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import pickle
import sqlite3
import zlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from remi.jobs.store import CANCELLED, COMPLETED, FAILED, INTERRUPTED, QUEUED, Job

log = logging.getLogger(__name__)

AGENT_SCHEDULER = "agent-scheduler"
PROMPT_QUEUE = "prompt-queue"
SOURCES = (AGENT_SCHEDULER, PROMPT_QUEUE)

_SIGNATURE_MAGIC = b"ASSIG1"
_DIGEST_SIZE = 32
MAX_PICKLE_BYTES = 64 * 1024 * 1024
MAX_REPORT_ITEMS = 500
OPAQUE_PREVIEW = 8

TASK_STATUS = {"pending": QUEUED, "running": INTERRUPTED, "done": COMPLETED, "failed": FAILED, "interrupted": INTERRUPTED}

# The Gradio handlers' argument names that differ from the core form's params.
LEGACY_RENAMES = {
    "prompt_styles": "styles",
    "n_iter": "batch_count",
    "override_settings_texts": "override_settings",
    "selected_scale_tab": "resize_tab",
    "img2img_batch_input_dir": "batch_input_dir",
    "img2img_batch_output_dir": "batch_output_dir",
    "img2img_batch_inpaint_mask_dir": "batch_inpaint_mask_dir",
    "img2img_batch_use_png_info": "batch_use_png_info",
    "img2img_batch_png_info_props": "batch_png_info_props",
    "img2img_batch_png_info_dir": "batch_png_info_dir",
    "img2img_batch_source_type": "batch_source",
    "img2img_batch_upload": "batch_upload",
}
# Image layers of each img2img mode: (init_img, sketch, mask, mask_upload)
IMG2IMG_LAYERS = {
    0: ("init_img", None, None, None),
    1: ("sketch", "sketch_fg", None, None),
    2: ("init_img_with_mask", None, "init_img_with_mask_fg", None),
    3: ("inpaint_color_sketch", "inpaint_color_sketch_fg", None, None),
    4: ("init_img_inpaint", None, None, "init_mask_inpaint"),
}
LEGACY_IMAGE_ARGS = {name for layers in IMG2IMG_LAYERS.values() for name in layers if name} | {"mask_alpha"}
# Handler arguments with no core param (dropped without a note)
IGNORED_ARGS = {"id_task", "request", "mode", "force_enable_hr"}
# A1111 API fields → core params / infotext keys the built-in scripts read
API_RENAMES = {"n_iter": "batch_count"}
API_PASTE = {
    "steps": "Steps",
    "sampler_name": "Sampler",
    "sampler_index": "Sampler",
    "scheduler": "Schedule type",
    "seed": "Seed",
    "subseed": "Variation seed",
    "subseed_strength": "Variation seed strength",
    "seed_resize_from_w": "Seed resize from-1",
    "seed_resize_from_h": "Seed resize from-2",
}
API_IGNORED = {
    "script_name", "script_args", "alwayson_scripts", "send_images", "save_images", "do_not_save_samples", "do_not_save_grid",
    "sampler_index", "init_images", "mask", "include_init_images", "infotext", "force_task_id", "firstpass_image",
    "override_settings_restore_afterwards", "eta", "s_min_uncond", "s_churn", "s_tmax", "s_tmin", "s_noise", "comments",
    "disable_extra_networks", "refiner_checkpoint", "refiner_switch_at", "restore_faces", "tiling", "firstphase_width",
    "firstphase_height", "hr_sampler_index", "latent_mask", "initial_noise_multiplier",
}


Note = dict[str, Any]


def note(message: str, **values: str | int) -> Note:
    """A note for the import report and the job: ``{message, values}``."""

    return {"message": message, "values": values}


def note_text(entry: Note | str) -> str:
    """A note as plain English text (logs)."""

    if isinstance(entry, str):
        return entry
    text = str(entry.get("message", ""))
    for key, value in (entry.get("values") or {}).items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


class ImportFailed(ValueError):
    """An import that cannot start (no database, unreadable file)."""

    def __init__(self, message: str, status: int = 404):
        super().__init__(message)
        self.status = status


@dataclass
class ImportItem:
    source_id: str
    title: str
    outcome: str
    """``queued``, ``history``, ``refused`` (not replayable), ``invalid``
    (queued values the form rejects: kept in the history to edit),
    ``skipped`` (imported before) or ``error``"""
    job_id: str | None = None
    notes: list[Note] = field(default_factory=list)


@dataclass
class ImportReport:
    source: str
    path: str
    total: int = 0
    queued: int = 0
    history: int = 0
    refused: int = 0
    invalid: int = 0
    skipped: int = 0
    errors: int = 0
    paused: bool = False
    items: list[ImportItem] = field(default_factory=list)

    def add(self, item: ImportItem) -> None:
        counter = {"queued": "queued", "history": "history", "refused": "refused", "invalid": "invalid", "skipped": "skipped", "error": "errors"}[item.outcome]
        setattr(self, counter, getattr(self, counter) + 1)
        if len(self.items) < MAX_REPORT_ITEMS and (item.outcome != "skipped" or item.notes):
            self.items.append(item)

    def json(self) -> dict[str, Any]:
        return asdict(self)


# region Safe unpickling


class SafeUnpickler(pickle.Unpickler):
    """Only plain data, numpy arrays and PIL images: any other global (a
    function, a class with side effects) refuses the whole pickle."""

    ALLOWED = {
        ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "complex"), ("builtins", "bytearray"),
        ("builtins", "slice"), ("builtins", "range"), ("collections", "OrderedDict"), ("collections", "defaultdict"),
        ("copyreg", "_reconstructor"), ("builtins", "object"),
        ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    }
    PIL_MODULES = ("PIL.Image", "PIL.PngImagePlugin", "PIL.JpegImagePlugin", "PIL.WebPImagePlugin", "PIL.GifImagePlugin", "PIL.BmpImagePlugin", "PIL.TiffImagePlugin")

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        if module in self.PIL_MODULES and name.endswith(("Image", "ImageFile")) and not name.startswith("_"):
            cls = super().find_class(module, name)
            from PIL import Image

            if isinstance(cls, type) and issubclass(cls, Image.Image):
                return cls
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def safe_loads(data: bytes) -> Any:
    return SafeUnpickler(io.BytesIO(data)).load()


def read_script_params(blob: bytes | None, key: bytes | None) -> tuple[list[Any] | None, list[Note]]:
    """The positional script args stored by the Agent Scheduler (signed or
    not), and notes. None when they cannot be read."""

    notes: list[Note] = []
    if not blob:
        return [], notes
    data = bytes(blob)
    if data.startswith(_SIGNATURE_MAGIC):
        mac, payload = data[len(_SIGNATURE_MAGIC) : len(_SIGNATURE_MAGIC) + _DIGEST_SIZE], data[len(_SIGNATURE_MAGIC) + _DIGEST_SIZE :]
        if key is None or not hmac.compare_digest(mac, hmac.new(key, payload, hashlib.sha256).digest()):
            notes.append(note("The script arguments are not signed by this installation; they were read with a restricted loader."))
        data = payload
    else:
        notes.append(note("The script arguments were saved unsigned; they were read with a restricted loader."))
    try:
        raw = zlib.decompressobj().decompress(data, MAX_PICKLE_BYTES)
        value = safe_loads(raw)
    except Exception as error:  # refused globals, corrupt data
        notes.append(note("The script arguments could not be read ({{error}}).", error=str(error)[:300]))
        return None, notes
    if not isinstance(value, (list, tuple)):
        notes.append(note("The script arguments are not a list."))
        return None, notes
    return list(value), notes


# endregion

# region Values


def describe(value: Any) -> str:
    """A short, safe description of an opaque argument."""

    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            return f"<image {value.width}×{value.height}>"
    except Exception:
        pass
    if isinstance(value, Mapping):
        if value.get("cls") in ("Image", "ndarray", "Tensor"):
            return f"<{value['cls']}>"
        keys = ", ".join(str(key) for key in list(value)[:6])
        return f"{{{keys}{', …' if len(value) > 6 else ''}}}"
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "…"


def deserialize_image(value: Any) -> Any:
    """A PIL image from an Agent Scheduler image dict, a data URL or an
    image (anything else is returned as is)."""

    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, Mapping) and value.get("cls") == "Image":
        data = zlib.decompressobj().decompress(base64.b64decode(value["data"]), MAX_PICKLE_BYTES)
        return Image.frombytes(value["mode"], tuple(value["size"]), data)
    if isinstance(value, Mapping) and value.get("cls") == "ndarray":
        import numpy as np

        data = zlib.decompressobj().decompress(base64.b64decode(value["data"]), MAX_PICKLE_BYTES)
        array = np.frombuffer(data, dtype=np.dtype(value.get("dtype", "uint8"))).reshape(tuple(value["shape"]))
        return Image.fromarray(array.astype("uint8"))
    if isinstance(value, str) and value.startswith("data:image/") and "," in value:
        image = Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1])))
        image.load()
        return image
    try:
        import numpy as np

        if isinstance(value, np.ndarray) and value.ndim in (2, 3):
            return Image.fromarray(value.astype("uint8"))
    except Exception:
        pass
    return value


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return describe(value)


@dataclass
class Mapper:
    """Maps one task's stored arguments to named form values for ``tab``."""

    tab: str
    store_image: Callable[[Any], str]
    """PIL image → upload id"""
    notes: list[Note] = field(default_factory=list)
    opaque: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        from remi.core.tabs import core_layout, runner_for
        from remi.ui import Context

        self.ctx = Context()
        self.layout = core_layout(self.tab)
        self.runner = runner_for(self.tab)

    # Core values

    def core_value(self, name: str, arg: Any) -> Any:
        """The wire value of a processing argument for core param ``name``
        (images are uploaded; a value that does not fit is kept as is, so
        validation reports it)."""

        from remi.ui.nodes import Image as ImageNode
        from remi.ui.nodes import ImageList

        node = self.layout.by_name[name]
        if isinstance(node, ImageNode):
            image = deserialize_image(arg)
            if image is None or isinstance(image, str):
                return image
            return self.store_image(image)
        if isinstance(node, ImageList):
            return []
        try:
            return node.coerce(node.from_arg(arg, self.ctx), self.ctx)
        except Exception:
            return _jsonable(arg)

    def core_values(self, processing: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        unknown: list[str] = []
        for name, arg in processing.items():
            if name in self.layout.by_name:
                values[name] = self.core_value(name, arg)
            elif name not in IGNORED_ARGS:
                unknown.append(name)
        if unknown:
            self.notes.append(note("Not mapped (no such parameter now): {{names}}.", names=", ".join(sorted(unknown))))
        return values

    def legacy_core(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Core values from a Gradio handler's named arguments."""

        processing = {LEGACY_RENAMES.get(name, name): value for name, value in args.items() if name not in LEGACY_IMAGE_ARGS}
        if self.tab == "img2img":
            mode = args.get("mode", 0)
            mode = mode if isinstance(mode, int) else 0
            processing["mode"] = mode
            init, sketch, mask, mask_upload = (args.get(name) if name else None for name in IMG2IMG_LAYERS.get(mode, (None,) * 4))
            if isinstance(init, Mapping) and "image" in init and "cls" not in init:  # the A1111 sketch dict {image, mask}
                init, mask = init.get("image"), mask or init.get("mask")
            processing.update(init_img=init, sketch=sketch, mask=None if mode == 4 else mask, mask_upload=mask_upload if mode == 4 else None)
        return self.core_values(processing)

    def api_core(self, args: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(core values, infotext params for the built-in scripts)`` from
        A1111 API fields."""

        processing: dict[str, Any] = {}
        paste: dict[str, Any] = {}
        unknown: list[str] = []
        for name, value in args.items():
            if name in API_PASTE:
                if value is not None and not (name == "sampler_index" and "sampler_name" in args):
                    paste[API_PASTE[name]] = value
            elif name == "override_settings":
                processing["override_settings"] = self.override_texts(value or {})
            elif name in API_IGNORED:
                continue
            elif API_RENAMES.get(name, name) in self.layout.by_name:
                processing[API_RENAMES.get(name, name)] = value
            elif value not in (None, "", [], {}, 0, False):
                unknown.append(name)
        if self.tab == "img2img":
            images = args.get("init_images") or []
            mask = args.get("mask")
            processing["mode"] = self._mode_index("Inpaint upload" if mask else "img2img")
            processing["init_img"] = images[0] if images else None
            processing["mask_upload"] = mask
            if len(images) > 1:
                self.notes.append(note("Only the first of {{count}} source images was kept.", count=len(images)))
        if unknown:
            self.notes.append(note("Not mapped (no such parameter now): {{names}}.", names=", ".join(sorted(unknown))))
        return self.core_values(processing), paste

    def _mode_index(self, value: str) -> Any:
        node = self.layout.by_name.get("mode")
        try:
            return node.to_arg(value, self.ctx) if node is not None else 0
        except Exception:
            return 0

    def override_texts(self, settings: Mapping[str, Any]) -> list[str]:
        from modules import infotext_utils, shared

        mapping = [(info.infotext, key) for key, info in shared.opts.data_labels.items() if info.infotext] + list(getattr(infotext_utils, "INFOTEXT_TO_SETTING", []))
        by_setting = {setting: label for label, setting in mapping}
        texts, unknown = [], []
        for setting, value in settings.items():
            label = by_setting.get(setting)
            if label is None:
                unknown.append(setting)
            else:
                texts.append(f"{label}: {value}")
        if unknown:
            self.notes.append(note("Setting overrides without an infotext name were dropped: {{names}}.", names=", ".join(sorted(unknown))))
        return texts

    # Script values

    def script_values(self, args: list[Any]) -> tuple[str | None, dict[str, dict[str, Any]]]:
        """``(selected script title, {script_id: values})`` for the full
        positional script args; arguments past the last slot are opaque."""

        if self.runner is None:
            return None, {}
        slots = self.runner.ensure_arg_slots()
        if len(args) > slots:
            extra = args[slots:]
            if any(value is not None for value in extra):
                self.opaque.extend(extra)
                self.notes.append(note("Script arguments past the current parameters could not be mapped: {{count}}.", count=len(extra)))
            args = args[:slots]
        elif len(args) < slots:
            self.notes.append(note("Script arguments were missing; their defaults are used: {{count}}.", count=slots - len(args)))
        args = [deserialize_image(value) if isinstance(value, Mapping) and value.get("cls") in ("Image", "ndarray") else value for value in args]
        selected, values = self.runner.script_values_from_args(args, self.ctx)
        return selected, {owner: {name: self._upload(value) for name, value in owner_values.items()} for owner, owner_values in values.items()}

    def _upload(self, value: Any) -> Any:
        from PIL import Image

        if isinstance(value, Image.Image):
            return self.store_image(value)
        return _jsonable(value)

    def api_script_values(self, args: Mapping[str, Any], paste: Mapping[str, Any]) -> tuple[str | None, dict[str, dict[str, Any]]]:
        if self.runner is None:
            return None, {}
        values, _problems = self.runner.paste_values(dict(paste), self.ctx)
        by_title = {script.title().lower(): script for script in self.runner.scripts}
        for title, entry in (args.get("alwayson_scripts") or {}).items():
            script = by_title.get(str(title).lower())
            if script is None:
                if str(title).lower() != "agent scheduler":
                    self.notes.append(note("The script “{{script}}” is not installed; its arguments were dropped.", script=str(title)))
                continue
            script_args = entry.get("args", []) if isinstance(entry, Mapping) else entry
            self._named_script(script, script_args, values)
        selected = None
        name = args.get("script_name")
        if name:
            script = by_title.get(str(name).lower())
            if script is None:
                self.notes.append(note("The script “{{script}}” is not installed; its arguments were dropped.", script=str(name)))
            else:
                selected = script.title()
                self._named_script(script, args.get("script_args") or [], values)
        return selected, values

    def _named_script(self, script: Any, script_args: Any, values: dict[str, dict[str, Any]]) -> None:
        layout = self.runner.layout_for(script)
        owner = values.setdefault(script.script_id, {})
        if isinstance(script_args, Mapping):
            owner.update({name: self._upload(value) for name, value in script_args.items() if name in layout})
            return
        script_args = list(script_args or [])
        if len(script_args) > len(layout):
            extra = script_args[len(layout) :]
            self.opaque.extend(extra)
            self.notes.append(note("{{script}}: arguments past its parameters could not be mapped: {{count}}.", script=script.title(), count=len(extra)))
            script_args = script_args[: len(layout)]
        owner.update({name: self._upload(value) for name, value in layout.args_to_values(script_args, self.ctx).items()})


# endregion

# region Agent Scheduler


def agent_scheduler_paths() -> list[str]:
    """Where the Agent Scheduler kept its database (first existing wins)."""

    from modules import paths_internal, shared

    configured = getattr(shared.cmd_opts, "agent_scheduler_sqlite_file", None)
    candidates = []
    if configured:
        candidates.append(configured if os.path.isabs(configured) else os.path.join(paths_internal.data_path, configured))
    candidates.append(os.path.join(paths_internal.data_path, "extension-data", "agent-scheduler", "task_scheduler.sqlite3"))
    candidates.append(os.path.join(paths_internal.script_path, "extensions-builtin", "agent-scheduler", "task_scheduler.sqlite3"))
    candidates.append(os.path.join(paths_internal.data_path, "extensions", "agent-scheduler", "task_scheduler.sqlite3"))
    return list(dict.fromkeys(os.path.abspath(path) for path in candidates))


def prompt_queue_paths() -> list[str]:
    from modules import paths_internal

    return [
        os.path.join(paths_internal.data_path, "extension-data", "sd-webui-prompt-queue", "queue.json"),
        os.path.join(paths_internal.script_path, "extensions-builtin", "sd-webui-prompt-queue", "queue.json"),
    ]


def find_source(source: str) -> str | None:
    paths = agent_scheduler_paths() if source == AGENT_SCHEDULER else prompt_queue_paths() if source == PROMPT_QUEUE else []
    return next((path for path in paths if os.path.isfile(path)), None)


def _connect_readonly(path: str) -> sqlite3.Connection:
    uri = "file:" + path.replace("?", "%3F").replace("#", "%23") + "?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def count_agent_scheduler(path: str) -> int:
    try:
        with _connect_readonly(path) as db:
            return int(db.execute("SELECT COUNT(*) FROM task").fetchone()[0])
    except sqlite3.Error:
        return 0


def count_prompt_queue(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as file:
            items = json.load(file).get("items", [])
        return sum(1 for item in items if isinstance(item, dict) and item.get("status") in ("pending", "running"))
    except (OSError, ValueError, AttributeError):
        return 0


def _timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _signing_key(db_path: str) -> bytes | None:
    try:
        with open(os.path.join(os.path.dirname(db_path), "agent_scheduler_signing.key"), "rb") as file:
            key = file.read()
        return key if len(key) >= 32 else None
    except OSError:
        return None


def output_roots() -> list[str]:
    """The configured output folders (real paths): the only places an
    import indexes images from."""

    from remi.server.services import output_dirs

    return [os.path.realpath(path) for path in output_dirs()]


def _resolve_image_path(path: Any, roots: list[str]) -> str | None:
    """The real path of a task's image when it is a file inside one of the
    output folders ``roots`` (a relative path is taken from the WebUI
    folder or the data folder, as the extension saved it)."""

    from modules import paths_internal

    if not isinstance(path, str) or not path or "\x00" in path:
        return None
    bases = [None] if os.path.isabs(path) else [paths_internal.script_path, paths_internal.data_path]
    for base in bases:
        candidate = os.path.realpath(path if base is None else os.path.join(base, path))
        if os.path.isfile(candidate) and any(candidate.startswith(root.rstrip(os.sep) + os.sep) for root in roots):
            return candidate
    return None


def _is_image(path: str) -> bool:
    from remi.files import sniff

    try:
        with open(path, "rb") as file:
            found = sniff(file.read(64), path)
    except OSError:
        return False
    return found is not None and found.kind == "image"


def _summary(core: Mapping[str, Any], tab: str) -> str:
    prompt = core.get("prompt")
    text = prompt if isinstance(prompt, str) and prompt.strip() else tab
    return " ".join(text.split())[:200]


class Importer:
    """Imports one source into ``queue`` (a :class:`remi.jobs.JobQueue`)."""

    def __init__(self, queue: Any, *, user: str | None = None) -> None:
        self.queue = queue
        self.store = queue.store
        self.files = queue.files
        self.user = user

    def store_image(self, image: Any) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return self.files.add_upload_bytes(buffer.getvalue(), "imported.png").id

    def _check(self, tab: str, core: dict, scripts: dict) -> list[dict[str, Any]] | None:
        from remi.ui import ValidationError

        try:
            self.queue.validate(tab, core, scripts)
        except ValidationError as error:
            return error.to_json()["errors"]
        return None

    def _layout_hash(self, tab: str) -> str:
        try:
            return self.queue.hash_layout(tab)
        except Exception:
            return ""

    def _finish(self, report: ImportReport) -> ImportReport:
        self.queue.changed()
        return report

    # Agent Scheduler

    def agent_scheduler(self, path: str | None = None) -> ImportReport:
        path = path or find_source(AGENT_SCHEDULER)
        if not path or not os.path.isfile(path):
            raise ImportFailed("no Agent Scheduler database was found")
        report = ImportReport(AGENT_SCHEDULER, path)
        key = _signing_key(path)
        try:
            with _connect_readonly(path) as db:
                columns = {row[1] for row in db.execute("PRAGMA table_info(task)").fetchall()}
                if not {"id", "type", "params", "status"} <= columns:
                    raise ImportFailed("the database has no Agent Scheduler task table", 422)
                wanted = [name for name in ("id", "name", "type", "params", "script_params", "priority", "status", "result", "bookmarked", "created_at", "updated_at") if name in columns]
                order = "priority" if "priority" in columns else "rowid"
                rows = [dict(row) for row in db.execute(f"SELECT {', '.join(wanted)} FROM task ORDER BY {order} ASC").fetchall()]
        except sqlite3.Error as error:
            raise ImportFailed(f"cannot read the database: {error}", 422) from error
        report.total = len(rows)
        for row in rows:
            source_id = str(row["id"])
            if self.store.imported_job(AGENT_SCHEDULER, source_id):
                report.add(ImportItem(source_id, row.get("name") or source_id, "skipped"))
                continue
            try:
                report.add(self._agent_task(row, key))
            except Exception as error:
                log.exception("importing Agent Scheduler task %s", source_id)
                report.add(ImportItem(source_id, row.get("name") or source_id, "error", notes=[note("The task could not be imported: {{error}}", error=f"{type(error).__name__}: {error}"[:300])]))
        return self._finish(report)

    def _agent_task(self, row: Mapping[str, Any], key: bytes | None) -> ImportItem:
        source_id = str(row["id"])
        params = json.loads(row["params"]) if isinstance(row["params"], str) else {}
        args = params.get("args") or {}
        tab = "img2img" if row.get("type") == "img2img" or params.get("is_img2img") else "txt2img"
        mapper = Mapper(tab, self.store_image)
        if params.get("is_ui", True):
            core = mapper.legacy_core(args)
            script_args, read_notes = read_script_params(row.get("script_params"), key)
            mapper.notes.extend(read_notes)
            if script_args is None:
                selected, scripts = None, {}
                mapper.opaque.append("<unreadable script arguments>")
                mapper.notes.append(note("The script parameters show their defaults; the stored ones could not be read."))
            else:
                selected, scripts = mapper.script_values(script_args)
        else:
            core, paste = mapper.api_core(args)
            selected, scripts = mapper.api_script_values(args, paste)
        if selected and "script" in mapper.layout.by_name:
            core["script"] = selected
        overrides = list(core.get("override_settings") or []) if isinstance(core.get("override_settings"), list) else []
        checkpoint, vae = params.get("checkpoint"), params.get("vae")
        if checkpoint and checkpoint != "System" and not any(text.startswith("Model hash:") for text in overrides):
            overrides.append(f"Model hash: {checkpoint}")
        if vae and vae != "System":
            overrides.append(f"VAE: {vae}")
        if overrides and "override_settings" in mapper.layout.by_name:
            core["override_settings"] = overrides

        status = TASK_STATUS.get(str(row.get("status")), INTERRUPTED)
        replayable = not mapper.opaque
        errors = None
        outcome = "history"
        if status == QUEUED:
            if not replayable:
                status, outcome = CANCELLED, "refused"
            else:
                errors = self._check(tab, core, scripts)
                if errors:
                    status, outcome = CANCELLED, "invalid"
                    mapper.notes.append(note("Some values do not fit the current parameters; edit the job to queue it."))
                else:
                    outcome = "queued"

        created = _timestamp(row.get("created_at"))
        finished = _timestamp(row.get("updated_at")) if status not in (QUEUED,) else None
        outputs, infotext, error_text = [], None, None
        result = row.get("result")
        if status == COMPLETED and result:
            noted = len(mapper.notes)
            outputs, infotext = self._agent_outputs(result, tab, mapper.notes)
            if not outputs and len(mapper.notes) == noted:
                mapper.notes.append(note("The images of this task are no longer on disk."))
        elif status in (FAILED, INTERRUPTED) and result:
            error_text = str(result)[:2000]
        # "refused" / "invalid" carry no error text: the job's notes (and ``errors``) explain them.

        meta: dict[str, Any] = {"source": AGENT_SCHEDULER, "source_id": source_id}
        if row.get("name"):
            meta["name"] = str(row["name"])[:200]
        if mapper.notes:
            meta["notes"] = mapper.notes
        if mapper.opaque:
            meta["opaque"] = {"count": len(mapper.opaque), "preview": [describe(value) for value in mapper.opaque[:OPAQUE_PREVIEW]]}
        job = Job(
            id="", seq=0, tab=tab, core=core, scripts=scripts, layout_hash=self._layout_hash(tab), status=status,
            created=created or 0.0, started=None, finished=finished, outputs=outputs, infotext=infotext, error=error_text,
            errors=errors, summary=_summary(core, tab), user=self.user, starred=bool(row.get("bookmarked")),
            replayable=replayable, meta=meta,
        )
        added = self.queue.add_imported(job, source=AGENT_SCHEDULER, source_id=source_id)
        return ImportItem(source_id, meta.get("name") or job.summary, outcome, added.id, list(mapper.notes))

    def _agent_outputs(self, result: str, tab: str, notes: list[Note] | None = None) -> tuple[list[dict[str, Any]], str | None]:
        try:
            data = json.loads(result)
        except ValueError:
            return [], None
        geninfo = data.get("geninfo") if isinstance(data, dict) else None
        infotexts = geninfo.get("infotexts") if isinstance(geninfo, dict) else None
        infotexts = [text for text in infotexts or [] if isinstance(text, str)]
        outputs = []
        roots = output_roots()
        skipped = 0
        paths = data.get("images") if isinstance(data, dict) else None
        for index, path in enumerate(paths if isinstance(paths, list) else []):
            real = _resolve_image_path(path, roots)
            if real is None:
                continue  # gone from disk (or never an output: not counted, the note below covers it)
            if not _is_image(real) or self.files.is_removed(real):
                skipped += 1  # not an image, or the user took it out of the gallery
                continue
            text = infotexts[min(index, len(infotexts) - 1)] if infotexts else None
            try:
                record = self.files.register_output(real, tab=tab, infotext=text)
            except OSError:
                continue
            outputs.append({"id": record.id, "kind": "image", "width": record.width, "height": record.height, "infotext": text})
        if skipped and notes is not None:
            notes.append(note("Images removed from the gallery, or that are not images, were left out: {{count}}.", count=skipped))
        return outputs, infotexts[0] if infotexts else None

    # Prompt Queue

    def prompt_queue(self, path: str | None = None) -> ImportReport:
        path = path or find_source(PROMPT_QUEUE)
        if not path or not os.path.isfile(path):
            raise ImportFailed("no Prompt Queue file was found")
        try:
            with open(path, encoding="utf-8") as file:
                items = json.load(file).get("items", [])
        except (OSError, ValueError, AttributeError) as error:
            raise ImportFailed(f"cannot read the queue file: {error}", 422) from error
        report = ImportReport(PROMPT_QUEUE, path)
        items = [item for item in items if isinstance(item, dict)]
        report.total = len(items)
        for item in items:
            source_id = str(item.get("id") or hashlib.sha1(json.dumps(item, sort_keys=True).encode()).hexdigest()[:12])
            title = " ".join(str(item.get("prompt") or "").split())[:80] or source_id
            if item.get("status") not in ("pending", "running"):
                report.add(ImportItem(source_id, title, "skipped", notes=[note("Finished prompts are not imported.")]))
                continue
            if self.store.imported_job(PROMPT_QUEUE, source_id):
                report.add(ImportItem(source_id, title, "skipped"))
                continue
            tab = item.get("tab") if item.get("tab") in ("txt2img", "img2img") else "txt2img"
            core = {"prompt": str(item.get("prompt") or ""), "negative_prompt": str(item.get("negative") or "")}
            notes = [note("Queued with the form's default parameters and this prompt.")]
            if item.get("anchor_enabled") and str(item.get("anchor") or "").strip():
                separator = item.get("anchor_separator") if isinstance(item.get("anchor_separator"), str) else ", "
                core["prompt"] = separator.join(part for part in (str(item["anchor"]).strip(), core["prompt"].strip()) if part)
                notes.append(note("The prompt anchor was written into the prompt."))
            errors = self._check(tab, core, {})
            status, outcome = (CANCELLED, "invalid") if errors else (QUEUED, "queued")
            if errors:
                notes.append(note("Some values do not fit the current parameters; edit the job to queue it."))
            job = Job(
                id="", seq=0, tab=tab, core=core, scripts={}, layout_hash=self._layout_hash(tab), status=status,
                created=_timestamp(item.get("created")) or 0.0, finished=None if status == QUEUED else _timestamp(item.get("created")),
                errors=errors,
                summary=_summary(core, tab), user=self.user, meta={"source": PROMPT_QUEUE, "source_id": source_id, "notes": notes},
            )
            added = self.queue.add_imported(job, source=PROMPT_QUEUE, source_id=source_id)
            report.add(ImportItem(source_id, title, outcome, added.id, notes))
        return self._finish(report)

    def run(self, source: str) -> ImportReport:
        """Import ``source``. The queue is paused first, so imported jobs do
        not start before the user reviewed them; it is resumed when nothing
        was queued and it was running."""

        if source not in SOURCES:
            raise ImportFailed(f"unknown source {source!r}", 422)
        was_paused = self.queue.paused
        if not was_paused:
            self.queue.set_paused(True)
        report: ImportReport | None = None
        try:
            report = self.agent_scheduler() if source == AGENT_SCHEDULER else self.prompt_queue()
            report.paused = not was_paused and report.queued > 0
            return report
        finally:
            if not was_paused and (report is None or not report.paused):
                self.queue.set_paused(False)


def sources() -> list[dict[str, Any]]:
    """What can be imported: ``{id, label, path, available, count}``."""

    found = []
    for source, label, counter in ((AGENT_SCHEDULER, "Agent Scheduler", count_agent_scheduler), (PROMPT_QUEUE, "Prompt Queue", count_prompt_queue)):
        path = find_source(source)
        found.append({"id": source, "label": label, "path": path, "available": path is not None, "count": counter(path) if path else 0})
    return found
