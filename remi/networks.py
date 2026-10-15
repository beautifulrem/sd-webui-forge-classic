"""Extra networks and model files for ``/api/v2/networks``: LoRA (incl.
LyCORIS), Textual Inversion embeddings, checkpoints, VAE and text encoder
modules, and every extension-registered extra-networks page.

Each kind has a provider listing :class:`NetworkEntry` items from the
WebUI's own registries (``networks.available_networks`` of sd_forge_lora,
an embedding database, ``sd_models.checkpoints_list``, the module folders,
``ExtraNetworksPage.list_items()``). :class:`NetworkIndex` keeps one
:class:`Record` per entry (folder, size, date, architecture, user metadata,
search text) until the kind is refreshed, and answers searches (every term
must match), folder / architecture / compatibility filters, sorting, cursor
pages and facet counts (folder tree, architectures). Items get preview file
ids from the :class:`remi.files.FileStore` (WebP thumbnails are served by
``/thumbs``) and a tiny blurred placeholder computed in the background.
Paths are only exposed relative to their model folder.

Extensions add kinds with :func:`register_provider` or, without Gradio,
:func:`register_extra_page` (any object with ``title``, ``list_items()`` and
the ``ExtraNetworksPage`` item dictionaries).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import queue
import re
import stat
import sys
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from remi import network_metadata as nm
from remi.files import FileStore, decode_cursor, encode_cursor

log = logging.getLogger(__name__)

PREVIEW_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "gif", "avif", "jxl", "heif", "heic", "mp4", "webm")
"""Preview files next to a model (images first; as upstream, short videos too)."""
MODEL_EXTENSIONS = (".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".sft", ".gguf")
USER_METADATA_LIMIT = 64 * 1024
PREVIEW_SIZE = 1024
PLACEHOLDER_SIZE = 16
SORTS = ("name", "date", "created", "size", "path")
SORT_ALIASES = {"mtime": "date"}
HIDDEN_MODES = ("Always", "When searched", "Never")
"""``extra_networks_hidden_models``: when items in dot-folders are listed."""
HIDDEN_SEARCH_MIN = 4
""""When searched": the search text must be at least this long."""


# region Kinds and entries


@dataclass(frozen=True)
class KindInfo:
    """How the browser treats a kind's cards."""

    id: str
    title: str
    action: str = "prompt"
    """``prompt`` (insert text), ``checkpoint`` (load the model), ``module``
    (toggle a VAE / text encoder) or ``none``"""
    negative: bool = False
    """cards may go to the negative prompt"""
    fields: tuple[str, ...] = ("description", "tags", "notes")
    """user-metadata keys the editor offers"""
    order: int = 100
    builtin: bool = False


BUILTIN_KINDS: dict[str, KindInfo] = {
    "lora": KindInfo("lora", "LoRA", "prompt", True, ("description", "sd version", "activation text", "preferred weight", "negative text", "tags", "notes"), 0, True),
    "embeddings": KindInfo("embeddings", "Textual Inversion", "prompt", True, ("description", "sd version", "tags", "notes"), 10, True),
    "checkpoints": KindInfo("checkpoints", "Checkpoints", "checkpoint", False, ("description", "sd version", "tags", "notes"), 20, True),
    "vae": KindInfo("vae", "VAE", "module", False, ("description", "tags", "notes"), 30, True),
    "text_encoders": KindInfo("text_encoders", "Text Encoders", "module", False, ("description", "tags", "notes"), 40, True),
}


PromptSource = str | Callable[[Mapping[str, Any]], str | None] | None
"""static text, or a callable of the entry's user metadata (evaluated per request)"""


@dataclass
class NetworkEntry:
    kind: str
    name: str
    """unique within the kind (the API addresses items by it)"""
    filename: str
    title: str = ""
    root: str | None = None
    """the model folder ``filename`` is in (its relative path is shown)"""
    alias: str | None = None
    hash: str | None = None
    prompt: PromptSource = None
    """text a card inserts"""
    negative_prompt: PromptSource = None
    value: str | None = None
    """what applying the card selects (checkpoint title, module file name)"""
    arch: str | None = None
    """architecture known by the provider (else read from the metadata)"""
    metadata: Callable[[], Mapping[str, Any]] | Mapping[str, Any] | None = None
    search_terms: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def base(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def on_disk(self) -> bool:
        """Whether ``filename`` is an absolute path: only then are files
        next to it (sidecar, preview, description) read. Extension items
        without a file must never touch the working directory."""

        return bool(self.filename) and os.path.isabs(self.filename)

    @property
    def relpath(self) -> str:
        if not self.filename:
            return self.name
        if self.root:
            try:
                rel = os.path.relpath(self.filename, self.root)
                if not rel.startswith(".."):
                    return rel.replace("\\", "/")
            except ValueError:
                pass
        return os.path.basename(self.filename)

    @property
    def folder(self) -> str:
        """The relative folder ("" for the model folder itself)."""

        return self.relpath.rpartition("/")[0]

    def read_metadata(self) -> dict[str, Any]:
        source = self.metadata
        if callable(source):
            source = source()
        if not source and self.on_disk and self.filename.lower().endswith(".safetensors"):
            source = read_safetensors_metadata(self.filename)
        return dict(source or {})


def _resolve(source: PromptSource, user: Mapping[str, Any]) -> str | None:
    value = source(user) if callable(source) else source
    return value or None


class Provider(Protocol):
    def list(self) -> Iterable[NetworkEntry]: ...

    def refresh(self) -> None: ...


_providers: dict[str, Provider] = {}
_infos: dict[str, KindInfo] = {}
_extra_pages: list[Any] = []


def register_provider(kind: str, provider: Provider, info: KindInfo | None = None) -> None:
    """Add (or replace) the provider of a network kind."""

    _providers[kind] = provider
    if info is not None:
        _infos[kind] = info


def register_extra_page(page: Any) -> None:
    """Add an extra-networks page without Gradio (``ExtraNetworksPage``
    protocol: ``title``, ``list_items()``, optional ``refresh()``,
    ``allow_negative_prompt``)."""

    if page not in _extra_pages:
        _extra_pages.append(page)


def kind_info(kind: str, provider: Any = None) -> KindInfo:
    info = _infos.get(kind) or getattr(provider, "info", None) or BUILTIN_KINDS.get(kind)
    return info if isinstance(info, KindInfo) else KindInfo(kind, kind.replace("_", " ").title())


def providers() -> dict[str, Provider]:
    """Built-in kinds, registered providers and extension pages."""

    if not _providers:
        register_default_providers()
    found: dict[str, Provider] = dict(_providers)
    for page in extra_pages():
        adapter = ExtraPageProvider.for_page(page)
        if adapter.kind not in found:
            found[adapter.kind] = adapter
    return found


def extra_pages() -> list[Any]:
    """Pages from ``modules.ui_extra_networks`` (when something loaded it;
    never imported here: it imports Gradio) and :func:`register_extra_page`,
    except the built-in LoRA / Textual Inversion / Checkpoints pages."""

    module = sys.modules.get("modules.ui_extra_networks")
    pages = list(getattr(module, "extra_pages", None) or []) + _extra_pages
    builtin = {"lora", "textual inversion", "checkpoints"}
    return [page for page in pages if str(getattr(page, "name", getattr(page, "title", ""))).lower() not in builtin]


# endregion

# region Files next to a model


def read_safetensors_metadata(filename: str) -> dict[str, Any]:
    try:
        with open(filename, "rb") as file:
            size = int.from_bytes(file.read(8), "little")
            if not 0 < size < 64 * 1024 * 1024:
                return {}
            header = json.loads(file.read(size))
    except (OSError, ValueError):
        return {}
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def read_user_metadata(entry: NetworkEntry) -> dict[str, Any]:
    """The JSON sidecar ``<model>.json`` (sd_forge_lora's format)."""

    if not entry.on_disk:
        return {}
    try:
        with open(entry.base + ".json", encoding="utf8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_user_metadata(entry: NetworkEntry, data: dict[str, Any]) -> None:
    require_file(entry)
    text = json.dumps(data, indent=4, ensure_ascii=False)
    if len(text.encode()) > USER_METADATA_LIMIT:
        raise ValueError("user metadata is too large")
    temp = entry.base + ".json.tmp"
    with open(temp, "w", encoding="utf8") as file:
        file.write(text)
    os.replace(temp, entry.base + ".json")


def editable(entry: NetworkEntry) -> bool:
    """Files may be written next to the entry (an absolute path to an
    existing file)."""

    return entry.on_disk and os.path.isfile(entry.filename)


def require_file(entry: NetworkEntry) -> None:
    if not editable(entry):
        raise NetworkReadOnly(f"{entry.name!r} has no model file to keep metadata or a preview next to")


def find_preview(entry: NetworkEntry) -> str | None:
    if not entry.on_disk:
        return None
    for ext in PREVIEW_EXTENSIONS:
        for candidate in (f"{entry.base}.preview.{ext}", f"{entry.base}.{ext}"):
            if os.path.isfile(candidate):
                return candidate
    return None


def find_description(entry: NetworkEntry) -> str | None:
    if not entry.on_disk:
        return None
    for candidate in (f"{entry.base}.txt", f"{entry.base}.description.txt"):
        try:
            with open(candidate, encoding="utf-8", errors="replace") as file:
                return file.read()
        except OSError:
            continue
    return None


def preview_image(image: Any) -> Any:
    """``image`` as a preview: the first frame, RGB or RGBA (CMYK, palette
    and 16-bit modes converted), downscaled to fit ``PREVIEW_SIZE``."""

    from PIL import Image

    if getattr(image, "is_animated", False):
        image.seek(0)
    has_alpha = image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info)
    image = image.convert("RGBA" if has_alpha else "RGB")
    if max(image.size) > PREVIEW_SIZE:
        image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
    return image


def placeholder_data_uri(path: str) -> str | None:
    """A ``PLACEHOLDER_SIZE`` px WebP of the image as a data URI (a few
    hundred bytes; the browser blurs it while the thumbnail loads)."""

    from PIL import Image

    try:
        with Image.open(path) as image:
            if getattr(image, "is_animated", False):
                image.seek(0)
            image.draft("RGB", (PLACEHOLDER_SIZE * 4, PLACEHOLDER_SIZE * 4))
            image = image.convert("RGB")
            image.thumbnail((PLACEHOLDER_SIZE, PLACEHOLDER_SIZE), Image.Resampling.BILINEAR)
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=40, method=2)
    except Exception:
        return None
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()


def _stat(entry: NetworkEntry) -> tuple[int | None, float | None, float | None, bool]:
    """``(size, modified, created, is a file)`` of the model file (one
    ``stat``); created is the birth time where the system keeps one, else
    ctime."""

    if not entry.on_disk:
        return None, None, None, False
    try:
        result = os.stat(entry.filename)
    except OSError:
        return None, None, None, False
    return result.st_size, result.st_mtime, getattr(result, "st_birthtime", None) or result.st_ctime, stat.S_ISREG(result.st_mode)


def is_hidden(relpath: str) -> bool:
    """In a hidden ("."-prefixed) folder, or a hidden file itself."""

    return any(part.startswith(".") for part in relpath.split("/") if part)


# endregion

# region Providers


def _opts() -> Any:
    from modules import shared

    return shared.opts


def _cmd_opts() -> Any:
    from modules import shared

    return shared.cmd_opts


def _root_for(filename: str, roots: Iterable[str | None]) -> str | None:
    real = os.path.abspath(filename)
    for root in sorted({os.path.abspath(root) for root in roots if root}, key=len, reverse=True):
        if real.startswith(root.rstrip(os.sep) + os.sep):
            return root
    return None


def format_weight(weight: float) -> str:
    """``1.0`` → ``1``, ``0.70`` → ``0.7`` (as the old cards wrote them)."""

    return f"{float(weight):g}"


def default_multiplier() -> float:
    try:
        return float(getattr(_opts(), "extra_networks_default_multiplier", 1.0))
    except (TypeError, ValueError):
        return 1.0


class LoraProvider:
    """sd_forge_lora's ``networks.available_networks`` (LoRA and LyCORIS
    files; empty when the extension is not loaded)."""

    @staticmethod
    def _module() -> Any:
        return sys.modules.get("networks")

    def refresh(self) -> None:
        module = self._module()
        if module is not None:
            module.list_available_networks()

    def list(self) -> Iterable[NetworkEntry]:
        module = self._module()
        if module is None:
            return []
        cmd_opts = _cmd_opts()
        roots = [getattr(cmd_opts, "lora_dir", None), *(getattr(cmd_opts, "lora_dirs", None) or [])]
        entries = []
        for name, on_disk in list(module.available_networks.items()):
            entry = NetworkEntry("lora", name, os.path.abspath(on_disk.filename), title=name, root=_root_for(on_disk.filename, roots), hash=on_disk.shorthash or None, metadata=on_disk.metadata)
            entry.alias = on_disk.get_alias()
            entry.prompt = lambda user, on_disk=on_disk: lora_prompt(on_disk.get_alias(), user)
            entry.negative_prompt = lambda user: str(user.get("negative text") or "").strip() or None
            entries.append(entry)
        return entries


def lora_prompt(alias: str, user: Mapping[str, Any]) -> str:
    """``<lora:alias:weight>`` (the preferred weight, else the default
    multiplier) followed by the activation text."""

    weight = nm.preferred_weight(user) or default_multiplier()
    text = f"<lora:{alias}:{format_weight(weight)}>"
    activation = str(user.get("activation text") or "").strip()
    return f"{text} {activation}" if activation else text


_embedding_db: Any = None
_embedding_lock = threading.Lock()


def embedding_database() -> Any:
    """The process' embedding list (loaded once; ``/sdapi/v1/embeddings``
    shares it). The model's own databases live in the text engines and
    exist only while a model is loaded."""

    global _embedding_db
    with _embedding_lock:
        if _embedding_db is None:
            from modules.textual_inversion.textual_inversion import EmbeddingDatabase

            db = EmbeddingDatabase()
            db.add_embedding_dir(_cmd_opts().embeddings_dir)
            db.load_textual_inversion_embeddings(force_reload=True, sync_with_sd_model=False)
            _embedding_db = db
        return _embedding_db


def embedding_arch(embedding: Any) -> str | None:
    """SD 1.x (768-wide vectors) or SDXL (``clip_g`` + ``clip_l``)."""

    vec = getattr(embedding, "vec", None)
    if isinstance(vec, Mapping):
        return "xl" if "clip_g" in vec else None
    return "sd" if getattr(embedding, "shape", None) == 768 else None


class EmbeddingProvider:
    def refresh(self) -> None:
        embedding_database().load_textual_inversion_embeddings(force_reload=True, sync_with_sd_model=False)

    def list(self) -> Iterable[NetworkEntry]:
        db = embedding_database()
        roots = list(db.embedding_dirs)
        return [
            NetworkEntry("embeddings", name, embedding.filename, title=name, root=_root_for(embedding.filename, roots), hash=embedding.shorthash or None, prompt=embedding.name, arch=embedding_arch(embedding))
            for name, embedding in list(db.word_embeddings.items())
            if getattr(embedding, "filename", None)
        ]


class CheckpointProvider:
    def refresh(self) -> None:
        from modules import shared

        shared.refresh_checkpoints()

    def list(self) -> Iterable[NetworkEntry]:
        from modules import sd_models

        roots = [sd_models.model_path, *(_cmd_opts().ckpt_dirs or [])]
        return [
            NetworkEntry("checkpoints", info.name, info.filename, title=info.name_for_extra, root=_root_for(info.filename, roots), hash=info.shorthash, metadata=info.metadata, value=info.title)
            for info in list(sd_models.checkpoints_list.values())
        ]


def module_roots() -> dict[str, list[str]]:
    """The VAE and text encoder folders (``main_entry.refresh_models``)."""

    from modules import paths

    cmd_opts = _cmd_opts()
    return {
        "vae": [os.path.abspath(os.path.join(paths.models_path, "VAE")), *(getattr(cmd_opts, "vae_dirs", None) or [])],
        "text_encoders": [os.path.abspath(os.path.join(paths.models_path, "text_encoder")), *(getattr(cmd_opts, "text_encoder_dirs", None) or [])],
    }


def module_files(kind: str) -> list[tuple[str, str, str]]:
    """``(file name, path, folder)`` of the modules of ``kind``."""

    found: dict[str, tuple[str, str, str]] = {}
    for root in map(os.path.abspath, module_roots()[kind]):
        if not os.path.isdir(root):
            continue
        for current, _dirs, files in os.walk(root):
            for name in files:
                if name.endswith(MODEL_EXTENSIONS):
                    found[name] = (name, os.path.join(current, name), root)
    return sorted(found.values())


class ModuleProvider:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def refresh(self) -> None:
        pass  # listed from the folders on every refresh of the index

    def list(self) -> Iterable[NetworkEntry]:
        return [NetworkEntry(self.kind, name, path, title=os.path.splitext(name)[0], root=root, value=name) for name, path, root in module_files(self.kind)]


_JS_TOKEN = re.compile(r'\s*(?:("(?:[^"\\]|\\.)*")|(opts\.\w+)|(-?\d+(?:\.\d+)?)|(\+))\s*')


def js_text(expression: Any) -> str:
    """The text of an ``ExtraNetworksPage`` prompt: a JavaScript
    concatenation of ``quote_js`` strings, numbers and ``opts.<name>``
    (``"<lora:x:" + opts.extra_networks_default_multiplier + ">"``).
    Anything else is taken literally."""

    if not isinstance(expression, str):
        return "" if expression is None else str(expression)
    parts: list[str] = []
    position = 0
    expect_operand = True
    while position < len(expression):
        match = _JS_TOKEN.match(expression, position)
        if match is None or match.end() == position:
            return expression
        string, option, number, plus = match.groups()
        if plus:
            if expect_operand:
                return expression
            expect_operand = True
        elif not expect_operand:
            return expression
        else:
            expect_operand = False
            if string is not None:
                try:
                    parts.append(json.loads(string))
                except ValueError:
                    return expression
            elif option is not None:
                value = getattr(_opts(), option[5:], "")
                parts.append(format_weight(value) if isinstance(value, float) else str(value))
            else:
                parts.append(number)
        position = match.end()
    return expression if expect_operand and parts else "".join(parts)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "page"


class ExtraPageProvider:
    """An ``ExtraNetworksPage`` of an extension as a kind (generic adapter
    of ``list_items()`` / ``refresh()``)."""

    _adapters: dict[int, ExtraPageProvider] = {}

    def __init__(self, page: Any) -> None:
        self.page = page
        title = str(getattr(page, "title", None) or getattr(page, "name", "Extra networks"))
        self.kind = _slug(str(getattr(page, "extra_networks_tabname", None) or title))
        if self.kind in BUILTIN_KINDS:
            self.kind = f"page_{self.kind}"
        self.info = KindInfo(self.kind, title, "prompt" if getattr(page, "allow_prompt", True) else "none", bool(getattr(page, "allow_negative_prompt", False)))

    @classmethod
    def for_page(cls, page: Any) -> ExtraPageProvider:
        adapter = cls._adapters.get(id(page))
        if adapter is None or adapter.page is not page:
            adapter = cls._adapters[id(page)] = cls(page)
        return adapter

    def refresh(self) -> None:
        refresh = getattr(self.page, "refresh", None)
        if callable(refresh):
            refresh()

    def list(self) -> Iterable[NetworkEntry]:
        roots: list[str] = []
        allowed = getattr(self.page, "allowed_directories_for_previews", None)
        if callable(allowed):
            try:
                roots = [str(root) for root in allowed() or [] if root]
            except Exception:
                log.exception("listing the folders of %s", self.info.title)
        entries: list[NetworkEntry] = []
        seen: set[str] = set()
        try:
            items = list(self.page.list_items())
        except Exception:
            log.exception("listing %s", self.info.title)
            return []
        for item in items:
            if not isinstance(item, Mapping) or not item.get("name"):
                continue
            name = str(item["name"])
            if name in seen:
                continue
            seen.add(name)
            # no file: no sidecar, preview or folder (never relative to the working directory)
            filename = str(item.get("filename") or "")
            if filename and not os.path.isabs(filename):
                filename = os.path.abspath(filename) if os.path.isfile(filename) else ""
            terms = tuple(str(term) for term in item.get("search_terms") or () if term)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else None
            entry = NetworkEntry(self.kind, name, filename, title=name, root=_root_for(filename, roots) if roots else None, hash=item.get("shorthash") or None, metadata=metadata, search_terms=terms)
            entry.prompt = js_text(item.get("prompt")) or None
            entry.negative_prompt = js_text(item.get("negative_prompt")) or None
            if isinstance(item.get("description"), str):
                entry.extra["description"] = item["description"]
            entries.append(entry)
        return entries


def register_default_providers() -> None:
    register_provider("lora", LoraProvider())
    register_provider("embeddings", EmbeddingProvider())
    register_provider("checkpoints", CheckpointProvider())
    register_provider("vae", ModuleProvider("vae"))
    register_provider("text_encoders", ModuleProvider("text_encoders"))


# endregion
# region Index


class NetworkError(LookupError):
    pass


class NetworkReadOnly(Exception):
    """The entry has no model file to keep a sidecar or preview next to."""


@dataclass
class Record:
    """What searching, filtering and sorting need, per entry (kept until
    the kind is refreshed or the entry edited)."""

    entry: NetworkEntry
    user: dict[str, Any]
    folder: str
    relpath: str
    size: int | None
    mtime: float | None
    created: float | None
    sd_version: str
    arch: str | None
    subtype: str | None
    description: str | None
    tags: list[str]
    haystack: str
    sort_name: str
    hidden: bool = False
    """in a "."-folder (``extra_networks_hidden_models`` decides when it is listed)"""
    editable: bool = False
    """a sidecar and preview can be written next to the model file"""


@dataclass(frozen=True)
class Filters:
    query: str | None = None
    folder: str | None = None
    archs: tuple[str, ...] = ()
    """architecture names; ``unknown`` matches items without one"""
    compatible: bool = False
    """only items made for the current model's architecture (or unknown)"""
    hidden: str = "Always"
    """``extra_networks_hidden_models``: ``Always``, ``When searched`` or ``Never``"""

    def terms(self) -> list[str]:
        return [term.casefold() for term in (self.query or "").split()]

    def shows_hidden(self) -> bool:
        if self.hidden == "Never":
            return False
        if self.hidden == "When searched":
            return len((self.query or "").strip()) >= HIDDEN_SEARCH_MIN
        return True


class PlaceholderCache:
    """Blurred-placeholder data URIs by preview file id, computed on a
    background thread and kept in ``<root>/placeholders.json`` (with the
    preview's path, so entries of deleted or replaced previews are pruned)."""

    LIMIT = 50_000
    IDLE_TIMEOUT = 5.0
    """seconds the worker waits for more work before it exits"""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, tuple[str, str]] = {}
        """file id → (data URI or "" when unreadable, source path)"""
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._pending: set[str] = set()
        self._thread: threading.Thread | None = None
        self._dirty = 0
        if path is not None:
            try:
                data = json.loads(path.read_text(encoding="utf8"))
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict):
                # the old format (id → URI) has no paths: those are recomputed
                self._data = {str(key): (str(value[0]), str(value[1])) for key, value in data.items() if isinstance(value, list) and len(value) == 2}

    def get(self, file_id: str, path: str) -> str | None:
        with self._lock:
            found = self._data.get(file_id)
            if found is not None:
                return found[0]
            if file_id in self._pending:
                return None
            self._pending.add(file_id)
            # queued under the lock: the worker only exits after seeing an
            # empty queue under the same lock, so nothing queued is left behind
            self._queue.put((file_id, path))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="remi-network-placeholders", daemon=True)
                self._thread.start()
        return None

    def compute_now(self) -> None:
        """Process the queue on this thread (tests)."""

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            self._process(*item)
        self.flush()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self.IDLE_TIMEOUT)
            except queue.Empty:
                self.flush()
                with self._lock:
                    if self._queue.empty():
                        self._thread = None
                        return
                continue
            self._process(*item)
            if self._dirty >= 64:
                self.flush()

    def _process(self, file_id: str, path: str) -> None:
        uri = placeholder_data_uri(path) or ""
        with self._lock:
            self._pending.discard(file_id)
            if len(self._data) >= self.LIMIT:
                self._data.clear()
            self._data[file_id] = (uri, path)
            self._dirty += 1

    def prune(self) -> int:
        """Forget placeholders whose preview file is gone or was replaced
        (a newer id for the same path); returns how many were dropped."""

        with self._lock:
            items = list(self._data.items())
        latest: dict[str, str] = {}
        for file_id, (_uri, path) in items:
            latest[path] = file_id  # insertion order: the last one is the newest
        gone = [file_id for file_id, (_uri, path) in items if latest[path] != file_id or not os.path.isfile(path)]
        if gone:
            with self._lock:
                for file_id in gone:
                    self._data.pop(file_id, None)
                self._dirty += len(gone)
            self.flush()
        return len(gone)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def flush(self) -> None:
        if self.path is None:
            return
        with self._lock:
            if not self._dirty:
                return
            text = json.dumps({key: list(value) for key, value in self._data.items()})
            self._dirty = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(text, encoding="utf8")
            os.replace(temp, self.path)
        except OSError:
            log.exception("saving network placeholders")


_indexes: weakref.WeakSet[NetworkIndex] = weakref.WeakSet()


def notify_changed(*kinds: str) -> None:
    """A registry changed outside the API (``/sdapi/v1/refresh-loras``):
    every index forgets those kinds, lists them again when asked and tells
    its clients (``networks.changed``)."""

    for index in list(_indexes):
        index.invalidate(*kinds)
        index.changed(*kinds)


def _current_arch() -> str | None:
    try:
        return getattr(_opts(), "forge_preset", None) or None
    except Exception:
        return None


def _option(name: str, default: Any) -> Any:
    try:
        return getattr(_opts(), name, default)
    except Exception:
        return default


def cover_key(filename: str, mtime: float | None) -> str:
    """The cache name of a model's embedded cover (changes with the file)."""

    return hashlib.sha1(f"{os.path.abspath(filename)}\0{mtime}".encode()).hexdigest()[:24]


class NetworkIndex:
    """Cached, searchable lists of every kind."""

    def __init__(
        self,
        files: FileStore,
        provider_map: Callable[[], Mapping[str, Provider]] = providers,
        *,
        current_arch: Callable[[], str | None] = _current_arch,
        option: Callable[[str, Any], Any] = _option,
        on_change: Callable[[str | None], None] | None = None,
    ) -> None:
        """``option(name, default)`` reads a WebUI option; ``on_change(kind)``
        tells clients a kind changed outside the API (None: every kind)."""

        self.files = files
        self._providers = provider_map
        self._current_arch = current_arch
        self._option = option
        self._on_change = on_change
        self._records: dict[str, list[Record]] = {}
        self._items: dict[tuple[str, str], dict[str, Any]] = {}
        """preview ids per entry, until a refresh"""
        self._lock = threading.Lock()
        self._build_locks: dict[str, threading.Lock] = {}
        """one listing per kind at a time (the list and facet requests arrive together)"""
        self._generation: dict[str, int] = {}
        """bumped by :meth:`invalidate`: a listing started before is not kept"""
        self.cache_dir = Path(files.root) / "networks"
        self.placeholders = PlaceholderCache(self.cache_dir / "placeholders.json")
        _indexes.add(self)

    # region Kinds

    def kinds(self) -> list[str]:
        return [info.id for info in self.kind_infos()]

    def kind_infos(self) -> list[KindInfo]:
        infos = []
        for kind, provider in self._providers().items():
            info = kind_info(kind, provider)
            infos.append(info if info.id == kind else KindInfo(kind, info.title, info.action, info.negative, info.fields, info.order, info.builtin))
        return sorted(infos, key=lambda info: (info.order, info.title.casefold()))

    def info(self, kind: str) -> KindInfo:
        return kind_info(kind, self._provider(kind))

    def current_arch(self) -> str | None:
        return self._current_arch()

    def hidden_mode(self) -> str:
        mode = self._option("extra_networks_hidden_models", "When searched")
        return mode if mode in HIDDEN_MODES else "When searched"

    def _provider(self, kind: str) -> Provider:
        provider = self._providers().get(kind)
        if provider is None:
            raise NetworkError(f"unknown network kind {kind!r}")
        return provider

    # endregion

    # region Records

    def entries(self, kind: str) -> list[NetworkEntry]:
        return [record.entry for record in self.records(kind)]

    def records(self, kind: str) -> list[Record]:
        provider = self._provider(kind)
        with self._lock:
            cached = self._records.get(kind)
            if cached is not None:
                return cached
            build_lock = self._build_locks.setdefault(kind, threading.Lock())
        with build_lock:
            with self._lock:
                cached = self._records.get(kind)
                generation = self._generation.get(kind, 0)
            if cached is not None:
                return cached
            built = [self._record(entry) for entry in provider.list()]
            with self._lock:
                if self._generation.get(kind, 0) == generation:
                    self._records[kind] = built
            return built

    @staticmethod
    def _record(entry: NetworkEntry) -> Record:
        user = read_user_metadata(entry)
        try:
            metadata = entry.read_metadata() if entry.metadata is not None or entry.filename.lower().endswith(".safetensors") else {}
        except Exception:
            metadata = {}
        sd_version = str(user.get("sd version") or "Unknown")
        arch = sd_version if sd_version in nm.ARCHS else entry.arch or nm.detect_arch(metadata)
        description = user.get("description") or entry.extra.get("description") or find_description(entry)
        description = description if isinstance(description, str) and description.strip() else None
        tags = nm.parse_tags(user.get("tags"))
        relpath = entry.relpath
        size, mtime, created, is_file = _stat(entry)
        parts = [entry.name, entry.title, relpath, entry.alias or "", entry.hash or "", description or "", str(user.get("activation text") or ""), *tags, *entry.search_terms]
        return Record(
            entry=entry,
            user=user,
            folder=relpath.rpartition("/")[0],
            relpath=relpath,
            size=size,
            mtime=mtime,
            created=created,
            sd_version=sd_version,
            arch=arch,
            subtype=nm.network_type(metadata) if entry.kind == "lora" else None,
            description=description,
            tags=tags,
            haystack="\n".join(parts).casefold(),
            sort_name=(entry.title or entry.name).casefold(),
            hidden=is_hidden(relpath),
            editable=is_file,
        )

    def refresh(self, kind: str | None = None) -> None:
        for name in [kind] if kind else self.kinds():
            self._provider(name).refresh()
            self.invalidate(name)

    def invalidate(self, *kinds: str) -> None:
        """Forget the cached lists of ``kinds`` (their registries changed)."""

        with self._lock:
            for kind in kinds:
                self._records.pop(kind, None)
                self._generation[kind] = self._generation.get(kind, 0) + 1
                for key in [key for key in self._items if key[0] == kind]:
                    del self._items[key]

    def changed(self, *kinds: str) -> None:
        """Tell clients (``on_change``) that ``kinds`` changed."""

        if self._on_change is None:
            return
        for kind in kinds or (None,):
            try:
                self._on_change(kind)
            except Exception:
                log.exception("announcing a network change")

    def _forget(self, entry: NetworkEntry) -> None:
        """Re-read one entry (its user metadata or preview changed)."""

        record = self._record(entry)
        with self._lock:
            self._items.pop((entry.kind, entry.name), None)
            records = self._records.get(entry.kind)
            if records is not None:
                self._records[entry.kind] = [record if existing.entry is entry else existing for existing in records]

    def record(self, kind: str, name: str) -> Record:
        for record in self.records(kind):
            if record.entry.name == name:
                return record
        raise NetworkError(f"no {kind} named {name!r}")

    def entry(self, kind: str, name: str) -> NetworkEntry:
        return self.record(kind, name).entry

    # endregion

    # region Search

    def _filters(self, query: str | None, folder: str | None, archs: Iterable[str], compatible: bool) -> Filters:
        return Filters(query, folder, tuple(archs), compatible, self.hidden_mode())

    def _match(self, records: Iterable[Record], filters: Filters, *, skip: str | None = None) -> list[Record]:
        terms = filters.terms()
        folder = (filters.folder or "").strip("/")
        archs = {arch.lower() for arch in filters.archs}
        current = self.current_arch() if filters.compatible else None
        hidden = filters.shows_hidden()
        out = []
        for record in records:
            if record.hidden and not hidden:
                continue
            if terms and not all(term in record.haystack for term in terms):
                continue
            if skip != "folder" and folder and not (record.folder == folder or record.folder.startswith(folder + "/")):
                continue
            if skip != "arch":
                if archs and (record.arch or "unknown") not in archs:
                    continue
                if current and record.arch not in (None, current):
                    continue
            out.append(record)
        return out

    def search(self, kind: str, *, query: str | None = None, cursor: str | None = None, offset: int | None = None, limit: int = 100, sort: str = "name", descending: bool = False, folder: str | None = None, archs: Iterable[str] = (), compatible: bool = False) -> tuple[list[dict[str, Any]], str | None, int]:
        """A page of matches from ``offset`` (or ``cursor``, the opaque
        form of an offset); the next cursor and the number of matches."""

        filters = self._filters(query, folder, archs, compatible)
        matches = self._match(self.records(kind), filters)
        sort = SORT_ALIASES.get(sort, sort)
        by_name = sorted(matches, key=lambda record: (record.sort_name, record.relpath))
        if sort == "date":
            matches = sorted(by_name, key=lambda record: record.mtime or 0.0, reverse=descending)
        elif sort == "created":
            matches = sorted(by_name, key=lambda record: record.created or 0.0, reverse=descending)
        elif sort == "size":
            matches = sorted(by_name, key=lambda record: record.size or 0, reverse=descending)
        elif sort == "path":
            matches = sorted(by_name, key=lambda record: record.relpath.casefold(), reverse=descending)
        else:
            matches = by_name[::-1] if descending else by_name
        if offset is None:
            position = decode_cursor(cursor, 1)
            offset = int(position[0]) if position and position[0].isdigit() else 0
        offset = max(0, int(offset))
        limit = min(max(int(limit), 1), 500)
        page = matches[offset : offset + limit]
        next_cursor = encode_cursor(offset + limit) if offset + limit < len(matches) else None
        return [self.item(record) for record in page], next_cursor, len(matches)

    def facets(self, kind: str, *, query: str | None = None, folder: str | None = None, archs: Iterable[str] = (), compatible: bool = False) -> dict[str, Any]:
        """Counts for the filter controls: each facet counts the matches
        of every other filter (the folder tree, and its "All" row, ignore
        the folder filter; the architecture counts ignore the architecture
        filters)."""

        records = self.records(kind)
        filters = self._filters(query, folder, archs, compatible)
        in_folders = self._match(records, filters, skip="folder")
        by_arch = self._match(records, filters, skip="arch")
        arch_counts: dict[str, int] = {}
        for record in by_arch:
            key = record.arch or "unknown"
            arch_counts[key] = arch_counts.get(key, 0) + 1
        hide_dirs = not self._option("extra_networks_show_hidden_directories", True)
        return {
            "kind": kind,
            "total": len(self._match(records, filters)),
            "folder_total": len(in_folders),
            "count": len(records),
            "folders": folder_tree((record.folder for record in in_folders), hide_hidden=hide_dirs),
            "archs": [{"id": arch, "count": count} for arch, count in sorted(arch_counts.items(), key=lambda item: (item[0] == "unknown", -item[1], item[0]))],
            "current_arch": self.current_arch(),
        }

    # endregion

    # region Items

    def item(self, record: Record) -> dict[str, Any]:
        entry = record.entry
        key = (entry.kind, entry.name)
        with self._lock:
            files = self._items.get(key)
        if files is None:
            files = self._preview(record)
            with self._lock:
                self._items[key] = files
        user = record.user
        preview_id = files.get("preview")
        placeholder = self.placeholders.get(preview_id, files["preview_path"]) if preview_id and files["preview_type"] == "image" else None
        return {
            "name": entry.name,
            "title": entry.title or entry.name,
            "kind": entry.kind,
            "path": record.relpath,
            "folder": record.folder,
            "preview": preview_id,
            "preview_type": files["preview_type"],
            "placeholder": placeholder or None,
            "description": record.description,
            "prompt": _resolve(entry.prompt, user),
            "negative_prompt": _resolve(entry.negative_prompt, user),
            "activation_text": str(user.get("activation text") or "") or None,
            "preferred_weight": nm.preferred_weight(user),
            "negative_text": str(user.get("negative text") or "") or None,
            "tags": record.tags,
            "search_terms": [term for term in (record.relpath, entry.hash) if term],
            "hash": entry.hash,
            "alias": entry.alias,
            "sd_version": record.sd_version,
            "arch": record.arch,
            "subtype": record.subtype,
            "value": entry.value,
            "mtime": record.mtime,
            "created": record.created,
            "size": record.size,
            "editable": record.editable,
            "hidden": record.hidden,
        }

    def _preview(self, record: Record) -> dict[str, Any]:
        path = find_preview(record.entry) or self._cover(record)
        preview = self.files.register_path(path) if path else None
        kind = None if preview is None else "video" if preview.mime.startswith("video/") else "image"
        return {"preview": preview.id if preview else None, "preview_path": preview.path if preview else None, "preview_type": kind}

    def _cover(self, record: Record) -> str | None:
        """The embedded cover image (``ssmd_cover_images``) extracted once
        into the cache folder (keyed by the model file and its mtime)."""

        entry = record.entry
        if not entry.on_disk or not entry.filename.lower().endswith(".safetensors"):
            return None
        target = self.cache_dir / "covers" / f"{cover_key(entry.filename, record.mtime)}.png"
        if target.is_file():
            return str(target)
        try:
            data = nm.cover_image(entry.read_metadata())
        except Exception:
            data = None
        if not data:
            return None
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                image.load()
                target.parent.mkdir(parents=True, exist_ok=True)
                preview_image(image).save(target, format="PNG", optimize=True)
        except Exception:
            log.warning("unreadable cover image in %s", entry.filename)
            return None
        return str(target)

    def prune_cache(self) -> None:
        """Drop placeholders of previews that are gone or replaced, and
        extracted covers no listed model uses any more (models changed or
        removed). Covers are kept when a kind cannot be listed."""

        self.placeholders.prune()
        covers = self.cache_dir / "covers"
        if not covers.is_dir():
            return
        keep: set[str] = set()
        for kind in self.kinds():
            try:
                records = self.records(kind)
            except Exception:
                log.exception("listing %s to prune the cover cache", kind)
                return
            keep.update(cover_key(record.entry.filename, record.mtime) for record in records if record.entry.on_disk and record.entry.filename.lower().endswith(".safetensors"))
        for path in covers.glob("*.png"):
            if path.stem not in keep:
                path.unlink(missing_ok=True)

    def meta(self, kind: str, name: str) -> dict[str, Any]:
        record = self.record(kind, name)
        metadata = record.entry.read_metadata()
        return {
            **self.item(record),
            "metadata": nm.visible_metadata(metadata),
            "user_metadata": record.user,
            "training": nm.training_info(metadata),
            "dataset_tags": [{"tag": tag, "count": count} for tag, count in nm.build_tags(metadata)[:400]],
        }

    def update_user_metadata(self, kind: str, name: str, values: Mapping[str, Any], *, replace: bool = False) -> dict[str, Any]:
        entry = self.entry(kind, name)
        require_file(entry)
        changes = nm.normalize_user_metadata(values)
        data = {} if replace else read_user_metadata(entry)
        for key, value in changes.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        write_user_metadata(entry, data)
        self._forget(entry)
        return self.meta(kind, name)

    def set_preview(self, kind: str, name: str, image: Any) -> dict[str, Any]:
        """Save ``image`` (PIL) as the entry's preview (``<name>.preview.png``,
        which :func:`find_preview` prefers to any other preview file),
        converted to RGB(A) and at most ``PREVIEW_SIZE`` pixels wide/high."""

        entry = self.entry(kind, name)
        require_file(entry)
        preview_image(image).save(f"{entry.base}.preview.png", optimize=True)
        self._forget(entry)
        return self.item(self.record(kind, name))

    def folder_of(self, kind: str, name: str) -> str:
        """The absolute folder of an item (for "open folder")."""

        entry = self.entry(kind, name)
        require_file(entry)
        return os.path.dirname(entry.filename)

    # endregion


def folder_tree(folders: Iterable[str], *, hide_hidden: bool = False) -> list[dict[str, Any]]:
    """``[{path, name, count, children}]`` from item folders; a node counts
    the items in it and in its subfolders. ``hide_hidden`` leaves out
    "."-folders (their items still count in the visible parents)."""

    counts: dict[str, int] = {}
    for folder in folders:
        if not folder:
            continue
        parts = folder.split("/")
        for depth in range(1, len(parts) + 1):
            if hide_hidden and parts[depth - 1].startswith("."):
                break
            path = "/".join(parts[:depth])
            counts[path] = counts.get(path, 0) + 1
    nodes: dict[str, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []
    for path in sorted(counts, key=lambda path: [part.casefold() for part in path.split("/")]):
        parent, _, name = path.rpartition("/")
        node = {"path": path, "name": name, "count": counts[path], "children": []}
        nodes[path] = node
        (nodes[parent]["children"] if parent else roots).append(node)
    return roots


# endregion
