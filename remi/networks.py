"""Extra networks and model files for ``/api/v2/networks/{kind}``: LoRA,
embeddings, checkpoints, VAE and text encoder modules.

Each kind has a provider that lists :class:`NetworkEntry` items from the
WebUI's own registries (``networks.available_networks`` of the LoRA
extension, an embedding database, ``sd_models.checkpoints_list``, the module
folders). :class:`NetworkIndex` caches the lists, searches and pages them,
and gives previews file ids from the :class:`remi.files.FileStore`; paths
are only exposed relative to their model folder. Extensions add kinds with
:func:`register_provider`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from remi.files import FileStore, decode_cursor, encode_cursor

PREVIEW_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "gif", "avif", "jxl")
MODEL_EXTENSIONS = (".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".sft", ".gguf")
USER_METADATA_LIMIT = 64 * 1024


@dataclass
class NetworkEntry:
    kind: str
    name: str
    filename: str
    title: str = ""
    root: str | None = None
    """the model folder ``filename`` is in (its relative path is shown)"""
    alias: str | None = None
    hash: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    metadata: Callable[[], Mapping[str, Any]] | Mapping[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def base(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def relpath(self) -> str:
        if self.root:
            try:
                rel = os.path.relpath(self.filename, self.root)
                if not rel.startswith(".."):
                    return rel.replace("\\", "/")
            except ValueError:
                pass
        return os.path.basename(self.filename)

    def read_metadata(self) -> dict[str, Any]:
        source = self.metadata
        if callable(source):
            source = source()
        if source is None and self.filename.lower().endswith(".safetensors"):
            source = read_safetensors_metadata(self.filename)
        return dict(source or {})


class Provider(Protocol):
    def list(self) -> Iterable[NetworkEntry]: ...

    def refresh(self) -> None: ...


_providers: dict[str, Provider] = {}


def register_provider(kind: str, provider: Provider) -> None:
    """Add (or replace) the provider of a network kind."""

    _providers[kind] = provider


def providers() -> dict[str, Provider]:
    if not _providers:
        register_default_providers()
    return _providers


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
    try:
        with open(entry.base + ".json", encoding="utf8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_user_metadata(entry: NetworkEntry, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=4, ensure_ascii=False)
    if len(text.encode()) > USER_METADATA_LIMIT:
        raise ValueError("user metadata is too large")
    temp = entry.base + ".json.tmp"
    with open(temp, "w", encoding="utf8") as file:
        file.write(text)
    os.replace(temp, entry.base + ".json")


def find_preview(entry: NetworkEntry) -> str | None:
    for ext in PREVIEW_EXTENSIONS:
        for candidate in (f"{entry.base}.preview.{ext}", f"{entry.base}.{ext}"):
            if os.path.isfile(candidate):
                return candidate
    return None


def find_description(entry: NetworkEntry) -> str | None:
    for candidate in (f"{entry.base}.txt", f"{entry.base}.description.txt"):
        try:
            with open(candidate, encoding="utf-8", errors="replace") as file:
                return file.read()
        except OSError:
            continue
    return None


# region Providers


def _opts() -> Any:
    from modules import shared

    return shared.opts


def _cmd_opts() -> Any:
    from modules import shared

    return shared.cmd_opts


def _root_for(filename: str, roots: Iterable[str]) -> str | None:
    real = os.path.abspath(filename)
    for root in sorted({os.path.abspath(root) for root in roots if root}, key=len, reverse=True):
        if real.startswith(root.rstrip(os.sep) + os.sep):
            return root
    return None


class LoraProvider:
    """The LoRA extension's ``networks.available_networks`` (empty when the
    extension is not loaded)."""

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
        opts, cmd_opts = _opts(), _cmd_opts()
        roots = [getattr(cmd_opts, "lora_dir", None), *(getattr(cmd_opts, "lora_dirs", None) or [])]
        entries = []
        for name, on_disk in list(module.available_networks.items()):
            entry = NetworkEntry("lora", name, on_disk.filename, title=name, root=_root_for(on_disk.filename, roots), alias=on_disk.get_alias(), hash=on_disk.shorthash or None, metadata=on_disk.metadata)
            user = read_user_metadata(entry)
            weight = user.get("preferred weight") or getattr(opts, "extra_networks_default_multiplier", 1.0)
            prompt = f"<lora:{entry.alias}:{weight}>"
            if user.get("activation text"):
                prompt += f" {user['activation text']}"
            entry.prompt = prompt
            entry.negative_prompt = user.get("negative text") or None
            version = user.get("sd version") or "Unknown"
            entry.extra["sd_version"] = version
            if getattr(opts, "lora_preset_filter", False) and version not in ("Unknown", getattr(opts, "forge_preset", None)):
                continue
            entries.append(entry)
        return entries


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


class EmbeddingProvider:
    def refresh(self) -> None:
        embedding_database().load_textual_inversion_embeddings(force_reload=True, sync_with_sd_model=False)

    def list(self) -> Iterable[NetworkEntry]:
        db = embedding_database()
        roots = list(db.embedding_dirs)
        return [NetworkEntry("embeddings", name, embedding.filename, title=name, root=_root_for(embedding.filename, roots), hash=embedding.shorthash or None, prompt=embedding.name) for name, embedding in list(db.word_embeddings.items())]


class CheckpointProvider:
    def refresh(self) -> None:
        from modules import shared

        shared.refresh_checkpoints()

    def list(self) -> Iterable[NetworkEntry]:
        from modules import sd_models

        roots = [sd_models.model_path, *(_cmd_opts().ckpt_dirs or [])]
        return [NetworkEntry("checkpoints", info.name_for_extra, info.filename, title=info.title, root=_root_for(info.filename, roots), hash=info.shorthash, metadata=info.metadata, extra={"model": info.title}) for info in list(sd_models.checkpoints_list.values())]


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
    for root in module_roots()[kind]:
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
        return [NetworkEntry(self.kind, name, path, title=name, root=root) for name, path, root in module_files(self.kind)]


def register_default_providers() -> None:
    register_provider("lora", LoraProvider())
    register_provider("embeddings", EmbeddingProvider())
    register_provider("checkpoints", CheckpointProvider())
    register_provider("vae", ModuleProvider("vae"))
    register_provider("text_encoders", ModuleProvider("text_encoders"))


# endregion


class NetworkError(LookupError):
    pass


class NetworkIndex:
    """Cached, searchable lists of every kind."""

    def __init__(self, files: FileStore, provider_map: Callable[[], dict[str, Provider]] = providers) -> None:
        self.files = files
        self._providers = provider_map
        self._cache: dict[str, list[NetworkEntry]] = {}
        self._items: dict[tuple[str, str], dict[str, Any]] = {}
        """:meth:`item` per entry (file reads and preview ids), until a refresh"""
        self._lock = threading.Lock()

    def kinds(self) -> list[str]:
        return list(self._providers())

    def _provider(self, kind: str) -> Provider:
        provider = self._providers().get(kind)
        if provider is None:
            raise NetworkError(f"unknown network kind {kind!r}")
        return provider

    def entries(self, kind: str) -> list[NetworkEntry]:
        provider = self._provider(kind)
        with self._lock:
            cached = self._cache.get(kind)
        if cached is None:
            cached = list(provider.list())
            with self._lock:
                self._cache[kind] = cached
        return cached

    def refresh(self, kind: str | None = None) -> None:
        for name in [kind] if kind else self.kinds():
            self._provider(name).refresh()
            self.invalidate(name)

    def invalidate(self, *kinds: str) -> None:
        """Forget the cached lists of ``kinds`` (their registries changed)."""

        with self._lock:
            for kind in kinds:
                self._cache.pop(kind, None)
                for key in [key for key in self._items if key[0] == kind]:
                    del self._items[key]

    def _forget(self, entry: NetworkEntry) -> None:
        with self._lock:
            self._cache.pop(entry.kind, None)
            self._items.pop((entry.kind, entry.name), None)

    def entry(self, kind: str, name: str) -> NetworkEntry:
        for entry in self.entries(kind):
            if entry.name == name:
                return entry
        raise NetworkError(f"no {kind} named {name!r}")

    def search(self, kind: str, *, query: str | None = None, cursor: str | None = None, limit: int = 100, sort: str = "name", descending: bool = False) -> tuple[list[dict[str, Any]], str | None, int]:
        terms = [term.lower() for term in (query or "").split()]
        matches = [entry for entry in self.entries(kind) if all(term in self._haystack(entry) for term in terms)]
        key: Callable[[NetworkEntry], Any] = {
            "path": lambda entry: entry.relpath.lower(),
            "mtime": lambda entry: _mtime(entry.filename),
        }.get(sort, lambda entry: entry.name.lower())
        matches.sort(key=key, reverse=descending)
        position = decode_cursor(cursor, 1)
        offset = int(position[0]) if position and position[0].isdigit() else 0
        limit = min(max(int(limit), 1), 500)
        page = matches[offset : offset + limit]
        next_cursor = encode_cursor(offset + limit) if offset + limit < len(matches) else None
        return [self.item(entry) for entry in page], next_cursor, len(matches)

    @staticmethod
    def _haystack(entry: NetworkEntry) -> str:
        cached = entry.extra.get("_haystack")
        if cached is None:
            parts = [entry.name, entry.title, entry.relpath, entry.alias or "", entry.hash or "", entry.prompt or ""]
            cached = entry.extra["_haystack"] = " ".join(parts).lower()
        return cached

    def item(self, entry: NetworkEntry) -> dict[str, Any]:
        key = (entry.kind, entry.name)
        with self._lock:
            cached = self._items.get(key)
        if cached is None:
            cached = self._item(entry)
            with self._lock:
                self._items[key] = cached
        return dict(cached)

    def _item(self, entry: NetworkEntry) -> dict[str, Any]:
        user = read_user_metadata(entry)
        preview_path = find_preview(entry)
        preview = self.files.register_path(preview_path) if preview_path else None
        description = user.get("description") or find_description(entry)
        return {
            "name": entry.name,
            "title": entry.title or entry.name,
            "kind": entry.kind,
            "path": entry.relpath,
            "preview": preview.id if preview else None,
            "description": description or None,
            "prompt": entry.prompt,
            "negative_prompt": entry.negative_prompt,
            "search_terms": [term for term in (entry.relpath, entry.hash) if term],
            "hash": entry.hash,
            "alias": entry.alias,
            "sd_version": entry.extra.get("sd_version"),
            "mtime": _mtime(entry.filename),
            "size": _size(entry.filename),
        }

    def meta(self, kind: str, name: str) -> dict[str, Any]:
        entry = self.entry(kind, name)
        return {**self.item(entry), "metadata": entry.read_metadata(), "user_metadata": read_user_metadata(entry)}

    def update_user_metadata(self, kind: str, name: str, values: Mapping[str, Any], *, replace: bool = False) -> dict[str, Any]:
        entry = self.entry(kind, name)
        data = {} if replace else read_user_metadata(entry)
        for key, value in values.items():
            if value is None:
                data.pop(key, None)
            else:
                data[str(key)] = value
        write_user_metadata(entry, data)
        self._forget(entry)
        return self.meta(kind, name)

    def set_preview(self, kind: str, name: str, image: Any) -> dict[str, Any]:
        """Save ``image`` (PIL) as the entry's preview (``<name>.preview.png``,
        which :func:`find_preview` prefers to any other preview file),
        converted to RGB(A) and at most ``PREVIEW_SIZE`` pixels wide/high."""

        entry = self.entry(kind, name)
        preview_image(image).save(f"{entry.base}.preview.png", optimize=True)
        self._forget(entry)
        return self.item(self.entry(kind, name))


PREVIEW_SIZE = 1024


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


def _mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None
