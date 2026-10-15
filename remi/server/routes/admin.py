"""System administration for the Settings → System views: server status,
licenses (``html/licenses.html`` as structured data), sysinfo validation,
loading the selected model, the weights in memory, hashing every
checkpoint, and settings validation without applying them (settings
import)."""

from __future__ import annotations

import datetime
import html.parser
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException

from remi.server import models as m
from remi.server.deps import ServicesDep, opts

log = logging.getLogger(__name__)

router = APIRouter()


# region Server status


def loaded_model() -> str | None:
    """Title of the checkpoint in memory (None when nothing is loaded)."""

    sd_models = sys.modules.get("modules.sd_models")
    model = getattr(getattr(sd_models, "model_data", None), "sd_model", None)
    info = getattr(model, "sd_checkpoint_info", None)
    return getattr(info, "title", None) or getattr(info, "name", None)


@router.get("/server/status", response_model=m.ServerStatus, summary="What the server allows (restart, extension access, settings changes) and the loaded model")
def server_status() -> dict[str, Any]:
    from modules import restart, shared
    from remi import extensions_admin
    from remi.server.app import version

    return {
        "restartable": restart.is_restartable(),
        "extension_access": extensions_admin.access_allowed(),
        "settings_frozen": bool(getattr(shared.cmd_opts, "freeze_settings", False)),
        "model_loaded": loaded_model(),
        "version": version(),
    }


# endregion

# region Models


@router.post("/models/reload", response_model=m.Ok, responses={422: {"model": m.Detail}}, summary="Load the selected model now (behind generation; also after an unload; `force` loads it again)")
def reload_model(services: ServicesDep, body: m.ModelReloadRequest | None = None) -> dict[str, Any]:
    from remi.server import runtime

    model = getattr(opts(), "sd_model_checkpoint", None) or None
    force = bool(body and body.force)

    services.bus.publish("model.loading", stage="loading", model=model)
    try:
        with services.queue.lock:
            services.queue.dispatch(lambda: runtime.load_selected_model(force=force))
    except Exception as error:
        services.bus.publish("model.loaded", model=model, error=str(error) or type(error).__name__)
        raise HTTPException(status_code=422, detail=f"the model was not loaded: {error}") from error
    services.bus.publish("model.loaded", model=model)
    return {"ok": True, "message": loaded_model()}


def loaded_weights() -> list[dict[str, Any]]:
    """What the memory manager holds (the former "List loaded models")."""

    memory_management = sys.modules.get("backend.memory_management")
    items = []
    for entry in list(getattr(memory_management, "current_loaded_models", None) or []):
        try:
            patcher = entry.model
            inner = getattr(patcher, "model", None) if patcher is not None else None
            memory = int(entry.model_loaded_memory())
            items.append({"name": type(inner).__name__ if inner is not None else "?", "memory_mb": memory // 2**20 if memory > 0 else None, "device": str(entry.device) if entry.device is not None else None})
        except Exception as error:  # a model being unloaded meanwhile
            items.append({"name": f"? ({type(error).__name__})", "memory_mb": None, "device": None})
    return items


@router.get("/models/loaded", response_model=m.LoadedModels, summary="The checkpoint and the weights in memory (UNet, VAE, text encoders) with their size and device")
def loaded_models() -> dict[str, Any]:
    return {"checkpoint": loaded_model(), "items": loaded_weights()}


class CheckpointHasher:
    """Hashes every checkpoint without a hash yet (the former "Calculate hash
    for all checkpoints"), in the background; one run at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"running": False, "done": 0, "total": 0, "failed": [], "current": [], "finished_at": None}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {**self._state, "failed": list(self._state["failed"]), "current": list(self._state["current"])}

    def start(self, threads: int, checkpoints: list[Any] | None = None) -> dict[str, Any]:
        if checkpoints is None:
            from modules import sd_models

            checkpoints = list(sd_models.checkpoints_list.values())
        with self._lock:
            if self._state["running"]:
                raise HTTPException(status_code=409, detail="the checkpoints are being hashed already")
            self._state = {"running": True, "done": 0, "total": len(checkpoints), "failed": [], "current": [], "finished_at": None}
        threading.Thread(target=self._run, args=(checkpoints, threads), name="remi-checkpoint-hashes", daemon=True).start()
        return self.status()

    def _hash(self, checkpoint: Any) -> None:
        name = getattr(checkpoint, "name", None) or str(checkpoint)
        with self._lock:
            self._state["current"].append(name)
        try:
            checkpoint.calculate_shorthash()
        except Exception:
            log.exception("hashing %s", name)
            with self._lock:
                self._state["failed"].append(name)
        finally:
            with self._lock:
                self._state["current"].remove(name)
                self._state["done"] += 1

    def _run(self, checkpoints: list[Any], threads: int) -> None:
        try:
            with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
                list(pool.map(self._hash, checkpoints))
        finally:
            with self._lock:
                self._state["running"] = False
                self._state["finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()


hasher = CheckpointHasher()


@router.get("/models/hashes", response_model=m.HashProgress, summary="Progress of hashing every checkpoint")
def hash_status() -> dict[str, Any]:
    return hasher.status()


@router.post("/models/hashes", response_model=m.HashProgress, status_code=202, responses={409: {"model": m.Detail}}, summary="Hash every checkpoint (sha256, in the background; poll GET for progress)")
def hash_checkpoints(body: m.HashRequest | None = None) -> dict[str, Any]:
    return hasher.start(body.threads if body else 1)


# endregion

# region Licenses


class _LicenseParser(html.parser.HTMLParser):
    """Reads ``html/licenses.html``: ``<h4>`` group titles, then per project
    ``<h2><a href>name</a></h2>``, ``<small>`` description and
    ``<pre id="<id>-license-content" data-license="url">``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.groups: list[dict[str, Any]] = []
        self._tag: str | None = None
        self._entry: dict[str, Any] | None = None
        self._text: list[str] = []
        self._skip = 0

    def _group(self) -> dict[str, Any]:
        if not self.groups:
            self.groups.append({"title": "", "items": []})
        return self.groups[-1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in ("style", "script"):
            self._skip += 1
        elif tag in ("h4", "h2", "small"):
            self._tag, self._text = tag, []
            if tag == "h2":
                self._entry = {"id": "", "name": "", "url": None, "description": "", "license_url": None}
                self._group()["items"].append(self._entry)
        elif tag == "a" and self._tag == "h2" and self._entry is not None:
            href = values.get("href") or ""
            self._entry["url"] = href if href.startswith(("https://", "http://")) else None
        elif tag == "pre" and self._entry is not None:
            ident = values.get("id") or ""
            self._entry["id"] = ident.removesuffix("-license-content")
            url = values.get("data-license") or ""
            self._entry["license_url"] = url if url.startswith("https://") else None

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script"):
            self._skip = max(0, self._skip - 1)
        elif tag == self._tag:
            text = " ".join("".join(self._text).split())
            if tag == "h4":
                self.groups.append({"title": text, "items": []})
            elif tag == "h2" and self._entry is not None:
                self._entry["name"] = text
            elif tag == "small" and self._entry is not None:
                self._entry["description"] = text
            self._tag = None

    def handle_data(self, data: str) -> None:
        if self._tag is not None and not self._skip:
            self._text.append(data)


_licenses_cache: tuple[float, dict[str, Any]] | None = None


def parse_licenses(text: str) -> dict[str, Any]:
    parser = _LicenseParser()
    parser.feed(text)
    parser.close()
    groups = []
    for group in parser.groups:
        items = [item for item in group["items"] if item["name"]]
        for index, item in enumerate(items):
            item["id"] = item["id"] or f"{len(groups)}-{index}"
        if items:
            groups.append({"title": group["title"], "items": items})
    return {"groups": groups}


@router.get("/licenses", response_model=m.Licenses, summary="Third-party code and its licenses (from html/licenses.html)")
def licenses() -> dict[str, Any]:
    global _licenses_cache

    from modules.paths_internal import script_path

    path = os.path.join(script_path, "html", "licenses.html")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {"groups": []}
    if _licenses_cache is None or _licenses_cache[0] != mtime:
        with open(path, encoding="utf-8") as file:
            _licenses_cache = (mtime, parse_licenses(file.read()))
    return _licenses_cache[1]


# endregion

# region Sysinfo and settings


@router.post("/sysinfo/check", response_model=m.SysinfoCheck, summary="Whether a sysinfo file is unmodified (its checksum matches)")
def check_sysinfo(body: m.SysinfoCheckRequest) -> dict[str, Any]:
    from modules import sysinfo

    return {"valid": bool(sysinfo.check(body.text))}


@router.post("/options/validate", response_model=m.Ok, responses={422: {"model": m.ValidationErrorBody}}, summary="Validate setting values without applying them (422 lists every rejected value)")
def validate_options(values: dict[str, Any]) -> dict[str, Any]:
    from modules import shared
    from remi.server.options import coerce_values

    coerce_values(opts(), values, shared.cmd_opts)
    return {"ok": True}


# endregion
