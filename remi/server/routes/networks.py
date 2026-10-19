"""Extra networks and model files: ``/networks`` (see ``remi.networks``)."""

from __future__ import annotations

import logging
import threading
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from remi.networks import NetworkError, NetworkReadOnly
from remi.server import models as m
from remi.server import network_models as nm
from remi.server.deps import ServicesDep, not_found, require_local_folders

log = logging.getLogger(__name__)

router = APIRouter()

ERRORS: dict[int | str, dict[str, Any]] = {404: {"model": m.Detail}}
WRITE_ERRORS: dict[int | str, dict[str, Any]] = {**ERRORS, 409: {"model": m.Detail, "description": "the item has no model file to write next to"}}

Folder = Annotated[str | None, Query(description="only items in this folder (relative path) and its subfolders")]
Archs = Annotated[list[str] | None, Query(description="architecture names (``unknown``: items without one); any of them")]
Compatible = Annotated[bool, Query(description="only items for the current UI preset's architecture (or unknown)")]


def _guard(call: Any) -> Any:
    try:
        return call()
    except NetworkError as error:
        raise not_found(str(error)) from error
    except NetworkReadOnly as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _archs(values: list[str] | None) -> list[str]:
    """``?arch=a&arch=b`` or ``?arch=a,b``."""

    return [part.strip() for value in values or () for part in value.split(",") if part.strip()]


@router.get("/networks", response_model=list[nm.NetworkKind], summary="Network kinds (built-in and extension pages), in display order")
def kinds(services: ServicesDep) -> list[dict[str, Any]]:
    return [{"id": info.id, "title": info.title, "action": info.action, "negative": info.negative, "fields": list(info.fields), "builtin": info.builtin} for info in services.networks.kind_infos()]


@router.post("/networks/refresh", response_model=m.Ok, summary="Re-scan every kind (and prune the preview caches)")
async def refresh_all(services: ServicesDep) -> dict[str, Any]:
    await run_in_threadpool(services.networks.refresh)
    services.bus.publish("networks.changed")
    threading.Thread(target=services.networks.prune_cache, name="remi-network-prune", daemon=True).start()
    return {"ok": True}


@router.get("/networks/{kind}", response_model=m.Page[nm.NetworkItem], responses=ERRORS, summary="Search, filter, sort and page a kind (every search term must match)")
def search(
    kind: str,
    services: ServicesDep,
    folder: Folder = None,
    arch: Archs = None,
    compatible: Compatible = False,
    q: str | None = None,
    cursor: str | None = None,
    offset: int | None = Query(None, ge=0, description="start at this match (instead of ``cursor``: clients jump straight to the window they show)"),
    limit: int = Query(100, ge=1, le=500),
    sort: str = Query("name", pattern="^(name|date|created|size|path|mtime)$", description="``date``: modified, ``created``: added; ``mtime`` is ``date``"),
    order: str = Query("asc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    items, next_cursor, total = _guard(lambda: services.networks.search(kind, query=q, cursor=cursor, offset=offset, limit=limit, sort=sort, descending=order == "desc", folder=folder, archs=_archs(arch), compatible=compatible))
    return {"items": items, "next_cursor": next_cursor, "total": total}


@router.get("/networks/{kind}/facets", response_model=nm.NetworkFacets, responses=ERRORS, summary="Folder tree and architecture counts for the filters")
def facets(kind: str, services: ServicesDep, folder: Folder = None, arch: Archs = None, compatible: Compatible = False, q: str | None = None) -> dict[str, Any]:
    return _guard(lambda: services.networks.facets(kind, query=q, folder=folder, archs=_archs(arch), compatible=compatible))


@router.post("/networks/{kind}/refresh", response_model=m.Ok, responses=ERRORS, summary="Re-scan a kind")
async def refresh(kind: str, services: ServicesDep) -> dict[str, Any]:
    await run_in_threadpool(_guard, lambda: services.networks.refresh(kind))
    services.bus.publish("networks.changed", kind=kind)
    return {"ok": True}


@router.get("/networks/{kind}/{name:path}/meta", response_model=nm.NetworkMeta, responses=ERRORS, summary="An item with its file metadata, training facts and dataset tags, and user metadata")
def meta(kind: str, name: str, services: ServicesDep) -> dict[str, Any]:
    return _guard(lambda: services.networks.meta(kind, name))


@router.patch("/networks/{kind}/{name:path}/meta", response_model=nm.NetworkMeta, responses={**WRITE_ERRORS, 422: {"model": m.Detail}}, summary="Edit the user metadata (the `<model>.json` sidecar)")
def edit_meta(kind: str, name: str, body: nm.UserMetadataPatch, services: ServicesDep) -> dict[str, Any]:
    try:
        result = _guard(lambda: services.networks.update_user_metadata(kind, name, body.user_metadata, replace=body.replace))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    services.bus.publish("networks.changed", kind=kind)
    return result


@router.put("/networks/{kind}/{name:path}/preview", response_model=nm.NetworkItem, responses={**WRITE_ERRORS, 422: {"model": m.Detail}}, summary="Use an uploaded or generated image as the preview")
def set_preview(kind: str, name: str, body: nm.PreviewRequest, services: ServicesDep) -> dict[str, Any]:
    from PIL import Image, UnidentifiedImageError

    record = services.files.get(body.file_id)
    if record is None or not record.is_image:
        raise not_found("no such image")
    try:
        image = Image.open(record.path)
        try:
            image.load()
        except BaseException:
            image.close()  # a truncated file: do not leave it open
            raise
    except FileNotFoundError as error:
        raise not_found("no such image") from error
    except (UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=422, detail="the file is not an image the WebUI can read") from error
    except OSError as error:  # truncated or otherwise broken image data
        log.debug("cannot read the preview image: %s", error)  # (the message may name the file)
        raise HTTPException(status_code=422, detail="cannot read the image") from error
    with image:
        result = _guard(lambda: services.networks.set_preview(kind, name, image))
    services.bus.publish("networks.changed", kind=kind)
    return result


@router.post("/networks/{kind}/{name:path}/open-folder", response_model=m.Ok, responses={**WRITE_ERRORS, 403: {"model": m.Detail}}, summary="Open the item's folder in the file manager of the WebUI machine")
def open_folder(kind: str, name: str, services: ServicesDep, request: Request) -> dict[str, Any]:
    from remi.server.gallery import GalleryError
    from remi.server.gallery import open_folder as open_path

    require_local_folders(request)
    folder = _guard(lambda: services.networks.folder_of(kind, name))
    try:
        open_path(folder)
    except GalleryError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error
    return {"ok": True}
