"""Extra networks and model files: ``/networks/{kind}``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from remi.networks import NetworkError
from remi.server import models as m
from remi.server.deps import ServicesDep, not_found

router = APIRouter()

ERRORS = {404: {"model": m.Detail}}


def _guard(call: Any) -> Any:
    try:
        return call()
    except NetworkError as error:
        raise not_found(str(error)) from error


@router.get("/networks", response_model=list[str], summary="Network kinds")
def kinds(services: ServicesDep) -> list[str]:
    return services.networks.kinds()


@router.get("/networks/{kind}", response_model=m.Page[m.NetworkItem], responses=ERRORS, summary="Search a kind (all terms must match); `sort` is name, path or mtime")
def search(
    kind: str,
    services: ServicesDep,
    q: str | None = None,
    cursor: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    sort: str = Query("name", pattern="^(name|path|mtime)$"),
    order: str = Query("asc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    items, next_cursor, total = _guard(lambda: services.networks.search(kind, query=q, cursor=cursor, limit=limit, sort=sort, descending=order == "desc"))
    return {"items": items, "next_cursor": next_cursor, "total": total}


@router.post("/networks/{kind}/refresh", response_model=m.Ok, responses=ERRORS, summary="Re-scan a kind")
async def refresh(kind: str, services: ServicesDep) -> dict[str, Any]:
    await run_in_threadpool(_guard, lambda: services.networks.refresh(kind))
    services.bus.publish("networks.changed", kind=kind)
    return {"ok": True}


@router.get("/networks/{kind}/{name}/meta", response_model=m.NetworkMeta, responses=ERRORS, summary="An item with its file metadata and user metadata")
def meta(kind: str, name: str, services: ServicesDep) -> dict[str, Any]:
    return _guard(lambda: services.networks.meta(kind, name))


@router.patch("/networks/{kind}/{name}/meta", response_model=m.NetworkMeta, responses={**ERRORS, 422: {"model": m.Detail}}, summary="Edit the user metadata (description, activation text, preferred weight, ...)")
def edit_meta(kind: str, name: str, body: m.UserMetadataPatch, services: ServicesDep) -> dict[str, Any]:
    try:
        result = _guard(lambda: services.networks.update_user_metadata(kind, name, body.user_metadata, replace=body.replace))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    services.bus.publish("networks.changed", kind=kind)
    return result


@router.put("/networks/{kind}/{name}/preview", response_model=m.NetworkItem, responses=ERRORS, summary="Use an uploaded or generated image as the preview")
def set_preview(kind: str, name: str, body: m.PreviewRequest, services: ServicesDep) -> dict[str, Any]:
    from PIL import Image

    record = services.files.get(body.file_id)
    if record is None or not record.is_image:
        raise not_found("no such image")
    with Image.open(record.path) as image:
        image.load()
        result = _guard(lambda: services.networks.set_preview(kind, name, image))
    services.bus.publish("networks.changed", kind=kind)
    return result
