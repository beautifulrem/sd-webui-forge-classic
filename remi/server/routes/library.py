"""The Gallery page (every output, filtered, sorted, starred, packed,
deleted, shown in its folder) and PNG Info (all metadata of an image)."""

from __future__ import annotations

import ipaddress
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import Field

from remi.library import CATEGORIES, SORTS, GalleryQuery, gallery_for, item_json
from remi.server import models as m
from remi.server.deps import ServicesDep, iso, not_found

router = APIRouter()

Category = Literal["txt2img", "img2img", "extras", "grids", "saved", "other"]
Sort = Literal["newest", "oldest", "name", "largest"]
assert set(CATEGORIES) == set(Category.__args__) and set(SORTS) == set(Sort.__args__)  # type: ignore[attr-defined]


class GalleryItem(m.Model):
    id: str
    name: str
    width: int | None = None
    height: int | None = None
    size: int
    mime: str
    created_at: str
    job_id: str | None = None
    tab: str | None = None
    category: Category
    model: str | None = None
    starred: bool = False
    infotext: str | None = None


class FacetCount(m.Model):
    name: str
    count: int


class GalleryFacets(m.Model):
    models: list[FacetCount]
    categories: list[FacetCount]
    oldest: str | None = None
    newest: str | None = None
    total: int
    starred: int


class IdsRequest(m.Model):
    ids: list[str] = Field(..., min_length=1, max_length=2000)


class StarOutputsRequest(IdsRequest):
    starred: bool = True


class DeleteOutputsRequest(IdsRequest):
    delete_files: bool = Field(False, description="also delete the files on disk (else the folder scan skips them from now on)")


class OutputCount(m.Model):
    count: int


class ZipResult(m.Model):
    id: str = Field(description="file id of the zip (download with /files/{id}?download=1)")
    name: str
    count: int


class MetadataEntry(m.Model):
    group: str = Field(description='"Image", "Metadata" (PNG text, comments, ...), "EXIF", "EXIF Exif", "EXIF GPS", ...')
    key: str
    value: str
    kind: Literal["text", "number", "binary"] = "text"


class PngChunk(m.Model):
    type: str
    length: int
    count: int = Field(1, description="consecutive IDAT chunks are merged")
    key: str | None = Field(None, description="keyword of a text chunk")


class ImageMetadata(m.Model):
    name: str
    format: str | None = None
    mode: str
    width: int
    height: int
    size: int
    infotext: str | None = None
    params: dict[str, str] = Field({}, description="the parsed generation parameters (display strings)")
    entries: list[MetadataEntry]
    chunks: list[PngChunk]


class MetadataRequest(m.Model):
    file_id: str


def _instant(value: str | None, name: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"{name}: not an ISO date or time") from error


@router.get("/gallery", response_model=m.Page[GalleryItem], summary="Outputs for the gallery, filtered and sorted")
def gallery(
    services: ServicesDep,
    cursor: str | None = None,
    limit: int = Query(100, ge=1, le=200),
    q: str | None = Query(None, max_length=500, description="every word must appear in the infotext, the file name or the model"),
    category: Category | None = None,
    model: str | None = Query(None, max_length=300),
    since: str | None = Query(None, description="ISO date-time (inclusive)"),
    until: str | None = Query(None, description="ISO date-time (exclusive)"),
    starred: bool | None = None,
    sort: Sort = "newest",
) -> dict[str, Any]:
    query = GalleryQuery(q, category, model, _instant(since, "since"), _instant(until, "until"), starred, sort)
    items, next_cursor, total = gallery_for(services.files).list(query, cursor=cursor, limit=limit)
    return {"items": [item_json(item, iso) for item in items], "next_cursor": next_cursor, "total": total}


@router.get("/gallery/facets", response_model=GalleryFacets, summary="Filter values: models, categories, date bounds")
def gallery_facets(services: ServicesDep) -> dict[str, Any]:
    facets = gallery_for(services.files).facets()
    facets["oldest"], facets["newest"] = iso(facets["oldest"]), iso(facets["newest"])
    return facets


@router.get("/gallery/{file_id}", response_model=GalleryItem, responses={404: {"model": m.Detail}}, summary="One output")
def gallery_item(file_id: str, services: ServicesDep) -> dict[str, Any]:
    item = gallery_for(services.files).get(file_id)
    if item is None:
        raise not_found("no such output")
    return item_json(item, iso)


@router.post("/gallery/star", response_model=OutputCount, summary="Star or unstar outputs")
def star_outputs(body: StarOutputsRequest, services: ServicesDep) -> dict[str, Any]:
    count = gallery_for(services.files).set_starred(body.ids, body.starred)
    services.bus.publish("outputs.changed", ids=body.ids[:200])
    return {"count": count}


@router.post("/gallery/delete", response_model=OutputCount, summary="Remove outputs from the gallery (and from disk with `delete_files`)")
def delete_outputs(body: DeleteOutputsRequest, services: ServicesDep) -> dict[str, Any]:
    count = gallery_for(services.files).delete(body.ids, delete_files=body.delete_files)
    services.bus.publish("outputs.changed", ids=body.ids[:200])
    return {"count": count}


@router.post("/gallery/zip", response_model=ZipResult, responses={404: {"model": m.Detail}}, summary="Pack outputs into a zip to download (kept for an hour)")
async def zip_outputs(body: IdsRequest, services: ServicesDep) -> dict[str, Any]:
    index = gallery_for(services.files)
    try:
        path, count = await run_in_threadpool(index.export_zip, body.ids)
    except ValueError as error:
        raise not_found(str(error)) from error
    record = services.files.register_path(path, kind="export")
    if record is None:
        raise HTTPException(status_code=500, detail="the zip could not be stored")
    return {"id": record.id, "name": record.name, "count": count}


PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded")


def _local(request: Request) -> bool:
    """A browser on the WebUI machine: a loopback peer that came through no
    proxy. A reverse proxy or tunnel on the same machine also connects from
    loopback; it is told apart only by the forwarding headers it adds, so
    one that adds none (``ssh -L``, a plain TCP tunnel) passes as local —
    hide folders with ``--hide-ui-dir-config`` when sharing the WebUI that way."""

    headers = getattr(request, "headers", None) or {}
    if any(name in headers for name in PROXY_HEADERS):
        return False
    host = request.client.host if request.client else ""
    if host in ("testclient", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@router.post("/gallery/{file_id}/reveal", response_model=m.Ok, responses={403: {"model": m.Detail}, 404: {"model": m.Detail}}, summary="Show an output in the file manager of the WebUI machine (local clients only)")
def reveal_output(file_id: str, request: Request, services: ServicesDep) -> dict[str, Any]:
    from modules import shared
    from remi.library import reveal

    if getattr(shared.cmd_opts, "hide_ui_dir_config", False):
        raise HTTPException(status_code=403, detail="folders are hidden (--hide-ui-dir-config)")
    if not _local(request):
        raise HTTPException(status_code=403, detail="only a browser on the WebUI machine can open its folders")
    record = services.files.get(file_id)
    if record is None or record.kind != "output":
        raise not_found("no such output")
    try:
        reveal(record.path)
    except FileNotFoundError as error:
        raise not_found("the file is gone") from error
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"cannot open the file manager: {error}") from error
    return {"ok": True}


@router.post("/pnginfo", response_model=ImageMetadata, responses={404: {"model": m.Detail}, 415: {"model": m.Detail}}, summary="All metadata of an uploaded or generated image (PNG Info)")
async def png_info(body: MetadataRequest, services: ServicesDep) -> dict[str, Any]:
    from remi.image_meta import read_metadata

    record = services.files.get(body.file_id)
    if record is None:
        raise not_found("no such image")
    if not record.is_image:
        raise HTTPException(status_code=415, detail="not an image")
    try:
        return await run_in_threadpool(read_metadata, record.path, record.name)
    except Image.DecompressionBombError as error:  # not an OSError
        raise HTTPException(status_code=415, detail="the image has too many pixels to read") from error
    except OSError as error:
        raise HTTPException(status_code=415, detail=f"cannot read the image: {error}") from error
