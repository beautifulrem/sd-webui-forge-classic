"""Uploads, files and thumbnails by opaque id, and the outputs gallery."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.responses import FileResponse

from remi.files import FileError, FileRecord
from remi.server import models as m
from remi.server.deps import ServicesDep, iso, not_found

router = APIRouter()

UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {"multipart/form-data": {"schema": {"type": "object", "required": ["file"], "properties": {"file": {"type": "string", "format": "binary"}}}}},
    }
}
FILE_CACHE = "private, max-age=31536000, immutable"
REVALIDATE = "private, no-cache"
"""network previews: the file behind an id may change on disk"""
FORM_OVERHEAD = 64 * 1024
"""multipart boundaries and headers allowed on top of the upload limit"""


class BodyTooLarge(MultiPartException):
    """(a MultiPartException, so the parser closes its temporary files)"""

    def __init__(self) -> None:
        super().__init__("the body is too large")


async def limited(stream: AsyncIterator[bytes], limit: int) -> AsyncIterator[bytes]:
    """``stream``, raising :class:`BodyTooLarge` past ``limit`` bytes (also
    for chunked bodies, which have no ``Content-Length``)."""

    size = 0
    async for chunk in stream:
        size += len(chunk)
        if size > limit:
            raise BodyTooLarge()
        yield chunk


def upload_json(record: FileRecord) -> dict[str, Any]:
    return {"id": record.id, "width": record.width, "height": record.height, "name": record.name, "mime": record.mime, "size": record.size}


@router.post("/uploads", response_model=m.Upload, status_code=201, openapi_extra=UPLOAD_BODY, responses={413: {"model": m.Detail}, 415: {"model": m.Detail}}, summary="Upload an image or file (multipart field `file`)")
async def upload(request: Request, services: ServicesDep) -> dict[str, Any]:
    limit = services.files.max_upload_bytes
    too_large = HTTPException(status_code=413, detail=f"the file is larger than {limit // (1024 * 1024)} MB")
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit + FORM_OVERHEAD:
        raise too_large
    if not (request.headers.get("content-type") or "").startswith("multipart/form-data"):
        raise HTTPException(status_code=422, detail="the form needs a `file` field (multipart/form-data)")
    # Parse the body ourselves: Request.form() spools file parts of any size
    parser = MultiPartParser(request.headers, limited(request.stream(), limit + FORM_OVERHEAD), max_files=1, max_fields=8)
    try:
        form = await parser.parse()
    except BodyTooLarge as error:
        raise too_large from error
    except MultiPartException as error:
        raise HTTPException(status_code=400, detail=f"malformed upload: {error.message}") from error
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(status_code=422, detail="the form needs a `file` field")

        def store() -> FileRecord:
            return services.files.add_upload(iter(lambda: file.file.read(1024 * 1024), b""), file.filename)

        try:
            record = await run_in_threadpool(store)
        except FileError as error:
            raise HTTPException(status_code=error.status, detail=str(error)) from error
    finally:
        for _key, value in form.multi_items():
            if isinstance(value, UploadFile):
                await value.close()
    return upload_json(record)


def _record(services: Any, file_id: str) -> FileRecord:
    record = services.files.get(file_id)
    if record is None:
        raise not_found("no such file")
    return record


def _cache(record: FileRecord) -> str:
    return REVALIDATE if record.kind == "network" else FILE_CACHE


def _etag(record: FileRecord, extra: str = "") -> str:
    stat = os.stat(record.path)
    return f'"{record.id}-{int(stat.st_mtime)}-{stat.st_size}{extra}"'


@router.get("/files/{file_id}", response_class=FileResponse, responses={200: {"content": {"application/octet-stream": {}}}, 304: {}, 404: {"model": m.Detail}}, summary="A file by id")
def get_file(file_id: str, services: ServicesDep, download: bool = False, if_none_match: str | None = Header(None)) -> Response:
    record = _record(services, file_id)
    etag = _etag(record)
    headers = {"ETag": etag, "Cache-Control": _cache(record), "X-Content-Type-Options": "nosniff"}
    if if_none_match == etag:
        return Response(status_code=304, headers=headers)
    media_type = record.mime if record.mime.startswith(("image/", "video/")) else "application/octet-stream"
    disposition = "attachment" if download or media_type == "application/octet-stream" else "inline"
    return FileResponse(record.path, media_type=media_type, filename=record.name, content_disposition_type=disposition, headers=headers)


@router.get("/thumbs/{file_id}", response_class=FileResponse, responses={200: {"content": {"image/webp": {}}}, 304: {}, 404: {"model": m.Detail}, 415: {"model": m.Detail}}, summary="A WebP thumbnail (width snapped to a fixed set)")
def get_thumb(file_id: str, services: ServicesDep, w: int | None = Query(None, ge=1, le=4096), if_none_match: str | None = Header(None)) -> Response:
    from remi.files import snap_width

    record = _record(services, file_id)
    width = snap_width(w)
    etag = _etag(record, f"-w{width}")
    headers = {"ETag": etag, "Cache-Control": _cache(record), "X-Content-Type-Options": "nosniff"}
    if if_none_match == etag:
        return Response(status_code=304, headers=headers)
    try:
        path = services.files.thumbnail(record, width)
    except FileError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=415, detail="cannot read the image") from error
    return FileResponse(path, media_type="image/webp", headers=headers)


def output_json(record: FileRecord) -> dict[str, Any]:
    kind = "image" if record.is_image else "video" if record.mime.startswith("video/") else "file"
    return {"id": record.id, "kind": kind, "name": record.name, "width": record.width, "height": record.height, "infotext": record.infotext, "created_at": iso(record.created), "job_id": record.job_id, "tab": record.tab}


@router.get("/outputs", response_model=m.Page[m.OutputItem], summary="Generated images, newest first; `q` searches infotext and names")
def list_outputs(services: ServicesDep, cursor: str | None = None, q: str | None = None, tab: str | None = None, limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    records, next_cursor, total = services.files.list_outputs(cursor=cursor, limit=limit, query=q, tab=tab)
    return {"items": [output_json(record) for record in records], "next_cursor": next_cursor, "total": total}


@router.delete("/outputs/{file_id}", status_code=204, responses={404: {"model": m.Detail}}, summary="Remove an output from the gallery (`delete_file` also deletes it on disk)")
def delete_output(file_id: str, services: ServicesDep, delete_file: bool = False) -> Response:
    record = _record(services, file_id)
    if record.kind != "output":
        raise not_found("no such output")
    services.files.delete(file_id, delete_file=delete_file)
    return Response(status_code=204)


@router.post("/outputs/save", response_model=m.SaveResult, responses={404: {"model": m.Detail}}, summary='Save copies to the "Save" folder (`outdir_save`, log.csv); `zip` also packs them')
def save_outputs(body: m.SaveRequest, services: ServicesDep) -> dict[str, Any]:
    from remi.server.gallery import save_outputs as save

    records = [_record(services, file_id) for file_id in dict.fromkeys(body.ids)]
    result = save(records, make_zip=body.zip)
    archive = services.files.register_path(result.zip_path, kind="export") if result.zip_path else None
    return {"saved": result.saved, "zip": archive.id if archive else None}


@router.post("/outputs/open-folder", response_model=m.Ok, responses={403: {"model": m.Detail}, 404: {"model": m.Detail}}, summary="Open an output folder in the file manager of the WebUI machine")
def open_output_folder(body: m.OpenFolderRequest) -> dict[str, Any]:
    from modules import shared
    from remi.server.gallery import GalleryError, open_folder, output_folder

    if shared.cmd_opts.hide_ui_dir_config:
        raise HTTPException(status_code=403, detail="folders are hidden (--hide-ui-dir-config)")
    try:
        open_folder(output_folder(body.folder))
    except GalleryError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error
    return {"ok": True}


@router.post("/outputs/refresh", response_model=m.ScanResult, summary="Index images in the output folders")
def refresh_outputs(services: ServicesDep) -> dict[str, Any]:
    from remi.server.services import output_dirs

    return {"added": services.files.scan_outputs(output_dirs())}
