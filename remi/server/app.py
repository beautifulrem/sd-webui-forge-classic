"""The FastAPI application of the Remi WebUI (no Gradio on this path)."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from remi.files import FileError
from remi.jobs import JobError
from remi.server.auth import Auth, SecurityMiddleware, credentials_from_cmd_opts, load_secret
from remi.server.services import Services, default_services
from remi.server.spa import Spa
from remi.server.ws import Hub
from remi.ui import ValidationError

log = logging.getLogger(__name__)

API_PREFIX = "/api/v2"


def version() -> str:
    try:
        from modules_forge import forge_version

        return f"{forge_version.version} {forge_version.release}"
    except Exception:
        return "remi"


def auth_from_cmd_opts(root: Path) -> Auth:
    from modules.shared_cmd_options import cmd_opts

    credentials = credentials_from_cmd_opts(cmd_opts)
    secure = bool(cmd_opts.tls_keyfile and cmd_opts.tls_certfile)
    path = f"/{cmd_opts.subpath.strip('/')}" if cmd_opts.subpath else "/"
    if not credentials:
        return Auth(cookie_path=path)
    return Auth(credentials, load_secret(root / "secret.key"), secure_cookies=secure, cookie_path=path, revoked_path=root / "revoked-sessions.json")


def _phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def handle_exception(request: Request, error: Exception) -> JSONResponse:
    """Unhandled errors as JSON. A server error (5xx) answers only a generic
    message and a correlation id; the exception, its message and any body it
    carries (a ``ResponseValidationError`` holds the whole response) go to
    the server log under that id, never to the client. An exception that
    carries a 4xx ``status_code`` is a client error: its ``detail`` is
    meant for the client and is kept."""

    status = getattr(error, "status_code", None)
    status = status if isinstance(status, int) and 400 <= status < 600 else 500
    if status < 500:
        detail = getattr(error, "detail", None)
        return JSONResponse(status_code=status, content={"detail": detail if isinstance(detail, str) and detail else _phrase(status)})
    error_id = secrets.token_hex(6)
    log.error("%s %s failed (error id %s)", request.method, request.url.path, error_id, exc_info=error)
    return JSONResponse(status_code=status, content={"error": _phrase(status), "detail": f"Internal server error (id {error_id}); the details are in the server log", "id": error_id})


def _validation_error(request: Request, error: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content=error.to_json())


def _job_error(request: Request, error: JobError) -> JSONResponse:
    return JSONResponse(status_code=error.status, content={"detail": str(error)})


def _file_error(request: Request, error: FileError) -> JSONResponse:
    return JSONResponse(status_code=error.status, content={"detail": str(error)})


def snapshot_events(services: Services) -> list[dict[str, Any]]:
    """What a client connecting mid-job needs: the queue and the running job."""

    from remi.realtime import progress_snapshot

    events: list[dict[str, Any]] = [{"type": "queue.changed", **services.queue.status()}]
    ref = services.queue.current_ref()
    if ref is not None:
        events.append({"type": "job.started", **ref.fields()})
        try:
            from modules import shared

            events.append({"type": "job.progress", **ref.fields(), **progress_snapshot(shared.state)})
        except Exception:
            pass
    return events


def create_app(
    *,
    services: Services | None = None,
    auth: Auth | None = None,
    webui: bool = True,
    sdapi: bool | None = None,
    spa_dir: Path | None = None,
    start_services: bool = True,
) -> FastAPI:
    """The app: ``/api/v2`` (REST + WebSocket), ``/internal``, ``/sdapi/v1``
    (``sdapi``; default: ``--api`` or ``--nowebui``) and, with ``webui``,
    the SPA from ``webui/dist``. ``services`` default to the process-wide
    ones (worker, file index, ...), started with the app."""

    services = services or default_services()
    auth = auth or auth_from_cmd_opts(services.root)
    if sdapi is None:
        from modules.shared_cmd_options import cmd_opts

        sdapi = bool(cmd_opts.api or cmd_opts.nowebui)
    hub = Hub(version=version(), snapshot=lambda: snapshot_events(services))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hub.bind(asyncio.get_running_loop())
        unsubscribe = services.bus.subscribe(hub)
        if start_services:
            services.start()
        try:
            yield
        finally:
            unsubscribe()
            await hub.close_all()

    app = FastAPI(
        title="Remi WebUI",
        version=version(),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        exception_handlers={Exception: handle_exception, ValidationError: _validation_error, JobError: _job_error, FileError: _file_error},
    )
    app.state.services = services
    app.state.auth = auth
    app.state.hub = hub
    app.state.sdapi = sdapi
    # extension_route_guard asks the app who is logged in
    app.auth_dependency = auth.user_for  # type: ignore[attr-defined]

    from remi.server.routes import admin, core, extensions, files, jobs, library, networks, prompt_tools, queue_tools, system

    api = APIRouter(prefix=API_PREFIX)
    for module in (core, jobs, files, networks, extensions, system, admin, queue_tools, library, prompt_tools):
        api.include_router(module.router)
    app.include_router(api)
    app.include_router(system.internal, prefix="/internal")
    system.add_progress_routes(app)

    if sdapi:
        from modules.api.api import Api
        from modules.call_queue import queue_lock

        Api(app, queue_lock)

    app.add_middleware(SecurityMiddleware, auth=auth)
    from modules import initialize_util

    initialize_util.setup_middleware(app)

    if webui:
        from modules import shared

        app.router.default = Spa(auth, spa_dir, theme=getattr(shared.cmd_opts, "theme", None))  # type: ignore[assignment]
    return app
