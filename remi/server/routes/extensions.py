"""Extensions management (the former "Extensions" tab): the installed list,
enabling/disabling, update checks and updates, installing from a git URL or
the extension index, uninstalling, and config-state backups. The logic lives
in :mod:`remi.extensions_admin`; changes load with the next "Reload UI"
(``/server/reload``) or restart.

Everything that writes (and the index download, a URL the client names) is
refused with 403 when extension access is disabled (``--listen`` & co.
without ``--enable-insecure-extension-access``): an extension is arbitrary
code. Git and file work runs off the event loop.

Errors are ``{"detail", "template", "params"}`` (:class:`AdminRoute`):
``detail`` is the English message, ``template`` its translatable text with
``{{name}}`` placeholders.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from remi import extensions_admin as admin
from remi.server import models as m


class AdminRoute(APIRoute):
    """Answers :class:`~remi.extensions_admin.AdminError` with its status and
    ``{"detail", "template", "params"}``."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route(request: Request) -> Response:
            try:
                return await handler(request)
            except admin.AdminError as error:
                return JSONResponse(status_code=error.status, content=error.to_json())

        return route


router = APIRouter(route_class=AdminRoute)

ERRORS: dict[int | str, dict[str, Any]] = {403: {"model": m.AdminErrorBody}, 404: {"model": m.AdminErrorBody}, 422: {"model": m.AdminErrorBody}}
GIT_ERRORS: dict[int | str, dict[str, Any]] = {**ERRORS, 502: {"model": m.AdminErrorBody}}


# region Installed


@router.get("/extensions", response_model=m.ExtensionList, summary="Installed extensions")
async def list_installed() -> dict[str, Any]:
    return await run_in_threadpool(admin.listing)


def _configure_and_list(disabled: list[str] | None, disable_all: str | None) -> dict[str, Any]:
    admin.configure(disabled, disable_all)
    return admin.listing()


@router.patch("/extensions", response_model=m.ExtensionList, responses=ERRORS, summary="Enable/disable extensions (loaded with the next reload)")
async def configure(body: m.ExtensionsPatch) -> dict[str, Any]:
    return await run_in_threadpool(_configure_and_list, body.disabled, body.disable_all)


def _check_and_list(names: list[str] | None) -> dict[str, Any]:
    admin.check_updates(names)
    return admin.listing()


@router.post("/extensions/check-updates", response_model=m.ExtensionList, responses=ERRORS, summary="Fetch the remotes of extensions (default: every enabled one) and report which can update")
async def check_updates(body: m.ExtensionCheckRequest | None = None) -> dict[str, Any]:
    return await run_in_threadpool(_check_and_list, body.names if body else None)


def _check_one(name: str) -> dict[str, Any]:
    admin.check_access()
    ext = admin.find(name)
    admin.check_one(ext)
    return admin.info(ext)


@router.post(
    "/extensions/{name}/check",
    response_model=m.ExtensionInfo,
    responses=ERRORS,
    summary="Check one extension for updates (clients check one at a time to show progress; a failed fetch is the status \"unknown (remote error)\")",
)
async def check_one(name: str) -> dict[str, Any]:
    return await run_in_threadpool(_check_one, name)


def _update_and_list(names: list[str]) -> dict[str, Any]:
    admin.update(names)
    return admin.listing()


@router.post(
    "/extensions/update",
    response_model=m.ExtensionList,
    responses=GIT_ERRORS,
    summary="Update extensions to their remote's latest commit (backs up the config state first; loaded with the next reload; 502 names those git could not update)",
)
async def update(body: m.ExtensionUpdateRequest) -> dict[str, Any]:
    return await run_in_threadpool(_update_and_list, body.names)


def _install(url: str, branch: str | None, dirname: str | None) -> dict[str, Any]:
    return admin.info(admin.install(url, branch, dirname))


@router.post(
    "/extensions/install",
    response_model=m.ExtensionInfo,
    status_code=201,
    responses={**GIT_ERRORS, 409: {"model": m.AdminErrorBody}},
    summary="Install an extension from a git URL (runs its install.py; loaded with the next reload; 502 when git cannot clone it)",
)
async def install(body: m.ExtensionInstallRequest) -> dict[str, Any]:
    return await run_in_threadpool(_install, body.url, body.branch, body.dirname)


@router.delete("/extensions/{name}", status_code=204, responses=ERRORS, summary="Uninstall an extension (deletes its folder; built-in ones cannot be removed)")
async def uninstall(name: str) -> Response:
    await run_in_threadpool(admin.uninstall, name)
    return Response(status_code=204)


# endregion

# region Index


@router.get("/extensions/index", response_model=m.ExtensionIndex, responses={**ERRORS, 502: {"model": m.AdminErrorBody}}, summary="The extension index (cached for 10 minutes; `refresh` reloads it)")
async def extension_index(url: str | None = None, refresh: bool = False) -> dict[str, Any]:
    return await run_in_threadpool(admin.load_index, url, refresh=refresh)


# endregion

# region Config states


@router.get("/extensions/config-states", response_model=list[m.ConfigStateInfo], summary="Saved config states (backups of the WebUI and extension commits), newest first")
async def config_states() -> list[dict[str, Any]]:
    return await run_in_threadpool(admin.list_config_states)


def _save_config_state(name: str) -> dict[str, Any]:
    admin.check_access()
    with admin.lock:
        return admin.save_config_state(name)


@router.post("/extensions/config-states", response_model=m.ConfigStateInfo, status_code=201, responses=ERRORS, summary="Back up the current WebUI and extension commits")
async def save_config_state(body: m.ConfigStateSave) -> dict[str, Any]:
    return await run_in_threadpool(_save_config_state, body.name)


@router.get("/extensions/config-states/{state_id}", response_model=m.ConfigStateDetail, responses=ERRORS, summary="A backup compared with the current state")
async def config_state(state_id: str) -> dict[str, Any]:
    return await run_in_threadpool(admin.config_state_detail, state_id)


@router.delete("/extensions/config-states/{state_id}", status_code=204, responses=ERRORS, summary="Delete a backup")
async def delete_config_state(state_id: str) -> Response:
    await run_in_threadpool(admin.delete_config_state, state_id)
    return Response(status_code=204)


@router.post(
    "/extensions/config-states/{state_id}/restore",
    response_model=m.RestoreResult,
    responses=ERRORS,
    summary="Restore the extensions of a backup (each is reset to its recorded commit while the next Reload UI runs; the WebUI checkout is never reset)",
)
async def restore_config_state(state_id: str, body: m.ConfigStateRestore) -> dict[str, Any]:
    return await run_in_threadpool(admin.restore_config_state, state_id, body.what)


# endregion
