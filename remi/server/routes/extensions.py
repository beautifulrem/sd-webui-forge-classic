"""Extensions management (the Gradio UI's "Extensions" tab): the installed
list, enabling/disabling, checking for and applying updates, installing from
a git URL. Changes take effect after a restart (``/server/restart``).

Everything that writes is refused when extension access is disabled
(``--listen`` & co. without ``--enable-insecure-extension-access``): an
extension is arbitrary code.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from remi.server import models as m

router = APIRouter()

DIRNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
ERRORS = {403: {"model": m.Detail}, 404: {"model": m.Detail}, 422: {"model": m.Detail}}

_lock = threading.Lock()
"""one management operation at a time"""


def _check_access() -> None:
    from modules import shared

    if shared.cmd_opts.disable_extension_access:
        raise HTTPException(status_code=403, detail="extension access is disabled (the server is reachable from other machines; see --enable-insecure-extension-access)")


def _info(ext: Any) -> dict[str, Any]:
    if not ext.have_info_from_repo:
        ext.read_info_from_repo()
    date = datetime.fromtimestamp(ext.commit_date, timezone.utc).isoformat() if ext.commit_date else None
    return {
        "name": ext.name,
        "enabled": bool(ext.enabled),
        "builtin": bool(ext.is_builtin),
        "remote": ext.remote,
        "branch": ext.branch,
        "version": ext.version or None,
        "commit_date": date,
        "status": ext.status or None,
        "can_update": bool(ext.can_update),
    }


def _listing() -> dict[str, Any]:
    from modules import extensions, shared

    return {
        "items": [_info(ext) for ext in extensions.extensions],
        "disable_all": shared.opts.disable_all_extensions,
        "access": not shared.cmd_opts.disable_extension_access,
    }


def _find(name: str) -> Any:
    from modules import extensions

    for ext in extensions.extensions:
        if ext.name == name:
            return ext
    raise HTTPException(status_code=404, detail=f"no extension {name!r}")


@router.get("/extensions", response_model=m.ExtensionList, summary="Installed extensions")
def list_installed() -> dict[str, Any]:
    return _listing()


@router.patch("/extensions", response_model=m.ExtensionList, responses=ERRORS, summary="Enable/disable extensions (applies after a restart)")
def configure(body: m.ExtensionsPatch) -> dict[str, Any]:
    from modules import extensions, shared

    _check_access()
    known = {ext.name for ext in extensions.extensions}
    if body.disabled is not None:
        unknown = sorted(set(body.disabled) - known)
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown extension(s): {', '.join(unknown)}")
        shared.opts.disabled_extensions = list(dict.fromkeys(body.disabled))
    if body.disable_all is not None:
        shared.opts.disable_all_extensions = body.disable_all
    shared.opts.save(shared.config_filename)
    for ext in extensions.extensions:
        ext.enabled = ext.name not in shared.opts.disabled_extensions
    return _listing()


@router.post("/extensions/check-updates", response_model=m.ExtensionList, responses=ERRORS, summary="Fetch the remotes of the enabled extensions and report which can update")
async def check_updates() -> dict[str, Any]:
    _check_access()
    return await run_in_threadpool(_check_updates)


def _check_updates() -> dict[str, Any]:
    from modules import errors, extensions

    with _lock:
        exts = [ext for ext in extensions.extensions if ext.remote is not None and ext.enabled]

        def check(ext: Any) -> None:
            try:
                ext.check_updates()
            except FileNotFoundError as error:
                if "FETCH_HEAD" not in str(error):
                    errors.report(f"Error checking updates for {ext.name}", exc_info=True)
            except Exception:
                errors.report(f"Error checking updates for {ext.name}", exc_info=True)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(check, exts))
    return _listing()


@router.post("/extensions/update", response_model=m.ExtensionList, responses=ERRORS, summary="Update extensions to their remote's latest commit (backs up the config state first; applies after a restart)")
async def update(body: m.ExtensionUpdateRequest) -> dict[str, Any]:
    _check_access()
    targets = [_find(name) for name in dict.fromkeys(body.names)]
    return await run_in_threadpool(_update, targets)


def _update(targets: list[Any]) -> dict[str, Any]:
    from modules import errors

    with _lock:
        save_config_state("Backup (pre-update)")
        failed = []
        for ext in targets:
            try:
                ext.fetch_and_reset_hard()
            except Exception:
                errors.report(f"Error getting updates for {ext.name}", exc_info=True)
                failed.append(ext.name)
    if failed:
        raise HTTPException(status_code=422, detail=f"could not update: {', '.join(failed)}")
    return _listing()


def save_config_state(name: str) -> str:
    """Back up the WebUI and extension commits (``config_states``)."""

    from modules import config_states
    from modules.paths_internal import config_states_dir

    state = config_states.get_config()
    state["name"] = os.path.basename(name or "Config")
    os.makedirs(config_states_dir, exist_ok=True)
    filename = os.path.join(config_states_dir, f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_{state['name']}.json")
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4, ensure_ascii=False)
    config_states.list_config_states()
    return filename


@router.post("/extensions/install", response_model=m.ExtensionInfo, status_code=201, responses={**ERRORS, 409: {"model": m.Detail}}, summary="Install an extension from a git URL (runs its install.py; loads after a restart)")
async def install(body: m.ExtensionInstallRequest) -> dict[str, Any]:
    _check_access()
    return await run_in_threadpool(_install, body.url.strip(), (body.branch or "").strip() or None, (body.dirname or "").strip() or None)


def normalize_git_url(url: str | None) -> str:
    return (url or "").replace(".git", "")


def _install(url: str, branch: str | None, dirname: str | None) -> dict[str, Any]:
    import git

    from modules import extensions, launch_utils, paths

    if not re.match(r"^(https?|ssh|git)://|^git@", url):
        raise HTTPException(status_code=422, detail="expected a git URL (https://, ssh://, git:// or git@)")
    dirname = dirname or normalize_git_url(url.rstrip("/").rsplit("/", 1)[-1])
    if not DIRNAME_RE.match(dirname):
        raise HTTPException(status_code=422, detail=f"not a valid folder name: {dirname!r}")
    with _lock:
        target = os.path.join(extensions.extensions_dir, dirname)
        if os.path.exists(target):
            raise HTTPException(status_code=409, detail=f"the folder {dirname!r} already exists")
        if any(normalize_git_url(ext.remote) == normalize_git_url(url) for ext in extensions.extensions if ext.remote):
            raise HTTPException(status_code=409, detail="an extension with this URL is installed")
        temp = os.path.join(paths.data_path, "tmp", dirname)
        try:
            shutil.rmtree(temp, True)
            options: dict[str, Any] = {"filter": ["blob:none"]}
            if branch:
                options["branch"] = branch
            try:
                with git.Repo.clone_from(url, temp, **options) as repo:
                    repo.remote().fetch()
                    for submodule in repo.submodules:
                        submodule.update()
            except git.GitCommandError as error:
                raise HTTPException(status_code=422, detail=f"git clone failed: {error.stderr.strip() if error.stderr else error}") from error
            try:
                os.rename(temp, target)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.move(temp, target)  # tmp/ and extensions/ on different file systems
            launch_utils.run_extension_installer(target)
            extensions.list_extensions()
        finally:
            shutil.rmtree(temp, True)
    return _info(_find(dirname))
