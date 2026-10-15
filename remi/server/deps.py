"""Shared helpers of the route modules."""

from __future__ import annotations

import ipaddress
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request

from remi.server.services import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def current_user(request: Request) -> str | None:
    return request.scope.get("state", {}).get("remi_user")


UserDep = Annotated[str | None, Depends(current_user)]


def iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def not_found(message: str = "Not Found") -> HTTPException:
    return HTTPException(status_code=404, detail=message)


def opts() -> Any:
    from modules import shared

    return shared.opts


def publish_options_changed(services: Services, keys: list[str]) -> None:
    if keys:
        services.bus.publish("options.changed", keys=sorted(set(keys)))


PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded")
LOCAL_ONLY = "only a browser on the WebUI machine can open its folders"


def is_local_client(request: Request) -> bool:
    """A browser on the WebUI machine: a loopback peer that came through no
    proxy (folder actions open a window on this machine's desktop). A reverse
    proxy or tunnel on the same machine also connects from loopback; it is
    told apart only by the forwarding headers it adds, so one that adds none
    (``ssh -L``, a plain TCP tunnel) passes as local — hide folders with
    ``--hide-ui-dir-config`` when sharing the WebUI that way."""

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


def require_local_folders(request: Request) -> None:
    """Folder actions: refused with ``--hide-ui-dir-config`` and for browsers
    on other machines."""

    from modules import shared

    if getattr(shared.cmd_opts, "hide_ui_dir_config", False):
        raise HTTPException(status_code=403, detail="folders are hidden (--hide-ui-dir-config)")
    if not is_local_client(request):
        raise HTTPException(status_code=403, detail=LOCAL_ONLY)
