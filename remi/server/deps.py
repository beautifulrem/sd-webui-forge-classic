"""Shared helpers of the route modules."""

from __future__ import annotations

from datetime import datetime, timezone
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
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def not_found(message: str = "Not Found") -> HTTPException:
    return HTTPException(status_code=404, detail=message)


def opts() -> Any:
    from modules import shared

    return shared.opts


def publish_options_changed(services: Services, keys: list[str]) -> None:
    if keys:
        services.bus.publish("options.changed", keys=sorted(set(keys)))
