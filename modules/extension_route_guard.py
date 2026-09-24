"""Login and cross-site checks for routes extensions add to Gradio's app.

Routes added with ``app.get`` / ``app.post`` in an app_started callback live
on Gradio's FastAPI app but outside Gradio's own login check, so with
``--gradio-auth`` (e.g. behind ``--share``) anyone could call them; and CORS
only hides responses, so any web page could make a user's browser send them
state-changing requests. :func:`guard_routes` wraps such routes.
"""

from __future__ import annotations

from typing import Iterable
from urllib.parse import urlparse


def refusal(app, request, require_login: bool = True) -> tuple[int, str] | None:
    """(status, reason) when ``request`` must be refused, else None."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        hosts = {request.headers.get("host"), request.headers.get("x-forwarded-host")}
        if request.headers.get("sec-fetch-site") == "cross-site" or (
            origin and (origin == "null" or urlparse(origin).netloc not in hosts)
        ):
            return 403, "Cross-site request refused"
    if not require_login:
        return None
    auth_dependency = getattr(app, "auth_dependency", None)
    if auth_dependency is not None:
        user = auth_dependency(request)
    elif getattr(app, "auth", None):
        cookie_id = getattr(app, "cookie_id", "")
        token = request.cookies.get(f"access-token-{cookie_id}") or request.cookies.get(
            f"access-token-unsecure-{cookie_id}"
        )
        user = getattr(app, "tokens", {}).get(token)
    else:
        return None
    return None if user is not None else (401, "Not authenticated")


def _wrap(app, routes, require_login: bool) -> int:
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse

    wrapped = 0
    for route in routes:
        if getattr(route, "_forge_route_guarded", False) or not hasattr(route, "app"):
            continue
        original = route.app

        async def guarded(scope, receive, send, original=original):
            if scope.get("type") == "http":
                refused = refusal(app, Request(scope, receive), require_login)
                if refused is not None:
                    status, reason = refused
                    await PlainTextResponse(reason, status_code=status)(scope, receive, send)
                    return
            await original(scope, receive, send)

        route.app = guarded
        route._forge_route_guarded = True
        wrapped += 1
    return wrapped


def _routes(app):
    return list(getattr(app, "router", app).routes)


def guard_routes(app, prefixes: Iterable[str]) -> int:
    """Login + cross-site checks for UI-only routes under ``prefixes``;
    returns how many were wrapped."""
    prefixes = tuple(prefixes)
    return _wrap(app, [r for r in _routes(app) if getattr(r, "path", "").startswith(prefixes)], True)


def snapshot_routes(app) -> set:
    return {id(route) for route in _routes(app)}


def guard_new_routes(app, before: set, exclude: Iterable[str] = ()) -> int:
    """Cross-site checks for every route added since ``before`` (i.e. by
    extensions), except under ``exclude``. No login requirement: extension
    APIs may serve non-browser clients, which send no Origin header."""
    exclude = tuple(exclude)
    routes = [
        route
        for route in _routes(app)
        if id(route) not in before and not (exclude and getattr(route, "path", "").startswith(exclude))
    ]
    return _wrap(app, routes, False)
