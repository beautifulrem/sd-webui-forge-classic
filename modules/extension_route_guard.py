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


def refusal(app, request) -> tuple[int, str] | None:
    """(status, reason) when ``request`` must be refused, else None."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        hosts = {request.headers.get("host"), request.headers.get("x-forwarded-host")}
        if request.headers.get("sec-fetch-site") == "cross-site" or (
            origin and (origin == "null" or urlparse(origin).netloc not in hosts)
        ):
            return 403, "Cross-site request refused"
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


def guard_routes(app, prefixes: Iterable[str]) -> int:
    """Wrap every route of ``app`` under ``prefixes``; returns how many."""
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse

    prefixes = tuple(prefixes)
    wrapped = 0
    for route in getattr(app, "router", app).routes:
        path = getattr(route, "path", "")
        if not path.startswith(prefixes) or getattr(route, "_forge_route_guarded", False):
            continue
        original = route.app

        async def guarded(scope, receive, send, original=original):
            if scope.get("type") == "http":
                refused = refusal(app, Request(scope, receive))
                if refused is not None:
                    status, reason = refused
                    await PlainTextResponse(reason, status_code=status)(scope, receive, send)
                    return
            await original(scope, receive, send)

        route.app = guarded
        route._forge_route_guarded = True
        wrapped += 1
    return wrapped
