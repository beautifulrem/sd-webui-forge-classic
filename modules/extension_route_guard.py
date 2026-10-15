"""Login and cross-site checks for routes extensions add to the app.

Routes added with ``app.get`` / ``app.post`` in an app_started callback live
outside the WebUI's own routers, so without a guard anyone reaching the server
could call them; and CORS only hides responses, so any web page could make a
user's browser send them state-changing requests. :func:`guard_routes` wraps
such routes.

Routes extensions add without opting in (:func:`guard_new_routes`) always get
the cross-site checks. Under ``--nowebui`` (no WebUI) they require
``--api-auth`` HTTP Basic credentials whenever it is set, on any bind, exactly
like ``/sdapi``. With the WebUI they require a login when the server is
reachable beyond this machine (``--ngrok``, ``--listen`` or a non-loopback
``--server-name``) and ``--auth`` / ``--auth-file`` is set: the WebUI login
(``app.auth_dependency``), API clients (no session cookie) then
authenticating with ``--api-auth`` credentials; loopback-only WebUI servers
keep them open to local API clients.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import re
import sys
from hmac import compare_digest
from typing import Iterable
from urllib.parse import urlparse


def _cmd_opts():
    # Only if the WebUI loaded it: importing it parses the command line.
    return getattr(sys.modules.get("modules.shared_cmd_options"), "cmd_opts", None)


def _cors_origins(cmd_opts) -> list[str]:
    return [value.strip() for value in (getattr(cmd_opts, "cors_allow_origins", None) or "").split(",")]


def _allowed_by_cors(origin: str) -> bool:
    """Origins the admin allowed with --cors-allow-origins(-regex)."""
    cmd_opts = _cmd_opts()
    if cmd_opts is None:
        return False
    allowed = _cors_origins(cmd_opts)
    if "*" in allowed or origin in allowed:
        return True
    pattern = getattr(cmd_opts, "cors_allow_origins_regex", None)
    return bool(pattern) and re.fullmatch(pattern, origin) is not None


def _hostname(value: str) -> str | None:
    try:
        return urlparse("//" + value).hostname
    except ValueError:
        return None


def _is_loopback_name(hostname: str) -> bool:
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _bind_host(cmd_opts) -> str:
    """The address the server listens on (the server's own choice, with or
    without --nowebui)."""
    from remi.server.launch import bind_host

    return bind_host(cmd_opts)


def _is_loopback_only(cmd_opts) -> bool:
    if getattr(cmd_opts, "share", None) or getattr(cmd_opts, "ngrok", None):
        return False  # also reached through the tunnel's host name
    host = _bind_host(cmd_opts)
    return bool(host) and _is_loopback_name(host.strip("[]").lower())


def is_rebound_host(request) -> bool:
    """A request to a loopback-only server under a foreign ``Host`` name.

    DNS rebinding re-points a site's name at 127.0.0.1, so its requests look
    same-origin to the browser. A server that only listens on loopback (by
    default, or with a loopback --server-name; no --share / --ngrok) is only
    reached as localhost, or under a name allowed with
    --cors-allow-origins(-regex)."""

    cmd_opts = _cmd_opts()
    if cmd_opts is None or not _is_loopback_only(cmd_opts):
        return False
    host = request.headers.get("host") or ""
    hostname = _hostname(host)
    if not hostname or _is_loopback_name(hostname):
        return False
    allowed = {_hostname(urlparse(origin).netloc or "") for origin in _cors_origins(cmd_opts)}
    if hostname in allowed:
        return False
    # A page from an allowed origin served under this Host would send it as Origin
    return not any(_allowed_by_cors(f"{scheme}://{host}") for scheme in ("http", "https"))


_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _same_origin(origin: str, request) -> bool:
    target = urlparse(origin)
    default_port = _DEFAULT_PORTS.get(target.scheme)
    try:
        origin_port = target.port or default_port
    except ValueError:  # out-of-range or non-numeric port
        return False
    forwarded_port = request.headers.get("x-forwarded-port")
    candidates = [request.headers.get("host") or ""]
    # Proxies: X-Forwarded-Host may list several hops (first is the client's),
    # and Forwarded carries host=...
    candidates += [value.strip() for value in (request.headers.get("x-forwarded-host") or "").split(",")[:1]]
    for part in (request.headers.get("forwarded") or "").split(","):
        for pair in part.split(";"):
            key, _, value = pair.strip().partition("=")
            if key.lower() == "host":
                candidates.append(value.strip('"'))
    for host in filter(None, candidates):
        try:
            parsed = urlparse("//" + host)
            port = parsed.port
        except ValueError:
            continue
        if parsed.hostname is None or parsed.hostname != target.hostname:
            continue
        if port is not None:
            if port == origin_port:  # "a.com:443" is https://a.com
                return True
        # A proxy's "$host" has no port: only the scheme's default port (or
        # the one the proxy reports) is this site, not other local ports.
        elif origin_port == default_port or (forwarded_port or "").strip() == str(origin_port):
            return True
    return False


def _cross_site_read(request) -> bool:
    """A GET/HEAD a page of another site (or another port of this one) made
    the browser send as a subresource (``<img>``, ``fetch``...): fetch
    metadata says so. Their responses are hidden by CORS, but the request
    itself still runs (blind SSRF/CSRF through GETs that fetch or compute).
    Top-level navigations (links, bookmarks) and allowed CORS origins pass;
    clients without fetch metadata (API clients, old browsers) too."""
    headers = request.headers
    if headers.get("sec-fetch-site") not in ("cross-site", "same-site"):
        return False
    if headers.get("sec-fetch-mode") == "navigate" and headers.get("sec-fetch-dest") in (None, "document"):
        return False
    origin = headers.get("origin")
    return not (origin and origin != "null" and (_allowed_by_cors(origin) or _same_origin(origin, request)))


def is_cross_site(request) -> bool:
    """A state-changing request, or a WebSocket handshake (browsers send those
    with cookies and no CORS check), that a browser sent from another site;
    or a cross-site subresource read (:func:`_cross_site_read`)."""
    method = getattr(request, "method", None)
    if method == "OPTIONS":
        return False
    if method in ("GET", "HEAD"):
        return _cross_site_read(request)
    origin = request.headers.get("origin")
    if origin and origin != "null" and _allowed_by_cors(origin):
        return False
    if origin:
        return origin == "null" or not _same_origin(origin, request)
    return request.headers.get("sec-fetch-site") == "cross-site"


def _has_api_credentials(request) -> bool:
    """Valid ``--api-auth`` HTTP Basic credentials (how /sdapi clients log in)."""
    api_auth = getattr(_cmd_opts(), "api_auth", None)
    scheme, _, encoded = (request.headers.get("authorization") or "").partition(" ")
    if not api_auth or scheme.lower() != "basic":
        return False
    try:
        user, sep, password = base64.b64decode(encoded.strip(), validate=True).decode("utf-8").partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    if not sep:
        return False
    matches = False
    for credential in api_auth.split(","):
        known_user, _, known_password = credential.partition(":")
        matches |= compare_digest(known_user.encode(), user.encode()) and compare_digest(known_password.encode(), password.encode())
    return matches


def refusal(app, request, require_login: bool = True, accept_api_auth: bool = False) -> tuple[int, str] | None:
    """(status, reason) when ``request`` must be refused, else None.
    ``accept_api_auth``: valid --api-auth credentials stand in for the login."""
    if is_rebound_host(request):
        return 403, "Unknown host refused"
    if is_cross_site(request):
        return 403, "Cross-site request refused"
    if not require_login:
        return None
    auth_dependency = getattr(app, "auth_dependency", None)
    api_auth = accept_api_auth and getattr(_cmd_opts(), "api_auth", None)
    if api_auth and _valid_api_credentials(request):
        return None  # (checked first: they are not WebUI logins, which would count as failed ones)
    if auth_dependency is not None:
        user = auth_dependency(request)
    elif api_auth:
        # No WebUI login on this app (--nowebui): --api-auth is the login, as for /sdapi
        return _api_auth_refusal(request)
    else:
        return None
    if user is None and api_auth:
        return _api_auth_refusal(request)
    return None if user is not None else (401, "Not authenticated")


def _valid_api_credentials(request) -> bool:
    from remi.server.auth import API_AUTH_THROTTLE, client_key

    return not API_AUTH_THROTTLE.throttled(client_key(request)) and _has_api_credentials(request)


def _api_auth_refusal(request) -> tuple[int, str] | None:
    """--api-auth Basic credentials, failures throttled per client."""
    from remi.server.auth import api_auth_refusal

    return api_auth_refusal(request)


def _wrap(app, routes, require_login: bool, accept_api_auth: bool = False) -> int:
    from starlette.requests import HTTPConnection, Request
    from starlette.responses import PlainTextResponse
    from starlette.websockets import WebSocketClose

    wrapped = 0
    for route in routes:
        if getattr(route, "_forge_route_guarded", False) or not hasattr(route, "app"):
            continue
        original = route.app

        async def guarded(scope, receive, send, original=original):
            kind = scope.get("type")
            if kind == "http":
                refused = refusal(app, Request(scope, receive), require_login, accept_api_auth)
                if refused is not None:
                    status, reason = refused
                    await PlainTextResponse(reason, status_code=status)(scope, receive, send)
                    return
            elif kind == "websocket":
                refused = refusal(app, HTTPConnection(scope), require_login, accept_api_auth)
                if refused is not None:
                    await WebSocketClose(code=1008, reason=refused[1])(scope, receive, send)
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
    returns how many were wrapped. Under --nowebui there is no WebUI login:
    --api-auth credentials are the login, as for /sdapi. Not with the UI: on a
    server with --api-auth but no WebUI login they would refuse the UI's own
    browser requests, which carry no Basic credentials."""
    prefixes = tuple(prefixes)
    routes = [r for r in _routes(app) if getattr(r, "path", "").startswith(prefixes)]
    return _wrap(app, routes, True, accept_api_auth=bool(getattr(_cmd_opts(), "nowebui", False)))


def snapshot_routes(app) -> set:
    return {id(route) for route in _routes(app)}


def _exposed_with_login(cmd_opts) -> bool:
    """Whether extension routes need a login. Under --nowebui (no WebUI):
    whenever --api-auth is set, on any bind, as /sdapi requires it. Otherwise
    on a server reachable beyond this machine (--ngrok tunnel, or a
    non-loopback bind such as --listen / --server-name 0.0.0.0) with a WebUI
    login: others can reach it, and the admin asked for one."""
    if cmd_opts is None:
        return False
    if getattr(cmd_opts, "nowebui", False):
        return bool(getattr(cmd_opts, "api_auth", None))
    if _is_loopback_only(cmd_opts):
        return False
    return bool(getattr(cmd_opts, "auth", None) or getattr(cmd_opts, "auth_file", None))


def guard_new_routes(app, before: set, exclude: Iterable[str] = ()) -> int:
    """Cross-site checks for every route added since ``before`` (i.e. by
    extensions), except under ``exclude``. Login under --nowebui whenever
    --api-auth is set (Basic credentials, as /sdapi), or on an exposed server
    (--ngrok / non-loopback bind) with --auth (the WebUI session cookie,
    or --api-auth Basic credentials); otherwise, e.g. on a loopback-only UI
    server, extension APIs stay open to non-browser clients, which send no
    Origin header."""
    exclude = tuple(exclude)
    routes = [
        route
        for route in _routes(app)
        if id(route) not in before and not (exclude and getattr(route, "path", "").startswith(exclude))
    ]
    return _wrap(app, routes, _exposed_with_login(_cmd_opts()), accept_api_auth=True)
