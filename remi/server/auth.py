"""Login and request checks of the Remi server.

Credentials come from ``--auth user:pass[,user2:pass2]`` and ``--auth-file``. With
credentials, every request to the app (REST, WebSocket, files, the SPA
shell) needs a session cookie from ``POST /api/v2/auth/login`` or HTTP Basic
credentials; without them the server is open, as before. Independently of
login, :class:`SecurityMiddleware` refuses requests under a rebound ``Host``
(DNS rebinding) and cross-site state-changing requests and WebSocket
handshakes (``modules.extension_route_guard`` semantics), also for
``/sdapi`` (which keeps ``--api-auth`` as its login).

Session cookies are ``user.expiry.signature`` signed with a per-installation
secret and the user's password, so they survive restarts and die with a
password change; logging out revokes the cookie (a denylist of signatures
until they expire, kept next to the secret).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from starlette._utils import get_route_path
from starlette.datastructures import Headers
from starlette.requests import HTTPConnection
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

COOKIE_NAME = "remi_session"
SESSION_SECONDS = 30 * 24 * 3600
LOGIN_FAILURE_DELAY = 0.5

PUBLIC_PATHS = ("/api/v2/auth/", "/internal/ping", "/favicon.svg", "/assets/")
"""Reachable without a login (the login page needs its assets)."""

GUARDED_PREFIXES = ("/api/v2", "/internal")
"""Host, cross-site and login checks apply here."""

DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")
"""The API description: needs a login like the API itself."""

SDAPI_PREFIX = "/sdapi"
"""Host and cross-site checks only (``--api-auth`` is its login)."""

MAX_THROTTLED_CLIENTS = 4096


def parse_credentials(lines: Iterable[str]) -> dict[str, str]:
    """``user:pass`` entries, comma- or line-separated."""

    credentials: dict[str, str] = {}
    for line in lines:
        for item in line.split(","):
            item = item.strip()
            if not item:
                continue
            user, sep, password = item.partition(":")
            if sep and user:
                credentials[user] = password
    return credentials


def credentials_from_cmd_opts(cmd_opts: Any) -> dict[str, str]:
    lines: list[str] = []
    if getattr(cmd_opts, "auth", None):
        lines.append(cmd_opts.auth)
    path = getattr(cmd_opts, "auth_file", None)
    if path:
        with open(path, encoding="utf8") as file:
            lines += file.read().splitlines()
    return parse_credentials(lines)


def load_secret(path: Path) -> bytes:
    """The installation's cookie signing secret (created on first use)."""

    try:
        data = path.read_bytes()
        if len(data) >= 32:
            return data
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    data = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
    return data


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass
class Auth:
    credentials: dict[str, str] = field(default_factory=dict)
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    secure_cookies: bool = False
    """always mark cookies ``Secure`` (``--tls-*``); else when the request is HTTPS"""
    cookie_path: str = "/"
    revoked_path: Path | None = None
    """where revoked sessions persist (None: in memory)"""
    _failures: dict[str, list[float]] = field(default_factory=dict)
    _revoked: dict[str, int] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def required(self) -> bool:
        return bool(self.credentials)

    def check_password(self, user: str, password: str) -> bool:
        known = self.credentials.get(user)
        # compare even for unknown users, so timing does not reveal them
        ok = hmac.compare_digest((known if known is not None else secrets.token_hex(8)).encode(), password.encode())
        return ok and known is not None

    def _signature(self, user: str, expiry: int) -> str:
        password = self.credentials.get(user, "")
        message = f"{user}\0{expiry}\0{hashlib.sha256(password.encode()).hexdigest()}".encode()
        return _b64(hmac.new(self.secret, message, hashlib.sha256).digest())

    def issue(self, user: str, *, now: float | None = None) -> str:
        expiry = int((now or time.time()) + SESSION_SECONDS)
        return f"{_b64(user.encode())}.{expiry}.{self._signature(user, expiry)}"

    def verify(self, token: str | None, *, now: float | None = None) -> str | None:
        if not token or token.count(".") != 2:
            return None
        encoded, expiry_text, signature = token.split(".")
        try:
            user = _unb64(encoded).decode()
            expiry = int(expiry_text)
        except (ValueError, binascii.Error, UnicodeDecodeError):
            return None
        if expiry < (now or time.time()) or user not in self.credentials:
            return None
        if not hmac.compare_digest(signature, self._signature(user, expiry)):
            return None
        return None if signature in self._revocations() else user

    # region Revocation (logout)

    def _revocations(self) -> dict[str, int]:
        """Revoked signatures → their expiry."""

        with self._lock:
            if self._revoked is None:
                self._revoked = {}
                if self.revoked_path is not None:
                    try:
                        data = json.loads(self.revoked_path.read_text())
                        self._revoked = {str(key): int(value) for key, value in data.items()} if isinstance(data, dict) else {}
                    except (OSError, ValueError, TypeError):
                        pass
            return self._revoked

    def revoke(self, token: str | None) -> None:
        """Log out ``token`` for good (if it is a valid session)."""

        if self.verify(token) is None:
            return
        assert token is not None
        _user, expiry, signature = token.split(".")
        revoked = self._revocations()
        now = time.time()
        with self._lock:
            for key in [key for key, until in revoked.items() if until < now]:
                del revoked[key]
            revoked[signature] = int(expiry)
            if self.revoked_path is not None:
                try:
                    self.revoked_path.parent.mkdir(parents=True, exist_ok=True)
                    temp = self.revoked_path.with_suffix(".tmp")
                    temp.write_text(json.dumps(revoked))
                    os.replace(temp, self.revoked_path)
                except OSError:
                    pass

    # endregion

    def basic_user(self, headers: Headers) -> str | None:
        scheme, _, encoded = (headers.get("authorization") or "").partition(" ")
        if scheme.lower() != "basic":
            return None
        try:
            user, sep, password = base64.b64decode(encoded.strip(), validate=True).decode().partition(":")
        except (binascii.Error, UnicodeDecodeError):
            return None
        return user if sep and self.check_password(user, password) else None

    def user_for(self, conn: HTTPConnection) -> str | None:
        """The logged-in user of a request (session cookie or Basic), None
        when it has no valid login."""

        return self.verify(conn.cookies.get(COOKIE_NAME)) or self.basic_user(conn.headers)

    def throttled(self, client: str) -> bool:
        """Too many failed logins from ``client`` in the last minute."""

        now = time.time()
        with self._lock:
            recent = [moment for moment in self._failures.get(client, []) if now - moment < 60]
            if recent:
                self._failures[client] = recent
            else:
                self._failures.pop(client, None)
            return len(recent) >= 10

    def record_failure(self, client: str) -> None:
        now = time.time()
        with self._lock:
            if len(self._failures) >= MAX_THROTTLED_CLIENTS:  # forget clients idle for a minute
                for key in [key for key, moments in self._failures.items() if not moments or now - moments[-1] >= 60]:
                    del self._failures[key]
            self._failures.setdefault(client, []).append(now)

    def set_cookie(self, response: Response, user: str, *, https: bool = False) -> None:
        """``https``: the request came over HTTPS (also behind a TLS proxy
        that reports it), so the cookie is ``Secure``."""

        response.set_cookie(COOKIE_NAME, self.issue(user), max_age=SESSION_SECONDS, path=self.cookie_path, httponly=True, samesite="lax", secure=self.secure_cookies or https)

    def clear_cookie(self, response: Response) -> None:
        response.delete_cookie(COOKIE_NAME, path=self.cookie_path)


def route_path(scope: Scope) -> str:
    """The request path below ``root_path`` (Starlette's routing view)."""

    return get_route_path(scope) or "/"


def refusal(auth: Auth, scope: Scope, *, cross_site_check: bool = True, require_login: bool = True) -> tuple[int, str] | None:
    """``(status, reason)`` when the request must be refused. Stores the
    logged-in user in ``scope["state"]["remi_user"]``."""

    from modules import extension_route_guard as guard

    conn = HTTPConnection(scope)
    path = route_path(scope)
    user = auth.user_for(conn) if auth.required else None
    scope.setdefault("state", {})["remi_user"] = user
    if guard.is_rebound_host(conn):
        return 403, "Unknown host refused"
    if cross_site_check and _cross_site(scope, conn):
        return 403, "Cross-site request refused"
    if require_login and auth.required and user is None and not path.startswith(PUBLIC_PATHS):
        return 401, "Not authenticated"
    return None


async def refuse(scope: Scope, receive: Receive, send: Send, status: int, reason: str) -> None:
    """Answer a refused request: a WebSocket close, the login page for a
    browser page load, JSON otherwise."""

    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008, "reason": reason})
        return
    path = route_path(scope)
    accept = Headers(scope=scope).get("accept") or ""
    if status == 401 and scope.get("method") == "GET" and not path.startswith(GUARDED_PREFIXES) and "text/html" in accept:
        response: Response = HTMLResponse(LOGIN_PAGE.replace("{{root}}", scope.get("root_path", "")), status_code=401, headers={"Cache-Control": "no-store"})
    else:
        headers = {"WWW-Authenticate": 'Basic realm="Remi"'} if status == 401 and path.startswith("/internal") else None
        response = JSONResponse({"detail": reason}, status_code=status, headers=headers)
    await response(scope, receive, send)


class SecurityMiddleware:
    """Host / cross-site / login checks for ``/api/v2``, ``/internal`` and
    the API docs, HTTP and WebSocket alike (pure ASGI, so it sees WebSocket
    handshakes); host and cross-site checks for ``/sdapi``, whose login stays
    ``--api-auth`` (API clients send no ``Origin``). Extension routes keep
    ``extension_route_guard``; the SPA shell checks the login itself
    (:mod:`remi.server.spa`)."""

    def __init__(self, app: ASGIApp, auth: Auth) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = route_path(scope)
            refused = None
            if path.startswith(GUARDED_PREFIXES) or path in DOCS_PATHS:
                refused = refusal(self.auth, scope)
            elif path.startswith(SDAPI_PREFIX):
                refused = refusal(self.auth, scope, require_login=False)
            if refused is not None:
                await refuse(scope, receive, send, *refused)
                return
        await self.app(scope, receive, send)


def _cross_site(scope: Scope, conn: HTTPConnection) -> bool:
    from modules import extension_route_guard as guard

    class _Request:
        method = scope.get("method") if scope["type"] == "http" else None
        headers = conn.headers

    return guard.is_cross_site(_Request)


LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Remi WebUI · Sign in</title>
<style>
:root{color-scheme:dark light;font-family:system-ui,sans-serif}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#111317;color:#e8e9ec}
@media (prefers-color-scheme:light){body{background:#f4f5f7;color:#16181d}form{background:#fff!important}}
form{display:grid;gap:.75rem;padding:2rem;border-radius:.75rem;background:#1b1e24;min-width:18rem;box-shadow:0 8px 30px #0004}
h1{margin:0 0 .5rem;font-size:1.25rem}
label{display:grid;gap:.3rem;font-size:.9rem}
input{font:inherit;padding:.55rem .7rem;border-radius:.45rem;border:1px solid #8884;background:transparent;color:inherit}
button{font:inherit;padding:.6rem;border:0;border-radius:.45rem;background:#6d5efc;color:#fff;cursor:pointer}
p{margin:0;min-height:1.2em;color:#ff7a7a;font-size:.9rem}
</style></head>
<body><form id="f"><h1>Remi WebUI</h1>
<label><span data-t="user">Username</span><input name="username" autocomplete="username" required autofocus></label>
<label><span data-t="password">Password</span><input name="password" type="password" autocomplete="current-password" required></label>
<button data-t="submit">Sign in</button><p id="e" role="alert"></p></form>
<script>
const zh = /^zh\b/i.test(navigator.language)
const text = zh
  ? {title: "Remi WebUI · 登录", user: "用户名", password: "密码", submit: "登录", wrong: "用户名或密码错误。", throttled: "尝试次数过多，请稍等一分钟。"}
  : {title: "Remi WebUI · Sign in", user: "Username", password: "Password", submit: "Sign in", wrong: "Wrong username or password.", throttled: "Too many attempts, wait a minute."}
if (zh) {
  document.documentElement.lang = "zh-CN"
  document.title = text.title
  for (const node of document.querySelectorAll("[data-t]")) node.textContent = text[node.dataset.t]
}
document.getElementById("f").addEventListener("submit", async (event) => {
  event.preventDefault()
  const data = Object.fromEntries(new FormData(event.target))
  const response = await fetch("{{root}}/api/v2/auth/login", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)})
  if (response.ok) location.reload()
  else document.getElementById("e").textContent = response.status === 429 ? text.throttled : text.wrong
})
</script></body></html>
"""
