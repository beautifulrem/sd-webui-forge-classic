"""The built React SPA (``webui/dist``), served for every GET no route
matched (it is the router's fallback, so routes extensions add later win).

Hashed files under ``/assets`` are cached for a year and served from their
precompressed ``.br`` / ``.gz`` copies (written by the build) when the client
accepts them; ``index.html`` (also served for client routes) is never cached.
Behind ``--subpath`` the shell's absolute asset URLs get the prefix and
``window.__REMI_ROOT__`` tells the app where its API lives;
``window.__REMI_THEME__`` carries ``--theme`` (the theme of browsers that
have not picked one in the Appearance menu). Both are set at the top of
``<head>``, before the shell's own script applies the theme.

Without a login (``--auth``), a page load gets the same shell with status
401 (:meth:`Spa.login_page`): it carries no data, and the SPA asks
``/api/v2/auth/status`` and shows its own sign-in form, so the server never
renders a UI of its own.
"""

from __future__ import annotations

import html
import json
import mimetypes
import os
import threading
from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.types import Receive, Scope, Send

from remi.server.auth import Auth, html_headers, refusal, refuse, route_path

DIST = Path(__file__).resolve().parents[2] / "webui" / "dist"
API_PREFIXES = ("/api/", "/sdapi/", "/internal/")
RETIRED_APIS = {
    "/tacapi/": "/api/v2/prompt-tools",
    "/physton_prompt/": "/api/v2/prompt-tools",
    "/agent-scheduler/": "/api/v2/jobs and /api/v2/queue",
    "/prompt-queue/": "/api/v2/jobs and /api/v2/queue",
}
"""HTTP APIs of built-in extensions that Remi replaced (architecture.md,
"Decision: their HTTP APIs are not kept"): ``410 Gone`` naming the
replacement instead of the SPA shell. A route an extension registers under
one of these prefixes still wins (this is the fallback)."""
THEMES = ("light", "dark", "system")
"""``--theme`` values the SPA knows (anything else: its own default, dark)"""
IMMUTABLE = "public, max-age=31536000, immutable"
PRECOMPRESSED = (("br", ".br"), ("gzip", ".gz"))
"""content codings of the build's precompressed copies, preferred first"""

NOT_BUILT = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Remi WebUI</title>
<style>body{font-family:system-ui,sans-serif;max-width:40rem;margin:4rem auto;padding:0 1rem;line-height:1.5}code{background:#8882;padding:.1em .3em;border-radius:.25em}</style></head>
<body><h1>The web interface is not built</h1>
<p>Build it once with Node.js 20+ and pnpm (or npm):</p>
<pre><code>cd webui
pnpm install
pnpm build</code></pre>
<p>then reload this page. The API is available meanwhile (its description: <code>/openapi.json</code>; the <code>/docs</code> viewer comes with the build).</p></body></html>"""


def accepted_codings(header: str) -> set[str]:
    """The content codings an ``Accept-Encoding`` header allows (``q=0`` excluded)."""

    codings = set()
    for part in header.split(","):
        name, _, params = part.strip().partition(";")
        quality = params.strip().removeprefix("q=").strip() if params.strip().startswith("q=") else "1"
        try:
            allowed = float(quality) > 0
        except ValueError:
            allowed = False
        if name and allowed:
            codings.add(name.strip().lower())
    return codings


class Spa:
    def __init__(self, auth: Auth, dist: Path | None = None, *, theme: str | None = None) -> None:
        self.auth = auth
        self.dist = (dist or DIST).resolve()
        theme = theme.strip().lower() if isinstance(theme, str) else None
        self.theme = theme if theme in THEMES else None
        self._index: tuple[float, str, bytes] | None = None
        self._lock = threading.Lock()

    @property
    def built(self) -> bool:
        return (self.dist / "index.html").is_file()

    def index(self, root: str) -> bytes | None:
        path = self.dist / "index.html"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        with self._lock:
            if self._index is not None and self._index[:2] == (mtime, root):
                return self._index[2]
        text = path.read_text(encoding="utf8")
        if root:
            text = text.replace('="/assets/', f'="{root}/assets/').replace('="/favicon', f'="{root}/favicon')
        script = f"<script>window.__REMI_ROOT__={json.dumps(root)};window.__REMI_THEME__={json.dumps(self.theme)}</script>"
        injected = f'<meta name="remi-root" content="{html.escape(root)}">{script}'
        # first in <head>: the shell's inline script reads __REMI_THEME__ before the first paint
        text = text.replace("<head>", f"<head>{injected}", 1) if "<head>" in text else text.replace("</head>", f"{injected}</head>", 1)
        data = text.encode()
        with self._lock:
            self._index = (mtime, root, data)
        return data

    def _file(self, path: str) -> Path | None:
        relative = path.lstrip("/")
        if not relative or relative == "index.html":
            return None
        candidate = (self.dist / relative).resolve()
        if not candidate.is_relative_to(self.dist) or not candidate.is_file():
            return None
        return candidate

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1000})
            return
        if scope["type"] != "http":
            return
        path = route_path(scope)
        retired = next((replacement for prefix, replacement in RETIRED_APIS.items() if path.startswith(prefix) or path == prefix.rstrip("/")), None)
        if retired is not None:
            detail = f"this API of the old WebUI was removed; use {retired} (see docs/remi-webui/architecture.md)"
            await JSONResponse({"detail": detail}, status_code=410)(scope, receive, send)
            return
        if scope.get("method") not in ("GET", "HEAD") or path.startswith(API_PREFIXES):
            await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
            return
        refused = refusal(self.auth, scope, cross_site_check=False)
        if refused is not None:
            await refuse(scope, receive, send, *refused, login_page=self.login_page)
            return
        accept = Headers(scope=scope).get("accept-encoding", "")
        await self.response(path, scope.get("root_path", ""), accept)(scope, receive, send)

    def login_page(self, root: str) -> Response | None:
        """The shell for a page load without a login (status 401, never
        cached); the SPA renders the sign-in form. None when the SPA is not
        built (the server then offers HTTP Basic)."""

        index = self.index(root)
        if index is None:
            return None
        return HTMLResponse(index, status_code=401, headers={**html_headers(index), "Cache-Control": "no-store"})

    def response(self, path: str, root: str, accept_encoding: str = "") -> Response:
        file = self._file(path)
        if file is not None:
            if not path.startswith("/assets/"):
                return FileResponse(file, headers={"Cache-Control": "public, max-age=3600"})
            headers = {"Cache-Control": IMMUTABLE, "Vary": "Accept-Encoding"}
            accepted = accepted_codings(accept_encoding)
            for coding, suffix in PRECOMPRESSED:
                packed = file.with_name(file.name + suffix)
                if coding in accepted and packed.is_file():
                    media_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                    return FileResponse(packed, media_type=media_type, headers={**headers, "Content-Encoding": coding})
            return FileResponse(file, headers=headers)
        # Only existing build files under /assets: it is public (the shell
        # loads it to show the sign-in form), so it must never fall back to
        # the shell
        if path == "/assets" or path.startswith("/assets/") or (os.path.splitext(path)[1] and not path.endswith(".html")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        index = self.index(root)
        if index is None:
            return HTMLResponse(NOT_BUILT, status_code=503, headers={**html_headers(NOT_BUILT.encode()), "Cache-Control": "no-store"})
        return HTMLResponse(index, headers=html_headers(index))
