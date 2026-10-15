"""The built React SPA (``webui/dist``), served for every GET no route
matched (it is the router's fallback, so routes extensions add later win).

Hashed files under ``/assets`` are cached for a year and served from their
precompressed ``.br`` / ``.gz`` copies (written by the build) when the client
accepts them; ``index.html`` (also served for client routes) is never cached.
Behind ``--subpath`` the shell's absolute asset URLs get the prefix and
``window.__REMI_ROOT__`` tells the app where its API lives.
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
<p>then reload this page. The API is available meanwhile (<code>/docs</code>).</p></body></html>"""


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
    def __init__(self, auth: Auth, dist: Path | None = None) -> None:
        self.auth = auth
        self.dist = (dist or DIST).resolve()
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
        script = f"<script>window.__REMI_ROOT__={json.dumps(root)}</script>"
        text = text.replace("</head>", f'<meta name="remi-root" content="{html.escape(root)}">{script}</head>', 1)
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
        if scope.get("method") not in ("GET", "HEAD") or path.startswith(API_PREFIXES):
            await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
            return
        refused = refusal(self.auth, scope, cross_site_check=False)
        if refused is not None:
            await refuse(scope, receive, send, *refused)
            return
        accept = Headers(scope=scope).get("accept-encoding", "")
        await self.response(path, scope.get("root_path", ""), accept)(scope, receive, send)

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
        if os.path.splitext(path)[1] and not path.endswith(".html"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        index = self.index(root)
        if index is None:
            return HTMLResponse(NOT_BUILT, status_code=503, headers={**html_headers(NOT_BUILT.encode()), "Cache-Control": "no-store"})
        return HTMLResponse(index, headers=html_headers(index))
