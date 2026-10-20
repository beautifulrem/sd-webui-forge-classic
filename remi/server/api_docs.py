"""The API documentation pages: ``/docs`` (Swagger UI), ``/redoc`` (ReDoc)
and ``/docs/oauth2-redirect``.

FastAPI's own pages load floating CDN versions of their viewers
(``cdn.jsdelivr.net/npm/swagger-ui-dist@5``, ``redoc@2``), Google Fonts and a
favicon from ``fastapi.tiangolo.com``: an outside request on every visit and
third-party code, unpinned and without integrity checks, running on the
app's origin with the user's session. These pages load pinned copies the SPA
build vendors into ``webui/dist`` instead (``apiDocs`` in
``webui/vite.config.ts`` writes them and ``api-docs.json``, which names
them), with the CSP of the server's other pages (``html_headers``): scripts
only from the server, network requests only to it.

Without a build the pages say how to build it and link the raw API
description (``/openapi.json``). The login for all of them is the docs login
of the security middleware (``docs_refusal``).
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from starlette.responses import HTMLResponse, JSONResponse, Response

from remi.server.auth import Auth, html_headers
from remi.server.spa import Spa

MANIFEST = "api-docs.json"
"""written by the build next to ``index.html``"""
VIEWER_FILES = ("swaggerJs", "swaggerCss", "redocJs")
_ASSET_RE = re.compile(r"assets/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.(?:js|css)")

SWAGGER_PARAMETERS = {"validatorUrl": None}
"""Swagger UI's default validator badge is an image from validator.swagger.io"""

NOT_BUILT = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:40rem;margin:4rem auto;padding:0 1rem;line-height:1.5}}code{{background:#8882;padding:.1em .3em;border-radius:.25em}}</style></head>
<body><h1>The API documentation viewer is not built</h1>
<p>It comes with the web interface build (Node.js 20+ and pnpm or npm):</p>
<pre><code>cd webui
pnpm install
pnpm build</code></pre>
<p>then reload this page. The API description itself is available meanwhile: <a href="{openapi}">openapi.json</a>.</p></body></html>"""


def viewer_assets(dist: Path) -> dict[str, str] | None:
    """The build's viewer files (``VIEWER_FILES`` → path under ``dist``),
    or ``None`` without a (complete) build."""

    try:
        manifest = json.loads((dist / MANIFEST).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    files = {key: manifest.get(key) for key in VIEWER_FILES}
    for value in files.values():
        if not isinstance(value, str) or not _ASSET_RE.fullmatch(value) or not (dist / value).is_file():
            return None
    return files  # type: ignore[return-value]


def page(body: bytes, status: int = 200) -> HTMLResponse:
    return HTMLResponse(body, status_code=status, headers={**html_headers(body), "Cache-Control": "no-store"})


def add_docs_routes(app: FastAPI, dist: Path, *, serve_assets: bool = False) -> None:
    """``/docs``, ``/docs/oauth2-redirect`` and ``/redoc`` for ``app``
    (created with ``docs_url=None, redoc_url=None``), their viewers from the
    build in ``dist``. The SPA serves the viewer files with the rest of the
    build; ``serve_assets`` (``--nowebui``: no SPA) serves just them."""

    redirect = "/docs/oauth2-redirect"

    def context(request: Request) -> tuple[str, str, dict[str, str] | None]:
        root = request.scope.get("root_path", "").rstrip("/")
        return root, root + (app.openapi_url or "/openapi.json"), viewer_assets(dist)

    def not_built(title: str, openapi: str) -> HTMLResponse:
        return page(NOT_BUILT.format(title=html.escape(title), openapi=html.escape(openapi)).encode(), 503)

    async def swagger_ui(request: Request) -> HTMLResponse:
        root, openapi, assets = context(request)
        title = f"{app.title} - Swagger UI"
        if assets is None:
            return not_built(title, openapi)
        response = get_swagger_ui_html(
            openapi_url=openapi,
            title=title,
            swagger_js_url=f"{root}/{assets['swaggerJs']}",
            swagger_css_url=f"{root}/{assets['swaggerCss']}",
            swagger_favicon_url=f"{root}/favicon.svg",
            oauth2_redirect_url=root + redirect,
            swagger_ui_parameters=SWAGGER_PARAMETERS,
        )
        return page(bytes(response.body))

    async def swagger_ui_redirect(request: Request) -> HTMLResponse:
        return page(bytes(get_swagger_ui_oauth2_redirect_html().body))

    async def redoc(request: Request) -> HTMLResponse:
        root, openapi, assets = context(request)
        title = f"{app.title} - ReDoc"
        if assets is None:
            return not_built(title, openapi)
        response = get_redoc_html(
            openapi_url=openapi,
            title=title,
            redoc_js_url=f"{root}/{assets['redocJs']}",
            redoc_favicon_url=f"{root}/favicon.svg",
            with_google_fonts=False,
        )
        return page(bytes(response.body))

    build = Spa(Auth(), dist)  # only its file responses: precompressed copies, immutable caching

    async def viewer_file(request: Request) -> Response:
        package, name = request.path_params["package"], request.path_params["name"]
        assets = viewer_assets(dist)
        if assets is None or f"assets/{package}" not in {value.rpartition("/")[0] for value in assets.values()}:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return build.response(f"/assets/{package}/{name}", "", request.headers.get("accept-encoding", ""))

    app.add_route("/docs", swagger_ui, include_in_schema=False)
    app.add_route(redirect, swagger_ui_redirect, include_in_schema=False)
    app.add_route("/redoc", redoc, include_in_schema=False)
    if serve_assets:
        app.add_route("/assets/{package}/{name}", viewer_file, include_in_schema=False)
