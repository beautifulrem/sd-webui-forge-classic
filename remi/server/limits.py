"""The request body cap of every HTTP route: ``/api/v2`` JSON, ``/sdapi``
(base64 images), extension routes and uploads alike.

Uploads have their own, tighter limit (``remi_upload_max_mb``, checked by
the upload route). Every other body may carry base64 images (a third larger
than the file, several per request), so the cap is twice the upload limit,
and never below :data:`MIN_BODY_LIMIT` (a small upload limit must not break
ordinary jobs). It is enforced while the body streams in, also for chunked
bodies that announce no ``Content-Length``: the server never buffers more
than the cap, whoever the client is.
"""

from __future__ import annotations

from collections.abc import Callable

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MIN_BODY_LIMIT = 16 * 1024 * 1024


def body_limit(max_upload_bytes: int) -> int:
    """The body cap for an upload limit of ``max_upload_bytes``."""

    return max(2 * int(max_upload_bytes), MIN_BODY_LIMIT)


class BodyTooLarge(HTTPException):
    """Raised from ``receive`` past the cap: an ``HTTPException``, so the
    route's body parsing passes it on (FastAPI re-raises those) and it
    answers 413."""

    def __init__(self, limit: int) -> None:
        super().__init__(status_code=413, detail=f"the request body is larger than {limit // (1024 * 1024)} MB")


class BodyLimitMiddleware:
    """Refuses (413) an HTTP request whose ``Content-Length`` or streamed
    body exceeds ``limit()`` (read per request: the setting is live)."""

    def __init__(self, app: ASGIApp, limit: Callable[[], int]) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit()
        length = Headers(scope=scope).get("content-length")
        if length and length.strip().isdigit() and int(length) > limit:
            error = BodyTooLarge(limit)
            await JSONResponse({"detail": error.detail}, status_code=413, headers={"Connection": "close"})(scope, receive, send)
            return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(limit)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLarge as error:
            # (a route that read the body outside FastAPI's handlers)
            if started:
                raise
            await JSONResponse({"detail": error.detail}, status_code=413, headers={"Connection": "close"})(scope, receive, send)

