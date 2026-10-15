"""The WebSocket hub (``/api/v2/ws``).

Each client gets a bounded event queue and a single preview slot (a newer
frame replaces one not sent yet). A client that falls so far behind that its
queue overflows gets ``{"type": "resync", "dropped": n}`` and a fresh
snapshot (queue state, running job) instead of the stale backlog; it should
refetch what it shows. Events arrive from any thread through the
:class:`remi.realtime.EventBus`; the hub hands them to its event loop. A
failed send closes the socket (the client reconnects).

Protocol: the server sends ``{"type": "hello", "features": [...],
"version": ...}`` on connect, then the current queue state; the client may
send ``{"type": "hello", "features": [...]}`` (previews are only sent in the
formats it lists, when it lists any) and ``{"type": "ping"}`` (answered with
``pong``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Callable
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from remi.realtime import FORMAT_NAMES, PreviewFrame

log = logging.getLogger(__name__)

SERVER_FEATURES = ("events.v1", "preview.binary", "preview.jpeg", "preview.webp", "jobs.v1")
MAX_EVENTS = 256


Snapshot = Callable[[], list[dict[str, Any]]]


class Client:
    def __init__(self, websocket: WebSocket, max_events: int = MAX_EVENTS, snapshot: Snapshot | None = None) -> None:
        self.websocket = websocket
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.preview: PreviewFrame | None = None
        self.features: set[str] | None = None
        self.wake = asyncio.Event()
        self.snapshot = snapshot
        self.dropped = 0
        """events lost since the last resync"""

    def push_event(self, event: dict[str, Any]) -> None:
        if len(self.events) == self.events.maxlen:
            self.dropped += 1
        self.events.append(event)
        self.wake.set()

    def push_frame(self, frame: PreviewFrame) -> None:
        if self.accepts(frame):
            self.preview = frame
            self.wake.set()

    def accepts(self, frame: PreviewFrame) -> bool:
        if self.features is None:
            return True
        return "preview.binary" in self.features and f"preview.{FORMAT_NAMES[frame.fmt]}" in self.features

    async def send_loop(self) -> None:
        """Send until the socket fails (then close it and return)."""

        try:
            while True:
                await self.wake.wait()
                self.wake.clear()
                if self.dropped:
                    await self._resync()
                while self.events:
                    await self._send(self.events.popleft())
                frame, self.preview = self.preview, None
                if frame is not None:
                    await self.websocket.send_bytes(frame.pack())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.debug("websocket send failed: %s", error)
            with contextlib.suppress(Exception):
                await self.websocket.close(code=1011)

    async def _send(self, event: dict[str, Any]) -> None:
        try:
            text = json.dumps(event, ensure_ascii=False, allow_nan=False, default=str)
        except ValueError:  # NaN/inf in an event: not JSON
            log.warning("dropped an event that is not JSON: %s", event.get("type"))
            return
        await self.websocket.send_text(text)

    async def _resync(self) -> None:
        dropped, self.dropped = self.dropped, 0
        self.events.clear()
        await self._send({"type": "resync", "dropped": dropped})
        try:
            snapshot = self.snapshot() if self.snapshot else []
        except Exception:
            log.exception("websocket snapshot")
            snapshot = []
        for event in snapshot:
            await self._send(event)


class Hub:
    """Subscribes to the bus while its app runs (see ``create_app``)."""

    def __init__(self, *, version: str = "", snapshot: Callable[[], list[dict[str, Any]]] | None = None) -> None:
        self.version = version
        self.snapshot = snapshot
        self.clients: set[Client] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    # Subscriber (any thread)

    def _call(self, func: Callable[..., None], *args: Any) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(func, *args)
        except RuntimeError:  # the loop is shutting down
            pass

    def send_event(self, event: dict[str, Any]) -> None:
        self._call(self._fan_event, event)

    def send_frame(self, frame: PreviewFrame) -> None:
        self._call(self._fan_frame, frame)

    def _fan_event(self, event: dict[str, Any]) -> None:
        for client in self.clients:
            client.push_event(event)

    def _fan_frame(self, frame: PreviewFrame) -> None:
        for client in self.clients:
            client.push_frame(frame)

    # Connections

    async def serve(self, websocket: WebSocket) -> None:
        await websocket.accept()
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        client = Client(websocket, snapshot=self.snapshot)
        client.push_event({"type": "hello", "features": list(SERVER_FEATURES), "version": self.version})
        try:
            for event in self.snapshot() if self.snapshot else []:
                client.push_event(event)
        except Exception:
            log.exception("websocket snapshot")
        self.clients.add(client)
        sender = asyncio.create_task(client.send_loop())
        receiver = asyncio.create_task(self._receive(client))
        try:
            # until the client leaves or a send fails
            await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.clients.discard(client)
            for task in (sender, receiver):
                task.cancel()
            for task in (sender, receiver):
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, Exception):
                    await task

    async def _receive(self, client: Client) -> None:
        while True:
            message = await client.websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            text = message.get("text")
            if not text:
                continue
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("type") == "hello" and isinstance(data.get("features"), list):
                client.features = {str(item) for item in data["features"]}
            elif data.get("type") == "ping":
                client.push_event({"type": "pong"})

    def wants_frames(self) -> bool:
        return bool(self.clients)

    async def close_all(self) -> None:
        for client in list(self.clients):
            with contextlib.suppress(Exception):
                await client.websocket.close(code=1001)
        self.clients.clear()
