"""Keep the task database and signing key away from Gradio's file routes.

Forge adds the data dir to Gradio's allowed_paths, and Gradio's blocked_paths
compare paths lexically, which a different spelling of the same file defeats
on case-insensitive filesystems (or via Windows short names and streams).
The file routes are wrapped instead, refusing any request whose target is the
same file as a protected one.
"""

import os
import time
from typing import Callable, Iterable

# Gradio 4 (and its deprecated alias) and Gradio 5 file routes.
_FILE_ROUTE_PARAMS = {
    "/file={path_or_url:path}": "path_or_url",
    "/file/{path:path}": "path",
    "/gradio_api/file={path_or_url:path}": "path_or_url",
}
_REFRESH_SECONDS = 2.0


def _identity(path: str):
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino


class ProtectedFiles:
    """File identities (device, inode) of the protected paths, re-read at
    most every couple of seconds, so a request costs one stat."""

    def __init__(self, paths: Callable[[], Iterable[str]]):
        self._paths = paths
        self._ids = frozenset()
        self._checked = float("-inf")

    def ids(self):
        now = time.monotonic()
        if now - self._checked >= _REFRESH_SECONDS:
            ids = set()
            for path in self._paths():
                try:
                    ids.add(_identity(path))
                except (OSError, ValueError):
                    pass
            self._ids, self._checked = frozenset(ids), now
        return self._ids

    def contains(self, requested: str) -> bool:
        try:
            target = _identity(os.path.abspath(requested))
        except (OSError, ValueError):
            return False  # Gradio cannot serve what cannot be stat'ed either
        return target in self.ids()


def install(app, protected: Callable[[], Iterable[str]]) -> int:
    """Wrap Gradio's file routes on ``app``; returns how many were wrapped."""
    from starlette.responses import PlainTextResponse

    files = ProtectedFiles(protected)
    wrapped = 0
    for route in getattr(app, "router", app).routes:
        param = _FILE_ROUTE_PARAMS.get(getattr(route, "path", None))
        if param is None or getattr(route, "_agent_scheduler_guarded", False):
            continue
        original = route.app

        async def guarded(scope, receive, send, original=original, param=param):
            requested = scope.get("path_params", {}).get(param, "")
            if isinstance(requested, str) and files.contains(requested):
                await PlainTextResponse("File not allowed", status_code=403)(scope, receive, send)
                return
            await original(scope, receive, send)

        route.app = guarded
        route._agent_scheduler_guarded = True
        wrapped += 1
    return wrapped
