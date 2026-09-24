"""Keep the task database and signing key away from Gradio's file routes.

Forge adds the data dir to Gradio's allowed_paths, and Gradio's blocked_paths
compare paths lexically, which a different spelling of the same file defeats
on case-insensitive filesystems (or via Windows short names and streams).
The file routes are wrapped instead, refusing any request whose target is a
protected file (or one derived from it, such as a SQLite journal).
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


def _name_key(name: str) -> str:
    # Case-insensitive filesystems; on Windows also alternate data streams
    # ("x:stream") and the ignored trailing dots / spaces ("x. ").
    if os.name == "nt":
        name = name.split(":", 1)[0].rstrip(" .")
    return name.casefold()


class ProtectedFiles:
    """Refuses requests for protected files under any spelling.

    Requests are resolved like the file route resolves them: the OS follows
    symlinks before "..", so the path is not normalised first. Only requests
    that land in a folder holding protected files (identity cached; folders
    are stable) are checked further, by name, including files created since,
    such as SQLite journals, and by file identity (e.g. Windows short names).
    """

    def __init__(self, paths: Callable[[], Iterable[str]]):
        self._paths = paths
        self._dirs = frozenset()
        self._checked = float("-inf")

    def _folders(self):
        now = time.monotonic()
        if now - self._checked >= _REFRESH_SECONDS:
            dirs = set()
            for path in self._paths():
                try:
                    dirs.add(_identity(os.path.dirname(path)))
                except (OSError, ValueError):
                    pass
            self._dirs, self._checked = frozenset(dirs), now
        return self._dirs

    def contains(self, requested: str) -> bool:
        path = os.path.join(os.getcwd(), requested)
        try:
            if _identity(os.path.dirname(path)) not in self._folders():
                return False
            name = _name_key(os.path.basename(path))
            for protected in self._paths():
                protected_name = _name_key(os.path.basename(protected))
                # Prefix: SQLite journals ("db-journal", "db-wal", ...) and
                # temp files ("key.<pid>.tmp") of protected files.
                if name.startswith(protected_name):
                    return True
            target = _identity(path)
        except (OSError, ValueError):
            return False  # Gradio cannot serve what cannot be stat'ed either
        for protected in self._paths():
            try:
                if _identity(protected) == target:
                    return True
            except (OSError, ValueError):
                continue
        return False


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
