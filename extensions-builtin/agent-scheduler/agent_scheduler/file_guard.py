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


def _folder_key(path: str) -> str:
    return os.path.normcase(path).casefold()


def _served_path(requested: str) -> str:
    """The path Gradio's file route will open for ``requested``."""
    try:
        from gradio import utils

        return str(utils.abspath(requested))
    except ImportError:
        return os.path.abspath(requested)


class ProtectedFiles:
    """Refuses requests for protected files under any spelling.

    The request is first turned into the path the file route will open (with
    Gradio's own function), then canonicalised with realpath, which follows
    symlinks and, on Windows, expands short (8.3) names. A request is refused
    when it lands in a protected file's folder under a name starting with that
    file's name (so SQLite journals and temp files created later count too),
    or when it is the same file as a protected one.
    """

    def __init__(self, paths: Callable[[], Iterable[str]]):
        self._paths = paths
        self._entries = ()
        self._ids = frozenset()
        self._checked = float("-inf")

    def _refresh(self):
        now = time.monotonic()
        if now - self._checked < _REFRESH_SECONDS:
            return
        entries, ids = set(), set()
        for path in self._paths():
            real = os.path.realpath(path)
            entries.add((_folder_key(os.path.dirname(real)), _name_key(os.path.basename(real))))
            try:
                ids.add(_identity(real))
            except (OSError, ValueError):
                pass
        self._entries, self._ids, self._checked = tuple(entries), frozenset(ids), now

    def contains(self, requested: str) -> bool:
        try:
            path = _served_path(requested)
            if not os.path.exists(path):
                return False  # the route answers 404
            real = os.path.realpath(path)
            target = _identity(real)
        except (OSError, ValueError):
            return True  # exists, but cannot be checked: refuse
        self._refresh()
        if target in self._ids:
            return True
        folder = _folder_key(os.path.dirname(real))
        name = _name_key(os.path.basename(real))
        return any(folder == entry_folder and name.startswith(entry_name) for entry_folder, entry_name in self._entries)


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
