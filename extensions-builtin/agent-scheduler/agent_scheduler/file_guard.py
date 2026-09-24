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


def _served_path(requested: str) -> str:
    """The path Gradio's file route will open for ``requested``."""
    try:
        from gradio import utils

        return str(utils.abspath(requested))
    except Exception:
        return os.path.abspath(requested)


class ProtectedFiles:
    """Refuses requests for protected files under any spelling.

    The request is turned into the path the file route opens (with Gradio's
    own function). Only names starting with a protected file's name (so also
    SQLite journals and key temp files, which come and go) or Windows short
    names are looked at further: the path is canonicalised with realpath
    (symlinks; short names on Windows) and refused when its folder is a
    protected file's folder (by identity: bind mounts, aliases) and its name
    matches, or when it is the same file as a protected one.
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
            name = _name_key(os.path.basename(real))
            try:
                entries.add((_identity(os.path.dirname(real)), name))
            except (OSError, ValueError):
                entries.add((None, name))  # folder missing: nothing to serve
            try:
                ids.add(_identity(real))
            except (OSError, ValueError):
                pass
        self._entries, self._ids, self._checked = tuple(entries), frozenset(ids), now

    def contains(self, requested: str) -> bool:
        self._refresh()
        path = _served_path(requested)
        name = _name_key(os.path.basename(path))
        short_name = os.name == "nt" and "~" in name
        if not short_name and not any(name.startswith(entry_name) for _, entry_name in self._entries):
            return False
        try:
            real = os.path.realpath(path)
            name = _name_key(os.path.basename(real))
            folder = _identity(os.path.dirname(real))
        except FileNotFoundError:
            return False  # no such folder: nothing to serve
        except Exception:
            return True  # cannot be checked: refuse
        # By name even if the file does not exist yet: it may by the time the
        # route opens it (SQLite journals exist only during writes).
        if any(folder == entry_folder and name.startswith(entry_name) for entry_folder, entry_name in self._entries):
            return True
        try:
            return _identity(real) in self._ids
        except (OSError, ValueError):
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
