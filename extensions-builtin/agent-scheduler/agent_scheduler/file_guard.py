"""Keep the task database and signing key away from Gradio's file routes.

Forge adds the data dir to Gradio's allowed_paths, and Gradio's blocked_paths
compare paths lexically, which a different spelling of the same file defeats
on case-insensitive filesystems (or via Windows short names and streams).
The file routes are wrapped instead, refusing any request whose target is a
protected file (or one derived from it, such as a SQLite journal).
"""

import os
import time
import unicodedata
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
    # Case- and normalization-insensitive filesystems (NTFS, APFS); on
    # Windows also alternate data streams ("x:stream") and the ignored
    # trailing dots / spaces ("x. ").
    if os.name == "nt":
        name = name.split(":", 1)[0].rstrip(" .")
    return unicodedata.normalize("NFC", name).casefold()


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
    own function) and refused when:

    - it is the same file as a protected one (stat follows symlinks the way
      the route's open does: aliases, hard links, short names, case), or
    - it lies in a protected file's folder (by identity) under a name that
      starts with a protected name, or a short name ("~"): SQLite journals
      and key temp files come and go, so they are matched whether or not
      they exist yet.
    """

    def __init__(self, paths: Callable[[], Iterable[str]]):
        self._paths = paths
        self._ids = frozenset()
        self._folders = frozenset()
        self._names = ()
        self._checked = float("-inf")

    def _refresh(self):
        now = time.monotonic()
        if now - self._checked < _REFRESH_SECONDS:
            return
        ids, folders, names = set(), set(), set()
        for path in self._paths():
            # Both the configured path and its target (it may be a symlink;
            # SQLite puts journals next to the resolved database).
            for spelling in {path, os.path.realpath(path)}:
                names.add(_name_key(os.path.basename(spelling)))
                try:
                    folders.add(_identity(os.path.dirname(spelling)))
                except (OSError, ValueError):
                    pass
            try:
                ids.add(_identity(path))
            except (OSError, ValueError):
                pass
        self._ids, self._folders, self._names = frozenset(ids), frozenset(folders), tuple(names)
        self._checked = now

    def contains(self, requested: str) -> bool:
        self._refresh()
        path = _served_path(requested)
        try:
            if _identity(path) in self._ids:
                return True
        except (FileNotFoundError, NotADirectoryError):
            pass  # may still be a journal about to be created
        except Exception:
            return True  # cannot be checked: refuse
        name = _name_key(os.path.basename(path))
        if "~" not in name and not any(name.startswith(protected) for protected in self._names):
            return False
        try:
            return _identity(os.path.dirname(path)) in self._folders
        except (FileNotFoundError, NotADirectoryError):
            return False  # no such folder: nothing to serve
        except Exception:
            return True


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
