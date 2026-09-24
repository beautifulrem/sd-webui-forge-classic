"""Keep the task database and signing key away from Gradio's file routes.

Forge adds the data dir to Gradio's allowed_paths, and Gradio's blocked_paths
compare paths lexically, which a different spelling of the same file defeats
on case-insensitive filesystems (or via Windows short names and streams).
The file routes are wrapped instead, refusing any request whose target is the
same file as a protected one.
"""

import os
from typing import Callable, Iterable

_FILE_ROUTE_PARAMS = {"/file={path_or_url:path}": "path_or_url", "/file/{path:path}": "path"}


def is_protected(requested: str, protected: Iterable[str]) -> bool:
    try:
        target = os.path.abspath(requested)
        if not os.path.isfile(target):
            return False
        for path in protected:
            if os.path.exists(path) and os.path.samefile(target, path):
                return True
    except (OSError, ValueError):
        return False  # Gradio cannot serve what cannot be stat'ed either
    return False


def install(app, protected: Callable[[], Iterable[str]]) -> int:
    """Wrap Gradio's file routes on ``app``; returns how many were wrapped."""
    from starlette.responses import PlainTextResponse

    wrapped = 0
    for route in getattr(app, "router", app).routes:
        param = _FILE_ROUTE_PARAMS.get(getattr(route, "path", None))
        if param is None or getattr(route, "_agent_scheduler_guarded", False):
            continue
        original = route.app

        async def guarded(scope, receive, send, original=original, param=param):
            requested = scope.get("path_params", {}).get(param, "")
            if isinstance(requested, str) and is_protected(requested, protected()):
                await PlainTextResponse("File not allowed", status_code=403)(scope, receive, send)
                return
            await original(scope, receive, send)

        route.app = guarded
        route._agent_scheduler_guarded = True
        wrapped += 1
    return wrapped
