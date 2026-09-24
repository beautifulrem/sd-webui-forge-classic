"""Access checks for the Agent Scheduler routes, beyond --api-auth."""

from fastapi import Request
from fastapi.exceptions import HTTPException

from modules.extension_route_guard import refusal


def make_request_guard(app, api_auth: bool):
    """A route dependency that refuses cross-site state changes and, unless
    --api-auth protects the API, requires the Gradio login when one is set."""

    def request_guard(request: Request):
        refused = refusal(app, request, require_login=not api_auth)
        if refused is not None:
            status, reason = refused
            raise HTTPException(status_code=status, detail=reason)

    return request_guard
