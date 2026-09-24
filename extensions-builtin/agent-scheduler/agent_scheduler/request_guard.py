"""Access checks for the Agent Scheduler routes, beyond --api-auth."""

from urllib.parse import urlparse

from fastapi import Request
from fastapi.exceptions import HTTPException


def make_request_guard(app, api_auth: bool):
    """A route dependency that refuses cross-site state changes and, unless
    --api-auth protects the API, requires the Gradio login when one is set."""

    def request_guard(request: Request):
        # CORS only hides responses: a page elsewhere could still clear the
        # queue or history with a simple cross-site POST.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            hosts = {request.headers.get("host"), request.headers.get("x-forwarded-host")}
            if request.headers.get("sec-fetch-site") == "cross-site" or (
                origin and (origin == "null" or urlparse(origin).netloc not in hosts)
            ):
                raise HTTPException(status_code=403, detail="Cross-site request refused")
        # These routes live on Gradio's app but outside its login check.
        if api_auth:
            return
        auth_dependency = getattr(app, "auth_dependency", None)
        if auth_dependency is not None:
            user = auth_dependency(request)
        elif getattr(app, "auth", None):
            cookie_id = getattr(app, "cookie_id", "")
            token = request.cookies.get(f"access-token-{cookie_id}") or request.cookies.get(
                f"access-token-unsecure-{cookie_id}"
            )
            user = getattr(app, "tokens", {}).get(token)
        else:
            return
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")

    return request_guard
