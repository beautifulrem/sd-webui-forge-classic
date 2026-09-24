import importlib.util
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

SPEC = importlib.util.spec_from_file_location(
    "agent_scheduler_request_guard", Path(__file__).parents[1] / "agent_scheduler" / "request_guard.py"
)
request_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(request_guard)


def _client(gradio_auth=None, api_auth=False):
    app = FastAPI()
    app.auth = gradio_auth
    app.cookie_id = "cid"
    app.tokens = {"good": "user"}
    deps = [Depends(request_guard.make_request_guard(app, api_auth))]

    @app.get("/read", dependencies=deps)
    def read():
        return "ok"

    @app.post("/clear", dependencies=deps)
    def clear():
        return "cleared"

    return TestClient(app, base_url="http://127.0.0.1:7860")


def test_cross_site_posts_are_refused():
    client = _client()
    assert client.post("/clear", headers={"origin": "https://evil.example"}).status_code == 403
    assert client.post("/clear", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert client.post("/clear", headers={"origin": "null"}).status_code == 403
    assert client.post("/clear", headers={"origin": "http://127.0.0.1:7860"}).json() == "cleared"
    assert client.post("/clear").json() == "cleared"  # API clients send no Origin


def test_gradio_login_is_required_without_api_auth():
    client = _client(gradio_auth=[("user", "pw")])
    assert client.get("/read").status_code == 401
    client.cookies.set("access-token-cid", "good")
    assert client.get("/read").json() == "ok"

    # --api-auth has its own Basic auth dependency.
    assert _client(gradio_auth=[("user", "pw")], api_auth=True).get("/read").json() == "ok"
    assert _client().get("/read").json() == "ok"  # no login configured
