from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.extension_route_guard import guard_routes


def _client(gradio_auth=None):
    app = FastAPI()
    app.auth = gradio_auth
    app.cookie_id = "cid"
    app.tokens = {"good": "user"}

    @app.get("/ext/read")
    def read():
        return "ok"

    @app.post("/ext/write")
    def write():
        return "written"

    @app.get("/other")
    def other():
        return "other"

    assert guard_routes(app, ("/ext/",)) == 2
    assert guard_routes(app, ("/ext/",)) == 0
    return TestClient(app, base_url="http://127.0.0.1:7860")


def test_cross_site_writes_are_refused():
    client = _client()
    assert client.post("/ext/write", headers={"origin": "https://evil.example"}).status_code == 403
    assert client.post("/ext/write", headers={"origin": "http://127.0.0.1:7860"}).json() == "written"
    assert client.get("/ext/read", headers={"origin": "https://evil.example"}).json() == "ok"


def test_gradio_login_is_required_when_configured():
    client = _client(gradio_auth=[("user", "pw")])
    assert client.get("/ext/read").status_code == 401
    assert client.get("/other").json() == "other"  # not guarded
    client.cookies.set("access-token-cid", "good")
    assert client.get("/ext/read").json() == "ok"


def test_new_extension_routes_refuse_cross_site_writes_only():
    from modules.extension_route_guard import guard_new_routes, snapshot_routes

    app = FastAPI()
    app.auth = [("user", "pw")]

    @app.post("/core")
    def core():
        return "core"

    before = snapshot_routes(app)

    @app.post("/ext/api")
    def ext():
        return "ext"

    assert guard_new_routes(app, before) == 1
    client = TestClient(app, base_url="http://127.0.0.1:7860")
    assert client.post("/ext/api", headers={"origin": "https://evil.example"}).status_code == 403
    assert client.post("/ext/api").json() == "ext"  # API clients: no Origin, no login needed
    assert client.post("/core", headers={"origin": "https://evil.example"}).json() == "core"


def test_proxies_and_cors_allow_list():
    import sys
    from types import SimpleNamespace

    from modules.extension_route_guard import is_cross_site

    def request(origin, host, **headers):
        return SimpleNamespace(method="POST", headers={"origin": origin, "host": host, **headers})

    # nginx "Host: $host" drops the port; "$host:$server_port" keeps 443.
    assert not is_cross_site(request("https://example.com", "example.com"))
    assert not is_cross_site(request("https://example.com", "example.com:443"))
    assert not is_cross_site(request("https://example.com:8443", "example.com", **{"x-forwarded-port": "8443"}))
    assert is_cross_site(request("http://localhost:3000", "localhost"))  # another local port
    assert not is_cross_site(request("https://a.com", "backend:7860", **{"x-forwarded-host": "a.com, a.com"}))
    assert not is_cross_site(request("https://a.com", "backend:7860", forwarded='for=1.2.3.4;host="a.com"'))
    assert is_cross_site(request("http://localhost:3000", "localhost:7860"))

    fake = SimpleNamespace(cmd_opts=SimpleNamespace(cors_allow_origins="http://localhost:3000", cors_allow_origins_regex=None))
    saved = sys.modules.get("modules.shared_cmd_options")
    sys.modules["modules.shared_cmd_options"] = fake
    try:
        assert not is_cross_site(request("http://localhost:3000", "localhost:7860"))
    finally:
        if saved is None:
            del sys.modules["modules.shared_cmd_options"]
        else:
            sys.modules["modules.shared_cmd_options"] = saved
