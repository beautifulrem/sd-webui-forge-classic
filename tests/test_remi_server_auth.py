"""Login, sessions, Basic auth, and the host / cross-site / WebSocket origin
checks of the Remi server."""

import base64
import time

import pytest
from starlette.datastructures import Headers
from starlette.websockets import WebSocketDisconnect

from remi.server.auth import COOKIE_NAME, Auth, credentials_from_cmd_opts, load_secret, parse_credentials
from remi.server.testing import WS_URL, server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


def basic(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


@pytest.fixture
def locked(webui, tmp_path):
    with server_app(tmp_path, auth=Auth({"ann": "s3cret", "bob": "pw:with:colons"}, b"k" * 32)) as server:
        yield server


# region Credentials and sessions


def test_credentials_parse_like_a1111_auth():
    assert parse_credentials(["ann:1, bob:2", "", "carol:x:y", "broken"]) == {"ann": "1", "bob": "2", "carol": "x:y"}


def test_auth_and_auth_file(tmp_path):
    from modules import cmd_args

    file = tmp_path / "logins.txt"
    file.write_text("ann:1\nbob:2,carol:3\n")
    options, _ = cmd_args.parser.parse_known_args(["--auth", "x:y", "--auth-file", str(file)])
    assert credentials_from_cmd_opts(options) == {"x": "y", "ann": "1", "bob": "2", "carol": "3"}


def test_session_tokens_are_signed_expire_and_die_with_the_password(tmp_path):
    secret = load_secret(tmp_path / "secret.key")
    assert load_secret(tmp_path / "secret.key") == secret and len(secret) == 32
    auth = Auth({"ann": "1"}, secret)
    token = auth.issue("ann")
    assert auth.verify(token) == "ann"
    assert auth.verify(token, now=time.time() + 31 * 86400) is None
    assert auth.verify(token[:-2] + "xx") is None and auth.verify("garbage") is None and auth.verify(None) is None
    assert Auth({"ann": "2"}, secret).verify(token) is None
    assert Auth({"ann": "1"}, b"other" * 8).verify(token) is None
    assert Auth({}, secret).verify(token) is None


# endregion

# region HTTP


def test_everything_is_open_without_credentials(webui, tmp_path):
    with server_app(tmp_path) as server:
        assert server.client.get("/api/v2/queue").status_code == 200
        assert server.client.get("/api/v2/auth/status").json() == {"required": False, "user": None}


def test_rest_needs_a_login(locked):
    response = locked.client.get("/api/v2/queue")
    assert response.status_code == 401 and response.json() == {"detail": "Not authenticated"}
    assert locked.client.get("/api/v2/auth/status").json() == {"required": True, "user": None}
    assert locked.client.get("/internal/ping").status_code == 200
    assert locked.client.post("/internal/progress", json={"id_task": "x"}).status_code == 401


def test_the_shell_shows_a_login_page(locked):
    page = locked.client.get("/", headers={"Accept": "text/html"})
    assert page.status_code == 401 and "Sign in" in page.text and "/api/v2/auth/login" in page.text


def test_login_sets_a_session_cookie_and_logout_clears_it(locked):
    client = locked.client
    assert client.post("/api/v2/auth/login", json={"username": "ann", "password": "nope"}).status_code == 401
    assert client.cookies.get(COOKIE_NAME) is None
    response = client.post("/api/v2/auth/login", json={"username": "ann", "password": "s3cret"})
    assert response.status_code == 204
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
    assert client.get("/api/v2/queue").status_code == 200
    assert client.get("/api/v2/auth/status").json() == {"required": True, "user": "ann"}
    assert client.get("/api/v2/app").json()["auth"] == {"required": True, "user": "ann"}
    client.post("/api/v2/auth/logout")
    client.cookies.clear()
    assert client.get("/api/v2/queue").status_code == 401


def test_logout_revokes_the_session_for_good(locked, tmp_path):
    client = locked.client
    client.post("/api/v2/auth/login", json={"username": "ann", "password": "s3cret"})
    stolen = client.cookies.get(COOKIE_NAME)
    assert client.post("/api/v2/auth/logout").status_code == 204
    client.cookies.clear()
    client.cookies.set(COOKIE_NAME, stolen)
    assert client.get("/api/v2/queue").status_code == 401

    # the revocation survives a restart (it is kept next to the secret)
    path = tmp_path / "revoked.json"
    auth = Auth({"ann": "pw"}, b"k" * 32, revoked_path=path)
    token = auth.issue("ann")
    auth.revoke(token)
    assert Auth({"ann": "pw"}, b"k" * 32, revoked_path=path).verify(token) is None
    assert Auth({"ann": "pw"}, b"k" * 32, revoked_path=path).verify(auth.issue("ann", now=time.time() + 1)) == "ann"


def test_session_cookies_are_secure_over_https(webui, tmp_path):
    with server_app(tmp_path, auth=Auth({"ann": "1"}, b"k" * 32), base_url="https://127.0.0.1:7860") as server:
        response = server.client.post("/api/v2/auth/login", json={"username": "ann", "password": "1"})
        assert "secure" in response.headers["set-cookie"].lower()


def test_failed_login_bookkeeping_is_bounded(monkeypatch):
    from remi.server import auth as module

    monkeypatch.setattr(module, "MAX_THROTTLED_CLIENTS", 10)
    auth = Auth({"ann": "1"})
    for index in range(50):
        auth._failures[f"10.0.0.{index}"] = [time.time() - 120]
    auth.record_failure("10.1.1.1")
    assert set(auth._failures) == {"10.1.1.1"}


def test_the_api_docs_need_a_login(locked):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert locked.client.get(path).status_code == 401, path
    assert locked.client.get("/openapi.json", headers=basic("ann", "s3cret")).status_code == 200
    assert "<label>" in locked.client.get("/docs", headers={"Accept": "text/html"}).text  # the login page


def test_http_basic_works_for_api_clients(locked):
    assert locked.client.get("/api/v2/queue", headers=basic("bob", "pw:with:colons")).status_code == 200
    assert locked.client.get("/api/v2/queue", headers=basic("bob", "wrong")).status_code == 401
    assert locked.client.post("/internal/progress", json={"id_task": "x"}, headers=basic("ann", "s3cret")).status_code == 200


def test_repeated_failed_logins_are_throttled(locked, monkeypatch):
    from remi.server import auth

    monkeypatch.setattr(auth, "LOGIN_FAILURE_DELAY", 0)
    monkeypatch.setattr("remi.server.routes.system.LOGIN_FAILURE_DELAY", 0)
    for _ in range(10):
        locked.client.post("/api/v2/auth/login", json={"username": "ann", "password": "x"})
    assert locked.client.post("/api/v2/auth/login", json={"username": "ann", "password": "s3cret"}).status_code == 429


def test_cross_site_writes_and_rebound_hosts_are_refused(webui, tmp_path):
    with server_app(tmp_path) as server:
        client = server.client
        assert client.post("/api/v2/queue/pause", headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post("/api/v2/queue/pause", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        assert client.post("/api/v2/queue/pause", headers={"Origin": "http://127.0.0.1:7860"}).status_code == 200
        assert client.get("/api/v2/queue", headers={"Origin": "https://evil.example"}).status_code == 200  # reads are CORS-protected
        assert client.get("/api/v2/queue", headers={"Host": "attacker.example"}).status_code == 403


def test_rebound_hosts_are_refused_under_nowebui_too(webui, tmp_path, monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "nowebui", True)
    with server_app(tmp_path) as server:
        assert server.client.get("/api/v2/queue", headers={"Host": "evil.example"}).status_code == 403
        assert server.client.patch("/api/v2/options", json={}, headers={"Host": "evil.example"}).status_code == 403
    monkeypatch.setattr(shared.cmd_opts, "listen", True)  # every interface: other names are legitimate
    with server_app(tmp_path / "listen") as server:
        assert server.client.get("/api/v2/queue", headers={"Host": "my-box.lan"}).status_code == 200


def test_sdapi_refuses_cross_site_and_rebound_requests(webui, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from remi.server.auth import SecurityMiddleware

    app = FastAPI()
    app.post("/sdapi/v1/interrupt")(lambda: {"ok": True})
    app.add_middleware(SecurityMiddleware, auth=Auth())
    client = TestClient(app, base_url="http://127.0.0.1:7860")
    assert client.post("/sdapi/v1/interrupt").json() == {"ok": True}  # API clients send no Origin
    assert client.post("/sdapi/v1/interrupt", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/sdapi/v1/interrupt", headers={"Host": "evil.example"}).status_code == 403
    # its login stays --api-auth's, not the WebUI login
    with_login = FastAPI()
    with_login.post("/sdapi/v1/interrupt")(lambda: {"ok": True})
    with_login.add_middleware(SecurityMiddleware, auth=Auth({"ann": "1"}))
    assert TestClient(with_login, base_url="http://127.0.0.1:7860").post("/sdapi/v1/interrupt").status_code == 200


def test_cors_allowed_origins_may_write(webui, tmp_path, monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "cors_allow_origins", "https://tools.example")
    with server_app(tmp_path) as server:
        response = server.client.post("/api/v2/queue/resume", headers={"Origin": "https://tools.example"})
        assert response.status_code == 200 and response.headers["access-control-allow-origin"] == "https://tools.example"


def test_extension_routes_keep_the_route_guard(webui, tmp_path):
    from modules import extension_route_guard

    with server_app(tmp_path, auth=Auth({"ann": "1"}, b"k" * 32)) as server:
        before = extension_route_guard.snapshot_routes(server.app)
        server.app.add_api_route("/my-extension/do", lambda: {"ok": True}, methods=["POST"])
        assert extension_route_guard.guard_new_routes(server.app, before) == 1
        # loopback server: no login required, cross-site writes refused
        assert server.client.post("/my-extension/do").json() == {"ok": True}
        assert server.client.post("/my-extension/do", headers={"Origin": "https://evil.example"}).status_code == 403
        # the guard asks the app who is logged in
        request = type("Request", (), {"cookies": {}, "headers": Headers(basic("ann", "1"))})()
        assert server.app.auth_dependency(request) == "ann"


# endregion

# region WebSocket


def test_websocket_needs_a_login_and_a_same_site_origin(locked):
    with pytest.raises(WebSocketDisconnect) as refused, locked.client.websocket_connect(WS_URL) as socket:
        socket.receive_json()
    assert refused.value.code == 1008
    with pytest.raises(WebSocketDisconnect), locked.client.websocket_connect(WS_URL, headers={**basic("ann", "s3cret"), "Origin": "https://evil.example"}) as socket:
        socket.receive_json()
    with locked.client.websocket_connect(WS_URL, headers={**basic("ann", "s3cret"), "Origin": "http://127.0.0.1:7860"}) as socket:
        assert socket.receive_json()["type"] == "hello"


def test_websocket_with_a_session_cookie(locked):
    locked.client.post("/api/v2/auth/login", json={"username": "ann", "password": "s3cret"})
    with locked.client.websocket_connect(WS_URL) as socket:
        assert socket.receive_json()["type"] == "hello"


# endregion
