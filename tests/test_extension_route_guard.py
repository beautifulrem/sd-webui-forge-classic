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


def test_websocket_handshakes_get_the_same_checks(monkeypatch):
    import sys
    from types import SimpleNamespace

    import pytest
    from starlette.websockets import WebSocket, WebSocketDisconnect

    # The test client's handshake always says "Host: testserver", which a
    # loopback-only server would refuse as a rebound name: act as --listen.
    options = SimpleNamespace(listen=True, server_name=None, cors_allow_origins=None, cors_allow_origins_regex=None)
    monkeypatch.setitem(sys.modules, "modules.shared_cmd_options", SimpleNamespace(cmd_opts=options))

    app = FastAPI()
    app.auth = [("user", "pw")]
    app.cookie_id = "cid"
    app.tokens = {"good": "user"}

    @app.websocket("/ext/ws")
    async def socket(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    assert guard_routes(app, ("/ext/",)) == 1
    client = TestClient(app, base_url="http://127.0.0.1:7860")
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/ext/ws"):
            pass
    assert refused.value.code == 1008  # no login

    client.cookies.set("access-token-cid", "good")
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/ext/ws", headers={"origin": "https://evil.example"}):
            pass
    assert refused.value.code == 1008  # cross-site

    with client.websocket_connect("/ext/ws", headers={"origin": "http://testserver"}) as websocket:
        assert websocket.receive_text() == "hello"


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


def test_loopback_only_server_refuses_rebound_host_names(monkeypatch):
    import sys
    from types import SimpleNamespace

    from modules.extension_route_guard import is_cross_site, refusal

    def request(host, origin=None, method="GET"):
        headers = {"host": host} | ({"origin": origin} if origin else {})
        return SimpleNamespace(method=method, headers=headers, cookies={})

    options = SimpleNamespace(listen=False, server_name=None, share=False, ngrok=None, cors_allow_origins="https://ui.example", cors_allow_origins_regex=None)
    monkeypatch.setitem(sys.modules, "modules.shared_cmd_options", SimpleNamespace(cmd_opts=options))
    app = SimpleNamespace()

    # DNS rebinding: same-origin to the browser, but a foreign Host name
    assert refusal(app, request("evil.example:7860", "http://evil.example:7860", "POST")) == (403, "Unknown host refused")
    assert refusal(app, request("evil.example:7860")) == (403, "Unknown host refused")  # reads too
    for host in ("127.0.0.1:7860", "localhost:7860", "[::1]:7860", "app.localhost:7860", "ui.example"):
        assert refusal(app, request(host)) is None
    options.listen = True  # reachable under any name: nothing to tell apart
    assert refusal(app, request("evil.example:7860")) is None

    # Malformed ports are refused, not a server error
    assert is_cross_site(SimpleNamespace(method="POST", headers={"origin": "http://a.com:99999", "host": "a.com"}))
    assert is_cross_site(SimpleNamespace(method="POST", headers={"origin": "http://a.com", "host": "a.com:x"}))


def _guard_options(monkeypatch, **overrides):
    import sys
    from types import SimpleNamespace

    values = dict(listen=False, server_name=None, share=False, ngrok=None, nowebui=False, cors_allow_origins=None, cors_allow_origins_regex=None)
    options = SimpleNamespace(**(values | overrides))
    monkeypatch.setitem(sys.modules, "modules.shared_cmd_options", SimpleNamespace(cmd_opts=options))
    monkeypatch.delenv("GRADIO_SERVER_NAME", raising=False)
    return options


def _get(host):
    from types import SimpleNamespace

    return SimpleNamespace(method="GET", headers={"host": host}, cookies={})


def test_loopback_server_name_keeps_dns_rebinding_protection(monkeypatch):
    from modules.extension_route_guard import is_rebound_host

    for server_name in ("127.0.0.1", "localhost", "[::1]", "::1"):
        _guard_options(monkeypatch, server_name=server_name)
        assert is_rebound_host(_get("evil.example:7860")), server_name
        assert not is_rebound_host(_get("127.0.0.1:7860"))
    for server_name in ("0.0.0.0", "192.168.1.5", "my-box.lan"):
        _guard_options(monkeypatch, server_name=server_name)
        assert not is_rebound_host(_get("my-box.lan:7860")), server_name


def test_bind_address_follows_what_gradio_and_uvicorn_do(monkeypatch):
    from modules.extension_route_guard import is_rebound_host

    _guard_options(monkeypatch)
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")  # Gradio then listens on every interface
    assert not is_rebound_host(_get("192.168.1.5:7860"))
    monkeypatch.setenv("GRADIO_SERVER_NAME", "127.0.0.1")
    assert is_rebound_host(_get("192.168.1.5:7860"))

    # --nowebui: uvicorn gets host=None and listens on every interface
    _guard_options(monkeypatch, nowebui=True)
    assert not is_rebound_host(_get("192.168.1.5:7861"))
    _guard_options(monkeypatch, share=True)
    assert not is_rebound_host(_get("abc.gradio.live"))


def test_rebound_host_check_honours_the_cors_regex(monkeypatch):
    from modules.extension_route_guard import is_rebound_host

    _guard_options(monkeypatch, cors_allow_origins_regex=r"https://[a-z]+\.ui\.example")
    assert not is_rebound_host(_get("studio.ui.example"))
    assert is_rebound_host(_get("studio.ui.example:7860"))  # the regex names no port
    assert is_rebound_host(_get("evil.example"))
    _guard_options(monkeypatch, cors_allow_origins_regex=r"http://studio\.lan:\d+")
    assert not is_rebound_host(_get("studio.lan:7860"))


def _exposed_app(monkeypatch, **overrides):
    """An app whose extension route was added after the snapshot, as webui.py does."""
    from modules.extension_route_guard import guard_new_routes, snapshot_routes

    _guard_options(monkeypatch, **(dict(gradio_auth="user:pw", gradio_auth_path=None, api_auth=None) | overrides))
    app = FastAPI()
    app.auth = [("user", "pw")]
    app.cookie_id = "cid"
    app.tokens = {"good": "user"}
    before = snapshot_routes(app)

    @app.post("/ext/api")
    def ext():
        return "ext"

    assert guard_new_routes(app, before) == 1
    return TestClient(app, base_url="http://127.0.0.1:7860")


def _basic(user, password):
    import base64

    return {"authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def test_exposed_servers_require_the_gradio_login_for_extension_routes(monkeypatch):
    for tunnel in (dict(share=True), dict(ngrok="token")):
        client = _exposed_app(monkeypatch, **tunnel)
        assert client.post("/ext/api").status_code == 401
        client.cookies.set("access-token-cid", "good")
        assert client.post("/ext/api").json() == "ext"


def test_lan_bound_servers_require_the_gradio_login_for_extension_routes(monkeypatch):
    # --listen / a non-loopback --server-name: reachable from the network, like a tunnel
    for bind in (dict(listen=True), dict(server_name="0.0.0.0"), dict(server_name="192.168.1.5")):
        client = _exposed_app(monkeypatch, **bind)
        assert client.post("/ext/api").status_code == 401, bind
        client.cookies.set("access-token-cid", "good")
        assert client.post("/ext/api").json() == "ext"
    client = _exposed_app(monkeypatch, listen=True, api_auth="bot:secret")
    assert client.post("/ext/api", headers=_basic("bot", "secret")).json() == "ext"
    # a loopback --server-name stays local
    assert _exposed_app(monkeypatch, server_name="127.0.0.1").post("/ext/api").json() == "ext"


def test_exposed_servers_accept_api_auth_credentials(monkeypatch):
    client = _exposed_app(monkeypatch, share=True, api_auth="bot:secret,other:pw2")
    assert client.post("/ext/api", headers=_basic("bot", "secret")).json() == "ext"
    assert client.post("/ext/api", headers=_basic("other", "pw2")).json() == "ext"
    assert client.post("/ext/api", headers=_basic("bot", "wrong")).status_code == 401
    assert client.post("/ext/api", headers={"authorization": "Basic !!notbase64"}).status_code == 401
    assert client.post("/ext/api").status_code == 401


def test_api_auth_credentials_do_not_count_without_api_auth(monkeypatch):
    client = _exposed_app(monkeypatch, share=True)
    assert client.post("/ext/api", headers=_basic("user", "pw")).status_code == 401


def test_local_or_login_free_servers_keep_extension_apis_open(monkeypatch):
    # loopback-only: local API clients keep working; exposed without --gradio-auth: nothing to log in with
    for options in (dict(share=False, ngrok=None), dict(share=True, gradio_auth=None), dict(listen=True, gradio_auth=None)):
        assert _exposed_app(monkeypatch, **options).post("/ext/api").json() == "ext"
    assert _exposed_app(monkeypatch, share=True, gradio_auth=None, gradio_auth_path="auth.txt").post("/ext/api").status_code == 401


def _api_only_app(monkeypatch, **overrides):
    """--nowebui: a plain FastAPI app, no Gradio login attributes."""
    from modules.extension_route_guard import guard_new_routes, snapshot_routes

    _guard_options(monkeypatch, **(dict(nowebui=True, gradio_auth=None, gradio_auth_path=None, api_auth=None) | overrides))
    app = FastAPI()
    before = snapshot_routes(app)

    @app.post("/ext/api")
    def ext():
        return "ext"

    assert guard_new_routes(app, before) == 1
    return TestClient(app, base_url="http://127.0.0.1:7861")


def test_api_only_servers_require_api_auth_for_extension_routes(monkeypatch):
    # --nowebui binds every interface (or --listen): /sdapi needs --api-auth, so do extension routes
    for options in (dict(), dict(listen=True, gradio_auth="u:p")):
        client = _api_only_app(monkeypatch, api_auth="api:secret", **options)
        assert client.post("/ext/api").status_code == 401, options
        assert client.post("/ext/api", headers=_basic("api", "wrong")).status_code == 401
        assert client.post("/ext/api", headers=_basic("u", "p")).status_code == 401  # no Gradio login here
        assert client.post("/ext/api", headers=_basic("api", "secret")).json() == "ext"


def test_api_only_servers_without_api_auth_or_on_loopback_stay_open(monkeypatch):
    # like /sdapi: no --api-auth, nothing to log in with (--gradio-auth has no UI to guard)
    assert _api_only_app(monkeypatch, listen=True).post("/ext/api").json() == "ext"
    assert _api_only_app(monkeypatch, listen=True, gradio_auth="u:p").post("/ext/api").json() == "ext"
    # a loopback --server-name keeps local API clients working
    assert _api_only_app(monkeypatch, server_name="127.0.0.1", api_auth="api:secret").post("/ext/api").json() == "ext"


def test_refusal_uses_api_auth_when_the_app_has_no_gradio_login(monkeypatch):
    from types import SimpleNamespace

    from modules.extension_route_guard import refusal

    _guard_options(monkeypatch, nowebui=True, listen=True, api_auth="api:secret")
    app = SimpleNamespace()  # no auth / auth_dependency

    def request(**headers):
        return SimpleNamespace(method="POST", headers={"host": "127.0.0.1:7861", **headers}, cookies={})

    assert refusal(app, request(), accept_api_auth=True) == (401, "Not authenticated")
    assert refusal(app, request(**_basic("api", "secret")), accept_api_auth=True) is None
    # UI-only guards (no API credentials accepted) keep their behaviour
    assert refusal(app, request()) is None


def test_opaque_null_origins_are_cross_site_even_when_allow_listed(monkeypatch):
    from types import SimpleNamespace

    from modules.extension_route_guard import is_cross_site

    # Sandboxed iframes, file:// pages and some redirects send "Origin: null".
    _guard_options(monkeypatch, cors_allow_origins="null,http://localhost:3000", cors_allow_origins_regex=".*")
    assert is_cross_site(SimpleNamespace(method="POST", headers={"origin": "null", "host": "127.0.0.1:7860"}))
    client = _client()
    assert client.post("/ext/write", headers={"origin": "null"}).status_code == 403


def test_fetch_metadata_refuses_cross_site_writes_without_an_origin():
    client = _client()
    assert client.post("/ext/write", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    for site in ("same-origin", "same-site", "none"):
        assert client.post("/ext/write", headers={"sec-fetch-site": site}).json() == "written", site
    assert client.get("/ext/read", headers={"sec-fetch-site": "cross-site"}).json() == "ok"  # reads stay open
    # An Origin, when present, decides
    assert client.post("/ext/write", headers={"sec-fetch-site": "cross-site", "origin": "http://127.0.0.1:7860"}).json() == "written"


def test_only_the_first_forwarded_host_hop_names_the_site():
    from types import SimpleNamespace

    from modules.extension_route_guard import is_cross_site

    def request(origin, forwarded_hosts):
        return SimpleNamespace(method="POST", headers={"origin": origin, "host": "backend:7860", "x-forwarded-host": forwarded_hosts})

    assert not is_cross_site(request("https://a.com", "a.com, proxy.internal"))
    assert not is_cross_site(request("https://a.com", " a.com "))
    # a later hop (appended by a proxy, or spoofed by the client) is not the site
    assert is_cross_site(request("https://evil.example", "a.com, evil.example"))
    assert is_cross_site(request("https://proxy.internal", "a.com,proxy.internal"))


def test_malformed_basic_credentials_are_refused(monkeypatch):
    import base64

    client = _exposed_app(monkeypatch, share=True, api_auth="bot:secret")

    def encoded(raw: bytes):
        return {"authorization": "Basic " + base64.b64encode(raw).decode()}

    for headers in (
        encoded(b"botsecret"),  # no colon
        encoded(b"bot:secret\xff"),  # not UTF-8
        encoded(b":"),
        encoded(b"bot:"),
        {"authorization": "Basic"},
        {"authorization": "Basic "},
        {"authorization": "Basic Ym90OnNlY3JldA"},  # missing padding
        {"authorization": "Bearer " + base64.b64encode(b"bot:secret").decode()},
        {"authorization": base64.b64encode(b"bot:secret").decode()},
    ):
        assert client.post("/ext/api", headers=headers).status_code == 401, headers
    assert client.post("/ext/api", headers={"authorization": "basic " + base64.b64encode(b"bot:secret").decode()}).json() == "ext"
