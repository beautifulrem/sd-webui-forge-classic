"""Security fixes of the audit (docs/remi-webui/security.md): login
throttling for Basic credentials, the /sdapi and docs logins, fetch-metadata
checks for reads, security headers and CSP, upload pixel caps, local-only
folder actions, extension-install validation, secret redaction and the
/sdapi URL fetcher."""

import base64
import hashlib
import io
import re
import subprocess
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from remi.server.auth import API_AUTH_THROTTLE, MAX_FAILURES_PER_MINUTE, Auth, SecurityMiddleware
from remi.server.testing import server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def options(webui, monkeypatch):
    """Sets ``cmd_opts`` attributes for one test."""

    from modules import shared

    def set_options(**values):
        for key, value in values.items():
            monkeypatch.setattr(shared.cmd_opts, key, value, raising=False)

    set_options(listen=False, server_name=None, ngrok=None, share=False, nowebui=False, api_auth=None, auth=None, auth_file=None, cors_allow_origins=None, cors_allow_origins_regex=None)
    return set_options


def basic(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def png(size=(32, 32)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


# region Logins


def test_basic_credentials_are_throttled_like_the_login_form(webui, tmp_path):
    with server_app(tmp_path, auth=Auth({"ann": "s3cret"}, b"k" * 32)) as server:
        for _ in range(MAX_FAILURES_PER_MINUTE):
            assert server.client.get("/api/v2/queue", headers=basic("ann", "guess")).status_code == 401
        # guessing on: refused before the password is even checked, right or wrong
        assert server.client.get("/api/v2/queue", headers=basic("ann", "s3cret")).status_code == 429
        # the throttle is shared with the login form
        assert server.client.post("/api/v2/auth/login", json={"username": "ann", "password": "s3cret"}).status_code == 429


def test_overlong_credentials_are_refused_unchecked(webui, tmp_path):
    with server_app(tmp_path, auth=Auth({"ann": "s3cret"}, b"k" * 32)) as server:
        assert server.client.post("/api/v2/auth/login", json={"username": "ann", "password": "x" * 5000}).status_code == 422
        assert not server.app.state.auth.check_password("ann", "x" * 5000)


def sdapi_app(auth=None):
    app = FastAPI()
    app.post("/sdapi/v1/interrupt")(lambda: {"ok": True})
    app.get("/sdapi/v1/loras")(lambda: ["an extension route"])
    app.get("/docs/oauth2-redirect")(lambda: "docs")
    app.get("/openapi.json")(lambda: {"openapi": "3"})
    app.add_middleware(SecurityMiddleware, auth=auth or Auth())
    return TestClient(app, base_url="http://127.0.0.1:7860")


def test_api_auth_guards_every_sdapi_route_and_is_throttled(options):
    options(api_auth="api:key")
    client = sdapi_app()
    assert client.get("/sdapi/v1/loras").status_code == 401  # extension routes too
    assert client.get("/sdapi/v1/loras", headers=basic("api", "key")).json() == ["an extension route"]
    assert client.post("/sdapi/v1/interrupt", headers=basic("api", "key")).json() == {"ok": True}
    for _ in range(MAX_FAILURES_PER_MINUTE):
        assert client.post("/sdapi/v1/interrupt", headers=basic("api", "nope")).status_code == 401
    assert client.post("/sdapi/v1/interrupt", headers=basic("api", "key")).status_code == 429
    API_AUTH_THROTTLE._failures.clear()


def test_exposed_sdapi_needs_the_webui_login_without_api_auth(options):
    locked = Auth({"ann": "1"}, b"k" * 32)
    assert sdapi_app(locked).post("/sdapi/v1/interrupt").status_code == 200  # loopback-only: as before
    options(listen=True)
    client = sdapi_app(locked)
    assert client.post("/sdapi/v1/interrupt").status_code == 401
    assert client.get("/sdapi/v1/loras").status_code == 401
    assert client.post("/sdapi/v1/interrupt", headers=basic("ann", "1")).json() == {"ok": True}
    assert sdapi_app().post("/sdapi/v1/interrupt").status_code == 200  # no --auth: nothing to log in with


def test_the_docs_need_a_login_including_their_subpages(options):
    locked = sdapi_app(Auth({"ann": "1"}, b"k" * 32))
    assert locked.get("/docs/oauth2-redirect").status_code == 401
    assert locked.get("/openapi.json", headers=basic("ann", "1")).status_code == 200
    options(api_auth="api:key")  # no WebUI login: --api-auth guards the docs
    client = sdapi_app()
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/openapi.json", headers=basic("api", "key")).status_code == 200


# endregion

# region Browser checks


def test_cross_site_subresource_reads_are_refused(webui, tmp_path):
    with server_app(tmp_path) as server:
        file_id = server.client.post("/api/v2/uploads", files={"file": ("a.png", png(), "image/png")}).json()["id"]
        url = f"/api/v2/files/{file_id}"
        image_tag = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "image"}
        assert server.client.get(url, headers=image_tag).status_code == 403
        # a blind GET that makes the server fetch a URL the page names
        assert server.client.get("/api/v2/extensions/index?url=http://192.168.1.1/", headers={**image_tag, "Sec-Fetch-Site": "same-site"}).status_code == 403
        assert server.client.get(url, headers={"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "image"}).status_code == 200
        assert server.client.get(url, headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}).status_code == 200
        assert server.client.get(url).status_code == 200  # API clients


def test_every_response_has_nosniff_and_a_same_origin_referrer_policy(webui, tmp_path):
    with server_app(tmp_path) as server:
        for response in (server.client.get("/api/v2/queue"), server.client.get("/api/v2/nothing-here"), server.client.post("/api/v2/queue/pause", headers={"Origin": "https://evil.example"})):
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["referrer-policy"] == "same-origin"


def script_hashes(csp):
    script_src = next(part for part in csp.split(";") if part.strip().startswith("script-src"))
    return set(re.findall(r"'sha256-([^']+)'", script_src))


def sha(text):
    return base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()


def test_the_spa_shell_gets_a_csp_that_allows_only_its_own_inline_scripts(options, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    theme = "document.documentElement.dataset.x = 1"
    (dist / "index.html").write_text(f'<!doctype html><html><head><script>{theme}</script><script type="module" src="/assets/app.js"></script></head><body></body></html>')
    options(cors_allow_origins="https://dash.example")
    with server_app(tmp_path / "data", spa_dir=dist) as server:
        response = server.client.get("/")
        csp = response.headers["content-security-policy"]
        root_script = re.search(r"<script>(window\.__REMI_ROOT__[^<]*)</script>", response.text).group(1)
        assert script_hashes(csp) == {sha(theme), sha(root_script)}
        assert "'unsafe-inline'" not in csp.split("style-src")[0] and "'unsafe-eval'" not in csp
        assert "object-src 'none'" in csp and "frame-ancestors 'self' https://dash.example" in csp


def test_the_login_page_has_a_csp_and_an_escaped_root(webui, tmp_path):
    from remi.server.auth import html_headers, login_page

    page = login_page('/x";alert(1);//</script>').decode()
    assert '/x\\";alert(1);//\\u003c/script>/api' in page and "</script>/api" not in page
    with server_app(tmp_path, auth=Auth({"ann": "1"}, b"k" * 32)) as server:
        response = server.client.get("/", headers={"Accept": "text/html"})
        assert response.status_code == 401 and "<label>" in response.text
        inline = re.search(r"<script>(.*?)</script>", response.text, re.S).group(1)
        assert script_hashes(response.headers["content-security-policy"]) == {sha(inline)}
    assert script_hashes(html_headers(b"<p>no scripts</p>")["Content-Security-Policy"]) == set()


# endregion

# region Files


def test_uploads_are_pixel_capped_even_when_pillows_limit_is_off(webui, tmp_path, monkeypatch):
    from remi import files

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)  # as the X/Y/Z plot leaves it
    monkeypatch.setattr(files, "MAX_UPLOAD_PIXELS", 32 * 32)
    with server_app(tmp_path) as server:
        assert server.client.post("/api/v2/uploads", files={"file": ("ok.png", png((32, 32)), "image/png")}).status_code == 201
        response = server.client.post("/api/v2/uploads", files={"file": ("bomb.png", png((33, 32)), "image/png")})
        assert response.status_code == 413 and "too large" in response.json()["detail"]


def test_folder_actions_are_for_a_browser_on_the_webui_machine(webui, tmp_path, monkeypatch):
    from modules import shared
    from remi.server import gallery

    opened = []
    monkeypatch.setattr(gallery, "open_folder", opened.append)
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", False)
    monkeypatch.setattr(shared.opts, "outdir_txt2img_samples", str(tmp_path), raising=False)
    with server_app(tmp_path / "data") as server:
        proxied = server.client.post("/api/v2/outputs/open-folder", json={"folder": "txt2img"}, headers={"X-Forwarded-For": "203.0.113.9"})
        assert proxied.status_code == 403 and "WebUI machine" in proxied.json()["detail"]
        assert opened == []


def test_macos_bundles_are_revealed_not_launched(monkeypatch, tmp_path):
    from remi.server import gallery

    bundle = tmp_path / "Calculator.app"
    bundle.mkdir()
    calls = []
    monkeypatch.setattr(gallery.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gallery.subprocess, "Popen", calls.append)
    gallery.open_folder(str(bundle))
    gallery.open_folder(str(tmp_path))
    assert calls == [["open", "-R", str(bundle)], ["open", str(tmp_path)]]


# endregion

# region Extension installs


def test_git_urls_branches_and_folder_names_are_validated():
    from remi import extensions_admin as admin

    for url in ("https://github.com/a/b", "https://github.com/a/b.git", "ssh://git@host/a/b", "git@github.com:a/b.git", "git://host/a/b"):
        assert admin.valid_git_url(url), url
    for url in ("ssh://-oProxyCommand=calc/x", "git@-oProxyCommand=calc:x", "ext::sh -c calc", "file:///etc", "https://host/a\nb", "https://host:99999/a", "--upload-pack=x", "https://"):
        assert not admin.valid_git_url(url), url
    for name in ("sd-demo", "a.b_c"):
        assert admin.valid_dirname(name)
    for name in ("..", ".hidden", "a/b", "CON", "nul.txt", "x.", ""):
        assert not admin.valid_dirname(name), name
    for branch in ("main", "release/1.2", "v1.0"):
        assert admin.valid_branch(branch)
    for branch in ("-c", "--upload-pack=x", "a..b", "a//b", "x.lock", "/main", "main/"):
        assert not admin.valid_branch(branch), branch


def test_installs_clone_with_network_transports_only(webui, monkeypatch, tmp_path):
    from remi import extensions_admin as admin

    seen = {}

    class Done(Exception):
        pass

    def fake_popen(args, env, **kwargs):
        seen.update(args=args, env=env)
        raise Done

    monkeypatch.setattr(admin, "check_access", lambda: None)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("modules.extensions.extensions_dir", str(tmp_path / "extensions"), raising=False)
    monkeypatch.setattr("modules.paths.data_path", str(tmp_path), raising=False)
    with pytest.raises(Done):
        admin.install("https://example.com/user/sd-demo")
    assert seen["env"]["GIT_ALLOW_PROTOCOL"] == "https:http:ssh:git"
    assert seen["args"][-3:-1] == ["--", "https://example.com/user/sd-demo"]
    with pytest.raises(admin.AdminError):
        admin.install("ssh://-oProxyCommand=touch%20/tmp/x/y")


# endregion

# region Secrets


def test_secrets_are_hidden_from_logged_arguments():
    from modules import redact

    hidden = redact.HIDDEN
    assert redact.argv(["--auth", "ann:pw", "--listen"]) == ["--auth", hidden, "--listen"]
    assert redact.argv(["--api-auth=a:b", "--ngrok", "tok", "--auth-file", "f.txt"]) == [f"--api-auth={hidden}", "--ngrok", hidden, "--auth-file", "f.txt"]
    assert redact.argv(["--api-a", "a:b"]) == ["--api-a", hidden]  # argparse abbreviations
    assert redact.argv(["--api", "--port", "7860"]) == ["--api", "--port", "7860"]  # exact flag, not an abbreviation of --api-auth
    assert redact.argv(["--ckpt", "https://u:tok@host/m"]) == ["--ckpt", f"https://{hidden}@host/m"]
    assert redact.command_line("--listen --auth 'a:b c'") == f"--listen --auth '{hidden}'"
    assert redact.mapping({"deepl_api_key": "k", "civitai_token": "t", "outdir": "x", "steps": 3, "index": "https://a:b@h/"}) == {
        "deepl_api_key": hidden,
        "civitai_token": hidden,
        "outdir": "x",
        "steps": 3,
        "index": f"https://{hidden}@h/",
    }


def test_sysinfo_and_cmd_flags_hide_logins(options, monkeypatch):
    import sys

    from modules import sysinfo
    from modules.api.api import Api

    options(auth="ann:pw", api_auth="api:key", ngrok="tok")
    monkeypatch.setattr(sys, "argv", ["launch.py", "--auth=ann:pw", "--api-auth", "api:key"])
    monkeypatch.setenv("COMMANDLINE_ARGS", "--ngrok tok --listen")
    text = str(sysinfo.get_argv()) + str(sysinfo.get_environment())
    assert "ann:pw" not in text and "api:key" not in text and "tok" not in text
    flags = Api.get_cmd_flags(SimpleNamespace())
    assert flags["auth"] == flags["api_auth"] == flags["ngrok"] == "<hidden>"


# endregion

# region /sdapi


def test_sdapi_options_cannot_change_restricted_settings(webui, monkeypatch):
    from modules import sysinfo
    from modules.api.api import Api

    calls = []
    monkeypatch.setattr(sysinfo, "set_config", lambda req, **kwargs: calls.append(kwargs))
    Api.set_config(SimpleNamespace(), {"api_forbid_local_requests": False})
    assert calls == [{"is_api": True}]


class FakeResponse:
    def __init__(self, status=200, location=None, body=b""):
        self.status_code, self.body = status, body
        self.headers = {"location": location} if location else {}
        self.is_redirect = location is not None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]


def test_sdapi_url_images_check_every_redirect_and_their_size(webui, monkeypatch):
    from modules.api import api

    addresses = {"public.example": "93.184.216.34", "internal.example": "10.0.0.5", "v6.example": "::1"}
    monkeypatch.setattr("socket.getaddrinfo", lambda host, *args, **kwargs: [(0, 0, 0, "", (addresses[host], 0))])
    monkeypatch.setattr(api.opts, "api_forbid_local_requests", True, raising=False)
    monkeypatch.setattr(api.opts, "api_useragent", "", raising=False)
    assert api.verify_url("https://public.example/a.png") and not api.verify_url("https://v6.example/a.png")
    assert not api.verify_url("https://user@internal.example:8080/a.png")

    replies = {"https://public.example/r": FakeResponse(302, "http://internal.example/admin"), "https://public.example/a.png": FakeResponse(body=png())}
    monkeypatch.setattr(api.requests, "get", lambda url, **kwargs: replies[url])
    with pytest.raises(api.HTTPException) as refused:
        api.fetch_url_image_bytes("https://public.example/r")
    assert "local resource" in refused.value.detail
    assert api.fetch_url_image_bytes("https://public.example/a.png") == png()
    monkeypatch.setattr(api, "MAX_URL_IMAGE_BYTES", 10)
    with pytest.raises(api.HTTPException) as too_large:
        api.fetch_url_image_bytes("https://public.example/a.png")
    assert too_large.value.status_code == 413


def test_sdapi_base64_images_are_pixel_capped(webui, monkeypatch):
    from modules.api import api
    from remi import files

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    monkeypatch.setattr(files, "MAX_UPLOAD_PIXELS", 16 * 16)
    assert api.decode_base64_to_image(base64.b64encode(png((16, 16))).decode()).size == (16, 16)
    with pytest.raises(api.HTTPException) as refused:
        api.decode_base64_to_image(base64.b64encode(png((17, 16))).decode())
    assert refused.value.status_code == 413


# endregion


def test_risky_launch_flags_are_warned_about(options):
    from modules import shared
    from remi.server.launch import exposure_warnings

    assert exposure_warnings(shared.cmd_opts) == []
    options(listen=True)
    assert any("without a login" in warning for warning in exposure_warnings(shared.cmd_opts))
    options(auth="a:b")
    assert any("unencrypted" in warning for warning in exposure_warnings(shared.cmd_opts))
    options(cors_allow_origins="*")
    assert any("every web site" in warning for warning in exposure_warnings(shared.cmd_opts))


def test_api_auth_credentials_do_not_lock_the_webui_login(options):
    # both logins set: /sdapi clients send --api-auth credentials, which are
    # wrong for the WebUI; they must not count as failed WebUI logins
    options(api_auth="api:key", listen=True)
    locked = Auth({"ann": "1"}, b"k" * 32)
    client = sdapi_app(locked)
    for _ in range(MAX_FAILURES_PER_MINUTE + 2):
        assert client.post("/sdapi/v1/interrupt", headers=basic("api", "key")).json() == {"ok": True}
    assert not locked.throttle.throttled("testclient")


def test_wrong_webui_basic_credentials_count_once_per_request(options):
    options(listen=True)
    locked = Auth({"ann": "1"}, b"k" * 32)
    client = sdapi_app(locked)
    client.post("/sdapi/v1/interrupt", headers=basic("ann", "wrong"))
    assert len(locked.throttle._failures["testclient"]) == 1
