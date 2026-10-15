"""Startup: scripts that cannot be loaded, launch helpers, the SPA build
step, model selection through PATCH /options and /sdapi on the same app."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules import scripts
from remi.server import runtime
from remi.server.testing import server_app
from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]


class LegacyScript(scripts.Script):
    """An A1111 script: Gradio controls in ui(), no params()."""

    def title(self):
        return "Legacy thing"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        raise AssertionError("ui() is never called")

    def process(self, p, on, amount):
        raise AssertionError("a disabled script must not run")


# region Scripts that cannot be loaded


def test_ui_only_scripts_are_disabled_and_listed(capsys):
    with webui_state(script_classes=[LegacyScript]) as state:
        assert "legacy_thing" not in state.txt2img.by_id and "legacy_thing" not in state.img2img.by_id
        assert "Legacy thing" not in state.txt2img.titles
        assert "declares ui() (Gradio controls) instead of params()" in capsys.readouterr().out
        found = [item for item in runtime.not_ported() if item["title"] == "Legacy thing"]
        assert {item["tab"] for item in found} == {"txt2img", "img2img"} and all(item["status"] == "disabled" and item["reason"] == "declares ui() instead of params()" for item in found)
        # the generation forms still validate and build their arguments
        from remi.core.tabs import processing_args

        assert processing_args("txt2img", {}, {})[1][0] == 0


def test_script_files_that_fail_to_import_are_listed(monkeypatch):
    with webui_state():
        folder = ROOT / "extensions" / "old-tab"
        broken = scripts.ScriptFile(str(folder), "tab.py", str(folder / "scripts" / "tab.py"))
        monkeypatch.setattr(scripts, "scripts_not_loaded", [(broken, "needs the module 'missing_toolkit', which is not installed")])
        assert {"kind": "script", "id": "tab.py", "title": "tab.py", "extension": "old-tab", "status": "disabled", "reason": "needs the module 'missing_toolkit', which is not installed"} in runtime.not_ported()


def test_a_script_that_imports_a_missing_module_is_skipped(tmp_path, monkeypatch):
    script = tmp_path / "scripts" / "needs_toolkit.py"
    script.parent.mkdir()
    script.write_text("import missing_toolkit_for_test\n")
    monkeypatch.setattr(scripts, "list_scripts", lambda dirname, ext, include_extensions=True: [scripts.ScriptFile(str(tmp_path), script.name, str(script))] if dirname == "scripts" else [])
    saved = (scripts.scripts_txt2img, scripts.scripts_img2img, scripts.scripts_postproc, list(scripts.scripts_data), list(scripts.postprocessing_scripts_data))
    try:
        scripts.load_scripts()
        assert [(item.filename, reason) for item, reason in scripts.scripts_not_loaded] == [("needs_toolkit.py", "needs the module 'missing_toolkit_for_test', which is not installed")]
    finally:
        scripts.scripts_txt2img, scripts.scripts_img2img, scripts.scripts_postproc = saved[:3]
        scripts.scripts_data[:], scripts.postprocessing_scripts_data[:] = saved[3], saved[4]
        scripts.scripts_not_loaded.clear()


# endregion

# region Startup


def run_python(code: str, timeout: int = 300) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "modules_forge" / "packages")])}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def test_the_webui_starts_and_generates(tmp_path):
    # Startup (forge init, imports, scripts), the app with /sdapi and every
    # module a job runs, in a fresh interpreter (the suites also run in an
    # environment with nothing but requirements.txt installed).
    code = f"""
import sys
sys.argv = ["launch.py", "--skip-prepare-environment", "--skip-version-check", "--skip-torch-cuda-test", "--data-dir", {str(tmp_path)!r}]
import webui
from modules import scripts
from remi.jobs import run_inline
from remi.server import Services, create_app, runtime
from remi.server.auth import Auth
runtime.prepare()
for name in ("modules.txt2img", "modules.img2img", "modules.postprocessing", "modules.extras", "modules.api.api", "remi.jobs.runner", "remi.server.gallery", "remi.server.routes.extensions"):
    __import__(name)
app = create_app(services=Services.create({str(tmp_path / "remi")!r}, dispatch=run_inline), auth=Auth(), sdapi=True, start_services=False)
assert any(getattr(route, "path", "") == "/sdapi/v1/txt2img" for route in app.routes)
# every built-in script and Extras operation loads
assert runtime.not_ported() == [], runtime.not_ported()
print("ok")
"""
    result = run_python(code)
    assert result.returncode == 0 and "ok" in result.stdout, result.stdout[-3000:] + result.stderr[-6000:]
    assert "Error loading script" not in result.stdout + result.stderr


def test_the_server_path_starts_alone():
    # The API app itself (queue, files, networks) starts without the WebUI runtime
    code = """
import sys, tempfile
sys.argv = sys.argv[:1]
from remi.jobs import run_inline
from remi.server import Services, create_app
from remi.server.auth import Auth
import remi.networks, remi.realtime, remi.files, remi.server.launch, remi.server.runtime
services = Services.create(tempfile.mkdtemp(), dispatch=run_inline)
app = create_app(services=services, auth=Auth(), sdapi=False, start_services=False)
from fastapi.testclient import TestClient
with TestClient(app, base_url="http://127.0.0.1:7860") as client:
    assert client.get("/api/v2/queue").status_code == 200
for name in ("modules.progress", "modules.initialize_util", "modules.script_callbacks", "modules.call_queue", "modules.cmd_args"):
    __import__(name)
print("ok")
"""
    result = run_python(code)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stdout[-2000:] + result.stderr[-4000:]


# endregion

# region Launch helpers


def test_bind_host_and_ports(monkeypatch):
    import socket
    from types import SimpleNamespace

    from remi.server import launch

    assert launch.bind_host(SimpleNamespace(server_name=None, listen=False)) == "127.0.0.1"
    assert launch.bind_host(SimpleNamespace(server_name=None, listen=True)) == "0.0.0.0"
    assert launch.bind_host(SimpleNamespace(server_name="me.local", listen=True)) == "me.local"
    assert launch.root_path(SimpleNamespace(subpath="/remi/")) == "/remi" and launch.root_path(SimpleNamespace(subpath=None)) == ""

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert not launch.port_free("127.0.0.1", port)
        assert launch.pick_port("127.0.0.1", None, port) != port
        assert launch.pick_port("127.0.0.1", 1234, port) == 1234


def test_the_server_thread_serves_and_stops(webui_tmp):
    import httpx

    from remi.server import launch

    with server_app(webui_tmp) as server:
        port = launch.pick_port("127.0.0.1", None, 18760)
        thread = launch.ServerThread(server.app, host="127.0.0.1", port=port, root="/sub")
        thread.start()
        try:
            assert thread.url == f"http://127.0.0.1:{port}/sub/"
            with httpx.Client(trust_env=False, timeout=5) as client:  # no proxy for loopback
                # behind a proxy that keeps the prefix, and one that strips it
                assert set(client.get(f"http://127.0.0.1:{port}/sub/internal/ping").json()) == {"instance"}
                assert set(client.get(f"http://127.0.0.1:{port}/internal/ping").json()) == {"instance"}
                assert client.get(f"http://127.0.0.1:{port}/sub/api/v2/queue").status_code == 200
        finally:
            thread.stop()
        assert not thread.thread.is_alive()


def test_stopping_does_not_wait_forever_for_a_blocked_request_and_frees_the_port():
    import threading
    import time

    import httpx
    from fastapi import FastAPI

    from remi.server import launch

    app = FastAPI()
    entered, release = threading.Event(), threading.Event()

    @app.get("/block")
    def block():  # e.g. waiting for the generation lock
        entered.set()
        release.wait(20)
        return {}

    port = launch.pick_port("127.0.0.1", None, 18790)
    server = launch.ServerThread(app, host="127.0.0.1", port=port, graceful_shutdown=0.5)
    server.start()

    def call():
        try:
            httpx.get(f"http://127.0.0.1:{port}/block", timeout=20, trust_env=False)
        except httpx.HTTPError:
            pass

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    try:
        assert entered.wait(5)
        started = time.monotonic()
        assert server.stop()
        assert time.monotonic() - started < 8
        assert launch.wait_port_free("127.0.0.1", port, 5)
        again = launch.ServerThread(app, host="127.0.0.1", port=port, graceful_shutdown=0.5)
        again.start()
        assert again.stop()
    finally:
        release.set()


def test_restart_answers_and_leaves_the_exit_to_the_serve_loop(webui_tmp, monkeypatch):
    from modules import restart, shared
    from modules.shared_state import State
    from remi.server import launch

    monkeypatch.setattr(shared, "state", State())
    monkeypatch.setattr(restart, "is_restartable", lambda: True)
    exited = []
    monkeypatch.setattr(restart, "restart_program", lambda: exited.append(True))
    with server_app(webui_tmp) as server:
        assert server.client.post("/api/v2/server/restart").json()["ok"] is True
        assert not exited
        assert shared.state.wait_for_server_command(1) == launch.RESTART_PROGRAM


def test_reload_holds_a_job_that_waits_for_the_lock(webui_tmp):
    import threading
    import time

    lock = threading.Lock()
    with server_app(webui_tmp, lock=lock) as server:
        queue = server.services.queue
        with lock:
            server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "x"}})
            assert wait_for(lambda: queue.busy)
            held = threading.Event()

            def reload():
                with queue.hold():
                    held.set()

            threading.Thread(target=reload, daemon=True).start()
            time.sleep(0.2)
            assert not held.is_set()  # the claimed job runs first
        assert held.wait(5)


def wait_for(predicate, timeout=5):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# endregion

# region Extensions and user files


def test_extensions_are_listed_and_toggled_only_with_access(webui_tmp, monkeypatch):
    from modules import extensions, shared

    fake = SimpleNamespace(name="ext-one", path=str(webui_tmp), enabled=True, is_builtin=False, remote=None, branch=None, version="", commit_date=None, status="", can_update=False, have_info_from_repo=True)
    monkeypatch.setattr(extensions, "extensions", [fake])
    monkeypatch.setattr(shared.opts, "disabled_extensions", [], raising=False)
    monkeypatch.setattr(shared, "config_filename", str(webui_tmp / "config.json"))
    with server_app(webui_tmp) as server:
        listing = server.client.get("/api/v2/extensions").json()
        assert listing["items"][0]["name"] == "ext-one" and listing["items"][0]["enabled"] is True
        monkeypatch.setattr(shared.cmd_opts, "disable_extension_access", True)
        assert server.client.patch("/api/v2/extensions", json={"disabled": ["ext-one"]}).status_code == 403
        assert server.client.post("/api/v2/extensions/install", json={"url": "https://example.com/x.git"}).status_code == 403
        monkeypatch.setattr(shared.cmd_opts, "disable_extension_access", False)
        assert server.client.patch("/api/v2/extensions", json={"disabled": ["nope"]}).status_code == 422
        toggled = server.client.patch("/api/v2/extensions", json={"disabled": ["ext-one"]}).json()
        assert toggled["items"][0]["enabled"] is False and shared.opts.disabled_extensions == ["ext-one"]
        assert server.client.post("/api/v2/extensions/install", json={"url": "file:///etc"}).status_code == 422
        assert server.client.post("/api/v2/extensions/install", json={"url": "https://example.com/x.git", "dirname": "../up"}).status_code == 422


def test_user_css_and_notification_sound_are_served_when_present(webui_tmp, monkeypatch):
    from modules import paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(webui_tmp))
    monkeypatch.setattr(paths_internal, "script_path", str(webui_tmp))
    with server_app(webui_tmp) as server:
        assert server.client.get("/api/v2/user-assets/user.css").status_code == 404
        assert server.client.get("/api/v2/app").json()["user_assets"] == []
        (webui_tmp / "user.css").write_text("body{color:red}")
        (webui_tmp / "notification.mp3").write_bytes(b"ID3")
        css = server.client.get("/api/v2/user-assets/user.css")
        assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
        assert server.client.get("/api/v2/app").json()["user_assets"] == ["notification.mp3", "user.css"]
        assert server.client.get("/api/v2/user-assets/..%2Fsecret").status_code == 404


@pytest.fixture
def webui_tmp(tmp_path):
    with webui_state():
        yield tmp_path


def test_webui_build_always_uses_pnpm_and_its_lockfile_or_explains(tmp_path, capsys, monkeypatch):
    from modules import launch_utils

    tools = {"node": "/n/node", "pnpm": "/n/pnpm", "npm": "/n/npm"}
    assert launch_utils.webui_build_commands(which=tools.get) == [["/n/pnpm", "install", "--frozen-lockfile"], ["/n/pnpm", "run", "build"]]
    via_npx = launch_utils.webui_build_commands(which={"node": "n", "npm": "/n/npm", "npx": "/n/npx"}.get)
    assert via_npx == [["/n/npx", "--yes", launch_utils.PNPM_FALLBACK, "install", "--frozen-lockfile"], ["/n/npx", "--yes", launch_utils.PNPM_FALLBACK, "run", "build"]]
    assert launch_utils.webui_build_commands(which={"pnpm": "p"}.get) is None

    webui = tmp_path / "webui"
    webui.mkdir()
    (webui / "package.json").write_text("{}")
    monkeypatch.delenv("REMI_SKIP_WEBUI_BUILD", raising=False)
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: None)
    assert launch_utils.build_webui(webui) is False
    assert "Node.js was not found" in capsys.readouterr().out

    ran = []
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: [["build"]])

    def fake_run(command, cwd, shell):
        ran.append((command, cwd))
        (webui / "dist").mkdir()
        (webui / "dist" / "index.html").write_text("<html>")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch_utils.subprocess, "run", fake_run)
    assert launch_utils.build_webui(webui) is True and ran == [(["build"], webui)]
    assert launch_utils.build_webui(webui) is True and len(ran) == 1  # already built

    # an update changes the sources: the old build is rebuilt
    (webui / "src").mkdir()
    (webui / "src" / "main.ts").write_text("export {}")
    ran.clear()

    def rebuild(command, cwd, shell):
        ran.append((command, cwd))
        (webui / "dist" / "index.html").write_text("<html>new")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch_utils.subprocess, "run", rebuild)
    assert launch_utils.build_webui(webui) is True and len(ran) == 1
    assert (webui / "dist" / "index.html").read_text() == "<html>new"
    assert launch_utils.build_webui(webui) is True and len(ran) == 1
    # without Node.js an existing (maybe stale) build is still used
    (webui / "src" / "main.ts").write_text("export const changed = 1")
    monkeypatch.setattr(launch_utils, "webui_build_commands", lambda: None)
    assert launch_utils.build_webui(webui) is True and "out of date" in capsys.readouterr().out


# endregion

# region Model selection and /sdapi


def test_model_settings_go_through_the_model_manager(webui_tmp, monkeypatch):
    from modules import sd_models, shared

    calls = []
    monkeypatch.setattr(runtime, "select_checkpoint", lambda name: calls.append(("checkpoint", name)) or True)
    monkeypatch.setattr(runtime, "select_modules", lambda names: calls.append(("modules", names)) or True)
    monkeypatch.setattr(runtime, "select_unet_dtype", lambda dtype: calls.append(("dtype", dtype)) or True)
    monkeypatch.setattr(runtime, "unet_dtypes", lambda: ["Automatic", "float8-e4m3fn"])
    monkeypatch.setattr(sd_models, "get_closet_checkpoint_match", lambda name: object() if name == "anima.safetensors" else None)
    monkeypatch.setattr("remi.networks.module_files", lambda kind: [("qwen_image_vae.safetensors", "/m/VAE/qwen_image_vae.safetensors", "/m/VAE")] if kind == "vae" else [("qwen.safetensors", "/m/te/qwen.safetensors", "/m/te")])
    monkeypatch.setattr(shared, "config_filename", str(webui_tmp / "config.json"))
    with server_app(webui_tmp) as server:
        body = {"sd_model_checkpoint": "anima.safetensors", "forge_additional_modules": ["/elsewhere/qwen.safetensors", "qwen_image_vae.safetensors"], "forge_unet_storage_dtype": "float8-e4m3fn"}
        assert server.client.patch("/api/v2/options", json=body).status_code == 200
        assert calls == [("checkpoint", "anima.safetensors"), ("modules", ["qwen.safetensors", "qwen_image_vae.safetensors"]), ("dtype", "float8-e4m3fn")]
        bad = server.client.patch("/api/v2/options", json={"sd_model_checkpoint": "missing", "forge_additional_modules": ["nope.bin"], "forge_unet_storage_dtype": "int3"})
        assert {error["param"] for error in bad.json()["errors"]} == {"sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype"}


def test_sdapi_is_mounted_on_the_same_app(webui_tmp):
    from fastapi.testclient import TestClient

    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    services = Services.create(webui_tmp, dispatch=run_inline, realtime=False, housekeeping=False)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=True, start_services=False)
        with TestClient(app, base_url="http://127.0.0.1:7860") as client:
            assert client.get("/sdapi/v1/samplers").status_code == 200
            info = client.get("/sdapi/v1/script-info").json()
            assert any(item["name"] == "sampler" for item in info)
            assert client.get("/api/v2/queue").status_code == 200
    finally:
        services.close()


def test_sdapi_infotext_fills_request_fields_and_script_args(webui_tmp):
    from modules.api import api, models

    with server_app(webui_tmp):
        request = models.StableDiffusionTxt2ImgProcessingAPI(infotext="a cat\nNegative prompt: dog\nSteps: 9, Sampler: Euler, CFG scale: 4, Seed: 77, Size: 640x512", cfg_scale=7)
        mentioned = {}
        api.Api.apply_infotext(None, request, "txt2img", script_runner=scripts.scripts_txt2img, mentioned_script_args=mentioned)
        assert request.prompt == "a cat" and request.negative_prompt == "dog" and request.steps == 9 and request.seed == 77
        assert (request.width, request.height) == (640, 512) and request.cfg_scale == 7  # set by the request: kept


# endregion
