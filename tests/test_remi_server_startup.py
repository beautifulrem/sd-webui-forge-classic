"""Startup without a model or a real server: remi.server.runtime (settings,
runners and model selection for a new app) and remi.server.launch (the
serve loop of webui.py, its server commands, ports, browser launch and
temp-dir cleanup). The slow end-to-end startup in a fresh interpreter is in
test_remi_server_runtime.py."""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

from remi.server import launch, runtime
from remi.testing import webui_state


@pytest.fixture
def webui():
    with webui_state() as state:
        yield state


class FakeMainEntry(types.ModuleType):
    """``modules_forge.main_entry`` (the real one scans model folders)."""

    def __init__(self, *, module_list=None):
        super().__init__("modules_forge.main_entry")
        self.calls: list[tuple] = []
        self.module_list = module_list if module_list is not None else {}
        self.forge_unet_storage_dtype_options = {"Automatic": None, "float8-e4m3fn": "fp8"}

    def refresh_models(self):
        self.calls.append(("refresh_models",))
        self.module_list = {"qwen.safetensors": "/m/qwen.safetensors"}

    def refresh_model_loading_parameters(self):
        self.calls.append(("refresh_model_loading_parameters",))

    def checkpoint_change(self, name, preset):
        self.calls.append(("checkpoint_change", name, preset))
        return True

    def modules_change(self, names, preset):
        self.calls.append(("modules_change", names, preset))
        return True

    def dtype_change(self, dtype, preset):
        self.calls.append(("dtype_change", dtype, preset))
        return False


@pytest.fixture
def main_entry(monkeypatch):
    fake = FakeMainEntry()
    monkeypatch.setitem(sys.modules, "modules_forge.main_entry", fake)
    import modules_forge

    monkeypatch.setattr(modules_forge, "main_entry", fake, raising=False)
    return fake


# region runtime: settings, runners, model selection


def test_prepare_registers_settings_runners_and_tab_names(webui, main_entry, monkeypatch):
    from modules import script_callbacks, shared
    from remi import pages

    calls = []
    monkeypatch.setattr(script_callbacks, "ui_settings_callback", lambda: calls.append("ui_settings"))
    monkeypatch.setattr(runtime, "initialize_runners", lambda: calls.append("runners"))
    monkeypatch.setattr(runtime, "initialize_model_selection", lambda: calls.append("models"))
    monkeypatch.setattr(pages, "tab_titles", lambda: ["Prompt Workshop", "Extras", "Prompt Workshop", "Joy Caption"])
    monkeypatch.setattr(shared, "tab_names", [])

    runtime.prepare(models=False)
    assert calls == ["ui_settings", "runners"]
    # the built-in tabs first, then each extension page once (a page titled like a built-in tab is that tab)
    assert shared.tab_names == ["txt2img", "img2img", "Extras", "PNG Info", "Gallery", "Networks", "Prompt Workshop", "Joy Caption"]
    assert "callbacks" in {getattr(info, "section", ("",))[0] for info in shared.opts.data_labels.values()}

    calls.clear()
    runtime.prepare()
    assert calls == ["ui_settings", "runners", "models"]
    assert shared.tab_names.count("Prompt Workshop") == 1  # rebuilt, not appended to (Reload UI)


def test_page_titles_without_the_pages_module(monkeypatch):
    import remi

    monkeypatch.setitem(sys.modules, "remi.pages", None)  # import fails
    monkeypatch.delattr(remi, "pages", raising=False)
    assert runtime._page_titles() == []


def test_initialize_runners_assigns_every_argument_slot(webui, monkeypatch):
    from modules import scripts
    from remi.core import extras

    assigned = []
    monkeypatch.setattr(extras, "assign_args", lambda runner: assigned.append(runner))
    runtime.initialize_runners()
    assert assigned == [scripts.scripts_postproc]
    for runner in (scripts.scripts_txt2img, scripts.scripts_img2img):
        assert runner.by_id["seed"].args_to > runner.by_id["seed"].args_from

    monkeypatch.setattr(scripts, "scripts_postproc", None)
    assigned.clear()
    runtime.initialize_runners()
    assert assigned == []


def test_a_checkpoint_is_picked_when_none_is_selected(webui, main_entry, monkeypatch):
    from modules import sd_models, shared

    listed = []
    monkeypatch.setattr(sd_models, "checkpoints_list", {})

    def list_models():
        listed.append(True)
        sd_models.checkpoints_list.update({"first": SimpleNamespace(name="anima.safetensors"), "second": SimpleNamespace(name="other.safetensors")})

    monkeypatch.setattr(sd_models, "list_models", list_models)
    monkeypatch.setattr(shared.opts, "set", lambda key, value: shared.opts.data.__setitem__(key, value))
    shared.opts.data["sd_model_checkpoint"] = "None"
    runtime.initialize_model_selection()
    assert listed == [True] and shared.opts.data["sd_model_checkpoint"] == "anima.safetensors"
    assert main_entry.calls == [("refresh_models",), ("refresh_model_loading_parameters",)]

    # a selected checkpoint is kept, and no scan happens for it
    shared.opts.data["sd_model_checkpoint"] = "mine.safetensors"
    runtime.initialize_model_selection()
    assert listed == [True] and shared.opts.data["sd_model_checkpoint"] == "mine.safetensors"


def test_no_checkpoint_at_all_leaves_the_selection_empty(webui, main_entry, monkeypatch):
    from modules import sd_models, shared

    monkeypatch.setattr(sd_models, "checkpoints_list", {})
    monkeypatch.setattr(sd_models, "list_models", lambda: None)
    shared.opts.data["sd_model_checkpoint"] = ""
    runtime.initialize_model_selection()
    assert shared.opts.data["sd_model_checkpoint"] == ""


def test_loading_the_selected_model(webui, main_entry, monkeypatch):
    from modules import sd_models

    reloads = []
    monkeypatch.setattr(sd_models, "model_data", SimpleNamespace(forge_hash="abc"))
    monkeypatch.setattr(sd_models, "forge_model_reload", lambda: reloads.append(sd_models.model_data.forge_hash))
    runtime.load_selected_model()
    runtime.load_selected_model(force=True)
    assert reloads == ["abc", ""]  # force: the loaded model counts as different
    assert main_entry.calls == [("refresh_model_loading_parameters",)] * 2


def test_selection_changes_go_through_the_main_entry_with_the_preset(webui, monkeypatch):
    from modules import shared

    fake = FakeMainEntry(module_list={})
    monkeypatch.setattr(runtime, "_main_entry", lambda: fake)
    monkeypatch.setattr(shared.opts, "forge_preset", "anima", raising=False)
    assert runtime.select_checkpoint("anima.safetensors") is True
    assert runtime.select_modules(["qwen.safetensors"]) is True
    assert runtime.select_unet_dtype("float8-e4m3fn") is False
    runtime.refresh_models()
    # the module list is scanned before the first module change
    assert fake.calls == [("checkpoint_change", "anima.safetensors", "anima"), ("refresh_models",), ("modules_change", ["qwen.safetensors"], "anima"), ("dtype_change", "float8-e4m3fn", "anima"), ("refresh_models",)]
    fake.calls.clear()
    runtime.select_modules([])
    assert fake.calls == [("modules_change", [], "anima")]


def test_the_main_entry_is_the_forge_module(monkeypatch):
    fake = types.ModuleType("modules_forge.main_entry")
    monkeypatch.setitem(sys.modules, "modules_forge.main_entry", fake)
    import modules_forge

    monkeypatch.setattr(modules_forge, "main_entry", fake, raising=False)
    assert runtime._main_entry() is fake


def test_unet_dtypes_with_and_without_the_main_entry(webui, monkeypatch):
    from modules import shared

    monkeypatch.setitem(sys.modules, "modules_forge.main_entry", FakeMainEntry())
    assert runtime.unet_dtypes() == ["Automatic", "float8-e4m3fn"]
    monkeypatch.delitem(sys.modules, "modules_forge.main_entry")
    monkeypatch.setattr(shared.opts, "forge_unet_storage_dtype", "bfloat16", raising=False)
    assert runtime.unet_dtypes() == ["bfloat16"]
    monkeypatch.setattr(shared.opts, "forge_unet_storage_dtype", "", raising=False)
    assert runtime.unet_dtypes() == ["Automatic"]


def test_the_extension_of_a_script_file(tmp_path, monkeypatch):
    from modules import paths_internal

    builtin, user = tmp_path / "extensions-builtin", tmp_path / "extensions"
    (builtin / "sd_forge_lora" / "scripts").mkdir(parents=True)
    (user / "my-ext" / "scripts").mkdir(parents=True)
    monkeypatch.setattr(paths_internal, "extensions_builtin_dir", str(builtin))
    monkeypatch.setattr(paths_internal, "extensions_dir", str(user))
    assert runtime._extension_of(str(builtin / "sd_forge_lora" / "scripts" / "lora.py")) == "sd_forge_lora"
    assert runtime._extension_of(str(user / "my-ext" / "scripts" / "x.py")) == "my-ext"
    assert runtime._extension_of(str(tmp_path / "scripts" / "loopback.py")) is None
    assert runtime._extension_of(str(tmp_path / "extensions-builtin-other" / "a.py")) is None  # a prefix is not a parent
    assert runtime._extension_of(None) is None and runtime._extension_of("") is None


def test_scripts_whose_title_fails_are_listed_by_file_name():
    class Broken:
        filename = "/x/scripts/broken.py"

        def title(self):
            raise RuntimeError("no title")

    assert runtime._title(Broken()) == "broken.py"
    assert runtime._title(SimpleNamespace(title=lambda: 42)) == "42"
    nameless = Broken()
    nameless.filename = None
    assert runtime._title(nameless) == "script"


def test_extras_operations_without_params_are_listed(webui, monkeypatch):
    from modules import scripts

    operation = SimpleNamespace(name=None, filename=None)
    monkeypatch.setattr(scripts.scripts_postproc, "disabled_scripts", [(operation, "declares ui() instead of params()")], raising=False)
    assert {"kind": "extras", "id": "", "title": "SimpleNamespace", "tab": "extras", "extension": None, "status": "disabled", "reason": "declares ui() instead of params()"} in runtime.not_ported()


# endregion

# region launch helpers


def test_pick_port_gives_up_after_the_attempts(monkeypatch):
    monkeypatch.setattr(launch, "port_free", lambda host, port: False)
    with pytest.raises(OSError, match="no free port in 7860-7879; use --port"):
        launch.pick_port("127.0.0.1", None, 7860)


def test_wait_port_free_times_out(monkeypatch):
    monkeypatch.setattr(launch, "port_free", lambda host, port: False)
    monkeypatch.setattr(launch.time, "sleep", lambda seconds: None)
    assert launch.wait_port_free("127.0.0.1", 1, timeout=0) is False
    monkeypatch.setattr(launch, "port_free", lambda host, port: True)
    assert launch.wait_port_free("127.0.0.1", 1, timeout=0) is True


def test_port_free_on_ipv6():
    assert launch.port_free("::1", 0) in (True, False)  # no error for IPv6 hosts (False where IPv6 is off)


@pytest.mark.parametrize(
    ("path", "kind", "expected"),
    [("/remi/remi/api/v2/queue", "http", "/remi/api/v2/queue"), ("/remi/remi", "websocket", "/remi"), ("/remi/api/v2/queue", "http", "/remi/api/v2/queue"), ("/remi/remi/x", "lifespan", "/remi/remi/x")],
)
def test_root_path_fix_drops_a_doubled_prefix(path, kind, expected):
    import asyncio

    seen = []

    async def app(scope, receive, send):
        seen.append(scope)

    scope = {"type": kind, "path": path, "raw_path": path.encode()}
    asyncio.run(launch.RootPathFix(app, "/remi")(scope, None, None))
    assert seen[0]["path"] == expected
    assert ("raw_path" in seen[0]) == (expected == path)


def test_cleanup_temp_dir_removes_only_pngs(webui, tmp_path, monkeypatch):
    from modules import shared

    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    for name in ("one.png", "a/two.png", "a/b/three.png", "keep.txt", "a/keep.jpg"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(shared.opts, "temp_dir", str(tmp_path), raising=False)
    launch.cleanup_temp_dir()
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()) == ["a/keep.jpg", "keep.txt"]

    for value in ("", str(tmp_path / "missing")):
        monkeypatch.setattr(shared.opts, "temp_dir", value, raising=False)
        launch.cleanup_temp_dir()  # nothing to do, no error


@pytest.mark.parametrize(
    ("autolaunch", "option", "non_local", "expected"),
    [(True, "Disable", False, True), (False, "Remote", True, True), (False, "Local", False, True), (False, "Local", True, False), (False, "Disable", False, False)],
)
def test_auto_launch(webui, monkeypatch, autolaunch, option, non_local, expected):
    from modules import shared

    monkeypatch.setattr(shared.opts, "auto_launch_browser", option, raising=False)
    assert launch._auto_launch(SimpleNamespace(autolaunch=autolaunch, webui_is_non_local=non_local)) is expected


def test_wait_for_command_skips_unknown_commands(webui, monkeypatch, capsys):
    from modules import shared

    commands = iter([None, "dance", "restart"])
    monkeypatch.setattr(shared, "state", SimpleNamespace(wait_for_server_command=lambda timeout: next(commands)))
    assert launch._wait_for_command() == "restart"
    assert "Unknown server command: dance" in capsys.readouterr().out

    def interrupted(timeout):
        raise KeyboardInterrupt

    monkeypatch.setattr(shared, "state", SimpleNamespace(wait_for_server_command=interrupted))
    assert launch._wait_for_command() == "stop"
    assert "Caught KeyboardInterrupt" in capsys.readouterr().out


class FakeServerThread:
    """``ServerThread`` without uvicorn: records its arguments."""

    instances: list[FakeServerThread] = []
    fail_starts = 0

    def __init__(self, app, *, host, port, root="", ssl_keyfile=None, ssl_certfile=None, timeout_keep_alive=30, graceful_shutdown=10):
        self.app, self.host, self.port, self.root = app, host, port, root
        self.ssl = (ssl_keyfile, ssl_certfile)
        self.timeout_keep_alive = timeout_keep_alive
        self.url = f"http://{host}:{port}{root}/"
        self.stops: list[float | None] = []
        self.started = False
        FakeServerThread.instances.append(self)

    def start(self, timeout=60):
        if FakeServerThread.fail_starts:
            FakeServerThread.fail_starts -= 1
            raise RuntimeError("the server failed to start")
        self.started = True

    def stop(self, timeout=None):
        self.stops.append(timeout)
        return True


@pytest.fixture
def fake_servers(monkeypatch):
    FakeServerThread.instances = []
    FakeServerThread.fail_starts = 0
    monkeypatch.setattr(launch, "ServerThread", FakeServerThread)
    monkeypatch.setattr(launch.time, "sleep", lambda seconds: None)
    return FakeServerThread


def opts_for_start(**overrides):
    return SimpleNamespace(**{"subpath": "/sub", "tls_keyfile": None, "tls_certfile": None, "timeout_keep_alive": 45, **overrides})


def test_start_server_retries_while_the_old_server_holds_the_port(fake_servers, monkeypatch):
    free = iter([False, True, True])
    monkeypatch.setattr(launch, "wait_port_free", lambda host, port: next(free))
    fake_servers.fail_starts = 1
    server = launch._start_server("app", "127.0.0.1", 7860, opts_for_start())
    assert len(fake_servers.instances) == 2 and server is fake_servers.instances[1] and server.started
    assert fake_servers.instances[0].stops == [5]  # the failed one is cleaned up
    assert (server.root, server.timeout_keep_alive) == ("/sub", 45)


def test_start_server_gives_up_after_the_attempts(fake_servers, monkeypatch):
    monkeypatch.setattr(launch, "wait_port_free", lambda host, port: True)
    fake_servers.fail_starts = 3
    with pytest.raises(RuntimeError):
        launch._start_server("app", "127.0.0.1", 7860, opts_for_start(), attempts=3)
    assert len(fake_servers.instances) == 3


# endregion

# region The serve loop


class FakeQueue:
    def __init__(self, stops_in_time=True):
        self.events: list[str] = []
        self.stops_in_time = stops_in_time

    def stop(self, timeout, interrupt):
        self.events.append(f"stop(timeout={timeout}, interrupt={interrupt})")
        return self.stops_in_time

    def hold(self):
        queue = self

        class Hold:
            def __enter__(self):
                queue.events.append("hold")

            def __exit__(self, *exc):
                queue.events.append("release")

        return Hold()


@pytest.fixture
def serve_env(webui, fake_servers, monkeypatch, tmp_path):
    """Everything ``serve`` calls, recorded in ``env.events``."""

    import remi.server
    from modules import extension_route_guard, initialize, script_callbacks, shared, timer
    from modules.shared_cmd_options import cmd_opts

    env = SimpleNamespace(events=[], commands=[], queue=FakeQueue(), browser=[], apps=[])
    services = SimpleNamespace(queue=env.queue, stop=lambda: env.events.append("services.stop"))

    def create_app(services, webui):
        app = SimpleNamespace(name=f"app{len(env.apps) + 1}", webui=webui)
        env.apps.append(app)
        env.events.append(f"create_app({app.name})")
        return app

    monkeypatch.setattr(remi.server, "default_services", lambda: services)
    monkeypatch.setattr(remi.server, "create_app", create_app)
    monkeypatch.setattr(runtime, "prepare", lambda: env.events.append("prepare"))
    monkeypatch.setattr(script_callbacks, "before_ui_callback", lambda: env.events.append("before_ui"))
    monkeypatch.setattr(script_callbacks, "app_started_callback", lambda demo, app: env.events.append(f"app_started({app.name})"))
    monkeypatch.setattr(script_callbacks, "app_reload_callback", lambda: env.events.append("app_reload"))
    monkeypatch.setattr(script_callbacks, "script_unloaded_callback", lambda: env.events.append("script_unloaded"))
    monkeypatch.setattr(extension_route_guard, "snapshot_routes", lambda app: {"before"})
    monkeypatch.setattr(extension_route_guard, "guard_new_routes", lambda app, before: env.events.append(f"guard({app.name}, {sorted(before)})") or 0)
    monkeypatch.setattr(initialize, "initialize_rest", lambda: env.events.append("initialize_rest"))
    monkeypatch.setattr(launch, "wait_port_free", lambda host, port: True)
    monkeypatch.setattr(launch, "pick_port", lambda host, requested, default: env.events.append(f"pick_port({host}, {requested}, {default})") or 7999)
    monkeypatch.setattr(launch, "cleanup_temp_dir", lambda: env.events.append("cleanup_temp_dir"))
    monkeypatch.setattr(launch, "_wait_for_command", lambda: env.commands.pop(0))
    monkeypatch.setattr(launch.webbrowser, "open", env.browser.append)
    monkeypatch.setattr(timer, "startup_record", None)
    monkeypatch.delenv("SD_WEBUI_RESTARTING", raising=False)
    monkeypatch.setattr(shared.opts, "clean_temp_dir_at_start", False, raising=False)
    monkeypatch.setattr(shared.opts, "auto_launch_browser", "Disable", raising=False)
    for name, value in {"port": None, "share": False, "ngrok": None, "listen": False, "server_name": None, "subpath": None, "tls_keyfile": None, "tls_certfile": None, "autolaunch": False}.items():
        monkeypatch.setattr(cmd_opts, name, value, raising=False)
    env.cmd_opts = cmd_opts
    env.timer = timer.Timer(print_log=False)
    yield env
    os.environ.pop("SD_WEBUI_RESTARTING", None)


def test_serve_builds_the_app_serves_and_stops(serve_env, capsys):
    serve_env.commands[:] = ["stop"]
    launch.serve(webui=True, startup_timer=serve_env.timer)
    assert serve_env.events == ["pick_port(127.0.0.1, None, 7860)", "before_ui", "prepare", "create_app(app1)", "app_started(app1)", "guard(app1, ['before'])", "services.stop"]
    assert serve_env.queue.events == ["stop(timeout=30, interrupt=True)"]
    (server,) = FakeServerThread.instances
    assert server.app is serve_env.apps[0] and server.app.webui is True and (server.host, server.port) == ("127.0.0.1", 7999)
    assert server.stops == [None]
    out = capsys.readouterr().out
    assert "Running on local URL:  http://127.0.0.1:7999/" in out and "Stopping server..." in out
    assert serve_env.browser == []
    from modules import timer

    assert timer.startup_record is not None


def test_serve_api_only_uses_the_api_port(serve_env):
    serve_env.commands[:] = ["stop"]
    launch.serve(webui=False, startup_timer=serve_env.timer)
    assert serve_env.events[0] == "pick_port(127.0.0.1, None, 7861)"
    assert FakeServerThread.instances[0].app.webui is False


def test_reload_ui_holds_the_queue_until_the_new_app_serves(serve_env, capsys):
    serve_env.commands[:] = ["restart", "stop"]
    launch.serve(webui=True, startup_timer=serve_env.timer)
    first_cycle = ["before_ui", "prepare", "create_app(app1)", "app_started(app1)", "guard(app1, ['before'])"]
    reload = ["app_reload", "script_unloaded", "initialize_rest"]
    second_cycle = ["before_ui", "prepare", "create_app(app2)", "app_started(app2)", "guard(app2, ['before'])"]
    assert serve_env.events == ["pick_port(127.0.0.1, None, 7860)", *first_cycle, *reload, *second_cycle, "services.stop"]
    # held from the reload until the second server started, then stopped at the end
    assert serve_env.queue.events == ["hold", "release", "stop(timeout=30, interrupt=True)"]
    first, second = FakeServerThread.instances
    assert first.stops == [None] and second.stops == [None] and second.app.name == "app2"
    assert os.environ["SD_WEBUI_RESTARTING"] == "1"
    assert "Reloading UI: waiting for the running job..." in capsys.readouterr().out


def test_restart_program_stops_everything_then_restarts(serve_env, monkeypatch, capsys):
    from modules import restart

    serve_env.queue.stops_in_time = False
    restarted = []
    monkeypatch.setattr(restart, "restart_program", lambda: restarted.append(True))
    serve_env.commands[:] = [launch.RESTART_PROGRAM]
    launch.serve(webui=True, startup_timer=serve_env.timer)
    assert restarted == [True] and serve_env.events[-1] == "services.stop"
    out = capsys.readouterr().out
    assert "Restarting..." in out and "The running job did not stop in time" in out


def test_serve_options_temp_dir_browser_share_ngrok_and_warnings(serve_env, monkeypatch, capsys):
    import modules
    from modules import shared

    connected = []
    ngrok = types.ModuleType("modules.ngrok")  # the real one opens a tunnel
    ngrok.connect = lambda token, port, options: connected.append((token, port, options))
    monkeypatch.setitem(sys.modules, "modules.ngrok", ngrok)
    monkeypatch.setattr(modules, "ngrok", ngrok, raising=False)
    monkeypatch.setattr(shared.opts, "clean_temp_dir_at_start", True, raising=False)
    monkeypatch.setattr(launch, "exposure_warnings", lambda cmd_opts: ["the WebUI is reachable from other machines without a login"])
    for name, value in {"share": True, "ngrok": "token", "ngrok_options": {"region": "eu"}, "autolaunch": True, "port": 9000}.items():
        monkeypatch.setattr(serve_env.cmd_opts, name, value, raising=False)
    serve_env.commands[:] = ["restart", "stop"]
    with pytest.MonkeyPatch.context() as warn:
        records = []
        warn.setattr(launch.log, "warning", lambda message, *args: records.append(message % args if args else message))
        launch.serve(webui=True, startup_timer=serve_env.timer)
    assert connected == [("token", 7999, {"region": "eu"})]
    assert serve_env.events.count("cleanup_temp_dir") == 2  # at every (re)start
    assert any("--share is not supported" in record for record in records)
    assert "Security warning: the WebUI is reachable from other machines without a login" in capsys.readouterr().out
    # the browser opens once: not again after "Reload UI"
    assert serve_env.browser == ["http://127.0.0.1:7999/"]


# endregion
