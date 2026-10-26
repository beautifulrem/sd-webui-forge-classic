"""Running the app with uvicorn on a background thread (the process main
thread belongs to ``modules_forge.main_thread``), and the serve loop of
``webui.py``: build the app, start it, wait for a server command, rebuild on
"Reload UI"."""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
import time
import webbrowser
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_PORT = 7860
DEFAULT_API_PORT = 7861
PORT_ATTEMPTS = 20
GRACEFUL_SHUTDOWN = 10
"""seconds uvicorn waits for open requests before cancelling them"""
RESTART_PROGRAM = "restart-program"
"""server command: stop and let the launcher script start the WebUI again"""


def bind_host(cmd_opts: Any) -> str:
    """``--server-name``, ``0.0.0.0`` with ``--listen``, else loopback, with
    or without ``--nowebui`` (``extension_route_guard`` reads it from here)."""

    if getattr(cmd_opts, "server_name", None):
        return cmd_opts.server_name
    return "0.0.0.0" if getattr(cmd_opts, "listen", False) else "127.0.0.1"


def exposure_warnings(cmd_opts: Any) -> list[str]:
    """Risky combinations of launch flags, printed at startup
    (``docs/remi-webui/security.md``, "Running exposed")."""

    from modules import extension_route_guard as guard

    warnings: list[str] = []
    exposed = not guard._is_loopback_only(cmd_opts)
    has_login = bool(getattr(cmd_opts, "auth", None) or getattr(cmd_opts, "auth_file", None))
    if exposed and not has_login and not getattr(cmd_opts, "nowebui", False):
        warnings.append("the WebUI is reachable from other machines without a login: anyone who can reach it (or make a browser on your network reach it) controls it; add --auth or --auth-file")
    if exposed and getattr(cmd_opts, "nowebui", False) and not getattr(cmd_opts, "api_auth", None):
        warnings.append("the API is reachable from other machines without a login; add --api-auth")
    if exposed and getattr(cmd_opts, "enable_insecure_extension_access", False):
        warnings.append("--enable-insecure-extension-access: logged-in users (everyone, without --auth) can install and run arbitrary code")
    if has_login and getattr(cmd_opts, "api_no_auth_on_localhost", False) and not getattr(cmd_opts, "api_auth", None):
        warnings.append("--api-no-auth-on-localhost: every program on this machine (and a tunnel that adds no proxy headers) can use /sdapi and extension APIs without the login")
    if "*" in guard._cors_origins(cmd_opts):
        warnings.append("--cors-allow-origins=*: every web site may send requests with your login and read the answers")
    tls = getattr(cmd_opts, "tls_keyfile", None) and getattr(cmd_opts, "tls_certfile", None)
    if exposed and has_login and not tls and not getattr(cmd_opts, "ngrok", None):  # ngrok serves HTTPS
        warnings.append("logins travel unencrypted: use --tls-keyfile/--tls-certfile or an HTTPS reverse proxy")
    return warnings


def port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def wait_port_free(host: str, port: int, timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while not port_free(host, port):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def pick_port(host: str, requested: int | None, default: int) -> int:
    """``requested`` as given; otherwise the first free port from ``default``
    (as Gradio did)."""

    if requested:
        return requested
    for port in range(default, default + PORT_ATTEMPTS):
        if port_free(host, port):
            return port
    raise OSError(f"no free port in {default}-{default + PORT_ATTEMPTS - 1}; use --port")


def root_path(cmd_opts: Any) -> str:
    return f"/{cmd_opts.subpath.strip('/')}" if cmd_opts.subpath else ""


class RootPathFix:
    """``--subpath`` behind proxies that strip the prefix and proxies that do
    not: uvicorn prefixes every path with ``root_path``, so an unstripped
    request arrives with it twice; drop one."""

    def __init__(self, app: Any, root: str) -> None:
        self.app, self.root = app, root

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if self.root and scope["type"] in ("http", "websocket"):
            doubled = self.root + self.root
            path = scope.get("path", "")
            if path == doubled or path.startswith(doubled + "/"):
                scope = {**scope, "path": path[len(self.root) :]}
                scope.pop("raw_path", None)
        await self.app(scope, receive, send)


class ServerThread:
    """One uvicorn server on a daemon thread."""

    def __init__(self, app: Any, *, host: str, port: int, root: str = "", ssl_keyfile: str | None = None, ssl_certfile: str | None = None, timeout_keep_alive: int = 30, graceful_shutdown: float = GRACEFUL_SHUTDOWN, forwarded_allow_ips: str | None = None) -> None:
        """``forwarded_allow_ips`` (``--forwarded-allow-ips``): the proxies
        whose ``X-Forwarded-For`` / ``-Proto`` replace the peer address and
        scheme. Without it no header is trusted: uvicorn would otherwise
        trust them from loopback peers, letting any client of a same-machine
        tunnel pick its own address and dodge the login throttle
        (``remi.server.auth.client_key``)."""

        import uvicorn

        config = uvicorn.Config(
            RootPathFix(app, root) if root else app,
            host=host,
            port=port,
            root_path=root,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            timeout_keep_alive=timeout_keep_alive,
            timeout_graceful_shutdown=graceful_shutdown,
            proxy_headers=bool(forwarded_allow_ips),
            forwarded_allow_ips=forwarded_allow_ips or None,
            # No permessage-deflate: the WebSocket's large frames are live previews, already
            # JPEG, so deflating them (about 2-12 ms of server CPU per frame, the same again
            # in the browser to inflate) saved nothing while costing the generation time on a
            # busy machine; the JSON events are a few hundred bytes.
            ws_per_message_deflate=False,
            log_level="warning",
            lifespan="on",
        )
        self.graceful_shutdown = graceful_shutdown
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="remi-uvicorn", daemon=True)
        self.url = f"{'https' if ssl_certfile else 'http'}://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}{root}/"

    def start(self, timeout: float = 60) -> None:
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("the server failed to start (see the log above)")
            if time.monotonic() > deadline:
                raise TimeoutError("the server did not start in time")
            time.sleep(0.05)

    def stop(self, timeout: float | None = None) -> bool:
        """Stop serving: open requests get ``graceful_shutdown`` seconds,
        then a forced exit; returns whether the server thread ended."""

        self.server.should_exit = True
        self.thread.join(self.graceful_shutdown + 5 if timeout is None else timeout)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(5)
        return not self.thread.is_alive()


def cleanup_temp_dir() -> None:
    """``clean_temp_dir_at_start``: delete PNGs left in ``opts.temp_dir``."""

    from modules import shared

    temp_dir = shared.opts.temp_dir
    if not temp_dir or not os.path.isdir(temp_dir):
        return
    for root, _dirs, files in os.walk(temp_dir, topdown=False):
        for name in files:
            if name.endswith(".png"):
                os.remove(os.path.join(root, name))


def serve(*, webui: bool, startup_timer: Any = None) -> None:
    """The server loop of ``webui.py`` (runs on its own thread)."""

    from modules import extension_route_guard, initialize, script_callbacks, shared, timer
    from modules.shared_cmd_options import cmd_opts
    from remi.server import create_app, default_services, runtime

    startup_timer = startup_timer or timer.startup_timer
    services = default_services()
    host = bind_host(cmd_opts)
    port = pick_port(host, cmd_opts.port, DEFAULT_PORT if webui else DEFAULT_API_PORT)
    if cmd_opts.share:
        log.warning("--share is not supported by the Remi server; use --listen, a reverse proxy or --ngrok")
    for warning in exposure_warnings(cmd_opts):
        print(f"Security warning: {warning}")
    if cmd_opts.ngrok is not None:
        from modules import ngrok

        print("ngrok authtoken detected, trying to connect...")
        ngrok.connect(cmd_opts.ngrok, port, cmd_opts.ngrok_options)

    holding = contextlib.ExitStack()  # the queue is held from a reload until the new app serves
    while True:
        if shared.opts.clean_temp_dir_at_start:
            cleanup_temp_dir()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        runtime.prepare()
        startup_timer.record("prepare runtime")

        app = create_app(services=services, webui=webui)
        startup_timer.record("create app")

        with startup_timer.subcategory("app_started_callback"):
            routes_before = extension_route_guard.snapshot_routes(app)
            script_callbacks.app_started_callback(None, app)
            # Routes added by extensions: refuse cross-site writes (CORS only
            # hides responses from other sites, it does not stop requests).
            extension_route_guard.guard_new_routes(app, routes_before)

        server = _start_server(app, host, port, cmd_opts)
        holding.close()
        startup_timer.record("start server")
        timer.startup_record = startup_timer.dump()
        print(f"Running on local URL:  {server.url}")
        print(f"Startup time: {startup_timer.summary()}.")

        if webui and os.getenv("SD_WEBUI_RESTARTING") != "1" and _auto_launch(cmd_opts):
            webbrowser.open(server.url)

        command = _wait_for_command()
        if command in ("stop", RESTART_PROGRAM):
            print("Restarting..." if command == RESTART_PROGRAM else "Stopping server...")
            server.stop()
            if not services.queue.stop(timeout=30, interrupt=True):
                print("The running job did not stop in time")
            services.stop()
            if command == RESTART_PROGRAM:
                from modules import restart

                restart.restart_program()  # marks tmp/restart and exits
            return

        os.environ.setdefault("SD_WEBUI_RESTARTING", "1")
        print("Reloading UI: waiting for the running job...")
        # Hold the queue while the old app still serves, so the running job
        # can still be watched and interrupted; then swap the app.
        holding.enter_context(services.queue.hold())
        if not server.stop():
            print("The server did not stop in time")
        startup_timer.reset()
        script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        initialize.initialize_rest()


def _start_server(app: Any, host: str, port: int, cmd_opts: Any, attempts: int = 3) -> ServerThread:
    """Start serving ``app``; after a reload the old server may still hold
    the port for a moment, so wait for it and retry."""

    for attempt in range(1, attempts + 1):
        if not wait_port_free(host, port):
            log.warning("port %d is still in use", port)
        server = ServerThread(app, host=host, port=port, root=root_path(cmd_opts), ssl_keyfile=cmd_opts.tls_keyfile, ssl_certfile=cmd_opts.tls_certfile, timeout_keep_alive=cmd_opts.timeout_keep_alive, forwarded_allow_ips=getattr(cmd_opts, "forwarded_allow_ips", None))
        try:
            server.start()
            return server
        except (RuntimeError, TimeoutError):
            if attempt == attempts:
                raise
            server.stop(timeout=5)
            time.sleep(2)
    raise AssertionError("unreachable")


def _auto_launch(cmd_opts: Any) -> bool:
    from modules import shared

    if cmd_opts.autolaunch or shared.opts.auto_launch_browser == "Remote":
        return True
    return shared.opts.auto_launch_browser == "Local" and not cmd_opts.webui_is_non_local


def _wait_for_command() -> str:
    from modules import shared

    try:
        while True:
            command = shared.state.wait_for_server_command(timeout=5)
            if command in ("stop", "restart", RESTART_PROGRAM):
                return command
            if command:
                print(f"Unknown server command: {command}")
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, stopping...")
        return "stop"
