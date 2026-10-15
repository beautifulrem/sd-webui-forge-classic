if __name__ == "__main__":
    raise SystemError("Call launch.py instead")


import os
from threading import Thread

from modules import initialize, timer
from modules_forge.initialization import initialize_forge

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.shush()

with startup_timer.subcategory("forge init"):
    initialize_forge()

initialize.imports()

initialize.check_versions()

initialize.initialize()


def _serve(webui: bool):
    import traceback

    from remi.server.launch import serve

    code = 0
    try:
        serve(webui=webui, startup_timer=startup_timer)
    except BaseException:
        traceback.print_exc()
        code = 1
    os._exit(code)  # stopped or failed: the main thread is busy in main_thread.loop()


def webui_worker():
    """The WebUI: the SPA, ``/api/v2`` and (with ``--api``) ``/sdapi/v1``."""

    _serve(webui=True)


def api_only_worker():
    """``--nowebui``: the APIs (``/api/v2``, ``/sdapi/v1``) without the SPA."""

    _serve(webui=False)


def api_only():
    Thread(target=api_only_worker, name="remi-server", daemon=True).start()


def webui():
    Thread(target=webui_worker, name="remi-server", daemon=True).start()
