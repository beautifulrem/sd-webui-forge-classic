"""The process-wide services (start / stop / close, the hourly housekeeping,
the default instance) and the queue completion actions' edge cases."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from remi.jobs import completion, run_inline
from remi.server import services as services_module
from remi.server.services import OUTPUT_DIR_OPTIONS, Services, default_services, output_dirs
from remi.server.testing import FakeState

# region Services


@pytest.fixture
def services(tmp_path):
    services = Services.create(tmp_path, dispatch=run_inline, realtime=False, housekeeping=False)
    yield services
    services.close()


def test_start_is_idempotent_and_stop_lets_it_start_again(services):
    services.start()
    worker = services.queue._thread
    services.start()  # a second app instance ("Reload UI") reuses the running services
    assert services.queue._thread is worker and services._threads == []
    services.stop()
    assert not services._started and services._stop.is_set()
    services.start()
    assert services._started and not services._stop.is_set()


def test_realtime_and_housekeeping_threads_start_and_stop(tmp_path, monkeypatch):
    import modules.shared  # noqa: F401 - the preview thread reads shared.opts: imported here, not on that thread under the join timeout

    ran = threading.Event()
    monkeypatch.setattr(Services, "housekeeping", lambda self: ran.set())
    state = FakeState()
    services = Services.create(tmp_path, dispatch=run_inline, state=lambda: state)
    try:
        before = set(threading.enumerate())
        services.start()
        assert ran.wait(5), "the housekeeping runs at start"
        [housekeeping] = [thread for thread in set(threading.enumerate()) - before if thread.name == "remi-housekeeping"]
        assert sorted(type(thread).__name__ for thread in services._threads) == ["PreviewEncoder", "ProgressMonitor"]
        assert all(thread.is_alive() for thread in services._threads)
        services.stop()
        for thread in [*services._threads, housekeeping]:
            thread.join(30)
            assert not thread.is_alive()
    finally:
        services.close()


def test_close_closes_every_database_including_the_gallery_index(tmp_path):
    from remi.library import gallery_for

    services = Services.create(tmp_path, dispatch=run_inline, realtime=False, housekeeping=False)
    index = gallery_for(services.files)  # opens its own connection to files.db
    services.close()
    for db in (index._db, services.files._db, services.store._db):
        with pytest.raises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")


def test_the_housekeeping_loop_repeats_until_stopped(services, monkeypatch):
    runs = []

    def housekeeping():
        runs.append(1)
        if len(runs) == 3:
            services._stop.set()

    monkeypatch.setattr(services, "housekeeping", housekeeping)
    monkeypatch.setattr(services, "HOUSEKEEPING_INTERVAL", 0.001)
    services._housekeeping_loop()
    assert len(runs) == 3


class Recorder:
    def __init__(self, result=0, error: Exception | None = None):
        self.calls: list[tuple] = []
        self.result, self.error = result, error

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _housekeeping_setup(services, monkeypatch, *, retention=7, prune_uploads=None, prune_unsaved=None, prune_exports=None, scan=None):
    import remi.library
    import remi.server.settings

    monkeypatch.setattr(remi.server.settings, "option", lambda name, default: {"remi_upload_retention_days": retention}.get(name, default))
    monkeypatch.setattr(services_module, "output_dirs", lambda: ["/outputs/txt2img"])
    recorders = {
        "prune_uploads": prune_uploads or Recorder(2),
        "prune_unsaved": prune_unsaved or Recorder(1),
        "prune_exports": prune_exports or Recorder(0),
        "scan_outputs": scan or Recorder(4),
    }
    monkeypatch.setattr(services.files, "prune_uploads", recorders["prune_uploads"])
    monkeypatch.setattr(services.files, "prune_unsaved", recorders["prune_unsaved"])
    monkeypatch.setattr(services.files, "scan_outputs", recorders["scan_outputs"])
    monkeypatch.setattr(remi.library, "gallery_for", lambda files: SimpleNamespace(prune_exports=recorders["prune_exports"]))
    return recorders


def test_housekeeping_prunes_with_the_retention_setting_and_indexes_outputs(services, monkeypatch, caplog):
    recorders = _housekeeping_setup(services, monkeypatch, retention=0.5)
    with caplog.at_level(logging.INFO, logger="remi.server.services"):
        services.housekeeping()
    [(args, kwargs)] = recorders["prune_uploads"].calls
    assert args == (0.5 * 86400,) and kwargs == {"keep": services.store.referenced_strings()}
    assert recorders["prune_unsaved"].calls == [((0.5 * 86400,), {})]
    assert recorders["prune_exports"].calls == [((), {})]
    assert recorders["scan_outputs"].calls == [((["/outputs/txt2img"],), {})]
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Removed 2 old upload(s)", "Removed 1 old unsaved output(s)", "Indexed 4 output image(s)"]


def test_housekeeping_is_quiet_when_there_is_nothing_to_do(services, monkeypatch, caplog):
    _housekeeping_setup(services, monkeypatch, prune_uploads=Recorder(0), prune_unsaved=Recorder(0), scan=Recorder(0))
    with caplog.at_level(logging.INFO, logger="remi.server.services"):
        services.housekeeping()
    assert caplog.records == []


def test_a_failing_housekeeping_step_does_not_stop_the_others(services, monkeypatch, caplog):
    recorders = _housekeeping_setup(
        services,
        monkeypatch,
        prune_uploads=Recorder(error=OSError("disk gone")),
        prune_exports=Recorder(error=OSError("exports gone")),
        scan=Recorder(error=PermissionError("no access")),
    )
    with caplog.at_level(logging.ERROR, logger="remi.server.services"):
        services.housekeeping()  # never raises: it runs on its own thread every hour
    assert recorders["prune_unsaved"].calls == []  # same step as the uploads
    assert len(recorders["prune_exports"].calls) == 1 and len(recorders["scan_outputs"].calls) == 1
    assert [record.getMessage() for record in caplog.records] == ["pruning uploads", "pruning gallery exports", "indexing outputs"]
    assert all(record.exc_info for record in caplog.records)


def test_output_dirs_are_the_set_output_folder_options(monkeypatch):
    from modules import shared

    values = dict.fromkeys(OUTPUT_DIR_OPTIONS, "")
    values |= {"outdir_samples": "/out", "outdir_grids": "/grids", "outdir_save": "/saved"}
    monkeypatch.setattr(shared, "opts", SimpleNamespace(**values))
    assert output_dirs() == ["/out", "/grids", "/saved"]


def test_default_services_are_created_once(monkeypatch, tmp_path):
    import modules.paths_internal

    monkeypatch.setattr(services_module, "_default", None)
    monkeypatch.setattr(modules.paths_internal, "data_path", str(tmp_path))
    roots = []

    def factory(root: Path):
        roots.append(root)
        return SimpleNamespace(root=root)

    first = default_services(factory)
    assert default_services(factory) is first and default_services() is first
    assert roots == [tmp_path / "remi-data"]


def test_default_services_read_the_upload_limit_setting(monkeypatch, tmp_path):
    import modules.paths_internal
    import remi.server.settings

    monkeypatch.setattr(services_module, "_default", None)
    monkeypatch.setattr(modules.paths_internal, "data_path", str(tmp_path))
    limit = {"remi_upload_max_mb": 64}
    monkeypatch.setattr(remi.server.settings, "option", lambda name, default: limit.get(name, default))
    services = default_services()
    try:
        assert services.root == tmp_path / "remi-data"
        assert services.files.max_upload_bytes == 64 * 1024 * 1024
        limit["remi_upload_max_mb"] = 0.5  # a live setting: read on every upload
        assert services.files.max_upload_bytes == 512 * 1024
    finally:
        services.close()


# endregion

# region Queue completion actions


def test_power_commands_per_platform():
    assert completion.power_command("Shut down", "win32") == ["shutdown", "/s", "/hybrid", "/t", "0"]
    assert completion.power_command("Hibernate", "win32") == ["shutdown", "/h"]
    assert completion.power_command("Sleep", "win32") is None  # a system call, not a command
    assert completion.power_command("Restart", "darwin")[0] == "osascript"
    assert completion.power_command("Hibernate", "darwin") == ["pmset", "sleepnow"]
    assert completion.power_command("Sleep", "linux") == ["sh", "-c", "systemctl hybrid-sleep || systemctl suspend"]
    assert completion.power_command("Stop webui", "linux") is None


def test_windows_sleep_is_a_system_call(monkeypatch):
    called = []
    monkeypatch.setattr(completion, "_windows_sleep", lambda: called.append(True))
    stops = []
    assert completion.perform("Sleep", platform="win32", run=lambda command: pytest.fail("no command"), stop_server=lambda: stops.append(1))
    assert called == [True] and stops == []


def test_an_unsupported_action_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(completion, "power_command", lambda action, platform: None)
    with caplog.at_level(logging.ERROR, logger="remi.jobs.completion"):
        assert not completion.perform("Hibernate", platform="sunos5", run=lambda command: pytest.fail("no command"), stop_server=lambda: pytest.fail("keeps running"))
    assert "Hibernate is not supported on sunos5" in caplog.text


def test_a_failing_command_keeps_the_server_running(caplog):
    def run(command):
        raise FileNotFoundError("systemctl")

    with caplog.at_level(logging.ERROR, logger="remi.jobs.completion"):
        assert not completion.perform("Shut down", platform="linux", run=run, stop_server=lambda: pytest.fail("the server must not stop"))
    assert "running the queue completion action Shut down" in caplog.text


def test_stop_webui_asks_the_server_loop_to_stop(monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared, "state", SimpleNamespace(server_command=None))
    assert completion.perform("Stop webui")
    assert shared.state.server_command == "stop"


def test_perform_later_runs_on_a_daemon_timer(monkeypatch):
    done = threading.Event()
    seen = []

    def perform(action):
        seen.append((action, threading.current_thread().name, threading.current_thread().daemon))
        done.set()

    monkeypatch.setattr(completion, "perform", perform)
    completion.perform_later("Sleep", delay=0.01)
    assert done.wait(5)
    assert seen == [("Sleep", "remi-queue-done", True)]


# endregion
