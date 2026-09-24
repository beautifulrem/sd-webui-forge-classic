import importlib.util
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

SPEC = importlib.util.spec_from_file_location(
    "agent_scheduler_file_guard", Path(__file__).parents[1] / "agent_scheduler" / "file_guard.py"
)
file_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(file_guard)


def _gradio_440_abspath(path):
    # gradio.utils.abspath as shipped with Gradio 4.40 (Forge Neo's pin).
    path = Path(path)
    if path.is_absolute():
        return path
    is_symlink = path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
    if is_symlink or path == path.resolve():
        return Path.cwd() / path
    return path.resolve()


def _app(abspath):
    app = FastAPI()

    # Same shapes and resolution as Gradio's file routes (without its
    # allow-list), independent of the guard's own code.
    @app.head("/file={path_or_url:path}")
    @app.get("/file={path_or_url:path}")
    async def file(path_or_url: str):
        path = abspath(path_or_url)
        if path.is_dir() or not path.exists():
            return PlainTextResponse("missing", status_code=404)
        return PlainTextResponse(path.read_text())

    @app.get("/file/{path:path}")
    async def file_deprecated(path: str):
        return await file(path)

    return app


@pytest.fixture(params=["installed", "4.40"])
def setup(tmp_path, monkeypatch, request):
    from gradio import utils

    if request.param == "4.40":
        abspath = _gradio_440_abspath
        monkeypatch.setattr(file_guard, "_served_path", lambda p: str(abspath(p)))
    else:
        abspath = utils.abspath  # the guard uses it itself
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    store = data / "extension-data" / "agent-scheduler"
    store.mkdir(parents=True)
    key = store / "signing.key"
    key.write_text("secret")
    # The configured database is a symlink to the real file elsewhere
    # (a plain file where symlinks are unavailable).
    (tmp_path / "volume").mkdir()
    (tmp_path / "volume" / "real.db").write_text("secret db")
    db = store / "tasks.sqlite3"
    try:
        db.symlink_to(tmp_path / "volume" / "real.db")
    except OSError:
        db.write_text("secret db")
    (data / "public.txt").write_text("hello")
    app = _app(abspath)
    assert file_guard.install(app, lambda: [str(key), str(db)]) == 3
    assert file_guard.install(app, lambda: [str(key), str(db)]) == 0  # idempotent
    return TestClient(app), data, store


def test_protected_files_are_refused_under_any_spelling(setup, tmp_path):
    client, data, store = setup
    (tmp_path / "alias").symlink_to(data)
    # data/link -> data/extension-data/agent-scheduler: "link/../.." is
    # resolved after the symlink by the OS, not collapsed by text.
    (data / "link").symlink_to(store)
    spellings = [
        str(store / "signing.key"),
        f"{store}/./signing.key",
        str(tmp_path / "alias" / "extension-data" / "agent-scheduler" / "signing.key"),
        f"{data}/link/../agent-scheduler/signing.key",
        f"{store}/signing.key/",
        f"{store}/signing.key/.",
        f"{store}/nope/../signing.key",
        f"{store}/tasks.sqlite3/../signing.key",
        # Relative forms (resolved against the working directory).
        "data/extension-data/agent-scheduler/signing.key/",
        "data/extension-data/agent-scheduler/nope/../signing.key",
        "data/link/../agent-scheduler/signing.key",
    ]
    for spelling in spellings:
        # %2E keeps HTTP clients from collapsing dot segments (curl --path-as-is).
        raw = spelling.replace(".", "%2E")
        for route in ("/file=", "/file/"):
            response = client.get(f"{route}{raw}")
            assert response.status_code in (403, 404) and "secret" not in response.text, (route, spelling)
    assert client.get(f"/file={store / 'signing.key'}").status_code == 403
    assert client.get(f"/file={data / 'public.txt'}").text == "hello"


def test_journals_are_refused_even_before_they_exist(setup):
    client, data, store = setup
    journal = store / "tasks.sqlite3-journal"

    # Refused by name: it may be created before the route opens it.
    assert client.get(f"/file={journal}").status_code == 403
    journal.write_text("secret pages")
    assert client.get(f"/file={journal}").status_code == 403
    assert client.get(f"/file={store / 'TASKS.SQLITE3-WAL'}").status_code == 403


def test_names_elsewhere_are_served(setup):
    client, data, store = setup
    (data / "tasks.sqlite3.png").write_text("image")

    assert client.get(f"/file={data / 'tasks.sqlite3.png'}").text == "image"


def test_aliases_under_other_names_are_refused(setup):
    client, data, store = setup
    try:
        (data / "pic.png").symlink_to(store / "signing.key")
        (data / "journal.png").symlink_to(store / "tasks.sqlite3-journal")  # not there yet
        os.link(store / "signing.key", data / "copy.txt")
    except OSError:
        pytest.skip("symlinks / hard links unavailable")

    for alias in ("pic.png", "copy.txt", "journal.png"):
        assert client.get(f"/file={data / alias}").status_code in (403, 404), alias
    assert client.get(f"/file={data / 'pic.png'}").status_code == 403
    (store / "tasks.sqlite3-journal").write_text("secret pages")
    assert client.get(f"/file={data / 'journal.png'}").status_code == 403
    assert client.get(f"/file={store / 'tasks.sqlite3'}").status_code == 403  # symlinked DB
    if (store / "tasks.sqlite3").is_symlink():
        assert client.get(f"/file={data.parent / 'volume' / 'real.db'}").status_code == 403


def test_short_names_of_protected_files_are_refused(setup):
    client, data, store = setup
    (data / "img~1.png").write_text("image")

    assert client.get(f"/file={store / 'TASKS~1.SQL'}").status_code == 403  # may appear later
    assert client.get(f"/file={store / 'SIGNIN~1.KEY'}").status_code == 403
    assert client.get(f"/file={data / 'img~1.png'}").text == "image"


def test_names_are_matched_across_unicode_forms():
    assert file_guard._name_key("t\u0061\u0302ches.sqlite3") == file_guard._name_key("t\u00e2ches.SQLITE3")
