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


def _app():
    app = FastAPI()

    # Same shapes as Gradio's file routes; the path is opened unnormalised.
    @app.head("/file={path_or_url:path}")
    @app.get("/file={path_or_url:path}")
    async def file(path_or_url: str):
        with open(path_or_url) as handle:
            return PlainTextResponse(handle.read())

    @app.get("/file/{path:path}")
    async def file_deprecated(path: str):
        return await file(path)

    return app


@pytest.fixture()
def setup(tmp_path):
    data = tmp_path / "data"
    store = data / "extension-data" / "agent-scheduler"
    store.mkdir(parents=True)
    key = store / "signing.key"
    key.write_text("secret")
    db = store / "tasks.sqlite3"
    db.write_text("db")
    (data / "public.txt").write_text("hello")
    app = _app()
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
    ]
    for spelling in spellings:
        assert os.path.exists(spelling), spelling
        # %2E keeps HTTP clients from collapsing dot segments (curl --path-as-is).
        raw = spelling.replace(".", "%2E")
        assert client.get(f"/file={raw}").status_code == 403, spelling
        assert client.get(f"/file/{raw}").status_code == 403, spelling
    assert client.get(f"/file={data / 'public.txt'}").text == "hello"


def test_journals_created_later_are_refused(setup):
    client, data, store = setup
    client.get(f"/file={data / 'public.txt'}")  # warm the cache
    journal = store / "tasks.sqlite3-journal"
    journal.write_text("pages")

    assert client.get(f"/file={journal}").status_code == 403
    assert client.get(f"/file={store / 'TASKS.SQLITE3-WAL'}").status_code in (403, 404)
