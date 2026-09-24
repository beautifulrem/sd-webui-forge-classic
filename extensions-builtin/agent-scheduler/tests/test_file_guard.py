import importlib.util
from pathlib import Path

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

    # Same shapes as Gradio's file routes.
    @app.head("/file={path_or_url:path}")
    @app.get("/file={path_or_url:path}")
    async def file(path_or_url: str):
        return PlainTextResponse(Path(path_or_url).read_text())

    @app.get("/file/{path:path}")
    async def file_deprecated(path: str):
        return await file(path)

    return app


def test_protected_files_are_refused_under_any_spelling(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    key = data / "signing.key"
    key.write_text("secret")
    (data / "public.txt").write_text("hello")
    (tmp_path / "alias").symlink_to(data)
    app = _app()

    assert file_guard.install(app, lambda: [str(key)]) == 3
    assert file_guard.install(app, lambda: [str(key)]) == 0  # idempotent
    client = TestClient(app)

    for spelling in (str(key), f"{data}/./signing.key", str(tmp_path / "alias" / "signing.key")):
        assert client.get(f"/file={spelling}").status_code == 403
        assert client.get(f"/file/{spelling}").status_code == 403
    assert client.get(f"/file={data / 'public.txt'}").text == "hello"
