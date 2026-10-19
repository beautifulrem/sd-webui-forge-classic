"""The layers of architecture.md hold: extension and script code, and the
layers they build on (``remi.ui``, ``remi.core``, ``remi.pages``), never reach
into the server (``remi.server``, ``request.app.state.services``). Actions get
files through ``ctx.files`` (``remi.ui.ActionFiles``) instead."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Code that must not know the server
CLIENT_CODE = ("extensions-builtin", "scripts", "modules/processing_scripts", "remi/ui", "remi/core", "remi/pages")

# WebUI glue in modules/ that is part of the server, each with its reason
SERVER_GLUE = {
    "modules/extension_route_guard.py": "the A1111 extension route guard the server installs (bind address, local-client and login checks)",
    "modules/util.py": "A1111's util.open_folder, delegating to the server's checked open-folder",
}

SERVICES_RE = re.compile(r"\.state\.services\b|[\"']services[\"']\)")


def _python_files(*roots: str):
    for root in roots:
        base = ROOT / root
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if "/tests/" in f"/{relative}" or "/node_modules/" in relative:
                continue
            yield relative, path


def _server_imports(source: str) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name == "remi.server" or alias.name.startswith("remi.server.") for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "remi.server" or node.module.startswith("remi.server.") or (node.module == "remi" and any(alias.name == "server" for alias in node.names)):
                lines.append(node.lineno)
    return lines


def test_extension_and_layer_code_does_not_import_the_server():
    found = []
    for relative, path in _python_files(*CLIENT_CODE):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found += [f"{relative}:{line}" for line in _server_imports(source)]
    assert found == [], f"import remi.server outside the server (use ctx.files / remi.ui): {found}"


def test_extension_and_layer_code_does_not_reach_the_server_services():
    found = []
    for relative, path in _python_files(*CLIENT_CODE):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found += [f"{relative}:{number}" for number, line in enumerate(source.splitlines(), 1) if SERVICES_RE.search(line)]
    assert found == [], f"request.app.state.services outside the server (use ctx.files): {found}"


def test_only_the_listed_webui_glue_imports_the_server():
    found = set()
    for relative, path in _python_files("modules", "modules_forge"):
        if relative.startswith("modules/processing_scripts/"):
            continue
        if _server_imports(path.read_text(encoding="utf-8")):
            found.add(relative)
    assert found == set(SERVER_GLUE), f"new server imports in modules/: {sorted(found - set(SERVER_GLUE))}; gone: {sorted(set(SERVER_GLUE) - found)}"


def test_action_files_reads_uploads_and_outputs_only(tmp_path):
    from remi.testing import webui_state

    with webui_state():  # images are read through modules.images (the WebUI's import order)
        _check_action_files(tmp_path)


def _check_action_files(tmp_path):
    from PIL import Image

    from remi import ui
    from remi.files import FileStore

    store = FileStore(tmp_path / "files")
    try:
        files = ui.ActionFiles(store)
        image_id = files.store_image(Image.new("RGB", (6, 4), (255, 0, 0)), "red.png")
        assert files.image_size(image_id) == (6, 4)
        assert files.image(image_id).size == (6, 4) and files.image(image_id).filename == "red.png"
        assert files.path(image_id) and files.get(image_id).is_image
        text_id = files.store_bytes(b"a,b\n", "table.csv")
        assert files.get(text_id) is not None and files.image(text_id) is None and files.image_size(text_id) is None
        preview = tmp_path / "net.png"
        Image.new("RGB", (2, 2)).save(preview)
        network = store.register_path(preview)
        assert network is not None and files.get(network.id) is None  # network previews are not action inputs
        for bad in (None, "", "unknown", 3):
            assert files.get(bad) is None and files.image(bad) is None and files.path(bad) is None
    finally:
        store.close()
