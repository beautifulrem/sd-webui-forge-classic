"""Harness for extension tabs (``on_ui_tabs``) ported to ``remi.pages``.

Not a test module; extension suites load it by file path. Nothing but the
recording CLI imports Gradio.

Before a tab is ported its Gradio UI is recorded once::

    /tmp/remi-testenv/bin/python tests/remi_page_harness.py \\
        extensions-builtin/<ext>/scripts/<script>.py

which builds the script's ``on_ui_tabs`` Blocks and writes
``extensions-builtin/<ext>/tests/fixtures/legacy_ui_page.json``: one entry
per input control in creation order (kind, label, default, bounds or
choices, ``elem_id``), then the buttons' labels. A tab has no txt2img /
img2img variant, no processing arguments, infotext keys or preset fields.

:func:`describe_page` reduces a page layout to the same shape and
:func:`assert_page_matches` compares it with the record through an explicit
``legacy index -> param name`` map, listing every intended difference.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path("tests") / "fixtures" / "legacy_ui_page.json"


def fixture_path(script_path: str | Path) -> Path:
    return Path(script_path).resolve().parents[1] / FIXTURE


def load_snapshot(script_path: str | Path) -> dict[str, Any]:
    return json.loads(fixture_path(script_path).read_text(encoding="utf8"))


def describe_node(node, ctx) -> dict[str, Any]:
    """One input as recorded: kind, label, default, bounds / choices."""

    from remi import ui
    from remi.ui.schema import jsonable

    entry: dict[str, Any] = {"kind": node.type, "label": node.label, "default": jsonable(node.resolve_default(ctx))}
    if isinstance(node, (ui.Slider, ui.Number)):
        entry.update(minimum=node.minimum, maximum=node.maximum, step=node.step)
    if isinstance(node, (ui.Choice, ui.MultiChoice)):
        entry["choices"] = jsonable(node.values(ctx))
    return entry


def describe_page(layout, ctx=None) -> dict[str, dict[str, Any]]:
    """Param name -> its description, for every input of a page layout."""

    from remi import ui

    ctx = ctx or ui.Context()
    return {node.name: describe_node(node, ctx) for node in layout.inputs}


def assert_page_matches(layout, snapshot: Mapping[str, Any], mapping: Mapping[int, str | None], changes: Mapping[int, Mapping[str, Any]] | None = None, *, ctx=None) -> None:
    """Every recorded input is mapped (``index -> param name``, or ``None``
    for a control the port drops on purpose) and matches its param except the
    documented ``changes`` (``{index: {key: value}}``; ``...`` skips a key)."""

    actual = describe_page(layout, ctx)
    inputs = snapshot["inputs"]
    assert sorted(mapping) == list(range(len(inputs))), f"map every recorded input exactly once (0..{len(inputs) - 1})"
    changes = dict(changes or {})
    for index, entry in enumerate(inputs):
        name = mapping[index]
        if name is None:
            continue
        assert name in actual, f"recorded input {index} ({entry['label']!r}) maps to unknown param {name!r}"
        want = {key: value for key, value in entry.items() if key != "elem_id"}
        want.update(changes.get(index, {}))
        got = actual[name]
        for key in sorted(set(want) | set(got)):
            if want.get(key) is ...:
                continue
            assert got.get(key) == want.get(key), f"input {index} ({entry['label']!r} -> {name}): {key} is {got.get(key)!r}, expected {want.get(key)!r}"


# region Loading, running and checking a page (no Gradio)


def load_page_script(path: str | Path, name: str):
    """Import a page script as Forge does (its extension folder on
    ``sys.path``); it registers its page(s) with ``remi.pages``."""

    import importlib.util as util

    path = Path(path).resolve()
    root = str(path.parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRecord:
    def __init__(self, id: str, path: Path, name: str):
        self.id, self.path, self.name, self.kind = id, str(path), name, "upload"

    @property
    def is_image(self) -> bool:
        return Path(self.path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")


class FakeFiles:
    """The slice of ``remi.files.FileStore`` pages use: uploads by id."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.records: dict[str, FakeRecord] = {}

    def add_upload_bytes(self, data: bytes, name: str | None = None) -> FakeRecord:
        import hashlib

        file_id = hashlib.sha1(data).hexdigest()[:16]
        path = self.root / f"{file_id}{Path(name or 'upload.png').suffix or '.png'}"
        path.write_bytes(data)
        self.records[file_id] = FakeRecord(file_id, path, name or path.name)
        return self.records[file_id]

    def add_image(self, image, name: str = "image.png") -> str:
        import io

        buffer = io.BytesIO()
        image.save(buffer, "PNG", pnginfo=getattr(image, "_pnginfo", None))
        return self.add_upload_bytes(buffer.getvalue(), name).id

    def get(self, file_id: str):
        return self.records.get(file_id)

    def resolve_upload(self, node, file_id: str):
        from PIL import Image

        from remi.ui import ParamError

        record = self.get(file_id)
        if record is None:
            raise ParamError(f"unknown upload {file_id!r}")
        image = Image.open(record.path)
        image.load()
        return image


def fake_ctx(page, files=None, *, user=None):
    """An action context whose request carries fake services (file store,
    a generation lock)."""

    import threading
    from types import SimpleNamespace

    from remi import ui

    services = SimpleNamespace(files=files, queue=SimpleNamespace(lock=threading.Lock()))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=services)))
    return ui.ActionContext(tab=page.owner, owner=page.owner, user=user, request=request)


def run_action(page, name: str, ctx, values=None, **extra):
    from remi import ui

    ctx.extra = dict(extra)
    return ui.call_action(page, name, ctx, values or {})


def wait_task(page, ctx, name: str, timeout: float = 10.0) -> dict:
    """The ``__task__`` answer once task ``name`` has ended (its result included)."""

    import time

    task = page.task(ctx, name)
    assert task is not None, f"no task {name!r}"
    deadline = time.time() + timeout
    while not task.done:
        assert time.time() < deadline, f"task {name!r} still {task.state}"
        time.sleep(0.01)
    return run_action(page, "__task__", ctx, {}, task=name, id=task.id)


def page_strings(page, script_path: str | Path) -> set[str]:
    """Every English string the SPA shows for the page: its title, layout
    texts (labels, infos, titles, placeholders, choices, Markdown), the
    ``label`` / ``placeholder`` props of its widgets and the message keys
    of the script (``ui.message`` / ``ui.toast`` / task ``progress``)."""

    import ast

    found = {page.title}
    found |= {target.label for target in page.send_to}

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") != "html":
            for key in ("label", "info", "title", "placeholder"):
                if isinstance(node.get(key), str) and node[key].strip():
                    found.add(node[key])
            if node.get("type") == "markdown":
                found.add(node["text"])
        for choice in node.get("choices") or ():
            if isinstance(choice, dict) and isinstance(choice.get("label"), str) and choice["label"].strip():
                found.add(choice["label"])
        for key in ("label", "placeholder"):
            value = (node.get("props") or {}).get(key)
            if isinstance(value, str) and value.strip():
                found.add(value)
        for key in ("children", "enable"):
            if key in node:
                visit(node[key])

    visit(page.schema()["layout"])
    calls = {"message": 0, "toast": 0, "progress": 1}
    for node in ast.walk(ast.parse(Path(script_path).read_text(encoding="utf8"))):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            index = calls.get(name)
            if index is not None and len(node.args) > index and isinstance(node.args[index], ast.Constant) and isinstance(node.args[index].value, str):
                found.add(node.args[index].value)
    # numbers and file extensions (".txt") are data
    return {text for text in found if text.strip() and not re.fullmatch(r"[\d.,\s-]+|\.\w+", text)}


def assert_page_gradio_free(extension: str | Path, scripts) -> None:
    """No Gradio import and no JS/CSS asset in the extension, and its scripts
    import with Gradio blocked (a fresh interpreter)."""

    import ast

    extension = Path(extension)
    offending = []
    for path in sorted(extension.rglob("*.py")):
        if "tests" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            offending += [f"{path.name}: {name}" for name in names if name.partition(".")[0] == "gradio" or name.startswith("modules.ui")]
    assert not offending, offending
    assert not list(extension.glob("javascript/*")) and not (extension / "style.css").exists(), "JS/CSS assets left"
    spec = importlib.util.spec_from_file_location("remi_port_harness", REPO / "tests" / "remi_port_harness.py")
    port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port)
    port.assert_gradio_free(scripts, extra_paths=[extension])


# endregion

# region Recording a legacy tab (needs Gradio)


def legacy_page_snapshot(blocks) -> dict[str, Any]:
    """Inputs and buttons of a built ``gr.Blocks``, in creation order."""

    import gradio as gr

    from remi.ui.legacy import gradio_kind, legacy_layout

    controls = [block for block in blocks.blocks.values() if gradio_kind(block) is not None]
    layout = legacy_layout("legacy", controls, [])
    entries = []
    for node, control in zip(layout.inputs, controls):
        entry = describe_node(node, __import__("remi.ui", fromlist=["Context"]).Context())
        if not entry["label"].strip() or entry["label"].startswith("#"):
            entry["label"] = str(getattr(control, "label", "") or "")
        entry["elem_id"] = getattr(control, "elem_id", None)
        entries.append(entry)
    buttons = [str(block.value) for block in blocks.blocks.values() if isinstance(block, gr.Button)]
    return {"inputs": entries, "buttons": buttons}


def _record(script_path: Path) -> Path:
    spec = importlib.util.spec_from_file_location("extension_script_harness", REPO / "tests" / "extension_script_harness.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.prepare_webui_imports()

    from modules import script_callbacks
    from remi.testing import webui_state

    with webui_state():
        script_callbacks.clear_callbacks()
        harness.load_script(script_path, f"legacy_page_{script_path.stem}")
        tabs = [tab for callback in script_callbacks.callback_map["callbacks_ui_tabs"] for tab in callback.callback() or []]
        if len(tabs) != 1:
            raise SystemExit(f"expected one tab from {script_path}, got {len(tabs)}")
        snapshot = legacy_page_snapshot(tabs[0][0])
        snapshot["title"] = tabs[0][1]
    target = fixture_path(script_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    return target


if __name__ == "__main__":
    import argparse

    sys.argv, argv = sys.argv[:1], sys.argv
    for path in (str(REPO / "modules_forge" / "packages"), str(REPO)):
        sys.path.insert(0, path)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("script", type=Path)
    options = parser.parse_args(argv[1:])
    print(_record(options.script.resolve()).relative_to(REPO))

# endregion
