"""Input flags of ``remi.ui``: ``persist`` / ``submit``, ``Toggle(locks=)``,
``ImageList(widget=)`` and ``ui.message``."""

import pytest

from remi import ui
from remi.ui import SchemaError


def schema(node):
    return node.schema(ui.Context())


def test_persist_and_submit_are_exported_only_when_off():
    assert "persist" not in schema(ui.Text("a", "")) and "submit" not in schema(ui.Text("a", ""))
    exported = schema(ui.Slider("s", 0, 10, 5, 1, persist=False, submit=False))
    assert exported["persist"] is False and exported["submit"] is False


def test_unsubmitted_params_take_their_defaults():
    layout = ui.ScriptLayout(ui.Group(ui.Toggle("on", True), ui.Text("notes", lambda: "saved", submit=False, persist=False)))
    assert layout.values_to_args({"on": False}) == [False, "saved"]
    # Still a param: actions and API clients may send it
    assert layout.values_to_args({"notes": "x"}) == [True, "x"]


def test_toggle_locks():
    assert schema(ui.Toggle("lock", False, locks="sampler.steps"))["locks"] == "sampler.steps"
    assert schema(ui.Toggle("lock", False, locks="steps"))["locks"] == "steps"
    assert "locks" not in schema(ui.Toggle("lock", False))
    for bad in ("", ".", "a..b", 3):
        with pytest.raises(SchemaError):
            ui.Toggle("lock", False, locks=bad)


def test_image_list_widget_hint():
    assert schema(ui.ImageList("refs")) == {"type": "image_list", "name": "refs", "label": "Refs", "default": []}
    node = ui.ImageList("refs", widget="image-list", props=lambda: {"columns": 2})
    assert schema(node) == {"type": "image_list", "name": "refs", "label": "Refs", "default": [], "widget": "image-list", "props": {"columns": 2}}
    with pytest.raises(SchemaError):
        ui.ImageList("refs", widget="")
    # Upload semantics are unchanged: ids resolved through the context
    layout = ui.ScriptLayout(node)
    ctx = ui.Context(resolve_upload=lambda _node, upload_id: f"image:{upload_id}")
    assert layout.values_to_args({"refs": ["u1", "u2"]}, ctx) == [["image:u1", "image:u2"]]


def test_message():
    assert ui.message("Saved.") == {"text": "Saved."}
    assert ui.message("Removed {{count}} tag(s)", count=2) == {"text": "Removed {{count}} tag(s)", "params": {"count": 2}}
    patch = ui.check_patch({"__outputs__": {"status": ui.message("Removed {{count}} tag(s)", count=2)}})
    assert patch["__outputs__"]["status"]["params"] == {"count": 2}
