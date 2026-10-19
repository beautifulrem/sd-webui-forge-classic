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


def test_tool_inputs_are_not_arguments():
    """``arg=False``: rendered and read by actions (a calculator's fields), never a processing argument."""

    layout = ui.ScriptLayout(
        ui.Group(ui.Toggle("on", True), ui.Text("tool_prompt", "", arg=False), ui.Slider("steps", 1, 50, 20, 1)),
        order=["steps", "on"],
    )
    assert layout.names == ["steps", "on"] and [node.name for node in layout.tools] == ["tool_prompt"]
    # Validated and defaulted like the others; accepted when sent, but never an argument
    assert layout.defaults() == {"steps": 20, "on": True, "tool_prompt": ""}
    assert layout.coerce({"tool_prompt": "x"})["tool_prompt"] == "x"
    assert layout.values_to_args({"tool_prompt": "x", "on": False}) == [20, False]
    assert layout.args_to_values([5, False]) == {"steps": 5, "on": False}
    assert "tool_prompt" in layout and schema(ui.Text("t", "", arg=False))["submit"] is False
    with pytest.raises(SchemaError):  # the order lists the arguments only
        ui.ScriptLayout(ui.Group(ui.Toggle("on", True), ui.Text("tool", "", arg=False)), order=["on", "tool"])
    for flags in ({"infotext": "Tool"}, {"preset": "x"}, {"paste": lambda params: None}):
        with pytest.raises(SchemaError):
            ui.Text("tool", "", arg=False, **flags)


def test_inactive_if_is_exported_checked_and_validated():
    """``inactive_if``: dimmed but editable while true (the Gradio UI's ``inactive`` class)."""

    assert "inactive_if" not in schema(ui.Slider("s", 0, 10, 5, 1))
    exported = schema(ui.Slider("s", 0, 10, 0, 1, inactive_if=ui.param("s", lte=0)))
    assert exported["inactive_if"] == {"param": "s", "lte": 0} and "enabled_if" not in exported
    with pytest.raises(SchemaError):
        ui.Slider("s", 0, 10, 5, 1, inactive_if={"param": "s", "bogus": 1})
    with pytest.raises(SchemaError, match="unknown param"):
        ui.ScriptLayout(ui.Group(ui.Slider("s", 0, 10, 5, 1, inactive_if=ui.param("missing"))))


def test_hires_target_marks_the_unused_controls():
    """Upscale by is inactive while a target size is set; a size left at 0 is inactive."""

    from remi.core.forms import hires_resolution_layout

    nodes = {node.name: node for node in hires_resolution_layout().inputs}

    def state(**values):
        values = {"hr_scale": 2.0, "hr_resize_x": 0, "hr_resize_y": 0, **values}
        return {name: ui.evaluate(nodes[name].inactive_if, values) for name in ("hr_scale", "hr_resize_x", "hr_resize_y")}

    assert state() == {"hr_scale": False, "hr_resize_x": True, "hr_resize_y": True}
    assert state(hr_resize_x=1024) == {"hr_scale": True, "hr_resize_x": False, "hr_resize_y": True}
    assert state(hr_resize_x=1024, hr_resize_y=768) == {"hr_scale": True, "hr_resize_x": False, "hr_resize_y": False}


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
