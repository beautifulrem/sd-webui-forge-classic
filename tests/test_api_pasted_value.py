"""``pasted_value``: the API's infotext replay (``modules/api/api.py``) turns
paste-field results into request / script values. modules.api.api cannot be
imported outside the WebUI, so the function is loaded from its source."""

import ast
from pathlib import Path

import gradio as gr

_SOURCE = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"
_function = next(node for node in ast.parse(_SOURCE.read_text(encoding="utf-8")).body if getattr(node, "name", None) == "pasted_value")
_ns = {}
exec(compile(ast.Module([_function], []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
pasted_value = _ns["pasted_value"]


def test_a_gradio_4_update_gives_its_value():
    # What gr.update() builds on this Gradio (4.x): {"__type__": "update", ...}
    assert gr.update(value=0.5, visible=True)["__type__"] == "update"
    assert pasted_value(gr.update(value="0.5", visible=True), float) == 0.5
    assert pasted_value(gr.update(value=3), int) == 3


def test_a_gradio_3_update_gives_its_value():
    assert pasted_value({"__type__": "generic_update", "value": "2"}, int) == 2


def test_an_update_without_a_value_sets_nothing():
    assert pasted_value(gr.update(visible=False), float) is None


def test_plain_values_are_converted_to_the_target_type():
    assert pasted_value("1.25", float) == 1.25
    assert pasted_value(True, bool) is True
    assert pasted_value({"a": 1}, dict) == {"a": 1}
