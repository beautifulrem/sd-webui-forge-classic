"""``ui.RangeSlider``: the server side of the shared ``range-slider`` widget."""

import json

import pytest

from remi import ui


def _layout(node):
    return ui.ScriptLayout(ui.Section("Range", children=[node]), owner="range")


def test_float_range_is_clamped_ordered_and_exported():
    node = ui.RangeSlider("span", 0.0, 1.0, (0.15, 0.85), 0.05, label="Span", infotext="Span")
    layout = _layout(node)
    assert layout.values_to_args({}) == [[0.15, 0.85]]
    assert layout.values_to_args({"span": [0.9, -1]}) == [[0.0, 0.9]]
    assert layout.values_to_args({"span": ["0.2", 1]}) == [[0.2, 1.0]]
    schema = layout.schema()["children"][0]
    assert json.loads(json.dumps(schema, allow_nan=False)) == {
        "type": "custom",
        "name": "span",
        "label": "Span",
        "default": [0.15, 0.85],
        "widget": "range-slider",
        "props": {"minimum": 0.0, "maximum": 1.0, "step": 0.05},
        "infotext": "Span",
    }


def test_integer_bounds_keep_integers():
    node = ui.RangeSlider("steps", 0, 100, (10, 90), 1)
    assert node.integer and node.coerce([90.4, 9.6], ui.Context()) == [10, 90]
    assert ui.RangeSlider("full", 0, 10).resolve_default(ui.Context()) == [0, 10]
    assert not ui.RangeSlider("mixed", 0, 10, (1, 2), 0.5).integer
    assert ui.RangeSlider("forced", 0.0, 1.0, (0.0, 1.0), 0.1, integer=True).coerce([0.4, 0.6], ui.Context()) == [0, 1]


@pytest.mark.parametrize("raw", ["0.1, 0.9", "(0.9, 0.1)", "[0.1,0.9]", (0.1, 0.9), [0.1, 0.9]])
def test_paste_accepts_text_and_pairs(raw):
    assert ui.RangeSlider("span", 0.0, 1.0).from_paste(raw, ui.Context()) == [0.1, 0.9]


@pytest.mark.parametrize("value", [None, "0.1, 0.9", [0.5], [0.1, 0.2, 0.3], [True, 0.5], ["x", 0.5], {"low": 0}, [float("nan"), 0.5]])
def test_bad_values_are_rejected(value):
    with pytest.raises(ui.ValidationError):
        _layout(ui.RangeSlider("span", 0.0, 1.0)).values_to_args({"span": value})


def test_bad_paste_is_rejected():
    with pytest.raises(ui.ParamError):
        ui.RangeSlider("span", 0.0, 1.0).from_paste("0.1", ui.Context())


def test_tuple_arguments_read_back_as_lists():
    assert ui.RangeSlider("span").from_arg((0.2, 0.4), ui.Context()) == [0.2, 0.4]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum": 1.0, "maximum": 0.0},
        {"step": 0},
        {"default": 0.5},
        {"default": (0.1, 0.2, 0.3)},
        {"default": "0.1, 0.2"},
        {"validate": lambda value: value},
    ],
)
def test_bad_declarations_fail_early(kwargs):
    with pytest.raises(ui.SchemaError):
        ui.RangeSlider("span", **kwargs)
