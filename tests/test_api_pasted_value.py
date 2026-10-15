"""``pasted_value``: the API's infotext replay (``modules/api/api.py``) turns
infotext values into request field values. modules.api.api cannot be
imported outside the WebUI, so the function is loaded from its source."""

import ast
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "modules" / "api" / "api.py"
_function = next(node for node in ast.parse(_SOURCE.read_text(encoding="utf-8")).body if getattr(node, "name", None) == "pasted_value")
_ns = {}
exec(compile(ast.Module([_function], []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
pasted_value = _ns["pasted_value"]


def test_none_sets_nothing():
    assert pasted_value(None, float) is None


def test_infotext_text_converts():
    # Infotext values arrive as text: bool("False") would be True
    assert pasted_value("False", bool) is False
    assert pasted_value("True", bool) is True
    assert pasted_value("3.0", int) == 3 and pasted_value("7", int) == 7


def test_plain_values_are_converted_to_the_target_type():
    assert pasted_value("1.25", float) == 1.25
    assert pasted_value(True, bool) is True
    assert pasted_value({"a": 1}, dict) == {"a": 1}
