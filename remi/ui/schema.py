"""Schema export (docs/remi-webui/schema-format.md)."""

from __future__ import annotations

import json
import math
from typing import Any

from remi.ui.context import Context, ensure_context
from remi.ui.layout import ScriptLayout
from remi.ui.nodes import Node

SCHEMA_VERSION = 1


def to_schema(node: Node | None, ctx: Context | None = None) -> dict | None:
    """The JSON-serialisable description of a layout tree (``None`` for none).

    The tree is validated as one owner's layout first (unique names, known
    condition references, tab bindings)."""

    if node is None:
        return None
    ScriptLayout(node, owner="schema")
    return node.schema(ensure_context(ctx))


def jsonable(value: Any) -> Any:
    """``value`` as strict JSON data (tuples become lists, non-finite floats
    and unknown objects become ``None``/strings)."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return str(value)


def dumps(data: Any) -> str:
    """Strict JSON text (raises on NaN/Infinity), stable key order as built."""

    return json.dumps(data, ensure_ascii=False, allow_nan=False)
