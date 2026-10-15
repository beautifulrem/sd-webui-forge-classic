"""``visible_if`` / ``enabled_if`` predicates (schema-format.md §2).

Conditions are plain JSON evaluated by the client; :func:`evaluate` mirrors
the client semantics for server-side checks and tests: ``eq``/``ne``/``in``
are Python ``==`` on JSON values (``True == 1``, ``"1" != 1``), and
``gt``/``gte``/``lt``/``lte`` take a finite number."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from remi.ui.errors import SchemaError

Condition = bool | Mapping[str, Any]

_ORDER = ("gt", "gte", "lt", "lte")
_COMPARE = ("eq", "ne", "in", *_ORDER)
_MISSING = object()  # a reference to a param that does not exist


def param(name: str, **operator: Any) -> dict:
    """A leaf: ``param("mode", in_=["a", "b"])``, ``param("enable")`` (truthy),
    ``param("cfg_scale", gt=1)``. ``in_`` stands for the ``in`` keyword."""

    if "in_" in operator:
        operator["in"] = operator.pop("in_")
    return check({"param": name, **operator})


def all_of(*conditions: Condition) -> dict:
    return check({"all": list(conditions)})


def any_of(*conditions: Condition) -> dict:
    return check({"any": list(conditions)})


def not_(condition: Condition) -> dict:
    return check({"not": condition})


def check(condition: Any) -> Condition:
    """Validate the grammar; returns the condition as plain dicts/lists."""

    if isinstance(condition, bool):
        return condition
    if not isinstance(condition, Mapping):
        raise SchemaError(f"a condition is a bool or an object, got {condition!r}")
    keys = set(condition)
    if keys == {"all"} or keys == {"any"}:
        (key,) = keys
        items = condition[key]
        if not isinstance(items, Sequence) or isinstance(items, str):
            raise SchemaError(f'"{key}" takes a list of conditions')
        return {key: [check(item) for item in items]}
    if keys == {"not"}:
        return {"not": check(condition["not"])}
    if "param" in keys:
        name = condition["param"]
        if not isinstance(name, str) or not name:
            raise SchemaError(f"bad param reference {name!r}")
        operators = keys - {"param"}
        if len(operators) > 1 or not operators <= set(_COMPARE):
            raise SchemaError(f"a param condition takes at most one of {_COMPARE}, got {sorted(operators)}")
        result = {"param": name}
        for operator in operators:
            operand = condition[operator]
            if operator == "in":
                if not isinstance(operand, Sequence) or isinstance(operand, str):
                    raise SchemaError('"in" takes a list')
                operand = list(operand)
            elif operator in _ORDER and not (isinstance(operand, (int, float)) and not isinstance(operand, bool) and math.isfinite(operand)):
                raise SchemaError(f'"{operator}" compares with a finite number, got {operand!r}')
            result[operator] = operand
        return result
    raise SchemaError(f"unknown condition {dict(condition)!r}")


def references(condition: Condition | None) -> set[str]:
    """Every param name a condition reads."""

    if condition is None or isinstance(condition, bool):
        return set()
    if "param" in condition:
        return {condition["param"]}
    if "not" in condition:
        return references(condition["not"])
    return set().union(*(references(item) for item in next(iter(condition.values()))))


def _truthy(value: Any) -> bool:
    return value is not _MISSING and value not in (None, False, 0, "") and value != []


_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda value, operand: value == operand,
    "ne": lambda value, operand: value != operand,
    "in": lambda value, operand: value in operand,
    "gt": lambda value, operand: value > operand,
    "gte": lambda value, operand: value >= operand,
    "lt": lambda value, operand: value < operand,
    "lte": lambda value, operand: value <= operand,
}


def evaluate(condition: Condition | None, values: Mapping[str, Any] | Callable[[str], Any]) -> bool:
    """Evaluate like the client. ``values`` maps param references to values,
    or is a function raising ``KeyError`` for a missing reference. A missing
    param is falsy (and ``ne`` is true)."""

    if isinstance(values, Mapping):
        mapping = values
        values = lambda name: mapping[name]  # noqa: E731
    if condition is None:
        return True
    if isinstance(condition, bool):
        return condition
    if "all" in condition:
        return all(evaluate(item, values) for item in condition["all"])
    if "any" in condition:
        return any(evaluate(item, values) for item in condition["any"])
    if "not" in condition:
        return not evaluate(condition["not"], values)

    try:
        value = values(condition["param"])
    except KeyError:
        value = _MISSING
    for operator, compare in _OPERATORS.items():
        if operator in condition:
            if value is _MISSING:
                return operator == "ne"
            try:
                return bool(compare(value, condition[operator]))
            except TypeError:
                return False
    return _truthy(value)
