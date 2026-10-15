"""Postprocessing (Extras) operations as ``remi.ui`` owners.

A ``ScriptPostprocessing`` receives its values by keyword, named by its
``params()`` layout. An operation whose ``params()`` fails, or that only has
A1111's ``ui()`` (Gradio controls, not supported), is dropped from the runner
(reported once), so the others keep working.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remi.ui import Context, ParamError, ScriptLayout, ValidationError, jsonable
from remi.ui.types import ScriptEntry


def _drop(runner: Any, script: Any, reason: str, *, trace: bool = True) -> None:
    from modules import errors

    errors.report(f"Extras operation {script.name or type(script).__name__} is disabled: {reason}", exc_info=trace)
    runner.scripts.remove(script)
    runner.disabled_scripts.append((script, reason))


def legacy_ui_only(script: Any) -> bool:
    """An A1111 operation that builds Gradio controls in ``ui()`` and has no ``params()``."""

    from modules.scripts_postprocessing import ScriptPostprocessing

    return callable(getattr(type(script), "ui", None)) and getattr(type(script), "params", None) is ScriptPostprocessing.params


def _layout(script: Any, owner: str) -> ScriptLayout:
    cached = getattr(script, "_remi_layout", None)
    if cached is None or cached.owner != owner:
        cached = script._remi_layout = ScriptLayout(script.params(), owner=owner)
    return cached


def assign_args(runner: Any) -> list[tuple[Any, ScriptLayout]]:
    """``(operation, layout)`` in run order; assigns each operation's
    argument slots and the name list ``ScriptPostprocessingRunner.run``
    zips its values with."""

    from modules.scripts import script_slug, unique_id

    result: list[tuple[Any, ScriptLayout]] = []
    taken: set[str] = set()
    for script in runner.scripts_in_preferred_order():
        if legacy_ui_only(script):
            _drop(runner, script, "declares ui() instead of params()", trace=False)
            continue
        owner = unique_id(script_slug(script.name or type(script).__name__), taken)
        try:
            layout = _layout(script, owner)
        except Exception as error:
            _drop(runner, script, f"params() failed: {error}")
            continue
        taken.add(owner)
        result.append((script, layout))

    position = 0
    for script, layout in result:
        script.arg_names = list(layout.names)
        script.args_from, script.args_to = position, position + len(layout)
        position += len(layout)
    return result


def postprocessing_args(runner: Any, values: Any, ctx: Context | None = None) -> list:
    """The positional ``*args`` of ``run_postprocessing`` for named values
    ``{operation id: {param: value}}``; missing params take their defaults."""

    ctx = ctx or Context()
    if values is None:
        values = {}
    elif not isinstance(values, Mapping):
        raise ValidationError([ParamError(f"scripts: expected an object of values by operation id, got {type(values).__name__}")])
    assigned = assign_args(runner)
    known = {layout.owner for _script, layout in assigned}
    problems: list = [ParamError("unknown postprocessing operation", owner=owner) for owner in values if owner not in known]
    args: list = [None] * (max((script.args_to for script, _layout in assigned), default=0))
    for script, layout in assigned:
        try:
            args[script.args_from : script.args_to] = layout.values_to_args(values.get(layout.owner), ctx)
        except ValidationError as error:
            problems.append(error)
    if problems:
        raise ValidationError.merge(problems)
    return args


def scripts_schema(runner: Any, ctx: Context | None = None) -> list[ScriptEntry]:
    ctx = ctx or Context()
    result: list[ScriptEntry] = []
    for order, (script, layout) in enumerate(assign_args(runner)):
        result.append(
            {
                "id": layout.owner,
                "title": script.name,
                "kind": "alwayson",
                "section": "scripts",
                "order": order,
                "layout": layout.schema(ctx),
                "defaults": jsonable(layout.defaults(ctx)),
                "args": layout.names,
            }
        )
    return result
