"""Postprocessing (Extras) operations as ``remi.ui`` owners.

A ``ScriptPostprocessing`` receives its values by keyword, so its params are
named either by its ``params()`` layout (when ported) or by the keys of the
control dict its legacy ``ui()`` returns. An operation whose ``params()`` or
``ui()`` fails is dropped from the runner (reported once), so the others
keep working.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remi.ui import Context, ParamError, ScriptLayout, ValidationError, jsonable
from remi.ui.legacy import legacy_layout
from remi.ui.types import ScriptEntry


def _has_params(script: Any) -> bool:
    return callable(getattr(script, "params", None))


def _drop(runner: Any, script: Any, reason: str) -> None:
    from modules import errors

    errors.report(f"Extras operation {script.name or type(script).__name__} is disabled: {reason}", exc_info=True)
    runner.scripts.remove(script)


def _ensure_legacy_controls(runner: Any) -> None:
    """Build the legacy operations' Gradio controls when no UI did, one
    operation at a time."""

    pending = [script for script in runner.scripts_in_preferred_order() if not _has_params(script) and script.controls is None]
    if runner.ui_created or not pending:
        return
    import gradio as gr

    from modules import gradio_extensions  # noqa: F401 - the Gradio patches ui() code expects

    with gr.Blocks(analytics_enabled=False):
        for script in pending:
            try:
                controls = script.ui()
                if not isinstance(controls, Mapping):
                    raise TypeError(f"ui() returned {type(controls).__name__}, not a dict of controls")
                script.controls = dict(controls)
            except Exception as error:
                _drop(runner, script, f"ui() failed: {error}")


def _layout(script: Any, owner: str) -> ScriptLayout:
    if _has_params(script):
        cached = getattr(script, "_remi_layout", None)
        if cached is None or cached.owner != owner:
            cached = script._remi_layout = ScriptLayout(script.params(), owner=owner)
        return cached
    controls = script.controls or {}
    return legacy_layout(owner, list(controls.values()), names=list(controls))


def assign_args(runner: Any) -> list[tuple[Any, ScriptLayout]]:
    """``(operation, layout)`` in run order. Unless the Gradio UI built the
    operations (it then owns ``args_from``/``args_to``), assigns their
    argument slots; a ported operation gets the name list
    ``ScriptPostprocessingRunner.run`` zips its values with."""

    from modules.scripts import script_slug, unique_id

    _ensure_legacy_controls(runner)
    result: list[tuple[Any, ScriptLayout]] = []
    taken: set[str] = set()
    for script in runner.scripts_in_preferred_order():
        owner = unique_id(script_slug(script.name or type(script).__name__), taken)
        try:
            layout = _layout(script, owner)
        except Exception as error:
            _drop(runner, script, f"params() failed: {error}")
            continue
        taken.add(owner)
        result.append((script, layout))

    if not runner.ui_created:
        position = 0
        for script, layout in result:
            if script.controls is None:
                script.controls = dict.fromkeys(layout.names)
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
                "legacy": layout.legacy,
                "layout": layout.schema(ctx),
                "defaults": jsonable(layout.defaults(ctx)),
                "args": layout.names,
            }
        )
    return result
