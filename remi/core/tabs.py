"""Per-tab bindings: the runner, the core layout, the tab schema and the
mapping of named request values to processing arguments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remi.core import forms
from remi.ui import SCHEMA_VERSION, Context, ParamError, ScriptLayout, ValidationError, jsonable
from remi.ui.types import TabSchema


def runner_for(tab: str):
    """The script runner of ``tab`` (None for extras, whose operations run
    through ``scripts.scripts_postproc``)."""

    from modules import scripts

    if tab == "txt2img":
        return scripts.scripts_txt2img
    if tab == "img2img":
        return scripts.scripts_img2img
    if tab == "extras":
        return None
    raise KeyError(f"unknown tab {tab!r} (one of {forms.TABS})")


def core_layout(tab: str) -> ScriptLayout:
    return forms.core_layout(tab, runner_for(tab))


def conflict_params(tab: str, layouts: Mapping[str, ScriptLayout] | None = None) -> list[str]:
    """``owner.param`` references whose changes should run the core
    ``__conflicts__`` action: every param bound to a conflict role or an
    exclusive group. ``layouts``: the tab's owners, when already built."""

    from modules.anima_feature_conflicts import CONFLICT_ROLES
    from remi.core.conflicts import bindings

    roles, groups = bindings(tab, layouts)
    refs = [f"{owner}.{name}" for role, (owner, name) in roles.items() if role in CONFLICT_ROLES]
    refs += [f"{owner}.{name}" for members in groups.values() for owner, name in members.values()]
    return sorted(set(refs))


def tab_schema(tab: str, ctx: Context | None = None) -> TabSchema:
    """``GET /api/v2/schema/{tab}`` (schema-format.md §4)."""

    from modules import scripts
    from remi.core import extras

    ctx = ctx or Context()
    runner = runner_for(tab)
    layout = forms.core_layout(tab, runner)
    if runner is None:
        script_entries = extras.scripts_schema(scripts.scripts_postproc, ctx)
    else:
        script_entries = runner.scripts_schema(ctx)
    schema: TabSchema = {
        "version": SCHEMA_VERSION,
        "tab": tab,
        "core": {"owner": "core", "layout": layout.schema(ctx), "defaults": jsonable(layout.defaults(ctx))},
        "scripts": script_entries,
    }
    if runner is not None:
        layouts = {"core": layout, **{script.script_id: script.layout for script in runner.scripts if script.layout is not None}}
        schema["conflict_params"] = conflict_params(tab, layouts)
    return schema


def _selected_script(layout: ScriptLayout, core: Any, ctx: Context) -> Any:
    """The core ``script`` value on its own, so the selected script's values
    are still checked when other core values are rejected."""

    node = layout.by_name.get("script")
    if node is None:
        return None
    try:
        return node.coerce(core["script"], ctx) if isinstance(core, Mapping) and "script" in core else node.resolve_default(ctx)
    except ParamError:
        return None


def processing_args(tab: str, core: Any, scripts_values: Any, ctx: Context | None = None) -> tuple[dict[str, Any], list]:
    """Validate a request: ``(core processing args by name, script args)``.

    Core values are coerced and converted (choice indices, uploads); script
    values become the full positional ``p.script_args`` (extras: the
    operations' positional args). Raises :class:`ValidationError` listing
    every rejected value (a body that is not an object of named values
    included)."""

    from modules import scripts
    from remi.core import extras

    ctx = ctx or Context()
    runner = runner_for(tab)
    layout = forms.core_layout(tab, runner)
    problems: list = []
    core_args: dict[str, Any] = {}
    try:
        values = layout.coerce(core, ctx)
    except ValidationError as error:
        problems.append(error)
        values = {"script": _selected_script(layout, core, ctx)}
    else:
        problems += forms.check_core(tab, values)
        try:
            core_args = dict(zip(layout.names, layout.args_for(values, ctx), strict=True))
        except ValidationError as error:
            problems.append(error)
    try:
        if runner is None:
            script_args = extras.postprocessing_args(scripts.scripts_postproc, scripts_values, ctx)
        else:
            script_args = runner.script_args_from_values(values.get("script"), scripts_values, ctx)
    except ValidationError as error:
        problems.append(error)
        script_args = []
    if problems:
        raise ValidationError.merge(problems)
    return core_args, script_args
