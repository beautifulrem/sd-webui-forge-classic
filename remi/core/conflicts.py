"""Anima conflict locks as an action patch (the core ``__conflicts__`` action).

The rules themselves are the pure resolvers of
``modules.anima_feature_conflicts``; this module only finds which params
play which role (from the layouts' ``preset=`` and ``exclusive=`` bindings)
and turns the resolved state into a patch relative to the core owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remi.ui import Context, ParamError
from remi.ui.types import Patch

# Roles without a preset field, bound by the built-in scripts
EXTRA_ROLES: dict[str, tuple[str, str]] = {"native.refiner_checkpoint": ("refiner", "checkpoint")}


def _layouts(tab: str) -> dict[str, Any]:
    from remi.core.tabs import core_layout, runner_for

    layouts = {"core": core_layout(tab)}
    runner = runner_for(tab)
    for script in runner.scripts if runner is not None else ():
        if script.layout is not None:
            layouts[script.script_id] = script.layout
    return layouts


def bindings(tab: str, layouts: Mapping[str, Any] | None = None) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, tuple[str, str]]]]:
    """``(role -> (owner, param), group -> {member -> (owner, param)})``."""

    layouts = layouts if layouts is not None else _layouts(tab)
    roles: dict[str, tuple[str, str]] = {}
    groups: dict[str, dict[str, tuple[str, str]]] = {}
    for owner, layout in layouts.items():
        for key, (name, _value) in layout.presets.items():
            roles.setdefault(key, (owner, name))
        for group, members in layout.exclusive.items():
            for member, name in members.items():
                groups.setdefault(group, {}).setdefault(member, (owner, name))
    for role, (owner, name) in EXTRA_ROLES.items():
        if owner in layouts and name in layouts[owner]:
            roles.setdefault(role, (owner, name))
    return roles, groups


def _ref(owner: str, name: str) -> str:
    return name if owner == "core" else f"{owner}.{name}"


def conflict_patch(tab: str, values: Mapping[str, Mapping[str, Any]], selected: str | None = None, *, ctx: Context | None = None, layouts: Mapping[str, Any] | None = None) -> Patch:
    """The patch locking conflicting controls for the current ``values``
    (``{"core": {...}, script_id: {...}}``; missing params read as their
    defaults). ``selected`` (``"owner.param"``, core params as
    ``"core.param"``) is what the user just changed: their last action wins.

    The exclusive groups are resolved first and the role rules see their
    result. A group member a role rule switches off leaves its group, which
    is then resolved again (so the next member in line may win), until no
    rule switches anything else off. Each control gets one entry: its final
    value, ``enabled`` only when every lock leaves it editable, and the
    narrowed ``choices``. Keys are relative to the core owner (core params
    bare, others ``owner.param``)."""

    from modules.anima_feature_conflicts import CONFLICT_ROLES, exclusive_group_order, resolve_conflict_locks, resolve_exclusive_features

    ctx = ctx or Context()
    layouts = layouts if layouts is not None else _layouts(tab)
    roles, groups = bindings(tab, layouts)
    roles = {role: ref for role, ref in roles.items() if role in CONFLICT_ROLES}
    groups = {group: members for group, members in groups.items() if len(members) >= 2}
    member_refs = {ref for members in groups.values() for ref in members.values()}

    def current(owner: str, name: str) -> Any:
        node = layouts[owner].by_name[name]
        owner_values = values.get(owner) or {}
        if isinstance(owner_values, Mapping) and name in owner_values:
            try:
                return node.coerce(owner_values[name], ctx)
            except ParamError:  # a draft value the form will reject anyway
                pass
        return node.resolve_default(ctx)

    selected_ref = tuple(selected.split(".", 1)) if isinstance(selected, str) and "." in selected else None
    selected_role = next((role for role, ref in roles.items() if ref == selected_ref), None)

    switched_off: set[tuple[str, str]] = set()
    messages: list[str] = []
    state: dict[tuple[str, str], Any] = {}
    enabled: dict[tuple[str, str], bool] = {}

    def lock(ref: tuple[str, str], editable: bool) -> None:
        enabled[ref] = enabled.get(ref, True) and editable

    while True:  # ends: switched_off only grows, and is bounded by the members
        state.clear()
        enabled.clear()
        for group, members in groups.items():
            names = exclusive_group_order(group, members)
            chosen = next((name for name in names if members[name] == selected_ref), None)
            wanted = tuple(members[name] not in switched_off and bool(current(*members[name])) for name in names)
            resolved = resolve_exclusive_features(names, wanted, selected=chosen)
            for name, on, editable in zip(names, resolved.enabled, resolved.interactive):
                state[members[name]] = on
                lock(members[name], editable)

        role_values = {role: state[ref] if ref in state else current(*ref) for role, ref in roles.items()}
        locks = resolve_conflict_locks(role_values, selected=selected_role)
        for role, resolved_value in locks.values.items():
            state[roles[role]] = resolved_value
        for role, editable in locks.enabled.items():
            lock(roles[role], editable)
        choices = {roles[role]: allowed for role, allowed in locks.choices.items()}
        messages += [message for message in locks.messages if message not in messages]

        newly_off = {roles[role] for role, resolved_value in locks.values.items() if roles[role] in member_refs and role_values[role] and not resolved_value}
        if newly_off <= switched_off:
            break
        switched_off |= newly_off

    patch: Patch = {}
    for ref, resolved_value in state.items():
        entry: dict[str, Any] = {"value": resolved_value}
        if ref in enabled:
            entry["enabled"] = enabled[ref]
        if ref in choices:
            entry["choices"] = [{"value": choice, "label": choice} for choice in choices[ref]]
        patch[_ref(*ref)] = entry
    if messages:
        patch["__toast__"] = {"level": "warning", "message": "Compatibility: " + "; ".join(messages)}
    return patch
