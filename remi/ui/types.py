"""Typed shapes of the exported JSON (docs/remi-webui/schema-format.md).

Plain ``TypedDict``s: the exporters build ordinary dicts, these only give
them names and keys for type checkers.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

Condition: TypeAlias = bool | dict[str, Any]
ChoiceValue: TypeAlias = str | int | float


class ChoiceEntry(TypedDict):
    value: ChoiceValue
    label: str


class NodeSchema(TypedDict, total=False):
    """One exported node (§1). ``type`` is always present; the other keys
    depend on the kind."""

    type: str
    visible_if: Condition
    enabled_if: Condition
    # inputs
    name: str
    label: str
    info: str
    default: Any
    on_change: str
    on_input: str
    width: str | int
    hidden: bool
    infotext: str
    preset: str
    exclusive: dict[str, str]
    minimum: float
    maximum: float
    step: float
    integer: bool
    precision: int
    lines: int
    max_lines: int
    placeholder: str
    prompt: bool
    choices: list[ChoiceEntry]
    style: str
    allow_custom: bool
    kind: str
    source: str
    mode: str
    accept: str
    widget: str
    props: dict[str, Any]
    persist: bool
    submit: bool
    locks: str
    # layout / display
    title: str
    open: bool
    enable: NodeSchema
    children: list[NodeSchema]
    bind: str
    value: Any
    text: str
    action: str
    variant: str
    section: str


class PatchEntry(TypedDict, total=False):
    """One param's part of an action patch (§5)."""

    value: Any
    choices: list[ChoiceEntry]
    visible: bool
    enabled: bool
    label: str
    info: str
    minimum: float
    maximum: float
    step: float


Patch: TypeAlias = dict[str, Any]
"""``{param | "owner.param": PatchEntry}`` plus the reserved ``__toast__``,
``__outputs__`` and ``__settings__`` entries."""


class ScriptEntry(TypedDict):
    """One script of a tab schema (§4)."""

    id: str
    title: str
    kind: Literal["alwayson", "selectable"]
    section: str | None
    order: int | float
    visible_if: NotRequired[Condition]
    legacy: bool
    layout: NodeSchema | None
    defaults: dict[str, Any]
    args: list[str]


class OwnerSchema(TypedDict):
    owner: str
    layout: NodeSchema | None
    defaults: dict[str, Any]


class TabSchema(TypedDict):
    version: int
    tab: str
    core: OwnerSchema
    scripts: list[ScriptEntry]
    conflict_params: NotRequired[list[str]]
