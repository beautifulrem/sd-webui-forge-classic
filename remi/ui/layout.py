"""Binding of a ``params()`` tree to positional processing arguments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from remi.ui import conditions
from remi.ui.context import Context, ensure_context
from remi.ui.errors import ParamError, SchemaError, ValidationError
from remi.ui.nodes import _NO_VALUE, Choice, Input, Node, Tabs
from remi.ui.types import NodeSchema

UNKNOWN_POLICIES = ("error", "ignore")


class ScriptLayout:
    """One owner's (a script's, or the core form's) parameters.

    ``inputs`` are the processing arguments in positional order: the
    depth-first declaration order of the tree, unless ``order`` (every
    argument param exactly once) overrides it for a script whose argument
    order differs from its visual order. Named values become positional
    arguments with :meth:`values_to_args`; missing names take their defaults.
    ``tools`` are the tool inputs (``arg=False``): validated and defaulted
    like the others, never an argument."""

    def __init__(self, root: Node | None, *, owner: str = "script", order: Sequence[str] | None = None):
        self.root = root
        self.owner = owner

        declared: dict[str, Input] = {}
        for node in root.walk() if root is not None else ():
            if isinstance(node, Input):
                if node.name in declared:
                    raise SchemaError(f"{owner}: duplicate param name {node.name!r}")
                declared[node.name] = node
        arguments = {name: node for name, node in declared.items() if node.arg}
        if order is not None:
            order = list(order)
            if len(order) != len(set(order)) or set(order) != set(arguments):
                raise SchemaError(f"{owner}: order must list every argument param exactly once (params: {list(arguments)})")
            self.inputs: list[Input] = [arguments[name] for name in order]
        else:
            self.inputs = list(arguments.values())
        self.tools: list[Input] = [node for node in declared.values() if not node.arg]
        self.by_name: dict[str, Input] = {node.name: node for node in (*self.inputs, *self.tools)}

        self.infotext: dict[str, str] = {}
        """infotext key -> param name"""
        self.paste_functions: dict[str, Any] = {}
        """param name -> ``paste(params)`` callable"""
        self.presets: dict[str, tuple[str, Any]] = {}
        """Anima preset field -> (param name, preset value or the table's)"""
        self.exclusive: dict[str, dict[str, str]] = {}
        """exclusive group -> {member -> param name}"""

        for node in self.inputs:
            if node.infotext:
                if node.infotext in self.infotext:
                    raise SchemaError(f"{owner}: infotext key {node.infotext!r} is used by {self.infotext[node.infotext]!r} and {node.name!r}")
                self.infotext[node.infotext] = node.name
            if node.paste is not None:
                self.paste_functions[node.name] = node.paste
            if node.preset_key:
                if node.preset_key in self.presets:
                    raise SchemaError(f"{owner}: preset field {node.preset_key!r} is bound twice")
                self.presets[node.preset_key] = (node.name, node.preset_value)
            if node.exclusive:
                group, member = node.exclusive
                self.exclusive.setdefault(group, {})[member] = node.name
        self._check_references()

    def _check_references(self) -> None:
        if self.root is None:
            return
        for node in self.root.walk():
            for condition in (node.visible_if, node.enabled_if, getattr(node, "inactive_if", None)):
                for name in conditions.references(condition):
                    if "." not in name and name not in self.by_name:
                        raise SchemaError(f"{self.owner}: condition refers to unknown param {name!r}")
            if isinstance(node, Tabs) and node.bind is not None:
                bound = self.by_name.get(node.bind)
                if not isinstance(bound, Choice):
                    raise SchemaError(f"{self.owner}: tabs bind to {node.bind!r}, which is not a Choice of this owner")
                if not callable(bound.choices):
                    values = bound.values(Context())
                    missing = [tab.value for tab in node.items if tab.value not in values]
                    if missing:
                        raise SchemaError(f"{self.owner}: tab values {missing} are not choices of {node.bind!r}")

    # region Introspection

    @property
    def names(self) -> list[str]:
        return [node.name for node in self.inputs]

    def __len__(self) -> int:
        return len(self.inputs)

    def __contains__(self, name: str) -> bool:
        return name in self.by_name

    def defaults(self, ctx: Context | None = None) -> dict[str, Any]:
        ctx = ensure_context(ctx)
        return {node.name: node.resolve_default(ctx) for node in (*self.inputs, *self.tools)}

    def schema(self, ctx: Context | None = None) -> NodeSchema | None:
        """The exported layout tree (``None`` for a script without params)."""

        return None if self.root is None else self.root.schema(ensure_context(ctx))

    # endregion

    # region Values

    def coerce(self, values: Mapping[str, Any] | None, ctx: Context | None = None, *, unknown: str = "error") -> dict[str, Any]:
        """Every param's validated wire value; missing names take defaults.

        Raises :class:`ValidationError` listing every rejected value. Names
        this layout does not declare are rejected (``unknown="error"``) or
        dropped (``unknown="ignore"``, for drafts of an older layout)."""

        if unknown not in UNKNOWN_POLICIES:
            raise ValueError(f"unknown is one of {UNKNOWN_POLICIES}")
        ctx = ensure_context(ctx)
        values = self.check_mapping(values)
        errors: list[ParamError] = []
        if unknown == "error":
            errors.extend(ParamError("unknown parameter", param=name, owner=self.owner) for name in values if name not in self.by_name)

        result: dict[str, Any] = {}
        for node in (*self.inputs, *self.tools):
            if node.name not in values:
                result[node.name] = node.resolve_default(ctx)
                continue
            try:
                result[node.name] = node.coerce(values[node.name], ctx)
            except ParamError as error:
                errors.append(self._located(error, node))
        if errors:
            raise ValidationError(errors)
        return result

    def check_mapping(self, values: Any) -> dict[str, Any]:
        """``values`` as a dict, or :class:`ValidationError` when the request
        sent something other than an object of named values."""

        if values is None:
            return {}
        if not isinstance(values, Mapping) or not all(isinstance(key, str) for key in values):
            raise ValidationError([ParamError(f"expected an object of named values, got {type(values).__name__}", owner=self.owner)])
        return dict(values)

    def values_to_args(self, values: Mapping[str, Any] | None, ctx: Context | None = None, *, unknown: str = "error") -> list[Any]:
        """The positional processing arguments for named ``values``."""

        ctx = ensure_context(ctx)
        given = self.check_mapping(values)
        return self.args_for(self.coerce(given, ctx, unknown=unknown), ctx, given=given)

    def args_for(self, coerced: Mapping[str, Any], ctx: Context | None = None, *, given: Mapping[str, Any] | None = None) -> list[Any]:
        """The positional processing arguments for already coerced values
        (:meth:`coerce`'s result). ``given`` are the values the request sent;
        the others are defaults (see :meth:`default_arg`)."""

        ctx = ensure_context(ctx)
        errors: list[ParamError] = []
        args: list[Any] = []
        for node in self.inputs:
            try:
                if given is not None and node.name not in given:
                    args.append(self.default_arg(node, coerced[node.name], ctx))
                else:
                    args.append(node.to_arg(coerced[node.name], ctx))
            except ParamError as error:
                errors.append(self._located(error, node))
        if errors:
            raise ValidationError(errors)
        return args

    def default_arg(self, node: Input, default: Any, ctx: Context) -> Any:
        """The processing argument of a param the request left out."""

        return node.to_arg(default, ctx)

    def args_to_values(self, args: Sequence[Any], ctx: Context | None = None) -> dict[str, Any]:
        """Named wire values for positional ``args`` (best effort: a value
        that does not validate is kept as given; missing trailing args take
        their defaults)."""

        ctx = ensure_context(ctx)
        args = list(args)
        if len(args) > len(self.inputs):
            raise ValueError(f"{self.owner} takes {len(self.inputs)} arguments, got {len(args)}")
        values: dict[str, Any] = {}
        for index, node in enumerate(self.inputs):
            if index >= len(args):
                values[node.name] = node.resolve_default(ctx)
                continue
            try:
                values[node.name] = node.coerce(node.from_arg(args[index], ctx), ctx)
            except ParamError:
                values[node.name] = args[index]
        return values

    def paste(self, params: Mapping[str, Any], ctx: Context | None = None) -> tuple[dict[str, Any], list[ParamError]]:
        """Values read from parsed infotext ``params``: ``(values, problems)``.

        Each param with a ``paste`` function or an ``infotext`` key is read;
        absent entries (``None``) are skipped, as are values that do not
        validate (reported in ``problems``)."""

        ctx = ensure_context(ctx)
        values: dict[str, Any] = {}
        problems: list[ParamError] = []
        for node in self.inputs:
            if node.paste is not None:
                try:
                    raw = node.paste(params)
                except Exception as error:  # an extension's paste function
                    problems.append(ParamError(f"paste function failed: {error}", param=node.name, owner=self.owner))
                    continue
            elif node.infotext:
                raw = params.get(node.infotext)
            else:
                continue
            if raw is None:
                continue
            try:
                values[node.name] = node.from_paste(raw, ctx)
            except ParamError as error:
                problems.append(self._located(error, node))
        return values, problems

    def preset_values(self, table: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Named values a preset gives this owner's bound params: the table
        value (or the param's own preset value), replaced by ``overrides``."""

        overrides = overrides or {}
        values = {}
        for key, (name, own) in self.presets.items():
            if key in overrides:
                values[name] = overrides[key]
            elif own is not _NO_VALUE:
                values[name] = own
            elif key in table:
                values[name] = table[key]
        return values

    def _located(self, error: ParamError, node: Input) -> ParamError:
        error.param = node.name
        error.owner = self.owner
        return error

    # endregion


def empty_layout(owner: str) -> ScriptLayout:
    return ScriptLayout(None, owner=owner)


__all__ = ["ScriptLayout", "ValidationError", "empty_layout"]
