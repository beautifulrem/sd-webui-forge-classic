"""Layouts of scripts still on the legacy ``ui()`` path.

Their controls are Gradio components; each becomes the matching ``remi.ui``
input (read by class name, as the settings exporter does), named ``#0``,
``#1``, … (or by the control keys of a postprocessing operation). A script
then receives exactly what Gradio passed it: choice indices for
``type="index"`` controls, ``Number`` precision, and for a param the request
left out or sent unchanged, Gradio's own conversion of the control's initial
value (``gr.State`` objects included). Controls without an equivalent are
``json`` params. Nothing here imports Gradio.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from remi.ui import nodes
from remi.ui.context import Context, ensure_context
from remi.ui.errors import ParamError, SchemaError
from remi.ui.layout import ScriptLayout
from remi.ui.schema import jsonable

# Gradio base classes this module maps (subclasses such as Forge's
# InputAccordion or DropdownMulti are found through their MRO)
_KINDS = ("Checkbox", "Slider", "Number", "Textbox", "Radio", "Dropdown", "CheckboxGroup", "ColorPicker", "State", "Image")
# Kinds whose preprocess() is a pure conversion of a value
_PURE_PREPROCESS = frozenset({"Checkbox", "Slider", "Number", "Textbox", "Radio", "Dropdown", "CheckboxGroup", "ColorPicker"})


def gradio_kind(control: Any) -> str | None:
    """The Gradio base class name of ``control`` among the mapped kinds."""

    for cls in type(control).__mro__:
        if cls.__name__ in _KINDS and (cls.__module__ or "").startswith("gradio"):
            return cls.__name__
    return None


def _same_json(a: Any, b: Any) -> bool:
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except (TypeError, ValueError):
        return False


def _copied(value: Any) -> Callable[[], Any]:
    """A factory returning fresh copies of ``value`` (Gradio copies a
    ``gr.State`` per session; scripts may mutate what they get)."""

    def make() -> Any:
        try:
            return copy.deepcopy(value)
        except Exception:  # an object that cannot be copied is passed as is
            return value

    return make


class LegacyParam(nodes.Json):
    """A control without a ``remi.ui`` equivalent: its value as JSON, cast on
    paste like Gradio's paste did (by the type of the control's value)."""

    def __init__(self, name: str, default: Any, *, value_type: type | None, **common):
        self.value_type = value_type
        super().__init__(name, default, **common)

    def resolve_default(self, ctx: Context) -> Any:
        return jsonable(self.default)

    def from_paste(self, raw: Any, ctx: Context) -> Any:
        if self.value_type is None or isinstance(raw, self.value_type):
            return raw
        try:
            if self.value_type is bool:
                return raw != "False" and bool(raw)
            if self.value_type is int:
                return float(raw)
            return self.value_type(raw)
        except (TypeError, ValueError):
            raise ParamError(f"cannot paste {raw!r} as {self.value_type.__name__}") from None


class LegacyState(nodes.State):
    """A ``gr.State``: never rendered; the script gets the state object."""

    def resolve_default(self, ctx: Context) -> Any:
        return jsonable(self.default)


class _LegacyChoiceMixin:
    """Gradio leniency: no selection (``None``) and the declared initial value
    are accepted even when not among the choices; an unknown value's index is
    ``None``, as in Gradio's preprocess."""

    def resolve_default(self, ctx: Context) -> Any:
        return jsonable(self.default)

    def _index(self, value: Any, ctx: Context) -> int | None:
        values = self.values(ctx)
        return values.index(value) if value in values else None


class LegacyChoice(_LegacyChoiceMixin, nodes.Choice):
    def coerce(self, value: Any, ctx: Context) -> Any:
        if value is None or _same_json(value, self.default):
            return value
        return super().coerce(value, ctx)


class LegacyMultiChoice(_LegacyChoiceMixin, nodes.MultiChoice):
    def coerce(self, value: Any, ctx: Context) -> list[Any]:
        if value is not None and _same_json(value, self.default):
            return list(value)
        return super().coerce(value, ctx)


class LegacyImage(nodes.Image):
    """A ``gr.Image``: the script gets a PIL image or a numpy array, in the
    control's ``image_mode``, as Gradio's ``type`` asked."""

    def __init__(self, name: str, *, returns: str, image_mode: str | None, **common):
        self.returns_kind = returns
        super().__init__(name, "image", mode=image_mode, **common)

    def to_arg(self, value: Any, ctx: Context) -> Any:
        image = super().to_arg(value, ctx)
        if self.returns_kind == "numpy" and image is not None and not isinstance(image, str):
            import numpy as np

            return np.asarray(image)
        return image


def _choice_kwargs(control: Any) -> dict[str, Any]:
    returns = getattr(control, "type", "value") if getattr(control, "type", "value") in ("value", "index") else "value"
    allow_custom = bool(getattr(control, "allow_custom_value", False)) and returns == "value"
    return {"type": returns, "allow_custom": allow_custom}


def legacy_node(name: str, control: Any, **common: Any) -> nodes.Input:
    """The ``remi.ui`` input for one Gradio control (``json`` when unknown)."""

    kind = gradio_kind(control)
    value = getattr(control, "value", None)
    try:
        if kind == "Checkbox":
            return nodes.Toggle(name, bool(value), **common)
        if kind == "Slider":
            minimum, maximum = control.minimum, control.maximum
            step = control.step or 1
            return nodes.Slider(name, minimum, maximum, minimum if value is None else value, step, **common)
        if kind == "Number" and value is not None:
            precision = control.precision if isinstance(control.precision, int) else None
            return nodes.Number(name, value, control.minimum, control.maximum, control.step, precision, **common)
        if kind == "Textbox":
            lines = control.lines if isinstance(control.lines, int) and control.lines >= 1 else 1
            return nodes.Text(name, "" if value is None else str(value), lines=lines, max_lines=control.max_lines, placeholder=control.placeholder, **common)
        if kind == "Radio":
            return LegacyChoice(name, list(control.choices or ()), value, style="radio", **_choice_kwargs(control), **common)
        if kind == "Dropdown":
            if getattr(control, "multiselect", False):
                return LegacyMultiChoice(name, list(control.choices or ()), list(value or ()), **_choice_kwargs(control), **common)
            return LegacyChoice(name, list(control.choices or ()), value, **_choice_kwargs(control), **common)
        if kind == "CheckboxGroup":
            return LegacyMultiChoice(name, list(control.choices or ()), list(value or ()), style="checkboxes", **_choice_kwargs(control), **common)
        if kind == "ColorPicker" and isinstance(value, str) and nodes.COLOR_RE.fullmatch(value):
            return nodes.Color(name, value, **common)
        if kind == "State":
            return LegacyState(name, value, **common)
        if kind == "Image" and getattr(control, "type", None) in ("pil", "numpy"):
            return LegacyImage(name, returns=control.type, image_mode=getattr(control, "image_mode", None), **common)
    except (SchemaError, TypeError, AttributeError):
        pass  # a control this mapping cannot express: plain JSON
    return LegacyParam(name, value, value_type=None if value is None else type(value), **common)


def initial_arg(control: Any) -> Callable[[], Any]:
    """What Gradio passed the script for the untouched ``control``."""

    value = getattr(control, "value", None)
    if gradio_kind(control) in _PURE_PREPROCESS:
        try:
            value = control.preprocess(value)
        except Exception:  # e.g. gr.Number's range check: Gradio would have failed, pass the value
            pass
    return _copied(value)


class LegacyLayout(ScriptLayout):
    """A legacy script's layout. Params the request leaves out, or sends
    with their exported default, become the control's initial argument."""

    legacy = True

    def __init__(self, root: nodes.Node | None, *, owner: str, initial_args: Mapping[str, Callable[[], Any]]):
        super().__init__(root, owner=owner)
        self.initial_args = dict(initial_args)

    def default_arg(self, node: nodes.Input, default: Any, ctx: Context) -> Any:
        return self.initial_args[node.name]()

    def args_for(self, coerced: Mapping[str, Any], ctx: Context | None = None, *, given: Mapping[str, Any] | None = None) -> list[Any]:
        ctx = ensure_context(ctx)
        if given is not None:
            given = {name: value for name, value in given.items() if name not in self.by_name or not _same_json(value, self.by_name[name].resolve_default(ctx))}
        return super().args_for(coerced, ctx, given=given)


def legacy_layout(owner: str, controls: Sequence[Any], infotext_fields: Sequence[Any] | None = None, *, names: Sequence[str] | None = None) -> LegacyLayout:
    """A positional layout for Gradio ``controls`` and their paste fields
    (``(component, key or function)`` pairs, any of which may be absent).
    Params are named ``#0``, ``#1``, … unless ``names`` are given (the
    control dict keys of a postprocessing operation)."""

    targets: dict[int, Any] = {}
    for field in infotext_fields or ():
        component, target = field[0], field[1]
        targets.setdefault(id(component), target)

    params: list[nodes.Input] = []
    initial: dict[str, Callable[[], Any]] = {}
    keys: set[str] = set()
    for index, control in enumerate(controls or ()):
        name = names[index] if names is not None else f"#{index}"
        target = targets.get(id(control))
        # A PasteKey (modules.anima_support) is both the key and the paste function
        key = str(target) if isinstance(target, str) and str(target) not in keys else None
        if key:
            keys.add(key)
        label = str(getattr(control, "label", None) or name)
        params.append(legacy_node(name, control, label=label, infotext=key, paste=target if callable(target) else None))
        initial[name] = initial_arg(control)
    root = nodes.Group(children=params) if params else None
    return LegacyLayout(root, owner=owner, initial_args=initial)
