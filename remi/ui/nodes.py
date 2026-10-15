"""Parameter and layout nodes of the ``remi.ui`` DSL.

A script's ``params()`` returns one tree of these nodes. Inputs hold values;
the depth-first declaration order of the inputs (a ``Section``'s ``enable``
toggle first) is the positional order of the script's processing arguments.
The JSON each node exports is specified in docs/remi-webui/schema-format.md.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, TypeAlias

from remi.ui import conditions
from remi.ui.context import Context
from remi.ui.errors import ParamError, SchemaError
from remi.ui.types import ChoiceValue, NodeSchema

NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|#\d+")
COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")
SEED_MAX = 2**63 - 1
"""Largest seed: per-image offsets (``seed + i``) stay inside torch's 64-bit range."""
# Infotext writes Python's str(bool); paste reads anything else present as on
_PASTE_FALSE_TEXT = frozenset({"false", "0", "no", "off", ""})


def _flatten(children: Iterable[Any]) -> list[Node]:
    """Children given positionally and/or as lists; ``None`` entries are
    skipped so conditional declarations read naturally."""

    flat: list[Node] = []
    for child in children:
        if child is None:
            continue
        if isinstance(child, Node):
            flat.append(child)
        elif isinstance(child, Iterable) and not isinstance(child, (str, bytes, Mapping)):
            flat.extend(_flatten(child))
        else:
            raise SchemaError(f"not a layout node: {child!r}")
    return flat


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any, what: str) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise SchemaError(f"{what} must be a finite number, got {value!r}")
    return value


def to_number(value: Any) -> int | float:
    """A finite number from a JSON number or a numeric string."""

    if isinstance(value, bool):
        raise ParamError(f"expected a number, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            value = int(text)
        except ValueError:
            try:
                value = float(text)
            except ValueError:
                raise ParamError(f"expected a number, got {value!r}") from None
    elif not isinstance(value, (int, float)):
        raise ParamError(f"expected a number, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ParamError(f"expected a finite number, got {value!r}")
    return value


def _as_int(value: int | float) -> int:
    return value if isinstance(value, int) else int(math.floor(value + 0.5))


def check_finite_json(value: Any, path: str = "value") -> Any:
    """``value`` when it holds no NaN/Infinity anywhere (JSON numbers only)."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ParamError(f"{path} must be a finite number, got {value!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            check_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_finite_json(item, f"{path}[{index}]")
    return value


def _is_image(value: Any) -> bool:
    try:
        from PIL import Image as PILImage
    except ImportError:  # pragma: no cover - Pillow ships with the WebUI
        return False
    return isinstance(value, PILImage.Image)


# region Base classes


class Node:
    """Anything that can appear in a layout tree."""

    type: ClassVar[str] = ""

    def __init__(self, *, visible_if: conditions.Condition | None = None, enabled_if: conditions.Condition | None = None):
        self.visible_if = None if visible_if is None else conditions.check(visible_if)
        self.enabled_if = None if enabled_if is None else conditions.check(enabled_if)

    @property
    def children(self) -> list[Node]:
        """Child nodes in declaration order (what :meth:`walk` visits)."""

        return []

    def walk(self) -> Iterator[Node]:
        """This node and its subtree, depth first, in declaration order."""

        yield self
        for child in self.children:
            yield from child.walk()

    def schema(self, ctx: Context) -> NodeSchema:
        data: dict[str, Any] = {"type": self.type}
        data.update(self._schema(ctx))
        if self.visible_if is not None:
            data["visible_if"] = self.visible_if
        if self.enabled_if is not None:
            data["enabled_if"] = self.enabled_if
        return data  # type: ignore[return-value]

    def _schema(self, ctx: Context) -> dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        name = getattr(self, "name", None)
        return f"{type(self).__name__}({name!r})" if name else f"{type(self).__name__}()"


class Input(Node):
    """A node holding one value; ``name`` is its param id within the owner.

    Common keyword arguments (architecture.md): ``label`` (English; default
    from the name, ``""`` hides it), ``info``, ``visible_if``, ``enabled_if``,
    ``infotext`` (infotext key), ``paste`` (``params -> value`` for derived
    paste; ``None`` skips), ``preset`` (Anima preset field, or ``(field,
    value)`` when this control's preset value differs from the table's),
    ``exclusive=(group, member)`` (``Toggle`` only), ``on_change`` (action
    name), ``width`` and ``hidden`` (submitted but not rendered)."""

    nullable: ClassVar[bool] = False
    """Whether ``None`` is a regular value (images, files, JSON)."""

    def __init__(
        self,
        name: str,
        default: Any = None,
        *,
        label: str | None = None,
        info: str | None = None,
        visible_if: conditions.Condition | None = None,
        enabled_if: conditions.Condition | None = None,
        infotext: str | None = None,
        paste: Callable[[dict], Any] | None = None,
        preset: str | tuple[str, Any] | None = None,
        exclusive: tuple[str, str] | None = None,
        on_change: str | None = None,
        width: str | int | None = None,
        hidden: bool = False,
    ):
        super().__init__(visible_if=visible_if, enabled_if=enabled_if)
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise SchemaError(f"bad param name {name!r} (use letters, digits and _)")
        if paste is not None and not callable(paste):
            raise SchemaError(f"{name}: paste must be callable")
        if infotext is not None and not isinstance(infotext, str):
            raise SchemaError(f"{name}: infotext must be a key string")
        self.name = name
        self.default = default
        self.label = _humanize(name) if label is None else str(label)
        self.info = info
        self.infotext = infotext
        self.paste = paste
        self.preset_key, self.preset_value = self._preset(preset)
        self.exclusive: tuple[str, str] | None = None
        if exclusive is not None:
            self.set_exclusive(exclusive)
        self.on_change = on_change
        self.width = self._width(width)
        self.hidden = bool(hidden)

    def _preset(self, preset: str | tuple[str, Any] | None) -> tuple[str | None, Any]:
        if preset is None:
            return None, None
        if isinstance(preset, str):
            return preset, _NO_VALUE
        if isinstance(preset, tuple) and len(preset) == 2 and isinstance(preset[0], str):
            return preset
        raise SchemaError(f"{self.name}: preset is a field name or (field, value)")

    def set_preset(self, preset: str | tuple[str, Any]) -> None:
        self.preset_key, self.preset_value = self._preset(preset)

    def set_exclusive(self, exclusive: tuple[str, str]) -> None:
        if not isinstance(self, Toggle):
            raise SchemaError(f"{self.name}: only a Toggle can be exclusive")
        if not (isinstance(exclusive, Sequence) and len(exclusive) == 2 and all(isinstance(part, str) and part for part in exclusive)):
            raise SchemaError(f"{self.name}: exclusive is (group, member)")
        self.exclusive = (exclusive[0], exclusive[1])

    def _width(self, width: str | int | None) -> str | int | None:
        if width is None or width in ("auto", "fill"):
            return width
        if isinstance(width, int) and not isinstance(width, bool) and 1 <= width <= 12:
            return width
        raise SchemaError(f"{self.name}: width is 'auto', 'fill' or 1-12, got {width!r}")

    # region Value protocol

    def resolve_default(self, ctx: Context) -> Any:
        """The default as the client sees it (callables evaluated, normalized)."""

        value = ctx.resolve(self, "default", self.default)
        try:
            return self.coerce(value, ctx)
        except ParamError:
            return self.fallback_default(value, ctx)

    def fallback_default(self, value: Any, ctx: Context) -> Any:
        """The default when the declared one does not validate (e.g. a stored
        sampler that is hidden now): kept as declared unless overridden."""

        return value

    def coerce(self, value: Any, ctx: Context) -> Any:
        """The canonical wire value for ``value``, or :class:`ParamError`."""

        return value

    def from_paste(self, raw: Any, ctx: Context) -> Any:
        """The value an infotext entry ``raw`` (usually a string) pastes as."""

        return self.coerce(raw, ctx)

    def to_arg(self, value: Any, ctx: Context) -> Any:
        """The processing argument for a coerced wire value."""

        return value

    def from_arg(self, arg: Any, ctx: Context) -> Any:
        """The wire value of a processing argument (inverse of :meth:`to_arg`)."""

        return arg

    # endregion

    def _schema(self, ctx: Context) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "label": self.label}
        if self.info:
            data["info"] = self.info
        data["default"] = self.resolve_default(ctx)
        data.update(self._input_schema(ctx))
        if self.on_change:
            data["on_change"] = self.on_change
        if self.width is not None:
            data["width"] = self.width
        if self.hidden:
            data["hidden"] = True
        if self.infotext:
            data["infotext"] = self.infotext
        if self.preset_key:
            data["preset"] = self.preset_key
        if self.exclusive:
            data["exclusive"] = {"group": self.exclusive[0], "member": self.exclusive[1]}
        return data

    def _input_schema(self, ctx: Context) -> dict[str, Any]:
        return {}


_NO_VALUE = object()  # ``preset="field"``: the preset table's value applies


# endregion

# region Inputs


class Toggle(Input):
    type = "toggle"

    def __init__(self, name: str, default: bool = False, **common):
        super().__init__(name, default, **common)

    def coerce(self, value: Any, ctx: Context) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if _is_int(value) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ParamError(f"expected true or false, got {value!r}")

    def from_paste(self, raw: Any, ctx: Context) -> bool:
        # Infotext writes Python's str(bool); anything else that is present
        # reads as "on", as Gradio's bool(text) did.
        if isinstance(raw, str):
            return raw.strip().lower() not in _PASTE_FALSE_TEXT
        return bool(raw)


class Slider(Input):
    """A bounded number. Integer-valued when the bounds, the step and a
    static default are all ``int`` (``Slider(0.0, 2.0, 1.0, 1.0)`` is a float
    slider; override with ``integer=``). Values are clamped to the bounds."""

    type = "slider"

    def __init__(self, name: str, minimum: float = 0, maximum: float = 100, default: float | None = None, step: float = 1, *, integer: bool | None = None, **common):
        minimum, maximum, step = (_finite(value, f"{name}: {what}") for value, what in ((minimum, "minimum"), (maximum, "maximum"), (step, "step")))
        if minimum > maximum:
            raise SchemaError(f"{name}: minimum {minimum} exceeds maximum {maximum}")
        if step <= 0:
            raise SchemaError(f"{name}: step must be positive")
        if default is None:
            default = minimum
        if integer is None:
            static_default = () if callable(default) else (default,)
            integer = all(_is_int(value) for value in (minimum, maximum, step, *static_default))
        self.minimum, self.maximum, self.step, self.integer = minimum, maximum, step, bool(integer)
        super().__init__(name, default, **common)

    def coerce(self, value: Any, ctx: Context) -> int | float:
        number = min(max(to_number(value), self.minimum), self.maximum)
        return _as_int(number) if self.integer else float(number)

    def _input_schema(self, ctx: Context) -> dict[str, Any]:
        return {"minimum": self.minimum, "maximum": self.maximum, "step": self.step, "integer": self.integer}


class Number(Input):
    """A free number. ``precision=0`` makes it an integer; out-of-bounds
    values are rejected (sliders clamp, numbers do not)."""

    type = "number"

    def __init__(self, name: str, default: float = 0, minimum: float | None = None, maximum: float | None = None, step: float | None = None, precision: int | None = None, **common):
        for value, what in ((minimum, "minimum"), (maximum, "maximum"), (step, "step")):
            if value is not None:
                _finite(value, f"{name}: {what}")
        if precision is not None and (not isinstance(precision, int) or precision < 0):
            raise SchemaError(f"{name}: precision must be a non-negative int")
        self.minimum, self.maximum, self.step, self.precision = minimum, maximum, step, precision
        super().__init__(name, default, **common)

    @property
    def integer(self) -> bool:
        return self.precision == 0

    def coerce(self, value: Any, ctx: Context) -> int | float:
        number = to_number(value)
        if self.minimum is not None and number < self.minimum:
            raise ParamError(f"must be at least {self.minimum}, got {number}")
        if self.maximum is not None and number > self.maximum:
            raise ParamError(f"must be at most {self.maximum}, got {number}")
        if self.precision == 0:
            return _as_int(number)
        if self.precision is not None:
            return round(float(number), self.precision)
        return number

    def _input_schema(self, ctx: Context) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key in ("minimum", "maximum", "step", "precision"):
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        data["integer"] = self.integer
        return data


class Text(Input):
    """A string. ``prompt=True`` renders the prompt editor."""

    type = "text"

    def __init__(self, name: str, default: str = "", *, lines: int = 1, max_lines: int | None = None, placeholder: str | None = None, prompt: bool = False, **common):
        if not isinstance(lines, int) or lines < 1:
            raise SchemaError(f"{name}: lines must be >= 1")
        self.lines, self.max_lines, self.placeholder, self.prompt = lines, max_lines, placeholder, bool(prompt)
        super().__init__(name, default, **common)

    def coerce(self, value: Any, ctx: Context) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if _is_number(value):
            return str(value)
        raise ParamError(f"expected text, got {type(value).__name__}")

    def _input_schema(self, ctx: Context) -> dict[str, Any]:
        data: dict[str, Any] = {"lines": self.lines}
        if self.max_lines is not None:
            data["max_lines"] = self.max_lines
        if self.placeholder is not None:
            data["placeholder"] = self.placeholder
        data["prompt"] = self.prompt
        return data


ChoiceItem: TypeAlias = ChoiceValue | tuple[str, ChoiceValue]
"""A choice value, or a ``(label, value)`` pair."""
Choices: TypeAlias = Iterable[ChoiceItem] | Callable[[], Iterable[ChoiceItem]]


def _choice_pair(item: ChoiceItem) -> tuple[str, Any]:
    if isinstance(item, (tuple, list)) and len(item) == 2:
        label, value = item
        return str(label), value
    return str(item), item


def _same_value(choice: Any, value: Any) -> bool:
    return choice == value and isinstance(choice, bool) == isinstance(value, bool)


def match_choice(pairs: Sequence[tuple[str, Any]], value: Any, allow_custom: bool) -> Any:
    """The choice value equal to ``value`` (numeric strings match numeric
    choices), any string when ``allow_custom``, else :class:`ParamError`."""

    for _label, choice in pairs:
        if _same_value(choice, value):
            return choice
    if isinstance(value, str):
        for _label, choice in pairs:
            if _is_number(choice):
                try:
                    if to_number(value) == choice:
                        return choice
                except ParamError:
                    break
    if allow_custom and isinstance(value, str):
        return value
    shown = ", ".join(repr(choice) for _label, choice in pairs[:8]) + (", ..." if len(pairs) > 8 else "")
    raise ParamError(f"{value!r} is not one of the choices ({shown})" if pairs else f"{value!r}: there are no choices")


class _ChoiceBase(Input):
    """Shared by :class:`Choice` and :class:`MultiChoice`. ``type="index"``
    hands the script choice indices (Gradio's ``type="index"``); the wire
    value stays the choice value."""

    styles: ClassVar[tuple[str, ...]] = ()

    def __init__(self, name: str, choices: Choices, default: Any, *, style: str, allow_custom: bool, type: str, **common):
        if not callable(choices) and (isinstance(choices, (str, bytes)) or not isinstance(choices, Iterable)):
            raise SchemaError(f"{name}: choices is a list or a callable returning one")
        if style not in self.styles:
            raise SchemaError(f"{name}: style is one of {self.styles}")
        if type not in ("value", "index"):
            raise SchemaError(f"{name}: type is 'value' or 'index'")
        if type == "index" and allow_custom:
            raise SchemaError(f"{name}: an index choice cannot allow custom values")
        self.choices = choices if callable(choices) else list(choices)
        self.style = style
        self.allow_custom = bool(allow_custom)
        self.returns = type
        super().__init__(name, default, **common)

    def resolve_choices(self, ctx: Context) -> list[tuple[str, Any]]:
        """``(label, value)`` pairs, resolved once per context."""

        return ctx.memo(self, "choice_pairs", lambda: [_choice_pair(item) for item in ctx.resolve(self, "choices", self.choices) or ()])

    def values(self, ctx: Context) -> list[Any]:
        return [value for _label, value in self.resolve_choices(ctx)]

    def _index(self, value: Any, ctx: Context) -> int:
        return self.values(ctx).index(value)

    def _value_at(self, index: Any, ctx: Context) -> Any:
        values = self.values(ctx)
        if _is_int(index) and 0 <= index < len(values):
            return values[index]
        raise ParamError(f"choice index {index!r} is out of range")

    def _input_schema(self, ctx: Context) -> dict[str, Any]:
        choices = [{"value": value, "label": label} for label, value in self.resolve_choices(ctx)]
        return {"choices": choices, "style": self.style, "allow_custom": self.allow_custom}


class Choice(_ChoiceBase):
    """One of ``choices``. The wire value is the choice value; with
    ``type="index"`` the processing argument is its index. A ``None``
    default means the first choice."""

    type = "choice"
    styles = ("dropdown", "radio")

    def __init__(self, name: str, choices: Choices, default: Any = None, *, style: str = "dropdown", allow_custom: bool = False, type: str = "value", **common):
        super().__init__(name, choices, default, style=style, allow_custom=allow_custom, type=type, **common)

    def coerce(self, value: Any, ctx: Context) -> Any:
        pairs = self.resolve_choices(ctx)
        if value is None:
            if not pairs:
                return None
            raise ParamError("a choice is required")
        return match_choice(pairs, value, self.allow_custom)

    def fallback_default(self, value: Any, ctx: Context) -> Any:
        values = self.values(ctx)
        return values[0] if values else None

    def to_arg(self, value: Any, ctx: Context) -> Any:
        if self.returns != "index" or value is None:
            return value
        return self._index(value, ctx)

    def from_arg(self, arg: Any, ctx: Context) -> Any:
        if self.returns != "index" or arg is None:
            return arg
        return self._value_at(arg, ctx)


class MultiChoice(_ChoiceBase):
    """Any number of ``choices`` (a list value)."""

    type = "multichoice"
    styles = ("dropdown", "checkboxes")

    def __init__(self, name: str, choices: Choices, default: Iterable[Any] = (), *, style: str = "dropdown", allow_custom: bool = False, type: str = "value", **common):
        super().__init__(name, choices, default, style=style, allow_custom=allow_custom, type=type, **common)

    def coerce(self, value: Any, ctx: Context) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
            raise ParamError(f"expected a list, got {type(value).__name__}")
        pairs = self.resolve_choices(ctx)
        return [match_choice(pairs, item, self.allow_custom) for item in value]

    def fallback_default(self, value: Any, ctx: Context) -> list[Any]:
        # Drop the stale entries of a declared default
        pairs = self.resolve_choices(ctx)
        kept = []
        for item in value or ():
            try:
                kept.append(match_choice(pairs, item, self.allow_custom))
            except ParamError:
                pass
        return kept

    def to_arg(self, value: Any, ctx: Context) -> list[Any]:
        if self.returns != "index":
            return list(value)
        return [self._index(item, ctx) for item in value]

    def from_arg(self, arg: Any, ctx: Context) -> list[Any]:
        if self.returns != "index":
            return list(arg or ())
        return [self._value_at(index, ctx) for index in arg or ()]

    def from_paste(self, raw: Any, ctx: Context) -> list[Any]:
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    raw = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    raise ParamError(f"cannot read a list from {raw!r}") from None
            else:
                raw = [part.strip() for part in text.split(",") if part.strip()]
        return self.coerce(raw, ctx)


class Seed(Input):
    """An integer seed in ``[-1, SEED_MAX]``; ``-1`` means random."""

    type = "seed"

    def __init__(self, name: str, default: int = -1, **common):
        super().__init__(name, default, **common)

    def coerce(self, value: Any, ctx: Context) -> int:
        if value is None:
            raise ParamError("a seed is required")
        number = to_number(value)
        if isinstance(number, float):
            if not number.is_integer():
                raise ParamError(f"a seed is a whole number, got {value!r}")
            number = int(number)
        if not -1 <= number <= SEED_MAX:
            raise ParamError(f"a seed is -1 (random) or between 0 and {SEED_MAX}, got {number}")
        return number


class Color(Input):
    type = "color"

    def __init__(self, name: str, default: str = "#000000", **common):
        super().__init__(name, default, **common)

    def coerce(self, value: Any, ctx: Context) -> str:
        if isinstance(value, str) and COLOR_RE.fullmatch(value.strip()):
            return value.strip()
        raise ParamError(f"expected a color like #rrggbb, got {value!r}")


class _UploadInput(Input):
    nullable = True

    def _check_item(self, value: Any) -> Any:
        if value is None or _is_image(value):
            return value
        if isinstance(value, str) and value:
            return value
        raise ParamError(f"expected an upload id, got {type(value).__name__}")

    def _resolve(self, value: Any, ctx: Context) -> Any:
        if isinstance(value, str) and ctx.resolve_upload is not None:
            return ctx.resolve_upload(self, value)
        return value

    def from_paste(self, raw: Any, ctx: Context) -> Any:
        raise ParamError("uploads are not pasted from infotext")


class Image(_UploadInput):
    """An uploaded image. ``kind="mask"``/``"sketch"`` layers name the image
    they are painted over as ``source`` (a ``mask`` without ``source`` is an
    uploaded mask image); ``mode`` converts the PIL image the script receives
    (e.g. ``"RGBA"``). Masks are alpha-coded, see :func:`mask_channel`."""

    type = "image"
    kinds = ("image", "mask", "sketch")

    def __init__(self, name: str, kind: str = "image", source: str | None = None, *, mode: str | None = None, **common):
        if kind not in self.kinds:
            raise SchemaError(f"{name}: kind is one of {self.kinds}")
        self.kind, self.source, self.mode = kind, source, mode
        super().__init__(name, None, **common)

    def coerce(self, value: Any, ctx: Context) -> Any:
        return self._check_item(value)

    def to_arg(self, value: Any, ctx: Context) -> Any:
        image = self._resolve(value, ctx)
        if self.mode and _is_image(image) and image.mode != self.mode:
            image = image.convert(self.mode)
        return image

    def from_arg(self, arg: Any, ctx: Context) -> str | None:
        return arg if isinstance(arg, str) else None

    def _input_schema(self, ctx: Context) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind}
        if self.source:
            data["source"] = self.source
        if self.mode:
            data["mode"] = self.mode
        return data


def mask_channel(mask: Any) -> Any:
    """The ``L`` mask of an uploaded mask image (schema-format.md §1.3): its
    alpha channel (opaque = masked), or its luminance (white = masked) when
    it has no alpha or the alpha is fully opaque (a plain black/white mask)."""

    if mask is None:
        return None
    if "A" in mask.getbands():
        alpha = mask.getchannel("A")
        if alpha.getextrema() != (255, 255):
            return alpha
        mask = mask.convert("RGB")
    return mask.convert("L")


class ImageList(_UploadInput):
    """Several uploaded images (batch inputs)."""

    type = "image_list"
    nullable = False

    def __init__(self, name: str, **common):
        super().__init__(name, [], **common)

    def coerce(self, value: Any, ctx: Context) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
            raise ParamError(f"expected a list of upload ids, got {type(value).__name__}")
        items = [self._check_item(item) for item in value]
        if any(item is None for item in items):
            raise ParamError("an image list cannot contain null")
        return items

    def to_arg(self, value: Any, ctx: Context) -> list[Any]:
        return [self._resolve(item, ctx) for item in value]

    def from_arg(self, arg: Any, ctx: Context) -> list[str]:
        return [item for item in arg or () if isinstance(item, str)]


class File(_UploadInput):
    """An uploaded file; the script receives its path."""

    type = "file"

    def __init__(self, name: str, *, accept: str | None = None, **common):
        self.accept = accept
        super().__init__(name, None, **common)

    def coerce(self, value: Any, ctx: Context) -> str | None:
        if value is not None and not (isinstance(value, str) and value):
            raise ParamError(f"expected an upload id, got {type(value).__name__}")
        return value

    def to_arg(self, value: Any, ctx: Context) -> Any:
        return self._resolve(value, ctx)

    def _input_schema(self, ctx: Context) -> dict[str, Any]:
        return {"accept": self.accept} if self.accept else {}


class Json(Input):
    """Any JSON value (NaN/Infinity are rejected); ``validate(value) ->
    value`` may normalize or raise ``ParamError``/``ValueError``."""

    type = "json"
    nullable = True

    def __init__(self, name: str, default: Any = None, *, validate: Callable[[Any], Any] | None = None, **common):
        self.validate = validate
        super().__init__(name, default, **common)

    def coerce(self, value: Any, ctx: Context) -> Any:
        check_finite_json(value)
        if self.validate is None:
            return value
        try:
            return self.validate(value)
        except ParamError:
            raise
        except (TypeError, ValueError) as error:
            raise ParamError(str(error)) from None


class Custom(Json):
    """A JSON value edited by the frontend widget registered as ``widget``."""

    type = "custom"

    def __init__(self, name: str, widget: str, default: Any = None, props: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None, **common):
        if not isinstance(widget, str) or not widget:
            raise SchemaError(f"{name}: widget is the registered widget name")
        self.widget = widget
        self.props = props if props is not None else {}
        super().__init__(name, default, **common)

    def _input_schema(self, ctx: Context) -> dict[str, Any]:
        return {"widget": self.widget, "props": dict(ctx.resolve(self, "props", self.props) or {})}


class State(Json):
    """A hidden JSON value: never rendered, always submitted."""

    type = "state"


# endregion

# region Layout


class Container(Node):
    """A node with children, given positionally and/or as ``children=[...]``."""

    def __init__(self, *items, children: Iterable[Node] | None = None, visible_if=None, enabled_if=None):
        super().__init__(visible_if=visible_if, enabled_if=enabled_if)
        self.items = _flatten((*items, children))

    @property
    def children(self) -> list[Node]:
        return self.items

    def _schema(self, ctx: Context) -> dict[str, Any]:
        return {"children": [child.schema(ctx) for child in self.items]}


class Row(Container):
    type = "row"


class Column(Container):
    type = "column"

    def __init__(self, *items, width: str | int | None = None, children: Iterable[Node] | None = None, visible_if=None, enabled_if=None):
        super().__init__(*items, children=children, visible_if=visible_if, enabled_if=enabled_if)
        self.width = width

    def _schema(self, ctx: Context) -> dict[str, Any]:
        data = {"width": self.width} if self.width is not None else {}
        return {**data, **super()._schema(ctx)}


class Group(Container):
    """A visual group; collapsible when it has a ``title``."""

    type = "group"

    def __init__(self, title: str | Node | Iterable[Node] | None = None, *items, open: bool = True, children: Iterable[Node] | None = None, visible_if=None, enabled_if=None):
        if title is not None and not isinstance(title, str):
            items, title = (title, *items), None
        super().__init__(*items, children=children, visible_if=visible_if, enabled_if=enabled_if)
        self.title = title
        self.open = bool(open)

    def _schema(self, ctx: Context) -> dict[str, Any]:
        data = {"title": self.title} if self.title else {}
        return {**data, "open": self.open, **super()._schema(ctx)}


class Section(Container):
    """A collapsible panel. With ``enable`` (a :class:`Toggle`) it is the
    former InputAccordion: the toggle is the first param of the subtree.
    ``exclusive=(group, member)`` is applied to the ``enable`` toggle."""

    type = "section"

    def __init__(self, title: str, *items, enable: Toggle | None = None, open: bool = False, exclusive: tuple[str, str] | None = None, children: Iterable[Node] | None = None, visible_if=None, enabled_if=None):
        super().__init__(*items, children=children, visible_if=visible_if, enabled_if=enabled_if)
        if enable is not None and not isinstance(enable, Toggle):
            raise SchemaError(f"section {title!r}: enable must be a Toggle")
        if exclusive is not None:
            if enable is None:
                raise SchemaError(f"section {title!r}: exclusive needs an enable toggle")
            enable.set_exclusive(exclusive)
        self.title = str(title)
        self.enable = enable
        self.open = bool(open)

    @property
    def children(self) -> list[Node]:
        return [self.enable, *self.items] if self.enable is not None else self.items

    def _schema(self, ctx: Context) -> dict[str, Any]:
        data: dict[str, Any] = {"title": self.title, "open": self.open}
        if self.enable is not None:
            data["enable"] = self.enable.schema(ctx)
        return {**data, **super()._schema(ctx)}


class Tab(Container):
    type = "tab"

    def __init__(self, title: str, *items, value: Any = None, children: Iterable[Node] | None = None, visible_if=None, enabled_if=None):
        super().__init__(*items, children=children, visible_if=visible_if, enabled_if=enabled_if)
        self.title = str(title)
        self.value = value

    def _schema(self, ctx: Context) -> dict[str, Any]:
        data: dict[str, Any] = {"title": self.title}
        if self.value is not None:
            data["value"] = self.value
        return {**data, **super()._schema(ctx)}


class Tabs(Container):
    """A tab strip. ``bind`` names a :class:`Choice` of the same owner whose
    value is the selected tab's ``value``."""

    type = "tabs"

    def __init__(self, *tabs, bind: str | None = None, visible_if=None, enabled_if=None):
        super().__init__(*tabs, visible_if=visible_if, enabled_if=enabled_if)
        if not all(isinstance(tab, Tab) for tab in self.items):
            raise SchemaError("Tabs only contains Tab nodes")
        if bind is not None and any(tab.value is None for tab in self.items):
            raise SchemaError(f"Tabs bound to {bind!r} need a value on every Tab")
        self.bind = bind

    def _schema(self, ctx: Context) -> dict[str, Any]:
        data = {"bind": self.bind} if self.bind else {}
        return {**data, **super()._schema(ctx)}


class Markdown(Node):
    type = "markdown"

    def __init__(self, text: str, *, visible_if=None):
        super().__init__(visible_if=visible_if)
        self.text = str(text)

    def _schema(self, ctx: Context) -> dict[str, Any]:
        return {"text": self.text}


class Html(Node):
    type = "html"

    def __init__(self, text: str, *, visible_if=None):
        super().__init__(visible_if=visible_if)
        self.text = str(text)

    def _schema(self, ctx: Context) -> dict[str, Any]:
        return {"text": self.text}


class Separator(Node):
    type = "separator"


class Button(Node):
    """Calls the owner's action ``action`` with the owner's current values."""

    type = "button"
    variants = ("primary", "secondary", "ghost", "tool")

    def __init__(self, label: str, action: str, variant: str = "secondary", *, visible_if=None, enabled_if=None):
        super().__init__(visible_if=visible_if, enabled_if=enabled_if)
        if variant not in self.variants:
            raise SchemaError(f"button {label!r}: variant is one of {self.variants}")
        self.label, self.action, self.variant = str(label), action, variant

    def _schema(self, ctx: Context) -> dict[str, Any]:
        return {"label": self.label, "action": self.action, "variant": self.variant}


class Gallery(Node):
    """Output-only image list, filled by an action's ``__outputs__``."""

    type = "gallery"

    def __init__(self, name: str, label: str | None = None, *, visible_if=None):
        super().__init__(visible_if=visible_if)
        self.name, self.label = name, label

    def _schema(self, ctx: Context) -> dict[str, Any]:
        return {"name": self.name, **({"label": self.label} if self.label else {})}


class Output(Node):
    """Output-only value, filled by an action's ``__outputs__``."""

    type = "output"
    kinds = ("text", "html", "image", "json")

    def __init__(self, name: str, kind: str = "text", label: str | None = None, *, visible_if=None):
        super().__init__(visible_if=visible_if)
        if kind not in self.kinds:
            raise SchemaError(f"output {name!r}: kind is one of {self.kinds}")
        self.name, self.kind, self.label = name, kind, label

    def _schema(self, ctx: Context) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, **({"label": self.label} if self.label else {})}


class Slot(Node):
    """Core forms only: where the scripts of ``section`` are rendered."""

    type = "slot"

    def __init__(self, section: str):
        super().__init__()
        self.section = section

    def _schema(self, ctx: Context) -> dict[str, Any]:
        return {"section": self.section}


# endregion
