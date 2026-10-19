"""Server actions: ``@ui.action`` callables returning patches.

An action receives ``(self, ctx, values)``: the owner's current named values
(plus ``ctx.core``, the generation form) and returns a patch
(schema-format.md §5)::

    {"param": {"value": 3, "enabled": False}, "core.width": {"value": 1024},
     "__toast__": {"level": "info", "message": "..."}, "__outputs__": {...},
     "__settings__": {setting: value}}

``__settings__`` reports settings the action has already changed on the
server (the client refreshes its copy; it must not write them back).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from remi.ui.errors import ParamError, SchemaError, ValidationError
from remi.ui.types import Patch, PatchEntry

PATCH_KEYS = frozenset({"value", "choices", "visible", "enabled", "label", "info", "minimum", "maximum", "step"})
SPECIAL_KEYS = frozenset({"__toast__", "__outputs__", "__settings__"})
TOAST_LEVELS = ("info", "success", "warning", "error")


@dataclass(frozen=True)
class ActionSpec:
    name: str
    func: Callable[..., Any]
    gpu: bool = False
    """Queue behind generation (runs under the generation lock)."""


@dataclass
class ActionContext:
    """What an action gets besides its owner's values."""

    tab: str | None = None
    owner: str = ""
    core: dict[str, Any] = field(default_factory=dict)
    scripts: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Other owners' values, by owner id, when the caller sent them."""
    user: str | None = None
    request: Any = None
    """The HTTP request (``fastapi.Request``) when the server runs the action."""
    extra: dict[str, Any] = field(default_factory=dict)
    files: Any = None
    """Uploads and outputs by id (``remi.ui.files.ActionFiles``: ``get``,
    ``image``, ``image_size``, ``path``, ``store_image``, ``store_bytes``);
    filled by the server, ``None`` elsewhere."""
    generation_lock: Any = None
    """The lock generation runs under (filled by the server; ``gpu=True``
    page tasks hold it). ``None``: the WebUI's ``call_queue.queue_lock``."""


def action(name: str | None = None, *, gpu: bool = False):
    """Register a method (or function) as an action named ``name``
    (default: the function name)."""

    def decorate(func):
        func.__remi_action__ = ActionSpec(name or func.__name__, func, gpu)
        return func

    if callable(name):  # used bare: @ui.action
        func, name = name, None
        return decorate(func)
    return decorate


def actions_of(owner: Any) -> dict[str, ActionSpec]:
    """The actions declared on ``owner``'s class (or module/namespace)."""

    found: dict[str, ActionSpec] = {}
    source = owner if inspect.ismodule(owner) or isinstance(owner, type) else type(owner)
    for attribute in dir(source):
        try:
            member = inspect.getattr_static(source, attribute)
        except AttributeError:
            continue
        member = getattr(member, "__func__", member)
        spec = getattr(member, "__remi_action__", None)
        if isinstance(spec, ActionSpec):
            if spec.name in found:
                raise SchemaError(f"action {spec.name!r} is declared twice on {source!r}")
            found[spec.name] = spec
    return found


def call_action(owner: Any, name: str, ctx: ActionContext, values: Mapping[str, Any] | None) -> Patch:
    """Run ``owner``'s action ``name``; returns the validated patch."""

    spec = actions_of(owner).get(name)
    if spec is None:
        raise KeyError(f"no action {name!r}")
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValidationError([ParamError(f"expected an object of named values, got {type(values).__name__}", owner=ctx.owner or None)])
    bound = spec.func if inspect.ismodule(owner) else spec.func.__get__(owner, type(owner))
    return check_patch(bound(ctx, dict(values)) or {})


def update(**props: Any) -> PatchEntry:
    """One param's patch: ``update(value=1, enabled=False)``."""

    unknown = set(props) - PATCH_KEYS
    if unknown:
        raise SchemaError(f"unknown patch keys {sorted(unknown)}")
    if "choices" in props:
        props["choices"] = normalize_choices(props["choices"])
    return props


def message(text: str, **params: Any) -> dict:
    """A translatable text for an ``output`` node (schema-format.md §5):
    ``text`` is the English catalog key with ``{{name}}`` placeholders the
    client fills from ``params`` after translating it."""

    if not isinstance(text, str):
        raise SchemaError("message text is a string")
    return {"text": text, "params": params} if params else {"text": text}


def toast(message: str | Mapping[str, Any], level: str = "info") -> dict:
    """A toast; the client translates ``message`` when a catalog has it.
    ``message`` may be a :func:`message` (a catalog key with ``{{name}}``
    placeholders and their ``params``) for texts with variable parts."""

    if level not in TOAST_LEVELS:
        raise SchemaError(f"toast level is one of {TOAST_LEVELS}")
    if isinstance(message, Mapping):
        return {"__toast__": {"level": level, "message": message_of(message)}}
    return {"__toast__": {"level": level, "message": str(message)}}


def message_of(value: Mapping[str, Any]) -> dict:
    """A validated :func:`message` dict."""

    params = value.get("params")
    if not isinstance(value.get("text"), str) or not (params is None or isinstance(params, Mapping)):
        raise SchemaError("a message is {'text': str, 'params'?: dict}")
    return message(value["text"], **dict(params or {}))


def message_english(value: str | Mapping[str, Any]) -> str:
    """The English text of a string or :func:`message` (placeholders filled)."""

    if not isinstance(value, Mapping):
        return str(value)
    text = str(value.get("text", ""))
    for name, param in (value.get("params") or {}).items():
        text = text.replace("{{" + str(name) + "}}", str(param))
    return text


def normalize_choices(choices: Any) -> list[dict]:
    """``[{"value", "label"}]`` from values, ``(label, value)`` pairs or
    already normalized entries."""

    from remi.ui.nodes import _choice_pair

    normalized = []
    for item in choices:
        if isinstance(item, Mapping) and set(item) == {"value", "label"}:
            normalized.append(dict(item))
        else:
            label, value = _choice_pair(item)
            normalized.append({"value": value, "label": label})
    return normalized


def check_patch(patch: Mapping[str, Any]) -> Patch:
    """Validate a patch's shape; choices are normalized to ``{value, label}``."""

    if not isinstance(patch, Mapping):
        raise SchemaError(f"an action returns a patch dict, got {type(patch).__name__}")
    result: Patch = {}
    for key, entry in patch.items():
        if key in SPECIAL_KEYS:
            if not isinstance(entry, Mapping):
                raise SchemaError(f"{key} takes an object")
            if key == "__toast__" and entry.get("level", "info") not in TOAST_LEVELS:
                raise SchemaError(f"toast level is one of {TOAST_LEVELS}")
            result[key] = dict(entry)
            continue
        if not isinstance(key, str) or key.startswith("__"):
            raise SchemaError(f"unknown patch entry {key!r}")
        if not isinstance(entry, Mapping):
            raise SchemaError(f"patch for {key!r} must be an object")
        result[key] = update(**entry)
    return result


def merge_patches(*patches: Mapping[str, Any]) -> Patch:
    """Later patches win per param property; toasts concatenate."""

    merged: dict[str, Any] = {}
    for patch in patches:
        for key, entry in patch.items():
            if key == "__toast__" and key in merged:
                previous = merged[key]
                merged[key] = {"level": entry.get("level", previous.get("level", "info")), "message": f"{message_english(previous['message'])}; {message_english(entry['message'])}"}
            else:
                merged[key] = {**merged.get(key, {}), **entry}
    return merged
