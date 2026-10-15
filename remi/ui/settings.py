"""Settings: widget kinds for ``OptionInfo(component=...)`` and the settings
schema exported from ``shared.opts.data_labels`` (schema-format.md §6).

``OptionInfo.component`` takes a ``remi.ui`` class (``ui.Slider``,
``ui.Choice``, ``ui.MultiChoice``, ``ui.Color``, ``ui.Text``, ...) with
A1111-style ``component_args`` (``minimum``/``maximum``/``step``,
``choices``, ``precision``, ``lines``, ``placeholder``, ``visible``,
``interactive``) plus ``style`` (``"radio"``, ``"checkboxes"``) and
``allow_custom``; any other component (or none) is picked from the default's
type.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from remi.ui import nodes
from remi.ui.context import Context, ensure_context
from remi.ui.schema import SCHEMA_VERSION, jsonable

REMI_WIDGETS = (nodes.Toggle, nodes.Slider, nodes.Number, nodes.Text, nodes.Choice, nodes.MultiChoice, nodes.Color, nodes.State, nodes.Json, nodes.Html, nodes.Separator, nodes.Row)


def is_remi_widget(component: Any) -> bool:
    return isinstance(component, type) and issubclass(component, REMI_WIDGETS)


def widget_kind(component: Any, args: Mapping[str, Any], default: Any) -> tuple[type, dict]:
    """``(remi.ui class, extra arguments)`` for an option's component."""

    if is_remi_widget(component):
        return component, {}
    if isinstance(default, bool):
        return nodes.Toggle, {}
    if isinstance(default, str):
        return nodes.Text, {}
    if isinstance(default, (int, float)):
        return nodes.Number, {}
    return nodes.Json, {}


def component_args(info) -> dict:
    args = info.component_args() if callable(info.component_args) else info.component_args
    return dict(args or {})


def setting_node(key: str, info, value: Any) -> nodes.Node:
    """The ``remi.ui`` node of one option, its default being ``value``."""

    args = component_args(info)
    kind, extra = widget_kind(info.component, args, info.default)
    extra.update({name: args[name] for name in ("style", "allow_custom") if name in args})
    if args.get("allow_custom_value"):
        extra["allow_custom"] = True

    if kind is nodes.Html:
        return nodes.Html(str(info.default))
    if kind in (nodes.Separator, nodes.Row):
        return kind()

    common: dict[str, Any] = {"label": info.label or ""}
    if args.get("visible") is False:
        common["hidden"] = True
    if args.get("interactive") is False:
        common["enabled_if"] = False
    if getattr(info, "infotext", None):
        common["infotext"] = info.infotext

    if kind is nodes.Slider:
        minimum, maximum, step = args.get("minimum", 0), args.get("maximum", 100), args.get("step", 1)
        # By the declared default: a stored value may have lost its int type
        integer = all(nodes._is_int(item) for item in (minimum, maximum, step, info.default))
        return nodes.Slider(key, minimum, maximum, value, step, integer=integer, **common)
    if kind is nodes.Number:
        precision = args.get("precision")
        if precision is None and isinstance(info.default, int) and not isinstance(info.default, bool):
            precision = 0
        return nodes.Number(key, value, args.get("minimum"), args.get("maximum"), args.get("step"), precision, **common)
    if kind is nodes.Text:
        return nodes.Text(key, value, lines=max(int(args.get("lines", 1) or 1), 1), max_lines=args.get("max_lines"), placeholder=args.get("placeholder"), **common)
    if kind in (nodes.Choice, nodes.MultiChoice):
        style = extra.get("style", "dropdown")
        return kind(key, args.get("choices") or [], value, style=style, allow_custom=bool(extra.get("allow_custom")), **common)
    if kind is nodes.Toggle:
        return nodes.Toggle(key, value, **common)
    if kind is nodes.Color:
        return nodes.Color(key, value, **common)
    if kind is nodes.State:
        return nodes.State(key, value, **common)
    return nodes.Json(key, value, **common)


_MARKUP_RE = re.compile(r"<[A-Za-z/!][^>]*>|&[A-Za-z#0-9]+;")


def _setting_extras(info) -> dict:
    data: dict[str, Any] = {}
    # ``notes`` are plain text (translated); a note with markup joins ``html``
    notes, html_blocks = [], list(getattr(info, "html_blocks", ()))
    for note in getattr(info, "notes", ()):
        if _MARKUP_RE.search(str(note)):
            html_blocks.append(f"<span class='info'>({note})</span>")
        else:
            notes.append(str(note))
    if notes:
        data["notes"] = notes
    links = list(getattr(info, "links", ()))
    if links:
        data["links"] = [{"label": label, "url": url} for label, url in links]
    html = "".join(html_blocks)
    if html.strip():
        data["html"] = html.strip()
    if getattr(info, "requires_restart", False):
        data["needs_restart"] = True
    if getattr(info, "requires_reload_ui", False):
        data["needs_reload_ui"] = True
    if info.refresh is not None:
        data["refresh"] = True
    if info.onchange is not None:
        data["onchange"] = True
    if info.restrict_api:
        data["restricted"] = True
    return data


def settings_schema(opts, ctx: Context | None = None, *, categories=None, quicksettings: list[str] | None = None, locked: Callable[[str], str | None] | None = None) -> dict:
    """The settings page: categories -> sections -> items, plus values.

    ``locked(key)``: why the server refuses changes to ``key`` (None: it
    accepts them); a locked setting is ``enabled_if: false`` with ``locked``
    set to that reason."""

    from modules import options as options_module

    ctx = ensure_context(ctx)
    categories = categories if categories is not None else options_module.categories
    result_categories: list[dict] = []
    by_category: dict[str, dict] = {}
    by_section: dict[tuple, dict] = {}
    open_rows: dict[tuple, dict | None] = {}

    for key, info in opts.data_labels.items():
        section = info.section
        if not section or section[0] is None:
            continue
        category = categories.mapping.get(info.category_id)
        category_id, category_label = (category.id, category.label) if category is not None else ("extensions", "Extensions")
        if category_id not in by_category:
            by_category[category_id] = {"id": category_id, "label": category_label, "sections": []}
            result_categories.append(by_category[category_id])
        if section not in by_section:
            by_section[section] = {"id": section[0], "label": section[1], "items": []}
            by_category[category_id]["sections"].append(by_section[section])
            open_rows[section] = None
        items = by_section[section]["items"]

        value = getattr(opts, key)  # the stored value, else the default
        try:
            node = setting_node(key, info, value)
        except Exception as error:  # an extension option this exporter cannot read
            node = nodes.Html(f"<i>{key}: {error}</i>")

        if isinstance(node, nodes.Row):  # OptionRow markers open and close a row
            if open_rows[section] is None:
                open_rows[section] = {"type": "row", "children": []}
                items.append(open_rows[section])
            else:
                open_rows[section] = None
            continue

        data = node.schema(ctx)
        if isinstance(node, nodes.Input):
            data["default"] = jsonable(value)
            data.update(_setting_extras(info))
            reason = locked(key) if locked is not None else None
            if reason:
                data["enabled_if"] = False
                data["locked"] = reason
        target = open_rows[section]["children"] if open_rows[section] is not None else items
        target.append(data)

    stored = {key: info for key, info in opts.data_labels.items() if not getattr(info, "do_not_save", False)}
    return {
        "version": SCHEMA_VERSION,
        "categories": result_categories,
        "quicksettings": list(quicksettings if quicksettings is not None else getattr(opts, "quicksettings_list", []) or []),
        "values": {key: jsonable(opts.data.get(key, info.default)) for key, info in stored.items()},
        "defaults": {key: jsonable(info.default) for key, info in stored.items()},
    }


def refresh_setting(opts, key: str) -> dict:
    """Run a setting's ``refresh`` and return its patch (new choices)."""

    info = opts.data_labels[key]
    if info.refresh is not None:
        info.refresh()
    args = component_args(info)
    patch: dict[str, Any] = {}
    if "choices" in args:
        from remi.ui.actions import normalize_choices

        patch["choices"] = normalize_choices(args["choices"])
    for name in ("minimum", "maximum", "step"):
        if name in args:
            patch[name] = args[name]
    return {key: patch}
