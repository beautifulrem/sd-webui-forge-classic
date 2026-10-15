"""Template editing behind the panel's actions: what selecting a template
does to a panel's timing, and saving, combining, renaming and deleting
templates. Pure functions over the template store; no UI toolkit."""

from modules.anima_user_data import UnreadableDataFile
from modules.anima_support import clamp as _clamp, to_float as _to_float
from lib_stage_scheduler.options import (
    BUILTIN_TEMPLATES,
    CURVE_CHOICES,
    CURVE_SMOOTH,
    PART_ALLOW,
    STAGE_CHARACTER,
    STAGE_CHOICES,
    STAGE_DEFAULT_CURVES,
    TEMPLATE_LABELS_EN,
    _msg,
    _t,
    _timing_value,
)
from lib_stage_scheduler.templates import (
    _TEMPLATES,
    _auto_stage_values,
    _part_preset_display_name,
    _parts_from_presets,
    _primary_template_part,
    _template_config,
    _template_names,
    _template_parts,
    _template_stage_from_name,
)


class TemplateEdit:
    """The outcome of a template edit: the ``status`` message, whether it
    changed the store (``ok``), and a ``(old, new)`` rename."""

    def __init__(self, status: str | dict, ok: bool = False, renamed: tuple[str, str] | None = None):
        self.status, self.ok, self.renamed = status, ok, renamed


def _edit(change, renamed=None) -> TemplateEdit:
    """Apply ``change(data)``, which returns a status (or raises
    ``_Refused(status)``), to the custom templates. An unreadable file is
    never overwritten with just the one change."""

    def apply(data):
        return None, change(data)

    try:
        _value, status = _TEMPLATES.edit(apply)
    except UnreadableDataFile as error:
        return TemplateEdit(_msg("templates_unreadable", error=error))
    except _Refused as refused:
        return TemplateEdit(str(refused))
    return TemplateEdit(status, True, renamed)


class _Refused(Exception):
    """A refused edit; the store is left as it was."""


def template_choices() -> list[tuple[str, str]]:
    """The template dropdown's ``(label, name)`` pairs: built-in templates
    by their English label, custom ones by name."""

    return [(TEMPLATE_LABELS_EN.get(name, name), name) for name in _template_names()]


def templates_payload() -> dict:
    """Every template as the stage-timeline widget draws it, by name."""

    payload = {}
    for name in _template_names():
        config = _template_config(name)
        payload[name] = {
            "label": TEMPLATE_LABELS_EN.get(name, name),
            "builtin": name in BUILTIN_TEMPLATES,
            "auto": bool(config.get("auto", True)),
            "stage": _template_stage_from_name(name),
            "curve": config.get("curve") if config.get("curve") in CURVE_CHOICES else None,
            # With auto timing these run instead of the panel's window (none: the window)
            "parts": _template_parts(config),
        }
    return payload


def apply_template(template_name, shift_value) -> dict:
    """The timing a panel takes when ``template_name`` is selected: the
    template's own timing, or with auto timing its stage window at the Shift."""

    config = _template_config(template_name)
    primary_part = _primary_template_part(config)
    auto = bool(config.get("auto", True))
    curve = config.get("curve") or primary_part.get("curve") or CURVE_SMOOTH
    if curve not in CURVE_CHOICES:
        curve = CURVE_SMOOTH

    if auto:
        start, end, peak, _ = _auto_stage_values(primary_part["stage"], shift_value)
    else:
        start = _clamp(_to_float(config.get("start"), 0.2), 0.0, 1.0)
        end = _clamp(_to_float(config.get("end"), 0.8), 0.0, 1.0)
        if end < start:
            start, end = end, start
        peak = _clamp(_to_float(config.get("peak"), (start + end) * 0.5), start, end)
    strength = _clamp(_to_float(config.get("strength"), 1.0), 0.0, 4.0)
    return {"auto": auto, "start": round(start, 4), "end": round(end, 4), "peak": round(peak, 4), "strength": round(strength, 4), "curve": curve}


def auto_timing(template_name, shift_value, timing) -> dict:
    """``timing`` with auto timing applied: the template's stage window at
    the Shift (the strength stays the user's multiplier)."""

    timing = _timing_value(timing)
    if not timing["auto"]:
        return timing
    start, end, peak, _ = _auto_stage_values(_primary_template_part(_template_config(template_name))["stage"], shift_value)
    return {**timing, "start": round(start, 4), "end": round(end, 4), "peak": round(peak, 4)}


def save_template(name, auto, template_name, timing) -> TemplateEdit:
    """Save a panel (its template and ``timing``) as the custom template
    ``name``; the source template's parts come along."""

    name = str(name or "").strip()
    if not name:
        return TemplateEdit(_t("empty_template_name"))
    if name in BUILTIN_TEMPLATES:
        return TemplateEdit(_t("builtin_template_kept"))

    timing = _timing_value(timing)
    template_name = str(template_name or "").strip()
    stage = _template_stage_from_name(template_name)
    source_parts = _template_parts(_template_config(template_name))
    record = {
        "auto": bool(auto),
        "stage": stage if stage in STAGE_CHOICES else STAGE_CHARACTER,
        "start": timing["start"],
        "end": timing["end"],
        "peak": timing["peak"],
        "strength": timing["strength"],
        "curve": timing["curve"] if timing["curve"] in CURVE_CHOICES else CURVE_SMOOTH,
    }
    if source_parts:
        record["parts"] = source_parts

    def save(data):
        data[name] = record
        return _msg("template_saved", name=name)

    return _edit(save)


def save_preset_combo_template(name, selected_presets) -> TemplateEdit:
    """Save the combination of base presets ``selected_presets`` (preset
    names or their builder labels) as the auto-timing template ``name``."""

    name = str(name or "").strip()
    if not name:
        return TemplateEdit(_t("empty_template_name"))
    if name in BUILTIN_TEMPLATES:
        return TemplateEdit(_t("builtin_preset_kept"))

    parts, error = _parts_from_presets(selected_presets)
    if error:
        return TemplateEdit(error)

    primary = next((part for part in parts if part["mode"] == PART_ALLOW), parts[0])
    presets_text = " + ".join(_part_preset_display_name(part) for part in parts)

    def save(data):
        data[name] = {
            "auto": True,
            "stage": primary["stage"],
            "parts": parts,
            # No start/end/peak: auto timing takes them from Shift when applied
            "strength": 1.0,
            "curve": primary.get("curve", STAGE_DEFAULT_CURVES.get(primary["stage"], CURVE_SMOOTH)),
        }
        return _msg("combo_saved", name=name, presets=presets_text)

    return _edit(save)


def rename_template(old_name, new_name) -> TemplateEdit:
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()
    if not old_name:
        return TemplateEdit(_t("no_template_selected"))
    if old_name in BUILTIN_TEMPLATES:
        return TemplateEdit(_t("builtin_not_renamed"))
    if not new_name:
        return TemplateEdit(_t("empty_new_name"))
    if new_name in BUILTIN_TEMPLATES:
        return TemplateEdit(_t("builtin_name_taken"))
    if new_name == old_name:
        return TemplateEdit(_t("same_name"))

    def rename(data):
        if old_name not in data:
            raise _Refused(_t("rename_missing"))
        if new_name in data:
            raise _Refused(_t("rename_taken"))
        data[new_name] = data.pop(old_name)
        return _msg("template_renamed", old=old_name, new=new_name)

    return _edit(rename, (old_name, new_name))


def delete_template(name) -> TemplateEdit:
    name = str(name or "").strip()
    if not name:
        return TemplateEdit(_t("no_template_selected"))
    if name in BUILTIN_TEMPLATES:
        return TemplateEdit(_t("builtin_not_deleted"))

    def delete(data):
        data.pop(name, None)
        return _msg("template_deleted", name=name)

    return _edit(delete)


def panel_template_after(current, choices, renamed=None):
    """A panel's template once the templates are ``choices``: it stays,
    follows a ``(old, new)`` rename, or falls back to the first template
    when its own is gone."""

    if current in choices:
        return current
    if renamed is not None and current == renamed[0] and renamed[1] in choices:
        return renamed[1]
    return choices[0] if choices else None
