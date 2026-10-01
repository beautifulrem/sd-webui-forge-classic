"""The template editor's Gradio handlers: panel template updates, saving,
renaming and deleting templates, and the help and builder dialogs."""


import gradio as gr

from modules.anima_user_data import UnreadableDataFile
from modules.anima_support import clamp as _clamp, to_float as _to_float
from lib_stage_scheduler.options import (
    BUILTIN_TEMPLATES,
    CURVE_CHOICES,
    CURVE_SMOOTH,
    PART_ALLOW,
    PART_BAN,
    STAGE_CHARACTER,
    STAGE_CHOICES,
    STAGE_DEFAULT_CURVES,
    _format_float,
    _t,
)


from lib_stage_scheduler.templates import (
    _TEMPLATES,
    _auto_stage_values,
    _builder_presets_from_template,
    _part_preset_display_name,
    _parts_from_presets,
    _primary_template_part,
    _template_config,
    _template_names,
    _template_parts,
    _template_stage_from_name,
)


def _panel_template_update(current, choices, renamed=None):
    """A panel's template dropdown showing ``choices``. Its selection stays
    (selecting a template re-applies its timing and strength to the panel),
    follows a ``(old, new)`` rename, or falls back to the first template when
    its own is gone."""
    if current in choices:
        return gr.update(choices=choices)
    if renamed is not None and current == renamed[0] and renamed[1] in choices:
        return gr.update(choices=choices, value=renamed[1])
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _template_updates(status, value=None, panels=(), renamed=None):
    """The template dropdowns: the base and hires panels' (``panels``: their
    selections; see ``_panel_template_update``, unknown ones stay), the
    manager's with ``value`` selected; and ``status``."""
    choices = _template_names()
    if value not in choices:
        value = choices[0] if choices else None
    if len(panels) == 2:
        panel_updates = [_panel_template_update(current, choices, renamed) for current in panels]
    else:
        panel_updates = [gr.update(choices=choices), gr.update(choices=choices)]
    return (*panel_updates, gr.update(choices=choices, value=value), status)


def _edit_templates(change, panels=(), renamed=None):
    """Apply ``change(data)``, which returns ``(template to select, status)``,
    to the custom templates; then ``_template_updates``. An unreadable file is
    never overwritten with just the one change."""
    try:
        value, status = _TEMPLATES.edit(change)
    except UnreadableDataFile as error:
        return _template_updates(_t("templates_unreadable").format(error=error), panels=panels)
    return _template_updates(status, value, panels, renamed)


def _apply_template_ui(template_name, shift_value):
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

    return (
        gr.update(value=auto),
        gr.update(value=round(start, 4)),
        gr.update(value=round(end, 4)),
        gr.update(value=round(peak, 4)),
        gr.update(value=round(strength, 4)),
        gr.update(value=curve),
    )


def _auto_timing_ui(template_name, shift_value, auto, start, end, peak, strength):
    if not auto:
        return gr.update(value=start), gr.update(value=end), gr.update(value=peak), gr.update(value=strength)

    config = _template_config(template_name)
    primary_part = _primary_template_part(config)
    start, end, peak, _ = _auto_stage_values(primary_part["stage"], shift_value)
    return (
        gr.update(value=round(start, 4)),
        gr.update(value=round(end, 4)),
        gr.update(value=round(peak, 4)),
        gr.update(value=strength),  # the user's multiplier, not the timing's
    )


def _reapplying_changed_templates(action):
    """Template management ``action`` (its last two inputs the base and hires
    panels' selected templates, its first two outputs their dropdowns), then,
    for a panel whose template it changed (renamed, or deleted and fallen
    back), what selecting that template does: _apply_template_ui. Takes the
    two panels' Shifts after ``action``'s inputs; returns the two panels'
    _apply_template_ui outputs (no-ops for the others) after its outputs.

    The dropdowns re-apply templates on user selection only (.input): a
    pasted template must keep the timing, strength and curve pasted with it."""

    def run(*args):
        *inputs, base_shift, hires_shift = args
        updates = action(*inputs)
        applied = []
        for before, update, shift in zip(inputs[-2:], updates[:2], (base_shift, hires_shift)):
            after = update.get("value", before) if isinstance(update, dict) else before
            applied.extend(_apply_template_ui(after, shift) if after != before else [gr.update() for _ in range(6)])
        return (*updates, *applied)

    return run


def _save_template_ui(name, auto, template_name, start, end, peak, strength, curve, *panels):
    """``panels``: the base and hires panels' selected templates."""
    name = str(name or "").strip()
    if not name:
        return _template_updates(_t("empty_template_name"), panels=panels)
    if name in BUILTIN_TEMPLATES:
        return _template_updates(_t("builtin_template_kept"), name, panels)

    template_name = str(template_name or "").strip()
    stage = _template_stage_from_name(template_name)
    source_parts = _template_parts(_template_config(template_name))
    record = {
        "auto": bool(auto),
        "stage": stage if stage in STAGE_CHOICES else STAGE_CHARACTER,
        "start": _clamp(_to_float(start), 0.0, 1.0),
        "end": _clamp(_to_float(end), 0.0, 1.0),
        "peak": _clamp(_to_float(peak), 0.0, 1.0),
        "strength": _clamp(_to_float(strength, 1.0), 0.0, 4.0),
        "curve": curve if curve in CURVE_CHOICES else CURVE_SMOOTH,
    }
    if source_parts:
        record["parts"] = source_parts

    def save(data):
        data[name] = record
        return name, _t("template_saved").format(name=name)

    return _edit_templates(save, panels)


def _save_preset_combo_template_ui(name, selected_presets, shift_value=3.0, *panels):
    """``panels``: the base and hires panels' selected templates."""
    name = str(name or "").strip()
    if not name:
        return _template_updates(_t("empty_template_name"), panels=panels)
    if name in BUILTIN_TEMPLATES:
        return _template_updates(_t("builtin_preset_kept"), name, panels)

    parts, error = _parts_from_presets(selected_presets)
    if error:
        return _template_updates(error, panels=panels)

    primary = next((part for part in parts if part["mode"] == PART_ALLOW), parts[0])
    presets_text = " + ".join(_part_preset_display_name(part) for part in parts)
    preview = _preset_combo_preview_text(selected_presets, shift_value)

    def save(data):
        data[name] = {
            "auto": True,
            "stage": primary["stage"],
            "parts": parts,
            # No start/end/peak: auto timing takes them from Shift when applied
            "strength": 1.0,
            "curve": primary.get("curve", STAGE_DEFAULT_CURVES.get(primary["stage"], CURVE_SMOOTH)),
        }
        return name, _t("combo_saved").format(name=name, presets=presets_text, preview=preview)

    return _edit_templates(save, panels)


def _open_template_builder_ui(template_name, shift_value):
    presets = _builder_presets_from_template(template_name)
    current = str(template_name or "").strip()
    default_name = "" if current in BUILTIN_TEMPLATES else current
    shift = _to_float(shift_value, 3.0)
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=default_name),
        gr.update(value=presets),
        gr.update(value=shift),
        _preset_combo_preview_text(presets, shift),
    )


def _close_template_modal_ui():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def _open_template_help_ui(template_name, shift_value):
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), _template_help_ui(template_name, shift_value)


def _preset_combo_preview_text(selected_presets, shift_value=3.0):
    parts, error = _parts_from_presets(selected_presets)
    if error:
        return error
    shift = _to_float(shift_value, 3.0)
    lines = [
        _t("combo_current") + " + ".join(_part_preset_display_name(part) for part in parts),
        "",
        f"{_t('intervention_preview')} (Shift `{_format_float(shift)}`):",
        "",
        f"| {_t('table_preset')} | {_t('table_action')} | {_t('table_start')} | {_t('table_end')} | {_t('table_peak')} | {_t('table_strength')} | {_t('table_curve')} |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for part in parts:
        start, end, peak, _ = _auto_stage_values(part["stage"], shift)
        action = _t("allow_action") if part["mode"] == PART_ALLOW else _t("ban_action")
        if part["mode"] == PART_BAN:
            strength_text = _t("zero_in_window")
        else:
            strength_text = _format_float(part["strength"])
        lines.append(
            "| {name} | {action} | {start} | {end} | {peak} | {strength} | {curve} |".format(
                name=_part_preset_display_name(part),
                action=action,
                start=_format_float(start),
                end=_format_float(end),
                peak=_format_float(peak),
                strength=strength_text,
                curve=part["curve"],
            )
        )
    return "\n".join(lines)


def _template_help_ui(template_name, shift_value):
    config = _template_config(template_name)
    # An unknown template reads as {}: one applying part of the character stage
    parts = _template_parts(config) or [_primary_template_part(config)]

    shift = _to_float(shift_value, 3.0)
    lines = [
        _t("help_title"),
        "",
        _t("help_parts"),
        _t("help_auto"),
        _t("help_combo"),
        _t("help_ratios"),
        "",
        f"{_t('current_template')}: `{template_name}`",
        f"{_t('current_shift')}: `{_format_float(shift)}`",
        "",
        _t("template_parts"),
    ]
    for part in parts:
        start, end, peak, _ = _auto_stage_values(part["stage"], shift)
        allow = part["mode"] == PART_ALLOW
        lines.append(
            _t("help_part").format(
                name=_part_preset_display_name(part),
                stage=part["stage"],
                action=_t("allow_action") if allow else _t("ban_action"),
                start=_format_float(start),
                end=_format_float(end),
                peak=_format_float(peak),
                strength=_format_float(part["strength"]) if allow else _t("zero_in_window"),
            )
        )

    return gr.update(value="\n".join(lines), visible=True)


def _rename_template_ui(old_name, new_name, *panels):
    """``panels``: the base and hires panels' selected templates."""
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()
    if not old_name:
        return _template_updates(_t("no_template_selected"), panels=panels)
    if old_name in BUILTIN_TEMPLATES:
        return _template_updates(_t("builtin_not_renamed"), old_name, panels)
    if not new_name:
        return _template_updates(_t("empty_new_name"), old_name, panels)
    if new_name in BUILTIN_TEMPLATES:
        return _template_updates(_t("builtin_name_taken"), old_name, panels)
    if new_name == old_name:
        return _template_updates(_t("same_name"), new_name, panels)

    def rename(data):
        if old_name not in data:
            return old_name, _t("rename_missing")
        if new_name in data:
            return old_name, _t("rename_taken")
        data[new_name] = data.pop(old_name)
        return new_name, _t("template_renamed").format(old=old_name, new=new_name)

    return _edit_templates(rename, panels, (old_name, new_name))


def _delete_template_ui(name, *panels):
    """``panels``: the base and hires panels' selected templates."""
    if not name:
        return _template_updates(_t("no_template_selected"), panels=panels)
    if name in BUILTIN_TEMPLATES:
        return _template_updates(_t("builtin_not_deleted"), name, panels)

    def delete(data):
        data.pop(name, None)
        return None, _t("template_deleted").format(name=name)

    return _edit_templates(delete, panels)
