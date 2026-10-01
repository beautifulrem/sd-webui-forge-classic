"""Stage templates: the bundled and user store, their normalisation, and the
template editor's UI handlers."""

import logging
from pathlib import Path

import gradio as gr

from modules.anima_user_data import TemplateStore, UnreadableDataFile
from modules.anima_lora_support import (
    stage_window_of,
)
from modules.anima_support import clamp as _clamp, to_float as _to_float
from lib_stage_scheduler.options import (
    BUILDER_PRESET_LABELS,
    BUILDER_PRESET_NAMES,
    BUILTIN_TEMPLATES,
    CURVE_CHOICES,
    CURVE_SMOOTH,
    PART_ALLOW,
    PART_BAN,
    PRESET_CHARACTER_ALLOW,
    PRESET_PARTS,
    STAGE_CHARACTER,
    STAGE_CHOICES,
    STAGE_COMPOSITION,
    STAGE_DEFAULT_CURVES,
    STAGE_STYLE,
    _format_float,
    _selector_names,
    _t,
)


logger = logging.getLogger("anima_lora_stage_scheduler")


EXTENSION_DIR = Path(__file__).resolve().parents[1]
# The bundled templates.json seeds the user copy and is never written back
BUNDLED_TEMPLATE_FILE = EXTENSION_DIR / "templates.json"


_TEMPLATES = TemplateStore("anima-lora-stage-scheduler", "templates.json", seed=BUNDLED_TEMPLATE_FILE)


def _parse_lora_template_overrides(text) -> dict[str, str]:
    templates = set(_template_names())
    overrides: dict[str, str] = {}

    for raw_line in str(text or "").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "->" in line:
            left, right = line.rsplit("->", 1)
        elif "=>" in line:
            left, right = line.rsplit("=>", 1)
        elif "=" in line:
            left, right = line.rsplit("=", 1)
        elif "＝" in line:
            left, right = line.rsplit("＝", 1)
        else:
            logger.warning("Ignored invalid Anima per-LoRA template line: %s", line)
            continue

        template_name = right.strip()
        if template_name not in templates:
            logger.warning("Ignored unknown Anima per-LoRA template: %s", template_name)
            continue

        names = _selector_names(left)
        if not names:
            logger.warning("Ignored empty Anima per-LoRA template selector: %s", line)
            continue

        for name in names:
            overrides[name] = template_name

    return overrides


def _load_custom_templates() -> dict:
    """Templates for display; an unreadable file reads as empty and is left alone."""
    return _TEMPLATES.load()


def _template_names() -> list[str]:
    names = list(BUILTIN_TEMPLATES.keys())
    for name in _load_custom_templates():
        if name not in names:
            names.append(name)
    return names


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


def _template_config(name: str) -> dict:
    if name in BUILTIN_TEMPLATES:
        return dict(BUILTIN_TEMPLATES[name])
    data = _load_custom_templates()
    value = data.get(str(name or ""), {})
    return dict(value) if isinstance(value, dict) else {}


def _normalize_template_part(part) -> dict | None:
    if isinstance(part, str):
        part = PRESET_PARTS.get(part) or PRESET_PARTS.get(BUILDER_PRESET_NAMES.get(part, ""))
    if not isinstance(part, dict):
        return None

    stage = part.get("stage")
    mode = part.get("mode", PART_ALLOW)
    if stage not in STAGE_CHOICES or mode not in {PART_ALLOW, PART_BAN}:
        return None

    return {
        "stage": stage,
        "mode": mode,
        "curve": part.get("curve") if part.get("curve") in CURVE_CHOICES else STAGE_DEFAULT_CURVES.get(stage, CURVE_SMOOTH),
        "strength": _clamp(_to_float(part.get("strength", 1.0), 1.0), 0.0, 4.0),
    }


def _part_preset_display_name(part: dict) -> str:
    normalized = _normalize_template_part(part)
    if not normalized:
        return ""
    canonical = _part_preset_name(normalized)
    return BUILDER_PRESET_LABELS.get(canonical, canonical)


def _template_parts(config: dict) -> list[dict]:
    raw_parts = config.get("parts")
    if isinstance(raw_parts, list):
        parts = [_normalize_template_part(part) for part in raw_parts]
        return [part for part in parts if part is not None]
    return []


def _part_preset_name(part: dict) -> str:
    normalized = _normalize_template_part(part)
    if not normalized:
        return ""
    for name, preset_part in PRESET_PARTS.items():
        if preset_part["stage"] == normalized["stage"] and preset_part["mode"] == normalized["mode"]:
            return name
    return ""


def _primary_template_part(config: dict, fallback_stage=STAGE_CHARACTER) -> dict:
    """The template's first applying part (else its first part), normalized;
    a template without parts reads as one applying part of its stage."""
    parts = _template_parts(config)
    for part in parts:
        if part["mode"] == PART_ALLOW:
            return part
    if parts:
        return parts[0]
    stage = config.get("stage") if config.get("stage") in STAGE_CHOICES else fallback_stage
    part = {"stage": stage, "mode": PART_ALLOW, "curve": config.get("curve"), "strength": config.get("strength", 1.0)}
    return _normalize_template_part(part) or part


def _template_preset_value(config: dict) -> str:
    parts = _template_parts(config)
    if len(parts) == 1:
        preset_name = _part_preset_name(parts[0])
        if preset_name:
            return preset_name
    return _part_preset_name(_primary_template_part(config)) or PRESET_CHARACTER_ALLOW


def _parts_from_presets(presets) -> tuple[list[dict], str | None]:
    selected = [str(item).strip() for item in (presets or []) if str(item).strip()]
    if not selected:
        return [], _t("no_presets")

    stages = {}
    parts = []
    for preset in selected:
        part = _normalize_template_part(preset)
        if part is None:
            return [], _t("unknown_preset").format(preset=preset)
        stage_modes = stages.setdefault(part["stage"], set())
        opposite = PART_BAN if part["mode"] == PART_ALLOW else PART_ALLOW
        if opposite in stage_modes:
            return [], _t("preset_conflict").format(stage=part["stage"])
        stage_modes.add(part["mode"])
        parts.append(part)

    return parts, None


def _presets_from_template(name: str) -> list[str]:
    config = _template_config(name)
    parts = _template_parts(config)
    presets = [_part_preset_name(part) for part in parts]
    presets = [preset for preset in presets if preset]
    if presets:
        return presets
    return [_template_preset_value(config)]


def _builder_presets_from_template(name: str) -> list[str]:
    return [BUILDER_PRESET_LABELS.get(preset, preset) for preset in _presets_from_template(name)]


def _template_stage_from_name(name: str, fallback=STAGE_CHARACTER) -> str:
    config = _template_config(name)
    primary_part = _primary_template_part(config, fallback)
    stage = primary_part["stage"]
    return stage if stage in STAGE_CHOICES else fallback


def _auto_stage_values(stage: str, shift_value) -> tuple[float, float, float, float]:
    return (*stage_window_of(stage, shift_value, composition=STAGE_COMPOSITION, style=STAGE_STYLE), 1.0)


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
    if not config:
        config = _template_config(PRESET_CHARACTER_ALLOW)
    parts = _template_parts(config)
    if not parts:
        parts = [_primary_template_part(config)]

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
