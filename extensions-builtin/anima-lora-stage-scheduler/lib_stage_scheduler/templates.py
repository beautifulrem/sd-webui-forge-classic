"""Stage templates: the bundled and user store, their normalisation, and how
a run reads them (each template's parts, stage and auto window)."""

import logging
from pathlib import Path


from modules.anima_user_data import TemplateStore
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
    _selector_names,
    _msg,
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
            return [], _msg("unknown_preset", preset=preset)
        stage_modes = stages.setdefault(part["stage"], set())
        opposite = PART_BAN if part["mode"] == PART_ALLOW else PART_ALLOW
        if opposite in stage_modes:
            return [], _msg("preset_conflict", stage=part["stage"])
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
