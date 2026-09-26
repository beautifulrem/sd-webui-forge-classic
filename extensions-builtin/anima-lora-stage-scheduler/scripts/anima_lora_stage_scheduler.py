import logging
import os
import weakref
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from modules import script_callbacks, scripts
from modules.anima_lora_files import adapter_kind_for_file, is_anima_adapter_file, lora_coverage
from modules.anima_online_lora import acquire_online_lora, online_lora_leased_by_other_than, release_online_lora, user_online_lora
from modules.anima_user_data import UnreadableDataFile, data_file, read_json_object, write_json_atomic
from modules.anima_lora_support import (
    LORA_TAG_RE,
    LayerWeights,
    invalidate_cond_caches,
    manual_window_bans,
    rewrite_batch_prompts,
    script_hooks,
    sigma_progress,
    window_curve_factor,
    looks_like_layer_weights as _looks_like_layer_weights,
    parse_layer_weights as _parse_layer_weights,
    parse_module_weights as _parse_module_weights,
    lora_file_names as _names_for_lora_file,
    name_keys,
    parse_targets as _parse_targets,
    resolve_network,
    split_weight_sections,
    active_template_parts,
    anima_lora_block_count,
    anima_source_block,
    current_sampling_position,
    is_plain_number as _is_plain_number,
    block_index_from_key as _block_index_from_key,
    module_factor as _module_factor,
    model_key as _model_key,
    qwen_layer_index as _qwen_layer_index,
    stage_window,
    intro_block,
    register_language_option,
    ui_language,
    ui_text,
)
from modules.anima_presets import register_preset_control


logger = logging.getLogger("anima_lora_stage_scheduler")
LANGUAGE_OPTION = "anima_lora_stage_scheduler_language"
LANG = {
    "zh": {
        "settings_label": "Anima LoRA 阶段介入界面语言",
        "title": "Anima LoRA 阶段介入",
        "enable": "启用阶段介入控制",
        "base_panel": "底图阶段",
        "hires_panel": "高分辨率修复阶段",
        "hires_lora_panel": "高分辨率修复 LoRA 阶段",
        "target_scope": "目标范围",
        "panel_weight": "面板权重，0=使用提示词/分层插件",
        "target_lora": "目标 LoRA",
        "target_lora_placeholder": "目标范围为“仅目标列表”时填写，多个用逗号或换行分隔",
        "template": "模板",
        "copy": "复制",
        "builder": "组合预设",
        "auto_timing": "根据 Shift 自动设置介入时间",
        "single_lora_template": "单 LoRA 模板",
        "single_lora_template_placeholder": "每行一个：LoRA名=模板名\n也支持：<lora:name:0.8>=仅人物特征阶段介入\n未填写的 LoRA 使用上方模板",
        "start": "开始比例",
        "end": "结束比例",
        "peak": "峰值比例",
        "strength": "阶段强度倍率",
        "curve": "强度曲线",
        "hr_independent": "高分辨率修复使用独立面板；关闭时继承底图设置，但按 Hires Shift 自动设置时间",
        "hr_disable": "高分辨率修复时禁用 LoRA",
        "hr_disable_tags": "禁用指定完整 LoRA 标签",
        "template_manager": "模板管理",
        "existing_template": "已有模板",
        "rename_to": "重命名为",
        "rename_template": "重命名模板",
        "delete_template": "删除模板",
        "new_template_name": "新模板名称",
        "save_auto_template": "保存为自动时间模板",
        "save_current_base": "保存当前底图面板为模板",
        "template_help": "### 模板教程",
        "close": "关闭",
        "combo_title": "### 组合预设",
        "template_name": "模板名称",
        "base_preset_combo": "基础预设组合",
        "preview_shift": "预览 Shift",
        "save_combo": "保存组合模板",
        "combo_current": "当前组合：",
        "intervention_preview": "介入时间预览",
        "table_preset": "基础预设",
        "table_action": "行为",
        "table_start": "开始",
        "table_end": "结束",
        "table_peak": "峰值",
        "table_strength": "强度",
        "table_curve": "曲线",
        "allow_action": "介入",
        "ban_action": "禁止",
        "zero_in_window": "窗口内 0",
        "help_title": "### Anima LoRA 阶段介入教程",
        "current_template": "当前模板",
        "current_shift": "当前 Shift",
        "template_parts": "当前模板组成：",
    },
    "en": {
        "settings_label": "Anima LoRA Stage Scheduler UI language",
        "title": "Anima LoRA Stage Scheduler",
        "enable": "Enable stage scheduling",
        "base_panel": "Base pass",
        "hires_panel": "Hires. fix stage",
        "hires_lora_panel": "Hires. fix LoRA stage",
        "target_scope": "Target scope",
        "panel_weight": "Panel weight, 0 = use prompt / layer-weight plugin",
        "target_lora": "Target LoRA",
        "target_lora_placeholder": "Fill only when target scope is target list; separate multiple names with commas or new lines",
        "template": "Template",
        "copy": "Copy",
        "builder": "Preset combo",
        "auto_timing": "Auto timing from Shift",
        "single_lora_template": "Per-LoRA templates",
        "single_lora_template_placeholder": "One per line: LoRA name=template name\nAlso supports: <lora:name:0.8>=仅人物特征阶段介入\nEmpty entries use the template above",
        "start": "Start ratio",
        "end": "End ratio",
        "peak": "Peak ratio",
        "strength": "Stage strength multiplier",
        "curve": "Strength curve",
        "hr_independent": "Use independent Hires. fix panel; disabled = inherit base settings with Hires Shift timing",
        "hr_disable": "Disable LoRA during Hires. fix",
        "hr_disable_tags": "Full LoRA tags to disable",
        "template_manager": "Template management",
        "existing_template": "Existing template",
        "rename_to": "Rename to",
        "rename_template": "Rename template",
        "delete_template": "Delete template",
        "new_template_name": "New template name",
        "save_auto_template": "Save as auto-timing template",
        "save_current_base": "Save current base panel as template",
        "template_help": "### Template Guide",
        "close": "Close",
        "combo_title": "### Preset Combo",
        "template_name": "Template name",
        "base_preset_combo": "Base preset combo",
        "preview_shift": "Preview Shift",
        "save_combo": "Save combo template",
        "combo_current": "Current combo: ",
        "intervention_preview": "Timing preview",
        "table_preset": "Base preset",
        "table_action": "Action",
        "table_start": "Start",
        "table_end": "End",
        "table_peak": "Peak",
        "table_strength": "Strength",
        "table_curve": "Curve",
        "allow_action": "Apply",
        "ban_action": "Block",
        "zero_in_window": "0 in window",
        "help_title": "### Anima LoRA Stage Scheduler Guide",
        "current_template": "Current template",
        "current_shift": "Current Shift",
        "template_parts": "Template parts:",
    },
}


def _language():
    return ui_language(LANGUAGE_OPTION)


def _t(key):
    return ui_text(LANG, LANGUAGE_OPTION, key)


INTRO_EN = (
    "Schedule when Anima LoRAs affect sampling: use composition, character, "
    "style, or custom timing windows, with separate base and Hires. fix controls."
)
INTRO_ZH = "控制 Anima LoRA 在采样中的介入时间：可使用构图、人物、画风或自定义时间窗口，并可分别设置底图和高分辨率修复阶段。"


def _target_mode_choices():
    return [TARGET_MODE_EN[item] for item in TARGET_MODE_CHOICES] if _language() == "en" else TARGET_MODE_CHOICES


def _display_target_mode(value):
    canonical = _normalize_target_mode(value)
    return TARGET_MODE_EN.get(canonical, canonical) if _language() == "en" else canonical


def _normalize_target_mode(value):
    value = str(value or "")
    if value in TARGET_MODE_CHOICES:
        return value
    return TARGET_MODE_ALIASES.get(value, MODE_AUTO_ANIMA)


def _disable_choices():
    return [DISABLE_EN[item] for item in DISABLE_CHOICES] if _language() == "en" else DISABLE_CHOICES


def _display_disable_mode(value):
    canonical = _normalize_disable_mode(value)
    return DISABLE_EN.get(canonical, canonical) if _language() == "en" else canonical


def _normalize_disable_mode(value):
    value = str(value or "")
    if value in DISABLE_CHOICES:
        return value
    return DISABLE_ALIASES.get(value, DISABLE_NONE)


def _register_ui_settings():
    register_language_option(
        LANGUAGE_OPTION,
        "Anima LoRA Stage Scheduler language / Anima LoRA 阶段介入语言",
        ("anima-lora-stage-scheduler", "Anima LoRA Stage Scheduler"),
    )


script_callbacks.on_ui_settings(_register_ui_settings)

EXTENSION_DIR = Path(__file__).resolve().parents[1]
# The bundled templates.json seeds the user copy and is never written back
BUNDLED_TEMPLATE_FILE = EXTENSION_DIR / "templates.json"


def _template_file():
    return data_file("anima-lora-stage-scheduler", "templates.json", seed=BUNDLED_TEMPLATE_FILE)


MODE_AUTO_ANIMA = "自动识别 Anima LoRA"
MODE_PROMPT_ALL = "提示词中的全部 LoRA"
MODE_TARGETS = "仅目标列表"
TARGET_MODE_CHOICES = [MODE_AUTO_ANIMA, MODE_PROMPT_ALL, MODE_TARGETS]
TARGET_MODE_EN = {
    MODE_AUTO_ANIMA: "Auto-detect Anima LoRA",
    MODE_PROMPT_ALL: "All LoRAs in prompt",
    MODE_TARGETS: "Target list only",
}
TARGET_MODE_ALIASES = {value: key for key, value in TARGET_MODE_EN.items()}

STAGE_COMPOSITION = "构图阶段"
STAGE_CHARACTER = "人物特征阶段"
STAGE_STYLE = "画风阶段"
STAGE_CHOICES = [STAGE_COMPOSITION, STAGE_CHARACTER, STAGE_STYLE]

PART_ALLOW = "allow"
PART_BAN = "ban"

PRESET_STYLE_ALLOW = "仅画风阶段介入"
PRESET_COMPOSITION_ALLOW = "仅构图阶段介入"
PRESET_CHARACTER_ALLOW = "仅人物特征阶段介入"
PRESET_STYLE_BAN = "仅禁止画风阶段介入"
PRESET_COMPOSITION_BAN = "仅禁止构图阶段"
PRESET_CHARACTER_BAN = "仅禁止人物特征阶段"
TEMPLATE_PRESET_CHOICES = [
    PRESET_STYLE_ALLOW,
    PRESET_COMPOSITION_ALLOW,
    PRESET_CHARACTER_ALLOW,
    PRESET_STYLE_BAN,
    PRESET_COMPOSITION_BAN,
    PRESET_CHARACTER_BAN,
]
BUILDER_PRESET_LABELS = {
    PRESET_STYLE_ALLOW: "单画风阶段介入",
    PRESET_COMPOSITION_ALLOW: "单构图阶段介入",
    PRESET_CHARACTER_ALLOW: "单人物特征阶段介入",
    PRESET_STYLE_BAN: "单禁止画风阶段介入",
    PRESET_COMPOSITION_BAN: "单禁止构图阶段",
    PRESET_CHARACTER_BAN: "单禁止人物特征阶段",
}
BUILDER_PRESET_CHOICES = list(BUILDER_PRESET_LABELS.values())
BUILDER_PRESET_NAMES = {value: key for key, value in BUILDER_PRESET_LABELS.items()}

CURVE_HOLD = "保持"
CURVE_SMOOTH = "平滑淡入淡出"
CURVE_TRIANGLE = "线性三角"
CURVE_FRONT = "前强后弱"
CURVE_BACK = "后强前弱"
CURVE_CHOICES = [CURVE_SMOOTH, CURVE_HOLD, CURVE_TRIANGLE, CURVE_FRONT, CURVE_BACK]

DISABLE_NONE = "不禁用"
DISABLE_ALL = "禁用全部 LoRA"
DISABLE_SELECTED = "禁用指定完整标签"
DISABLE_CHOICES = [DISABLE_NONE, DISABLE_ALL, DISABLE_SELECTED]
DISABLE_EN = {
    DISABLE_NONE: "Do not disable",
    DISABLE_ALL: "Disable all LoRAs",
    DISABLE_SELECTED: "Disable listed full tags",
}
DISABLE_ALIASES = {value: key for key, value in DISABLE_EN.items()}

STAGE_DEFAULT_CURVES = {
    STAGE_COMPOSITION: CURVE_FRONT,
    STAGE_CHARACTER: CURVE_SMOOTH,
    STAGE_STYLE: CURVE_SMOOTH,
}

PRESET_PARTS = {
    PRESET_STYLE_ALLOW: {"stage": STAGE_STYLE, "mode": PART_ALLOW},
    PRESET_COMPOSITION_ALLOW: {"stage": STAGE_COMPOSITION, "mode": PART_ALLOW},
    PRESET_CHARACTER_ALLOW: {"stage": STAGE_CHARACTER, "mode": PART_ALLOW},
    PRESET_STYLE_BAN: {"stage": STAGE_STYLE, "mode": PART_BAN},
    PRESET_COMPOSITION_BAN: {"stage": STAGE_COMPOSITION, "mode": PART_BAN},
    PRESET_CHARACTER_BAN: {"stage": STAGE_CHARACTER, "mode": PART_BAN},
}

BUILTIN_TEMPLATES = {
    PRESET_STYLE_ALLOW: {
        "stage": STAGE_STYLE,
        "parts": [PRESET_PARTS[PRESET_STYLE_ALLOW]],
        "auto": True,
        "start": 0.62,
        "end": 1.0,
        "peak": 0.86,
        "strength": 1.0,
        "curve": CURVE_SMOOTH,
    },
    PRESET_COMPOSITION_ALLOW: {
        "stage": STAGE_COMPOSITION,
        "parts": [PRESET_PARTS[PRESET_COMPOSITION_ALLOW]],
        "auto": True,
        "start": 0.0,
        "end": 0.32,
        "peak": 0.08,
        "strength": 1.0,
        "curve": CURVE_FRONT,
    },
    PRESET_CHARACTER_ALLOW: {
        "stage": STAGE_CHARACTER,
        "parts": [PRESET_PARTS[PRESET_CHARACTER_ALLOW]],
        "auto": True,
        "start": 0.20,
        "end": 0.74,
        "peak": 0.48,
        "strength": 1.0,
        "curve": CURVE_SMOOTH,
    },
    PRESET_STYLE_BAN: {
        "stage": STAGE_STYLE,
        "parts": [PRESET_PARTS[PRESET_STYLE_BAN]],
        "auto": True,
        "start": 0.62,
        "end": 1.0,
        "peak": 0.86,
        "strength": 1.0,
        "curve": CURVE_SMOOTH,
    },
    PRESET_COMPOSITION_BAN: {
        "stage": STAGE_COMPOSITION,
        "parts": [PRESET_PARTS[PRESET_COMPOSITION_BAN]],
        "auto": True,
        "start": 0.0,
        "end": 0.32,
        "peak": 0.08,
        "strength": 1.0,
        "curve": CURVE_FRONT,
    },
    PRESET_CHARACTER_BAN: {
        "stage": STAGE_CHARACTER,
        "parts": [PRESET_PARTS[PRESET_CHARACTER_BAN]],
        "auto": True,
        "start": 0.20,
        "end": 0.74,
        "peak": 0.48,
        "strength": 1.0,
        "curve": CURVE_SMOOTH,
    },
}

_ACTIVE_STATE = None
_CFG_CALLBACK_REGISTERED = False
# Name of this script's lease on Forge's online-LoRA switch
ONLINE_LORA_OWNER = "anima_lora_stage_scheduler"
# (current_lora_hash tag, load_networks' hash, _loaded_patch_config, strengths)
# of the patches the last run left loaded; see _prepare_for_extra_network_activation
_RELEASED_PATCHES = None
# PassConfig.text_encoder_signature of both passes of the last run
_TEXT_ENCODER_SIGNATURE = None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _cleanup_runtime(p):
    """Undo ``p``'s own run. Never another's: a nested run (ADetailer) may
    carry a copied reference to the outer run's state, or none at all."""
    global _ACTIVE_STATE, _RELEASED_PATCHES

    release_online_lora(p, ONLINE_LORA_OWNER)
    state = getattr(p, "_anima_lora_stage_scheduler_state", None)
    if state is None or state.p is not p:
        return

    state.set_factor(1.0)
    owner = getattr(state, "p", None)
    sd_model = getattr(owner, "sd_model", None)
    if state.loaded_pass is not None and sd_model is not None:
        # The next run must not keep this run's scheduled / panel-weighted
        # patches, unless it schedules the very same ones.
        tag = f"anima-stage-scheduler:released:{uuid.uuid4().hex}"
        _RELEASED_PATCHES = (tag, sd_model.current_lora_hash, _loaded_patch_config(state, state.loaded_pass), list(state.wrappers))
        sd_model.current_lora_hash = tag
        state.loaded_pass = None
    if owner is not None and getattr(owner, "_anima_lora_stage_scheduler_state", None) is state:
        owner._anima_lora_stage_scheduler_state = None
    if _ACTIVE_STATE is state:
        _ACTIVE_STATE = state.outer_state


def _note_text_encoder_signature(p, signature):
    """Drop the cached prompt conditioning when the panels' text-encoder
    weights change: the cache key only sees the prompts."""
    global _TEXT_ENCODER_SIGNATURE

    if signature is not None and not any(signature):
        signature = None
    if signature != _TEXT_ENCODER_SIGNATURE:
        _TEXT_ENCODER_SIGNATURE = signature
        invalidate_cond_caches(p)


def _to_float(value, fallback=0.0):
    try:
        return float(value)
    except Exception:
        return fallback


def _format_float(value):
    text = f"{_to_float(value):.8f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _normalize_name(name: str) -> str:
    return str(name or "").strip().lower()


@dataclass
class PanelWeightSpec:
    raw: str = "0"
    numeric: float | None = None
    block_weights: LayerWeights = field(default_factory=LayerWeights)
    module_weights: dict[str, float] = field(default_factory=dict)
    qwen_weights: LayerWeights = field(default_factory=LayerWeights)
    has_block: bool = False
    has_module: bool = False
    has_qwen: bool = False

    @property
    def active(self) -> bool:
        return self.numeric is not None or self.has_block or self.has_module or self.has_qwen

    def prompt_multiplier(self) -> float:
        return self.numeric if self.numeric is not None else 1.0

    def strength_for_key(self, key, fallback, lora_blocks=None, model_blocks=None):
        """This panel's strength for ``key``. Parts the panel does not set keep
        ``fallback``, the strength handed down the add_patches chain (e.g. a
        layer-weight factor)."""
        if self.numeric is not None:
            return self.numeric

        model_key = _model_key(key)
        qwen_layer = _qwen_layer_index(model_key)
        if qwen_layer is not None:
            return self.qwen_weights.factor(qwen_layer) if self.has_qwen else fallback

        block_index = _block_index_from_key(model_key)
        if block_index is not None:
            # Block weights are written in the LoRA's own layout (see
            # anima_source_block for depth-expanded models).
            block_index = anima_source_block(block_index, lora_blocks, model_blocks)
            factor = self.block_weights.factor(block_index) if self.has_block else fallback
            if self.has_module:
                factor = factor * _module_factor(model_key, self.module_weights)
            return factor

        return fallback


def _parse_panel_weight(text) -> PanelWeightSpec:
    raw = str(text or "").strip()
    spec = PanelWeightSpec(raw=raw or "0")
    if not raw:
        return spec

    if _is_plain_number(raw):
        value = _to_float(raw, 0.0)
        if abs(value) > 1e-12:
            spec.numeric = value
        return spec

    sections = split_weight_sections(raw)
    block_text = sections.get("blocks", "")
    module_text = sections.get("modules", "")
    qwen_text = sections.get("qwen", "")

    if block_text and _looks_like_layer_weights(block_text):
        spec.block_weights = _parse_layer_weights(block_text)
        spec.has_block = True
    if module_text:
        spec.module_weights = _parse_module_weights(module_text)
        spec.has_module = bool(spec.module_weights)
    if qwen_text and _looks_like_layer_weights(qwen_text):
        spec.qwen_weights = _parse_layer_weights(qwen_text)
        spec.has_qwen = True

    if not spec.active:
        logger.warning("Ignored invalid Anima stage panel weight syntax: %s", raw)
    return spec


def _parse_disable_tags(text) -> list[str]:
    tags = []
    for line in str(text or "").replace("\r", "\n").split("\n"):
        tag = line.strip()
        if tag:
            tags.append(tag)
    return tags


def _selector_names(selector: str) -> set[str]:
    value = str(selector or "").strip()
    if not value:
        return set()

    match = LORA_TAG_RE.fullmatch(value)
    if match:
        value = match.group(1)

    return _parse_targets(value)


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
    try:
        return read_json_object(_template_file())
    except UnreadableDataFile:
        logger.exception("Failed to read Anima LoRA stage templates")
        return {}


def _editable_templates():
    """``(templates, None)``, or ``(None, message)`` when the file cannot be
    read: it is then never overwritten with just the one change."""
    try:
        return read_json_object(_template_file()), None
    except UnreadableDataFile as error:
        return None, f"模板文件无法读取，未做修改：{error}"


def _save_custom_templates(data: dict):
    write_json_atomic(_template_file(), data)


def _template_names() -> list[str]:
    names = list(BUILTIN_TEMPLATES.keys())
    for name in _load_custom_templates():
        if name not in names:
            names.append(name)
    return names


def _template_dropdown_update(value=None):
    choices = _template_names()
    if value not in choices:
        value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


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


def _stage_from_template_value(value, fallback=STAGE_CHARACTER) -> str:
    if value in STAGE_CHOICES:
        return value
    part = PRESET_PARTS.get(str(value or ""))
    if part:
        return part["stage"]
    return fallback


def _primary_template_part(config: dict, fallback_stage=STAGE_CHARACTER) -> dict:
    parts = _template_parts(config)
    for part in parts:
        if part["mode"] == PART_ALLOW:
            return part
    if parts:
        return parts[0]
    stage = config.get("stage") if config.get("stage") in STAGE_CHOICES else fallback_stage
    return {"stage": stage, "mode": PART_ALLOW, "curve": config.get("curve", STAGE_DEFAULT_CURVES.get(stage, CURVE_SMOOTH)), "strength": _to_float(config.get("strength", 1.0), 1.0)}


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
        return [], "请选择至少一个基础预设。"

    stages = {}
    parts = []
    for preset in selected:
        part = _normalize_template_part(preset)
        if part is None:
            return [], f"未知基础预设：{preset}"
        stage_modes = stages.setdefault(part["stage"], set())
        opposite = PART_BAN if part["mode"] == PART_ALLOW else PART_ALLOW
        if opposite in stage_modes:
            return [], f"组合冲突：{part['stage']} 同时选择了介入和禁止，无法保存。"
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
    kind = {STAGE_COMPOSITION: "composition", STAGE_STYLE: "style"}.get(stage, "character")
    return (*stage_window(kind, _to_float(shift_value, 3.0)), 1.0)


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


def _save_template_ui(name, auto, template_name, start, end, peak, strength, curve):
    name = str(name or "").strip()
    if not name:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), "模板名称不能为空。"
    if name in BUILTIN_TEMPLATES:
        return _template_dropdown_update(name), _template_dropdown_update(name), _template_dropdown_update(name), "内置模板不能覆盖，请换一个名称。"

    data, error = _editable_templates()
    if error:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), error
    template_name = str(template_name or "").strip()
    stage = _template_stage_from_name(template_name)
    source_parts = _template_parts(_template_config(template_name))
    data[name] = {
        "auto": bool(auto),
        "stage": stage if stage in STAGE_CHOICES else STAGE_CHARACTER,
        "start": _clamp(_to_float(start), 0.0, 1.0),
        "end": _clamp(_to_float(end), 0.0, 1.0),
        "peak": _clamp(_to_float(peak), 0.0, 1.0),
        "strength": _clamp(_to_float(strength, 1.0), 0.0, 4.0),
        "curve": curve if curve in CURVE_CHOICES else CURVE_SMOOTH,
    }
    if source_parts:
        data[name]["parts"] = source_parts
    _save_custom_templates(data)
    return _template_dropdown_update(name), _template_dropdown_update(name), _template_dropdown_update(name), f"已保存模板：{name}"


def _save_preset_combo_template_ui(name, selected_presets, shift_value=3.0):
    name = str(name or "").strip()
    if not name:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), "模板名称不能为空。"
    if name in BUILTIN_TEMPLATES:
        return _template_dropdown_update(name), _template_dropdown_update(name), _template_dropdown_update(name), "内置基础预设不能覆盖，请换一个名称。"

    parts, error = _parts_from_presets(selected_presets)
    if error:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), error

    primary = next((part for part in parts if part["mode"] == PART_ALLOW), parts[0])
    start, end, peak, strength = _auto_stage_values(primary["stage"], 3.0)
    data, error = _editable_templates()
    if error:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), error
    data[name] = {
        "auto": True,
        "stage": primary["stage"],
        "parts": parts,
        "start": start,
        "end": end,
        "peak": peak,
        "strength": strength,
        "curve": primary.get("curve", STAGE_DEFAULT_CURVES.get(primary["stage"], CURVE_SMOOTH)),
    }
    _save_custom_templates(data)
    presets_text = " + ".join(_part_preset_display_name(part) for part in parts)
    preview = _preset_combo_preview_text(selected_presets, shift_value)
    return _template_dropdown_update(name), _template_dropdown_update(name), _template_dropdown_update(name), f"已保存组合模板：{name} = {presets_text}\n\n{preview}"


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
        start, end, peak, strength = _auto_stage_values(part["stage"], shift)
        action = _t("allow_action") if part["mode"] == PART_ALLOW else _t("ban_action")
        curve = part.get("curve") if part.get("curve") in CURVE_CHOICES else STAGE_DEFAULT_CURVES.get(part["stage"], CURVE_SMOOTH)
        part_strength = _clamp(_to_float(part.get("strength", strength), strength), 0.0, 4.0)
        if part["mode"] == PART_BAN:
            strength_text = _t("zero_in_window")
        else:
            strength_text = _format_float(part_strength)
        lines.append(
            "| {name} | {action} | {start} | {end} | {peak} | {strength} | {curve} |".format(
                name=_part_preset_display_name(part),
                action=action,
                start=_format_float(start),
                end=_format_float(end),
                peak=_format_float(peak),
                strength=strength_text,
                curve=curve,
            )
        )
    return "\n".join(lines)


def _preset_combo_status_ui(selected_presets, shift_value=3.0):
    return _preset_combo_preview_text(selected_presets, shift_value)


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
        "- 模板由基础预设组合而成；“介入”表示 LoRA 只在该阶段生效，“禁止”表示 LoRA 在该阶段强制为 0。" if _language() == "zh" else "- Templates are built from base presets. Apply means the LoRA is active only in that stage; Block forces it to 0 in that stage.",
        "- 开启“根据 Shift 自动设置介入时间”时，每个基础预设都会按当前 Shift 自动移动开始、结束、峰值比例。" if _language() == "zh" else "- When auto timing from Shift is enabled, every base preset moves its start, end, and peak ratios from the current Shift value.",
        "- 组合模板可混合多个阶段，但同一阶段不能同时选择“介入”和“禁止”。" if _language() == "zh" else "- Combo templates can mix stages, but the same stage cannot be both Apply and Block.",
        "- 开始比例/结束比例是采样进度 0-1；峰值比例是介入曲线最强的位置；阶段强度倍率会乘到最终 LoRA 强度上。" if _language() == "zh" else "- Start/end are sampling progress ratios from 0 to 1; peak is the strongest point of the curve; stage strength multiplies the final LoRA strength.",
        "",
        f"{_t('current_template')}: `{template_name}`",
        f"{_t('current_shift')}: `{_format_float(shift)}`",
        "",
        _t("template_parts"),
    ]
    for part in parts:
        start, end, peak, strength = _auto_stage_values(part["stage"], shift)
        action = _t("allow_action") if part["mode"] == PART_ALLOW else _t("ban_action")
        if _language() == "zh":
            lines.append(f"- `{_part_preset_display_name(part)}`：{part['stage']} {action}，自动窗口 {_format_float(start)} - {_format_float(end)}，峰值 {_format_float(peak)}，强度 {_format_float(strength)}")
        else:
            lines.append(f"- `{_part_preset_display_name(part)}`: {part['stage']} {action}, auto window {_format_float(start)} - {_format_float(end)}, peak {_format_float(peak)}, strength {_format_float(strength)}")

    return gr.update(value="\n".join(lines), visible=True)


def _rename_template_ui(old_name, new_name):
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()
    if not old_name:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), "没有选择模板。"
    if old_name in BUILTIN_TEMPLATES:
        return _template_dropdown_update(old_name), _template_dropdown_update(old_name), _template_dropdown_update(old_name), "内置模板不能重命名。"
    if not new_name:
        return _template_dropdown_update(old_name), _template_dropdown_update(old_name), _template_dropdown_update(old_name), "新名称不能为空。"
    if new_name in BUILTIN_TEMPLATES:
        return _template_dropdown_update(old_name), _template_dropdown_update(old_name), _template_dropdown_update(old_name), "不能重命名为内置模板名称。"
    if new_name == old_name:
        return _template_dropdown_update(new_name), _template_dropdown_update(new_name), _template_dropdown_update(new_name), "新旧名称相同，无需重命名。"

    data, error = _editable_templates()
    if error:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), error
    if old_name not in data:
        return _template_dropdown_update(old_name), _template_dropdown_update(old_name), _template_dropdown_update(old_name), "找不到要重命名的模板。"
    if new_name in data:
        return _template_dropdown_update(old_name), _template_dropdown_update(old_name), _template_dropdown_update(old_name), "目标名称已存在。"

    data[new_name] = data.pop(old_name)
    _save_custom_templates(data)
    return _template_dropdown_update(new_name), _template_dropdown_update(new_name), _template_dropdown_update(new_name), f"已重命名模板：{old_name} → {new_name}"


def _delete_template_ui(name):
    if not name:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), "没有选择模板。"
    if name in BUILTIN_TEMPLATES:
        return _template_dropdown_update(name), _template_dropdown_update(name), _template_dropdown_update(name), "内置模板不能删除。"

    data, error = _editable_templates()
    if error:
        return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), error
    if name in data:
        del data[name]
        _save_custom_templates(data)
    return _template_dropdown_update(), _template_dropdown_update(), _template_dropdown_update(), f"已删除模板：{name}"


def _is_anima_lora_file(filename) -> bool:
    """An adapter (LoRA or LoKr) that trains Anima DiT blocks."""
    return bool(filename) and is_anima_adapter_file(filename, adapter_kind_for_file(filename) or "lora")


def _lora_block_count(filename) -> int | None:
    """The Anima depth (28/40/52) a LoRA file was trained for."""
    if not filename:
        return None
    return anima_lora_block_count(lora_coverage(filename, adapter_kind_for_file(filename) or "lora").blocks)


def _resolve_lora_filename(name: str):
    return getattr(resolve_network(name), "filename", None)


def _is_unet_patch_key(key) -> bool:
    text = str(key or "").lower()
    if "qwen" in text or "text_encoder" in text or "text_encoders" in text:
        return False
    return text.startswith("diffusion_model.") or ".blocks." in text or "blocks_" in text


def _activation_pass_for_extra_network_data(state, extra_network_data) -> str | None:
    p = state.p
    if getattr(p, "enable_hr", False) and getattr(p, "hr_extra_network_data", None) is extra_network_data:
        return "hires"
    if getattr(p, "extra_network_data", None) is extra_network_data:
        return "base"
    return None


def _loaded_patch_config(state, patch_pass) -> tuple:
    """What the patches loaded for ``patch_pass`` depend on besides the LoRA
    set, which load_networks hashes itself."""
    return patch_pass, state.base.patch_signature, state.hires.patch_signature, state.original_online_lora, frozenset(_lora_control_files())


def _prepare_for_extra_network_activation(p, extra_network_data):
    global _RELEASED_PATCHES

    state = getattr(p, "_anima_lora_stage_scheduler_state", None)
    if state is None or not state.enabled or getattr(p, "sd_model", None) is None:
        return None

    pass_name = _activation_pass_for_extra_network_data(state, extra_network_data)
    patch_pass = state.patch_pass(pass_name) if pass_name is not None else None
    if patch_pass is not None and patch_pass != state.loaded_pass:
        config = _loaded_patch_config(state, patch_pass)
        released, _RELEASED_PATCHES = _RELEASED_PATCHES, None
        if state.loaded_pass is None and released is not None and released[0] == p.sd_model.current_lora_hash and released[2] == config:
            # The last run left these very patches loaded (nothing loaded
            # others since): keep them, and schedule their strengths again.
            p.sd_model.current_lora_hash = released[1]
            state.wrappers = released[3]
        else:
            # The loaded patches were built for the other pass's panel, or by
            # another run: make load_networks rebuild them. A changed LoRA set
            # is reloaded by load_networks by itself.
            p.sd_model.current_lora_hash = f"anima-stage-scheduler:{patch_pass}:{hash(config):x}"
        state.loaded_pass = patch_pass
    return pass_name


def _sampling_progress(params) -> float:
    """Progress 0..1 of the current model call in the full sigma schedule (as
    the artist mixer reads it): Hires. fix and img2img start part-way, as
    their windows were denoised by the pass before. By steps without one."""
    p = getattr(getattr(params, "denoiser", None), "p", None)
    unet = getattr(getattr(getattr(p, "sd_model", None), "forge_objects", None), "unet", None)
    options = (getattr(unet, "model_options", None) or {}).get("transformer_options") or {}
    progress = sigma_progress(getattr(params, "sigma", None), options.get("sampling_sigmas"))
    if progress is not None:
        return progress
    step, total_steps = current_sampling_position(params)
    return _clamp(step / (total_steps - 1), 0.0, 1.0) if total_steps > 1 else 0.0


_CURVE_SHAPES = {CURVE_HOLD: "hold", CURVE_TRIANGLE: "triangle", CURVE_FRONT: "front", CURVE_BACK: "back"}


def _curve_factor_for_window(progress, start, end, peak, strength, curve) -> float:
    factor = window_curve_factor(progress, start, end, peak, _CURVE_SHAPES.get(curve, "smooth"))
    return _clamp(factor * _clamp(strength, 0.0, 4.0), 0.0, 4.0)


def _stage_curve_factor(config, progress) -> float:
    stage_strength = _clamp(config.stage_strength, 0.0, 4.0)
    if config.parts:
        allow_factors = []
        ban_active = False

        for part in config.parts:
            start, end, peak, strength = _auto_stage_values(part["stage"], config.shift)
            curve = part.get("curve") if part.get("curve") in CURVE_CHOICES else STAGE_DEFAULT_CURVES.get(part["stage"], CURVE_SMOOTH)
            part_strength = _clamp(_to_float(part.get("strength", strength), strength), 0.0, 4.0)

            if part["mode"] == PART_BAN:
                window_start, window_end = (start, end) if start <= end else (end, start)
                if window_start <= progress <= window_end:
                    ban_active = True
                continue

            allow_factors.append(_curve_factor_for_window(progress, start, end, peak, part_strength, curve))

        # The panel's stage strength scales the template's curves
        factor = max(allow_factors) * stage_strength if allow_factors else stage_strength
        return 0.0 if ban_active else _clamp(factor, 0.0, 4.0)

    if config.ban_window:
        start, end = sorted((config.start, config.end))
        return 0.0 if start <= progress <= end else stage_strength
    return _curve_factor_for_window(progress, config.start, config.end, config.peak, config.stage_strength, config.curve)


@dataclass
class PassConfig:
    target_mode: str
    target_loras: str
    panel_weight: PanelWeightSpec
    template_name: str
    lora_templates: dict[str, str]
    auto_timing: bool
    stage: str
    parts: list[dict]
    start: float
    end: float
    peak: float
    stage_strength: float
    curve: str
    shift: float
    prompt_names: set[str] = field(default_factory=set)
    # Manual timing: the start-end window blocks the LoRA instead of applying it
    ban_window: bool = False
    _lora_template_cache: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def panel_weight_nonzero(self) -> bool:
        return self.panel_weight.active

    @property
    def targets(self) -> set[str]:
        return _parse_targets(self.target_loras)

    @property
    def patch_signature(self) -> tuple:
        """What LoRA patches loaded under this pass depend on: which files are
        scheduled and their panel weights. Timing is looked up every step."""
        return self.target_mode, frozenset(self.targets), self.panel_weight.raw

    @property
    def text_encoder_signature(self) -> tuple | None:
        """The part of patch_signature the prompt conditioning depends on: a
        panel weight reaches the text encoder as a number or a qwen= section."""
        panel = self.panel_weight
        return self.patch_signature if panel.numeric is not None or panel.has_qwen else None

    def finalize_timing(self):
        template = _template_config(self.template_name)
        template_parts = _template_parts(template)
        self.parts = active_template_parts(self.auto_timing, template_parts)
        self.ban_window = not self.auto_timing and manual_window_bans(template_parts)
        stage = template.get("stage") or _stage_from_template_value(self.stage) or STAGE_CHARACTER
        if self.parts:
            primary = next((part for part in self.parts if part["mode"] == PART_ALLOW), self.parts[0])
            stage = primary["stage"]
        if stage not in STAGE_CHOICES:
            stage = STAGE_CHARACTER
        self.stage = stage

        if self.auto_timing:
            self.start, self.end, self.peak, _ = _auto_stage_values(stage, self.shift)
        else:
            self.start = _clamp(_to_float(self.start), 0.0, 1.0)
            self.end = _clamp(_to_float(self.end), 0.0, 1.0)
            if self.end < self.start:
                self.start, self.end = self.end, self.start
            self.peak = _clamp(_to_float(self.peak), self.start, self.end)
        self.stage_strength = _clamp(_to_float(self.stage_strength, 1.0), 0.0, 4.0)

        if self.curve not in CURVE_CHOICES:
            self.curve = CURVE_SMOOTH

    def config_for_file(self, filename, names=None):
        names = _names_for_lora_file(filename) if names is None else names
        template_name = None
        for selector_name, selector_template in self.lora_templates.items():
            if selector_name in names:
                template_name = selector_template
                break

        if not template_name or template_name == self.template_name:
            return self

        cache_key = _normalize_name(template_name)
        cached = self._lora_template_cache.get(cache_key)
        if cached is not None:
            return cached

        template = _template_config(template_name)
        auto_timing = bool(template.get("auto", self.auto_timing))
        parts = _template_parts(template)
        stage = template.get("stage") or self.stage
        if parts:
            primary = next((part for part in parts if part["mode"] == PART_ALLOW), parts[0])
            stage = primary["stage"]
        if stage not in STAGE_CHOICES:
            stage = self.stage
        curve = template.get("curve") or self.curve
        if curve not in CURVE_CHOICES:
            curve = self.curve
        start = template.get("start", self.start) if not auto_timing else self.start
        end = template.get("end", self.end) if not auto_timing else self.end
        peak = template.get("peak", self.peak) if not auto_timing else self.peak
        stage_strength = template.get("strength", self.stage_strength) if not auto_timing else self.stage_strength

        config = PassConfig(
            target_mode=self.target_mode,
            target_loras=self.target_loras,
            panel_weight=self.panel_weight,
            template_name=template_name,
            lora_templates=self.lora_templates,
            auto_timing=auto_timing,
            stage=stage,
            parts=parts,
            start=start,
            end=end,
            peak=peak,
            stage_strength=stage_strength,
            curve=curve,
            shift=self.shift,
            prompt_names=self.prompt_names,
        )
        config.finalize_timing()
        self._lora_template_cache[cache_key] = config
        return config

    def prompt_name_matches(self, lora_name: str) -> bool:
        if self.target_mode == MODE_PROMPT_ALL:
            return True
        if self.target_mode == MODE_TARGETS:
            return bool(name_keys(lora_name) & self.targets)

        filename = _resolve_lora_filename(lora_name)
        return _is_anima_lora_file(filename)

    def file_matches(self, filename, names=None) -> bool:
        names = _names_for_lora_file(filename) if names is None else names
        if self.target_mode == MODE_PROMPT_ALL:
            return bool(names & self.prompt_names) if self.prompt_names else True
        if self.target_mode == MODE_TARGETS:
            return bool(names & self.targets)
        return _is_anima_lora_file(filename)


# Tag of patches that serve both passes (see RuntimeState.patch_pass)
BOTH_PASSES = "both"


class ScheduledStrength:
    """A LoRA patch strength that follows the schedule every step. It stands
    in for a float in the weight adapters' arithmetic (e.g. BOFT's
    ``1 - strength``); tensors defer to its reflected operators."""

    def __init__(self, base_strength, filename, pass_name: str):
        self.base_strength = _to_float(base_strength, 1.0)
        self.filename = filename
        # "base", "hires" or BOTH_PASSES: the passes these patches were built for
        self.pass_name = pass_name
        self.factor = 1.0

    def value(self) -> float:
        return float(self.base_strength) * float(self.factor)

    def __float__(self):
        return self.value()

    def __bool__(self):
        return self.value() != 0.0

    def __repr__(self):
        return _format_float(self.value())

    def __eq__(self, other):
        return self.value() == other

    def __ne__(self, other):
        return self.value() != other

    def __mul__(self, other):
        return self.value() * other

    def __rmul__(self, other):
        return other * self.value()

    def __truediv__(self, other):
        return self.value() / other

    def __rtruediv__(self, other):
        return other / self.value()

    def __add__(self, other):
        return self.value() + other

    def __radd__(self, other):
        return other + self.value()

    def __sub__(self, other):
        return self.value() - other

    def __rsub__(self, other):
        return other - self.value()

    def __neg__(self):
        return -self.value()

    def __abs__(self):
        return abs(self.value())

    def __lt__(self, other):
        return self.value() < other

    def __le__(self, other):
        return self.value() <= other

    def __gt__(self, other):
        return self.value() > other

    def __ge__(self, other):
        return self.value() >= other


@dataclass
class RuntimeState:
    p: object
    run_id: str
    enabled: bool
    base: PassConfig
    hires: PassConfig
    hr_disable_mode: str
    hr_disable_tags: list[str]
    original_online_lora: bool
    current_activation_pass: str | None = None
    wrappers: list[weakref.ref] = field(default_factory=list)
    warned_offline: bool = False
    yielded_to_lora_control: set = field(default_factory=set)
    # The patch_pass() tag of the loaded LoRA patches
    loaded_pass: str | None = None
    # The run this one interrupted (a nested process_images), active again after it
    outer_state: object = None
    _file_names: dict = field(default_factory=dict)
    _file_configs: dict = field(default_factory=dict)

    def current_config(self) -> PassConfig:
        pass_name = self.current_pass_name()
        if pass_name == "hires":
            return self.hires
        return self.base

    def current_pass_name(self) -> str:
        if self.current_activation_pass in {"base", "hires"}:
            return self.current_activation_pass
        if bool(getattr(self.p, "is_hr_pass", False)):
            return "hires"
        return "base"

    def patch_pass(self, pass_name: str) -> str:
        """Tag of the patches loaded for ``pass_name``: one set serves both
        passes unless their panels build different ones, so Hires. fix does
        not reload every LoRA from disk on each pass change."""
        if self.base.patch_signature == self.hires.patch_signature:
            return BOTH_PASSES
        return pass_name

    def file_names(self, filename) -> frozenset[str]:
        """lora_file_names, once per run: it scans every installed LoRA."""
        if filename not in self._file_names:
            self._file_names[filename] = _names_for_lora_file(filename)
        return self._file_names[filename]

    def file_config(self, config: PassConfig, filename) -> PassConfig:
        """``config`` with the per-LoRA template of ``filename`` applied."""
        key = (id(config), filename)
        if key not in self._file_configs:
            self._file_configs[key] = config.config_for_file(filename, self.file_names(filename))
        return self._file_configs[key]

    def register(self, wrapper: ScheduledStrength):
        # Weakly: a LoRA reload drops its patches (and their strengths), which
        # must then stop being updated every step.
        self.wrappers.append(weakref.ref(wrapper))

    def live_wrappers(self) -> list[ScheduledStrength]:
        alive = [ref() for ref in self.wrappers]
        if None in alive:
            self.wrappers = [ref for ref, wrapper in zip(self.wrappers, alive) if wrapper is not None]
        return [wrapper for wrapper in alive if wrapper is not None]

    def set_factor(self, factor: float):
        for wrapper in self.live_wrappers():
            wrapper.factor = factor


def _build_pass_config(target_mode, target_loras, panel_weight, template_name, lora_templates_text, auto_timing, stage, start, end, peak, stage_strength, curve, shift):
    target_mode = _normalize_target_mode(target_mode)
    config = PassConfig(
        target_mode=target_mode,
        target_loras=str(target_loras or ""),
        panel_weight=_parse_panel_weight(panel_weight),
        template_name=str(template_name or next(iter(BUILTIN_TEMPLATES))),
        lora_templates=_parse_lora_template_overrides(lora_templates_text),
        auto_timing=bool(auto_timing),
        stage=_stage_from_template_value(stage),
        parts=[],
        start=_to_float(start, 0.2),
        end=_to_float(end, 0.8),
        peak=_to_float(peak, 0.5),
        stage_strength=_to_float(stage_strength, 1.0),
        curve=curve if curve in CURVE_CHOICES else CURVE_SMOOTH,
        shift=_to_float(shift, 3.0),
    )
    config.finalize_timing()
    return config


def _remove_hires_lora_tags(p, state: RuntimeState):
    if not getattr(p, "enable_hr", False) or not hasattr(p, "all_hr_prompts") or not p.all_hr_prompts:
        return

    mode = state.hr_disable_mode
    if mode == DISABLE_NONE:
        return

    tags = state.hr_disable_tags

    def rewrite(prompt):
        text = str(prompt or "")
        if mode == DISABLE_ALL:
            return LORA_TAG_RE.sub("", text)
        if mode == DISABLE_SELECTED:
            for tag in tags:
                text = text.replace(tag, "")
        return text

    p.all_hr_prompts = [rewrite(prompt) for prompt in p.all_hr_prompts]


def _collect_prompt_names(prompts) -> set[str]:
    names = set()
    for prompt in prompts or []:
        for match in LORA_TAG_RE.finditer(str(prompt or "")):
            names |= name_keys(match.group(1))
    return names


def _rewrite_prompt_loras(prompts, config: PassConfig):
    rewritten = []

    def replace(match):
        name = match.group(1).strip()
        if config.panel_weight_nonzero and config.prompt_name_matches(name):
            return f"<lora:{name}:{_format_float(config.panel_weight.prompt_multiplier())}>"
        return match.group(0)

    for prompt in prompts or []:
        rewritten.append(LORA_TAG_RE.sub(replace, str(prompt or "")))
    return rewritten


def _rewrite_batch_prompts(p, state: RuntimeState):
    def rewrite(config):
        def apply(prompts):
            rewritten = _rewrite_prompt_loras(prompts, config)
            config.prompt_names.update(_collect_prompt_names(rewritten))
            return rewritten

        return apply

    rewrite_batch_prompts(p, rewrite(state.base), rewrite(state.hires) if getattr(p, "enable_hr", False) else None)


def _patch_added_strengths(patch_destination, before_counts, state: RuntimeState, config: PassConfig, online_mode, lora_blocks=None, model_blocks=None, filename=None):
    pass_name = state.patch_pass(state.current_pass_name())

    def scheduled_patch(key, patch):
        if not isinstance(patch, (tuple, list)) or len(patch) < 5:
            return None

        strength_patch, patch_value, strength_model, offset, function = patch[:5]
        forge_fields = tuple(patch[5:])
        model_key = _model_key(key)
        base_strength = config.panel_weight.strength_for_key(model_key, strength_patch, lora_blocks, model_blocks) if config.panel_weight_nonzero else strength_patch

        if _is_unet_patch_key(model_key):
            if online_mode:
                strength_obj = ScheduledStrength(base_strength, filename, pass_name)
                state.register(strength_obj)
                strength_patch = strength_obj
            else:
                if not state.warned_offline:
                    logger.warning("Anima stage scheduler needs Forge online LoRA for per-step strength control.")
                    state.warned_offline = True
                strength_patch = _to_float(base_strength, 1.0)
        elif config.panel_weight_nonzero:
            strength_patch = _to_float(base_strength, 1.0)
        else:
            strength_patch = base_strength

        updated = (strength_patch, patch_value, strength_model, offset, function, *forge_fields)
        # Online patches are mutable lists that LoRA Control edits in place.
        return list(updated) if isinstance(patch, list) else updated

    for key, current_patches in patch_destination.items():
        start = before_counts.get(key, 0)
        if start >= len(current_patches):
            continue

        for index in range(start, len(current_patches)):
            entry = current_patches[index]
            # Online LoRAs are OnlineLoRAPatch weight wrappers holding a patch list.
            wrapped = getattr(entry, "patch", None)
            if isinstance(wrapped, list):
                for slot, patch in enumerate(wrapped):
                    updated = scheduled_patch(key, patch)
                    if updated is not None:
                        wrapped[slot] = updated
                continue

            updated = scheduled_patch(key, entry)
            if updated is not None:
                current_patches[index] = updated


def _add_patches(original, patcher, patches, strength_patch, strength_model, filename, online_mode):
    state = _ACTIVE_STATE
    config = state.current_config() if state is not None and state.enabled else None
    should_control = config is not None and config.file_matches(filename, state.file_names(filename))
    if should_control and filename in _lora_control_files():
        # Forge's LoRA Control (<lora:name:[w@t,...]>) overwrites the
        # strength slot every step; leave that LoRA to it rather than
        # having one schedule silently replace the other.
        should_control = False
        _note_lora_control(state, filename)
    if state is not None and not should_control and online_mode and not state.original_online_lora and not online_lora_leased_by_other_than(ONLINE_LORA_OWNER):
        # Only scheduled LoRAs need the online patches this run switched on;
        # the others keep Forge's faster merged weights, unless another script
        # (FreeFuse) holds online LoRA on for them too.
        online_mode = False

    # Forge Neo stores offline patches in ``patches`` and online LoRAs
    # as ``OnlineLoRAPatch`` entries in ``weight_wrapper_patches``.
    patch_destination = patcher.weight_wrapper_patches if online_mode else patcher.patches
    before_counts = {key: len(value) for key, value in patch_destination.items()} if should_control else {}

    loaded = original(patcher, patches, strength_patch=strength_patch, strength_model=strength_model, filename=filename, online_mode=online_mode)

    if should_control:
        lora_blocks = model_blocks = None
        if config.panel_weight.has_block:
            blocks = getattr(getattr(getattr(patcher, "model", None), "diffusion_model", None), "blocks", None)
            model_blocks = len(blocks) if blocks is not None else None
            lora_blocks = _lora_block_count(filename)
        _patch_added_strengths(patch_destination, before_counts, state, config, bool(online_mode), lora_blocks, model_blocks, filename)

    return loaded


def _activate(original, p, extra_network_data):
    state = getattr(p, "_anima_lora_stage_scheduler_state", None)
    previous_pass = getattr(state, "current_activation_pass", None) if state is not None else None
    pass_name = _prepare_for_extra_network_activation(p, extra_network_data)
    if state is not None and pass_name is not None:
        state.current_activation_pass = pass_name
    try:
        return original(p, extra_network_data)
    finally:
        if state is not None:
            state.current_activation_pass = previous_pass


def install_patch():
    from backend.patcher.base import ModelPatcher
    from backend.patcher.function_chain import function_chain_contains
    from modules import extra_networks

    # The wrappers call these through script_hooks: after "Reload UI" the
    # wrappers an earlier copy of this module installed run this copy, which
    # holds the live run state.
    script_hooks["stage_scheduler.add_patches"] = _add_patches
    script_hooks["stage_scheduler.activate"] = _activate

    if not function_chain_contains(ModelPatcher.add_patches, "_anima_stage_scheduler_patched"):
        original_add_patches = ModelPatcher.add_patches

        def add_patches_with_stage_scheduler(self, patches: list[dict], strength_patch: float = 1.0, strength_model: float = 1.0, *, filename: str = None, online_mode: bool = None):
            return script_hooks["stage_scheduler.add_patches"](original_add_patches, self, patches, strength_patch, strength_model, filename, online_mode)

        add_patches_with_stage_scheduler._anima_stage_scheduler_patched = True
        add_patches_with_stage_scheduler.__wrapped__ = original_add_patches
        ModelPatcher.add_patches = add_patches_with_stage_scheduler
        logger.info("Installed Anima LoRA stage scheduler patch")

    if not function_chain_contains(extra_networks.activate, "_anima_stage_scheduler_patched"):
        original_activate = extra_networks.activate

        def activate_with_stage_scheduler(p, extra_network_data):
            return script_hooks["stage_scheduler.activate"](original_activate, p, extra_network_data)

        activate_with_stage_scheduler._anima_stage_scheduler_patched = True
        activate_with_stage_scheduler.__wrapped__ = original_activate
        extra_networks.activate = activate_with_stage_scheduler
        logger.info("Installed Anima LoRA stage scheduler extra network patch")


def _lora_control_files() -> set:
    """Files scheduled by Forge's built-in LoRA Control in this generation."""
    from modules import scripts as webui_scripts

    for runner in (webui_scripts.scripts_txt2img, webui_scripts.scripts_img2img):
        for script in getattr(runner, "alwayson_scripts", None) or ():
            if script.title() == "LoRA Control Integrated":
                return set(getattr(type(script), "mapping", None) or ())
    return set()


def _note_lora_control(state, filename):
    if filename in state.yielded_to_lora_control:
        return
    state.yielded_to_lora_control.add(filename)
    name = os.path.splitext(os.path.basename(str(filename)))[0]
    logger.warning("Anima stage scheduler skips %s: it is scheduled with LoRA Control syntax", name)
    extra = getattr(state.p, "extra_generation_params", None)
    if isinstance(extra, dict):
        skipped = [value for value in str(extra.get("Anima stage scheduler skipped", "")).split(", ") if value]
        extra["Anima stage scheduler skipped"] = ", ".join(sorted({*skipped, name}))


def _denoiser_callback(params):
    state = _ACTIVE_STATE
    if state is None or not state.enabled:
        return

    denoiser = getattr(params, "denoiser", None)
    p = getattr(denoiser, "p", None)
    if p is None:
        return
    if p is not state.p:
        # A nested run the scheduler is not part of (ADetailer) samples with
        # the loaded patches: at their plain strengths, not frozen at the
        # factor of the outer run's last step.
        state.set_factor(1.0)
        return

    pass_name = state.current_pass_name()
    pass_config = state.current_config()
    progress = _sampling_progress(params)
    # Hundreds of strengths share one or a few per-file configs: one curve each
    factors: dict[int, float] = {}
    for wrapper in state.live_wrappers():
        if wrapper.pass_name not in (pass_name, BOTH_PASSES):
            wrapper.factor = 0.0
            continue
        config = state.file_config(pass_config, wrapper.filename)
        if id(config) not in factors:
            factors[id(config)] = _stage_curve_factor(config, progress)
        wrapper.factor = factors[id(config)]


def _register_cfg_callback_once():
    global _CFG_CALLBACK_REGISTERED
    if _CFG_CALLBACK_REGISTERED:
        return
    script_callbacks.on_cfg_denoiser(_denoiser_callback, name="anima_lora_stage_scheduler")
    _CFG_CALLBACK_REGISTERED = True


class Script(scripts.Script):
    section = None

    def __init__(self):
        _register_cfg_callback_once()
        self.base_shift_component = None
        self.hires_shift_component = None

    def title(self):
        return _t("title")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        elem_id = kwargs.get("elem_id")
        if self.is_txt2img:
            if elem_id == "txt2img_distilled_cfg_scale":
                self.base_shift_component = component
            elif elem_id == "txt2img_hr_distilled_cfg":
                self.hires_shift_component = component
        elif self.is_img2img and elem_id == "img2img_distilled_cfg_scale":
            self.base_shift_component = component

    def _stage_panel(self, prefix: str, label: str, default_open: bool, default_target_mode=MODE_AUTO_ANIMA):
        choices = _template_names()
        default_template = choices[0] if choices else "仅人物特征阶段介入"

        with gr.Accordion(label, open=default_open, elem_id=self.elem_id(f"{prefix}_panel")):
            with gr.Row():
                target_mode = gr.Radio(_target_mode_choices(), value=_display_target_mode(default_target_mode), label=_t("target_scope"), elem_id=self.elem_id(f"{prefix}_target_mode"))
                panel_weight = gr.Textbox(
                    label=_t("panel_weight"),
                    value="0",
                    lines=1,
                    placeholder="0.8 或 0-18=1,19-27=0.2 或 blocks=0-18=1,19-27=0.2;modules=mlp=0.8",
                    elem_id=self.elem_id(f"{prefix}_panel_weight"),
                )
            target_loras = gr.Textbox(label=_t("target_lora"), value="", lines=1, placeholder=_t("target_lora_placeholder"), elem_id=self.elem_id(f"{prefix}_target_loras"))

            with gr.Row():
                template = gr.Dropdown(label=_t("template"), choices=choices, value=default_template, elem_id=self.elem_id(f"{prefix}_template"))
                copy_template = gr.Button(_t("copy"), elem_id=self.elem_id(f"{prefix}_copy_template"))
                help_template = gr.Button("?", elem_id=self.elem_id(f"{prefix}_template_help"))
                open_builder = gr.Button(_t("builder"), elem_id=self.elem_id(f"{prefix}_open_template_builder"))
            auto_timing = gr.Checkbox(label=_t("auto_timing"), value=True, elem_id=self.elem_id(f"{prefix}_auto_timing"))
            lora_templates = gr.Textbox(
                label=_t("single_lora_template"),
                value="",
                lines=3,
                placeholder=_t("single_lora_template_placeholder"),
                elem_id=self.elem_id(f"{prefix}_lora_templates"),
            )

            with gr.Row():
                start = gr.Slider(0.0, 1.0, value=0.62 if prefix == "base" else 0.20, step=0.01, label=_t("start"), elem_id=self.elem_id(f"{prefix}_start"))
                end = gr.Slider(0.0, 1.0, value=1.0 if prefix == "base" else 0.74, step=0.01, label=_t("end"), elem_id=self.elem_id(f"{prefix}_end"))
                peak = gr.Slider(0.0, 1.0, value=0.86 if prefix == "base" else 0.48, step=0.01, label=_t("peak"), elem_id=self.elem_id(f"{prefix}_peak"))

            with gr.Row():
                stage_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label=_t("strength"), elem_id=self.elem_id(f"{prefix}_stage_strength"))
                curve = gr.Dropdown(label=_t("curve"), choices=CURVE_CHOICES, value=CURVE_SMOOTH, elem_id=self.elem_id(f"{prefix}_curve"))

        return target_mode, target_loras, panel_weight, template, lora_templates, auto_timing, start, end, peak, stage_strength, curve, copy_template, help_template, open_builder

    def ui(self, is_img2img):
        template_css = f"""
        <style>
        #{self.elem_id("template_modal")} {{
            position: fixed;
            inset: 0;
            z-index: 5000;
            padding: 24px;
            overflow: auto;
            box-sizing: border-box;
            background: rgba(0, 0, 0, 0.92);
        }}
        #{self.elem_id("template_modal")}.hide {{
            display: none !important;
        }}
        #{self.elem_id("template_modal")}:not(.hide) {{
            display: flex !important;
            align-items: center;
            justify-content: center;
        }}
        #{self.elem_id("template_modal_box")} {{
            width: min(980px, calc(100vw - 48px));
            max-height: calc(100vh - 48px);
            margin: 0 auto;
            background: #111827 !important;
            border: 1px solid #475569;
            border-radius: 8px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.65);
            padding: 16px;
            overflow: auto;
            box-sizing: border-box;
        }}
        #{self.elem_id("template_modal_box")} > .form,
        #{self.elem_id("template_help_modal")},
        #{self.elem_id("template_builder_modal")} {{
            background: #111827 !important;
        }}
        </style>
        """
        with gr.Accordion(_t("title"), open=False, elem_id=self.elem_id("accordion")):
            intro_block(self.elem_id, INTRO_EN, INTRO_ZH)
            enabled = gr.Checkbox(label=_t("enable"), value=False, elem_id=self.elem_id("enabled"))
            gr.HTML(template_css)

            base_controls = self._stage_panel("base", _t("base_panel"), True)

            with gr.Accordion(_t("hires_panel"), open=False, elem_id=self.elem_id("hires_panel")):
                hr_independent = gr.Checkbox(label=_t("hr_independent"), value=False, elem_id=self.elem_id("hr_independent"))
                hires_controls = self._stage_panel("hires", _t("hires_lora_panel"), True)
                with gr.Row():
                    hr_disable_mode = gr.Radio(_disable_choices(), value=_display_disable_mode(DISABLE_NONE), label=_t("hr_disable"), elem_id=self.elem_id("hr_disable_mode"))
                hr_disable_tags = gr.Textbox(
                    label=_t("hr_disable_tags"),
                    value="",
                    lines=3,
                    placeholder="<lora:name:0.8>\n<lora:name:blocks=0-9=0.8,10-18=1.0,19-27=0.7>",
                    elem_id=self.elem_id("hr_disable_tags"),
                )

            template_choices = _template_names()
            manage_default = template_choices[0] if template_choices else None
            with gr.Accordion(_t("template_manager"), open=False, elem_id=self.elem_id("template_manager")):
                with gr.Row():
                    manage_template = gr.Dropdown(label=_t("existing_template"), choices=template_choices, value=manage_default, elem_id=self.elem_id("manage_template"))
                    rename_template_name = gr.Textbox(label=_t("rename_to"), lines=1, elem_id=self.elem_id("rename_template_name"))
                with gr.Row():
                    rename_template = gr.Button(_t("rename_template"), elem_id=self.elem_id("rename_template"))
                    delete_template = gr.Button(_t("delete_template"), elem_id=self.elem_id("delete_template"))
                with gr.Row():
                    custom_name = gr.Textbox(label=_t("new_template_name"), lines=1, elem_id=self.elem_id("custom_name"))
                    custom_auto = gr.Checkbox(label=_t("save_auto_template"), value=True, elem_id=self.elem_id("custom_auto"))
                with gr.Row():
                    save_current_template = gr.Button(_t("save_current_base"), elem_id=self.elem_id("save_template"))
                template_status = gr.Markdown(value="", elem_id=self.elem_id("template_status"))

            with gr.Group(visible=False, elem_id=self.elem_id("template_modal")) as template_modal:
                with gr.Group(elem_id=self.elem_id("template_modal_box")):
                    with gr.Group(visible=False, elem_id=self.elem_id("template_help_modal")) as template_help_modal:
                        with gr.Row():
                            gr.Markdown(_t("template_help"))
                            close_help = gr.Button(_t("close"), elem_id=self.elem_id("close_template_help"))
                        template_help = gr.Markdown(value="", elem_id=self.elem_id("template_help_text"))
                    with gr.Group(visible=False, elem_id=self.elem_id("template_builder_modal")) as template_builder:
                        with gr.Row():
                            gr.Markdown(_t("combo_title"))
                            close_builder = gr.Button(_t("close"), elem_id=self.elem_id("close_template_builder"))
                        builder_name = gr.Textbox(label=_t("template_name"), lines=1, elem_id=self.elem_id("builder_name"))
                        builder_presets = gr.CheckboxGroup(
                            label=_t("base_preset_combo"),
                            choices=BUILDER_PRESET_CHOICES,
                            value=[BUILDER_PRESET_LABELS[PRESET_CHARACTER_ALLOW]],
                            elem_id=self.elem_id("builder_presets"),
                        )
                        builder_shift = gr.Number(label=_t("preview_shift"), value=3.0, precision=4, elem_id=self.elem_id("builder_shift"))
                        with gr.Row():
                            save_combo_template = gr.Button(_t("save_combo"), variant="primary", elem_id=self.elem_id("save_combo_template"))
                        builder_status = gr.Markdown(value="", elem_id=self.elem_id("builder_status"))

        (
            base_target_mode,
            base_target_loras,
            base_panel_weight,
            base_template,
            base_lora_templates,
            base_auto,
            base_start,
            base_end,
            base_peak,
            base_stage_strength,
            base_curve,
            base_copy_template,
            base_help_template,
            base_open_builder,
        ) = base_controls
        (
            hr_target_mode,
            hr_target_loras,
            hr_panel_weight,
            hr_template,
            hr_lora_templates,
            hr_auto,
            hr_start,
            hr_end,
            hr_peak,
            hr_stage_strength,
            hr_curve,
            hr_copy_template,
            hr_help_template,
            hr_open_builder,
        ) = hires_controls

        base_shift = self.base_shift_component or gr.Number(value=3.0, visible=False)
        hires_shift = self.hires_shift_component or self.base_shift_component or gr.Number(value=3.0, visible=False)

        for template_component, shift_component, auto_component, start_component, end_component, peak_component, strength_component, curve_component, help_button, copy_button in [
            (base_template, base_shift, base_auto, base_start, base_end, base_peak, base_stage_strength, base_curve, base_help_template, base_copy_template),
            (hr_template, hires_shift, hr_auto, hr_start, hr_end, hr_peak, hr_stage_strength, hr_curve, hr_help_template, hr_copy_template),
        ]:
            template_component.change(
                fn=_apply_template_ui,
                inputs=[template_component, shift_component],
                outputs=[auto_component, start_component, end_component, peak_component, strength_component, curve_component],
                show_progress=False,
            )
            auto_component.change(
                fn=_auto_timing_ui,
                inputs=[template_component, shift_component, auto_component, start_component, end_component, peak_component, strength_component],
                outputs=[start_component, end_component, peak_component, strength_component],
                show_progress=False,
            )
            help_button.click(
                fn=_open_template_help_ui,
                inputs=[template_component, shift_component],
                outputs=[template_modal, template_help_modal, template_builder, template_help],
                show_progress=False,
            )
            copy_button.click(
                fn=None,
                _js="(value) => { if (navigator.clipboard) { navigator.clipboard.writeText(value || ''); } return []; }",
                inputs=[template_component],
                outputs=[],
                show_progress=False,
            )

        base_open_builder.click(
            fn=_open_template_builder_ui,
            inputs=[base_template, base_shift],
            outputs=[template_modal, template_help_modal, template_builder, builder_name, builder_presets, builder_shift, builder_status],
            show_progress=False,
        )
        hr_open_builder.click(
            fn=_open_template_builder_ui,
            inputs=[hr_template, hires_shift],
            outputs=[template_modal, template_help_modal, template_builder, builder_name, builder_presets, builder_shift, builder_status],
            show_progress=False,
        )
        close_builder.click(
            fn=_close_template_modal_ui,
            inputs=[],
            outputs=[template_modal, template_help_modal, template_builder],
            show_progress=False,
        )
        close_help.click(
            fn=_close_template_modal_ui,
            inputs=[],
            outputs=[template_modal, template_help_modal, template_builder],
            show_progress=False,
        )
        builder_presets.change(
            fn=_preset_combo_status_ui,
            inputs=[builder_presets, builder_shift],
            outputs=[builder_status],
            show_progress=False,
        )
        builder_shift.change(
            fn=_preset_combo_status_ui,
            inputs=[builder_presets, builder_shift],
            outputs=[builder_status],
            show_progress=False,
        )
        manage_template.change(
            fn=lambda name: gr.update(value="" if str(name or "") in BUILTIN_TEMPLATES else str(name or "")),
            inputs=[manage_template],
            outputs=[rename_template_name],
            show_progress=False,
        )

        if self.base_shift_component is not None:
            self.base_shift_component.change(
                fn=_auto_timing_ui,
                inputs=[base_template, self.base_shift_component, base_auto, base_start, base_end, base_peak, base_stage_strength],
                outputs=[base_start, base_end, base_peak, base_stage_strength],
                show_progress=False,
            )
            if hasattr(self.base_shift_component, "release"):
                self.base_shift_component.release(
                    fn=_auto_timing_ui,
                    inputs=[base_template, self.base_shift_component, base_auto, base_start, base_end, base_peak, base_stage_strength],
                    outputs=[base_start, base_end, base_peak, base_stage_strength],
                    show_progress=False,
                )

        if self.hires_shift_component is not None:
            self.hires_shift_component.change(
                fn=_auto_timing_ui,
                inputs=[hr_template, self.hires_shift_component, hr_auto, hr_start, hr_end, hr_peak, hr_stage_strength],
                outputs=[hr_start, hr_end, hr_peak, hr_stage_strength],
                show_progress=False,
            )
            if hasattr(self.hires_shift_component, "release"):
                self.hires_shift_component.release(
                    fn=_auto_timing_ui,
                    inputs=[hr_template, self.hires_shift_component, hr_auto, hr_start, hr_end, hr_peak, hr_stage_strength],
                    outputs=[hr_start, hr_end, hr_peak, hr_stage_strength],
                    show_progress=False,
                )

        save_current_template.click(
            fn=_save_template_ui,
            inputs=[custom_name, custom_auto, base_template, base_start, base_end, base_peak, base_stage_strength, base_curve],
            outputs=[base_template, hr_template, manage_template, template_status],
            show_progress=False,
        )
        save_combo_template.click(
            fn=_save_preset_combo_template_ui,
            inputs=[builder_name, builder_presets, builder_shift],
            outputs=[base_template, hr_template, manage_template, builder_status],
            show_progress=False,
        )
        rename_template.click(
            fn=_rename_template_ui,
            inputs=[manage_template, rename_template_name],
            outputs=[base_template, hr_template, manage_template, template_status],
            show_progress=False,
        )
        delete_template.click(
            fn=_delete_template_ui,
            inputs=[manage_template],
            outputs=[base_template, hr_template, manage_template, template_status],
            show_progress=False,
        )

        register_preset_control(self.tabname, "lora_stage.enabled", enabled)

        return [
            enabled,
            base_target_mode,
            base_target_loras,
            base_panel_weight,
            base_template,
            base_lora_templates,
            base_auto,
            base_start,
            base_end,
            base_peak,
            base_stage_strength,
            base_curve,
            hr_independent,
            hr_target_mode,
            hr_target_loras,
            hr_panel_weight,
            hr_template,
            hr_lora_templates,
            hr_auto,
            hr_start,
            hr_end,
            hr_peak,
            hr_stage_strength,
            hr_curve,
            hr_disable_mode,
            hr_disable_tags,
        ]

    def process(
        self,
        p,
        enabled,
        base_target_mode,
        base_target_loras,
        base_panel_weight,
        base_template,
        base_lora_templates,
        base_auto,
        base_start,
        base_end,
        base_peak,
        base_stage_strength,
        base_curve,
        hr_independent,
        hr_target_mode,
        hr_target_loras,
        hr_panel_weight,
        hr_template,
        hr_lora_templates,
        hr_auto,
        hr_start,
        hr_end,
        hr_peak,
        hr_stage_strength,
        hr_curve,
        hr_disable_mode,
        hr_disable_tags,
        *args,
    ):
        global _ACTIVE_STATE

        if not enabled:
            _cleanup_runtime(p)
            _note_text_encoder_signature(p, None)
            return

        base_shift = _to_float(getattr(p, "distilled_cfg_scale", 3.0), 3.0)
        hires_shift = _to_float(getattr(p, "hr_distilled_cfg", base_shift), base_shift)

        base_config = _build_pass_config(
            base_target_mode,
            base_target_loras,
            base_panel_weight,
            base_template,
            base_lora_templates,
            base_auto,
            _template_stage_from_name(base_template),
            base_start,
            base_end,
            base_peak,
            base_stage_strength,
            base_curve,
            base_shift,
        )

        if hr_independent:
            hires_config = _build_pass_config(
                hr_target_mode,
                hr_target_loras,
                hr_panel_weight,
                hr_template,
                hr_lora_templates,
                hr_auto,
                _template_stage_from_name(hr_template),
                hr_start,
                hr_end,
                hr_peak,
                hr_stage_strength,
                hr_curve,
                hires_shift,
            )
        else:
            hires_config = _build_pass_config(
                base_target_mode,
                base_target_loras,
                base_panel_weight,
                base_template,
                base_lora_templates,
                base_auto,
                _template_stage_from_name(base_template),
                base_start,
                base_end,
                base_peak,
                base_stage_strength,
                base_curve,
                hires_shift,
            )

        state = RuntimeState(
            p=p,
            run_id=uuid.uuid4().hex,
            enabled=True,
            base=base_config,
            hires=hires_config,
            hr_disable_mode=_normalize_disable_mode(hr_disable_mode),
            hr_disable_tags=_parse_disable_tags(hr_disable_tags),
            original_online_lora=user_online_lora(),
            outer_state=_ACTIVE_STATE,
        )

        acquire_online_lora(p, ONLINE_LORA_OWNER)
        p._anima_lora_stage_scheduler_state = state
        _ACTIVE_STATE = state
        _note_text_encoder_signature(p, (base_config.text_encoder_signature, hires_config.text_encoder_signature))

        _remove_hires_lora_tags(p, state)

        p.extra_generation_params["Anima LoRA Stage Scheduler"] = (
            f"base={base_config.template_name}:{base_config.start:.2f}-{base_config.end:.2f}, "
            f"hires={hires_config.template_name}:{hires_config.start:.2f}-{hires_config.end:.2f}"
        )
        if state.hr_disable_mode != DISABLE_NONE:
            p.extra_generation_params["Anima Hires LoRA Disable"] = state.hr_disable_mode

    def before_process_batch(self, p, *args, **kwargs):
        state = getattr(p, "_anima_lora_stage_scheduler_state", None)
        if state is None or not state.enabled:
            return

        install_patch()
        _rewrite_batch_prompts(p, state)

    def process_before_every_sampling(self, p, *args, **kwargs):
        state = getattr(p, "_anima_lora_stage_scheduler_state", None)
        if state is None or not state.enabled:
            return

        state.set_factor(0.0)

    def postprocess(self, p, processed, *args, **kwargs):
        _cleanup_runtime(p)

    def cleanup(self, p, *args, **kwargs):
        _cleanup_runtime(p)
