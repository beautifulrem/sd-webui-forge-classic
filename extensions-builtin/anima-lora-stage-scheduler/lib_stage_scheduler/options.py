"""The panels' language, choices, script-argument table and input parsing."""

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

from modules.anima_lora_support import (
    LORA_TAG_RE,
    LayerWeights,
    looks_like_layer_weights as _looks_like_layer_weights,
    parse_layer_weights as _parse_layer_weights,
    parse_module_weights as _parse_module_weights,
    parse_targets as _parse_targets,
    split_weight_sections,
    lora_key_factor,
)
from modules.anima_lora_ui import ui_language, ui_text
from modules.anima_support import plain_number as _plain_number, script_arguments, to_float as _to_float


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
        "panel_weight_placeholder": "0.8 或 0-18=1,19-27=0.2 或 blocks=0-18=1,19-27=0.2;modules=mlp=0.8",
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
        "help_parts": "- 模板由基础预设组合而成；“介入”表示 LoRA 只在该阶段生效，“禁止”表示 LoRA 在该阶段强制为 0。",
        "help_auto": "- 开启“根据 Shift 自动设置介入时间”时，每个基础预设都会按当前 Shift 自动移动开始、结束、峰值比例。",
        "help_combo": "- 组合模板可混合多个阶段，但同一阶段不能同时选择“介入”和“禁止”。",
        "help_ratios": "- 开始比例/结束比例是采样进度 0-1；峰值比例是介入曲线最强的位置；阶段强度倍率会乘到最终 LoRA 强度上。",
        "help_part": "- `{name}`：{stage} {action}，自动窗口 {start} - {end}，峰值 {peak}，强度 {strength}",
        "templates_unreadable": "模板文件无法读取，未做修改：{error}",
        "no_presets": "请选择至少一个基础预设。",
        "unknown_preset": "未知基础预设：{preset}",
        "preset_conflict": "组合冲突：{stage} 同时选择了介入和禁止，无法保存。",
        "empty_template_name": "模板名称不能为空。",
        "builtin_template_kept": "内置模板不能覆盖，请换一个名称。",
        "builtin_preset_kept": "内置基础预设不能覆盖，请换一个名称。",
        "template_saved": "已保存模板：{name}",
        "combo_saved": "已保存组合模板：{name} = {presets}\n\n{preview}",
        "no_template_selected": "没有选择模板。",
        "builtin_not_renamed": "内置模板不能重命名。",
        "empty_new_name": "新名称不能为空。",
        "builtin_name_taken": "不能重命名为内置模板名称。",
        "same_name": "新旧名称相同，无需重命名。",
        "rename_missing": "找不到要重命名的模板。",
        "rename_taken": "目标名称已存在。",
        "template_renamed": "已重命名模板：{old} → {new}",
        "builtin_not_deleted": "内置模板不能删除。",
        "template_deleted": "已删除模板：{name}",
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
        "panel_weight_placeholder": "0.8 or 0-18=1,19-27=0.2 or blocks=0-18=1,19-27=0.2;modules=mlp=0.8",
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
        "help_parts": "- Templates are built from base presets. Apply means the LoRA is active only in that stage; Block forces it to 0 in that stage.",
        "help_auto": "- When auto timing from Shift is enabled, every base preset moves its start, end, and peak ratios from the current Shift value.",
        "help_combo": "- Combo templates can mix stages, but the same stage cannot be both Apply and Block.",
        "help_ratios": "- Start/end are sampling progress ratios from 0 to 1; peak is the strongest point of the curve; stage strength multiplies the final LoRA strength.",
        "help_part": "- `{name}`: {stage} {action}, auto window {start} - {end}, peak {peak}, strength {strength}",
        "templates_unreadable": "The template file cannot be read; nothing was changed: {error}",
        "no_presets": "Select at least one base preset.",
        "unknown_preset": "Unknown base preset: {preset}",
        "preset_conflict": "Conflicting combo: {stage} is selected as both Apply and Block; not saved.",
        "empty_template_name": "The template name cannot be empty.",
        "builtin_template_kept": "Built-in templates cannot be overwritten; choose another name.",
        "builtin_preset_kept": "Built-in base presets cannot be overwritten; choose another name.",
        "template_saved": "Saved template: {name}",
        "combo_saved": "Saved combo template: {name} = {presets}\n\n{preview}",
        "no_template_selected": "No template selected.",
        "builtin_not_renamed": "Built-in templates cannot be renamed.",
        "empty_new_name": "The new name cannot be empty.",
        "builtin_name_taken": "Cannot rename to a built-in template name.",
        "same_name": "The new name is the same as the old one; nothing to rename.",
        "rename_missing": "The template to rename was not found.",
        "rename_taken": "The target name already exists.",
        "template_renamed": "Renamed template: {old} → {new}",
        "builtin_not_deleted": "Built-in templates cannot be deleted.",
        "template_deleted": "Deleted template: {name}",
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


def _lookup_choice(value, codes, aliases=None):
    """The choice ``value`` names, else None: the choice itself (a key of
    ``codes``, which maps choices to their infotext codes), its code, or its
    label in ``aliases`` (label -> choice). The API replays an infotext's
    codes straight into the run (it does not run on_infotext_pasted), so the
    run side takes codes as well as labels."""

    if value in codes:
        return value
    by_code = {code: choice for choice, code in codes.items()}
    return by_code.get(value) or (aliases or {}).get(value)


def _normalize_target_mode(value):
    return _lookup_choice(str(value or ""), TARGET_MODE_CODES, TARGET_MODE_ALIASES) or MODE_AUTO_ANIMA


def _disable_choices():
    return [DISABLE_EN[item] for item in DISABLE_CHOICES] if _language() == "en" else DISABLE_CHOICES


def _display_disable_mode(value):
    canonical = _normalize_disable_mode(value)
    return DISABLE_EN.get(canonical, canonical) if _language() == "en" else canonical


def _normalize_disable_mode(value):
    return _lookup_choice(str(value or ""), DISABLE_CODES, DISABLE_ALIASES) or DISABLE_NONE


def _normalize_curve(value):
    return _lookup_choice(str(value or ""), CURVE_CODES) or CURVE_SMOOTH


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
# Each curve's window_curve_factor shape, also its value in infotext
CURVE_CODES = {CURVE_SMOOTH: "smooth", CURVE_HOLD: "hold", CURVE_TRIANGLE: "triangle", CURVE_FRONT: "front", CURVE_BACK: "back"}

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

# Infotext values of the choices whose labels follow the UI language, so a
# zh UI and an en UI paste each other's images
TARGET_MODE_CODES = {MODE_AUTO_ANIMA: "auto", MODE_PROMPT_ALL: "all", MODE_TARGETS: "targets"}
DISABLE_CODES = {DISABLE_NONE: "none", DISABLE_ALL: "all", DISABLE_SELECTED: "selected"}

# The controls of a stage panel (_stage_panel) that change the run: (name
# after the panel's prefix, the field _panel_infotext records it under)
PANEL_ARGS = (
    ("target_mode", "scope"),
    ("target_loras", "targets"),
    ("panel_weight", "weight"),
    ("template", "template"),
    ("lora_templates", "per-LoRA templates"),
    ("auto", "auto timing"),
    ("start", "start"),
    ("end", "end"),
    ("peak", "peak"),
    ("stage_strength", "strength"),
    ("curve", "curve"),
)
PANEL_INFOTEXT_FIELDS = tuple(field for _, field in PANEL_ARGS)
INFOTEXT_ENABLED = "Anima LoRA stage enabled"
INFOTEXT_HIRES_INDEPENDENT = "Anima LoRA stage hires independent"
INFOTEXT_HIRES_DISABLE = "Anima LoRA stage hires disable"
INFOTEXT_HIRES_DISABLE_TAGS = "Anima LoRA stage hires disable tags"


def _panel_key(pass_name: str, field_name: str) -> str:
    return f"Anima LoRA stage {pass_name} {field_name}"


# One row per script argument, in the order ui() returns the components (the
# API's positional order): (name of the control in ui() and of the argument
# process reads, infotext key, preset name or None). ``infotext_fields``
# pastes each key back into its control.
SCRIPT_ARGS = (
    ("enabled", INFOTEXT_ENABLED, "lora_stage.enabled"),
    *((f"base_{name}", _panel_key("base", field), None) for name, field in PANEL_ARGS),
    ("hr_independent", INFOTEXT_HIRES_INDEPENDENT, None),
    *((f"hr_{name}", _panel_key("hires", field), None) for name, field in PANEL_ARGS),
    ("hr_disable_mode", INFOTEXT_HIRES_DISABLE, None),
    ("hr_disable_tags", INFOTEXT_HIRES_DISABLE_TAGS, None),
)


SCRIPT_ARG_NAMES = tuple(row[0] for row in SCRIPT_ARGS)
INFOTEXT_KEYS = tuple(row[1] for row in SCRIPT_ARGS)


def _arguments(values) -> SimpleNamespace:
    """The script args, one attribute per SCRIPT_ARGS name."""

    return script_arguments("Anima LoRA stage scheduler", SCRIPT_ARG_NAMES, values)


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

# Default (start, end, peak) of each stage's window, used until auto timing
# derives it from the Shift
STAGE_DEFAULT_TIMINGS = {
    STAGE_STYLE: (0.62, 1.0, 0.86),
    STAGE_COMPOSITION: (0.0, 0.32, 0.08),
    STAGE_CHARACTER: (0.20, 0.74, 0.48),
}

# One single-part auto-timing template per preset
BUILTIN_TEMPLATES = {
    name: {
        "stage": part["stage"],
        "parts": [part],
        "auto": True,
        **dict(zip(("start", "end", "peak"), STAGE_DEFAULT_TIMINGS[part["stage"]])),
        "strength": 1.0,
        "curve": STAGE_DEFAULT_CURVES[part["stage"]],
    }
    for name, part in PRESET_PARTS.items()
}


def _format_float(value):
    text = f"{_to_float(value):.8f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


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
        return lora_key_factor(
            key,
            fallback,
            qwen_weights=self.qwen_weights if self.has_qwen else None,
            block_weights=self.block_weights if self.has_block else None,
            module_weights=self.module_weights if self.has_module else None,
            lora_blocks=lora_blocks,
            model_blocks=model_blocks,
        )


def _parse_panel_weight(text) -> PanelWeightSpec:
    raw = str(text or "").strip()
    spec = PanelWeightSpec(raw=raw or "0")
    if not raw:
        return spec

    value = _plain_number(raw)  # also "０．５", "−0.5"
    if value is not None:
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
