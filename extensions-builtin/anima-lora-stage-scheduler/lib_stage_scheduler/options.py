"""The panels' texts, choices, script-argument tables and input parsing."""

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
from modules.anima_support import clamp as _clamp, plain_number as _plain_number, script_arguments, to_float as _to_float
from remi import ui


logger = logging.getLogger("anima_lora_stage_scheduler")

# Messages of the template actions (English catalog keys with {{name}}
# placeholders: the SPA translates them, and the panel's labels, through
# localizations/zh_CN.json)
TEXT = {
    "title": "Anima LoRA Stage Scheduler",
    "templates_unreadable": "The template file cannot be read; nothing was changed: {{error}}",
    "no_presets": "Select at least one base preset.",
    "unknown_preset": "Unknown base preset: {{preset}}",
    "preset_conflict": "Conflicting combo: {{stage}} is selected as both Apply and Block; not saved.",
    "empty_template_name": "The template name cannot be empty.",
    "builtin_template_kept": "Built-in templates cannot be overwritten; choose another name.",
    "builtin_preset_kept": "Built-in base presets cannot be overwritten; choose another name.",
    "template_saved": "Saved template: {{name}}",
    "combo_saved": "Saved combo template: {{name}} = {{presets}}",
    "no_template_selected": "No template selected.",
    "builtin_not_renamed": "Built-in templates cannot be renamed.",
    "empty_new_name": "The new name cannot be empty.",
    "builtin_name_taken": "Cannot rename to a built-in template name.",
    "same_name": "The new name is the same as the old one; nothing to rename.",
    "rename_missing": "The template to rename was not found.",
    "rename_taken": "The target name already exists.",
    "template_renamed": "Renamed template: {{old}} → {{new}}",
    "builtin_not_deleted": "Built-in templates cannot be deleted.",
    "template_deleted": "Deleted template: {{name}}",
}


def _t(key):
    return TEXT.get(key, key)


def _msg(key, **params):
    """A message of TEXT for a toast (a dict with ``params``): the SPA
    translates it, then fills the placeholders."""
    return ui.message(_t(key), **{name: str(value) for name, value in params.items()}) if params else _t(key)


INTRO_EN = (
    "Schedule when Anima LoRAs affect sampling: use composition, character, "
    "style, or custom timing windows, with separate base and Hires. fix controls."
)


def _target_mode_choices():
    """``(label, value)`` pairs: the English label of each canonical choice."""
    return [(TARGET_MODE_EN[item], item) for item in TARGET_MODE_CHOICES]


def _display_target_mode(value):
    """The control value of a target scope given as its choice, code or label."""
    return _normalize_target_mode(value)


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
    return [(DISABLE_EN[item], item) for item in DISABLE_CHOICES]


def _display_disable_mode(value):
    return _normalize_disable_mode(value)


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
# English labels of the canonical values the panels keep (template names,
# stages, curves stay what infotext and templates.json record)
TEMPLATE_LABELS_EN = {
    PRESET_STYLE_ALLOW: "Style stage only",
    PRESET_COMPOSITION_ALLOW: "Composition stage only",
    PRESET_CHARACTER_ALLOW: "Character stage only",
    PRESET_STYLE_BAN: "Block the style stage",
    PRESET_COMPOSITION_BAN: "Block the composition stage",
    PRESET_CHARACTER_BAN: "Block the character stage",
}
STAGE_LABELS_EN = {STAGE_COMPOSITION: "Composition stage", STAGE_CHARACTER: "Character stage", STAGE_STYLE: "Style stage"}

CURVE_HOLD = "保持"
CURVE_SMOOTH = "平滑淡入淡出"
CURVE_TRIANGLE = "线性三角"
CURVE_FRONT = "前强后弱"
CURVE_BACK = "后强前弱"
CURVE_CHOICES = [CURVE_SMOOTH, CURVE_HOLD, CURVE_TRIANGLE, CURVE_FRONT, CURVE_BACK]
# Each curve's window_curve_factor shape, also its value in infotext
CURVE_CODES = {CURVE_SMOOTH: "smooth", CURVE_HOLD: "hold", CURVE_TRIANGLE: "triangle", CURVE_FRONT: "front", CURVE_BACK: "back"}
CURVE_LABELS_EN = {CURVE_SMOOTH: "Smooth fade in/out", CURVE_HOLD: "Hold", CURVE_TRIANGLE: "Linear triangle", CURVE_FRONT: "Strong early, fading", CURVE_BACK: "Rising, strong late"}

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


# A panel's timing controls (PANEL_ARGS names), edited together by the
# stage-timeline widget as one value: {"auto", "start", "end", "peak",
# "strength", "curve"}
TIMING_ARGS = ("auto", "start", "end", "peak", "stage_strength", "curve")
TIMING_KEYS = ("auto", "start", "end", "peak", "strength", "curve")


def _panel_param_names(prefix: str) -> tuple[str, ...]:
    names = [f"{prefix}_{name}" for name, _ in PANEL_ARGS if name not in TIMING_ARGS]
    return (*names, f"{prefix}_timing")


# The params() inputs, in argument order: SCRIPT_ARGS with each panel's
# timing controls folded into its ``<prefix>_timing``
PARAM_NAMES = (
    "enabled",
    *_panel_param_names("base"),
    "hr_independent",
    *_panel_param_names("hr"),
    "hr_disable_mode",
    "hr_disable_tags",
)


SCRIPT_OWNER = "Anima LoRA stage scheduler"


def _arguments(values) -> SimpleNamespace:
    """The script args, one attribute per SCRIPT_ARGS name: from the
    params() arguments (PARAM_NAMES, each panel's timing one object), or
    from the Gradio panel's whole list of 26 (a caller handing process() all
    of them itself). The API gives a script only its params() slots, so an
    API request must use the params() layout: a Gradio-layout list cut to
    those slots is refused (TypeError) by its timing slots."""

    values = tuple(values)
    if len(values) == len(SCRIPT_ARG_NAMES):
        return script_arguments(SCRIPT_OWNER, SCRIPT_ARG_NAMES, values)
    given = script_arguments(SCRIPT_OWNER, PARAM_NAMES, values)
    for prefix in ("base", "hr"):
        timing = getattr(given, f"{prefix}_timing")
        if timing is not None and not isinstance(timing, dict):
            raise TypeError(
                f"{SCRIPT_OWNER}: argument {PARAM_NAMES.index(f'{prefix}_timing')} ({prefix}_timing) is a timing object, got {type(timing).__name__}; "
                f"send the {len(PARAM_NAMES)} params() arguments ({', '.join(PARAM_NAMES)}), not the Gradio panel's {len(SCRIPT_ARG_NAMES)}"
            )
    named = {}
    for name in SCRIPT_ARG_NAMES:
        prefix, _, arg = name.partition("_")
        if arg in TIMING_ARGS and prefix in ("base", "hr"):
            timing = _timing_value(getattr(given, f"{prefix}_timing"))
            named[name] = timing[TIMING_KEYS[TIMING_ARGS.index(arg)]]
        else:
            named[name] = getattr(given, name)
    return SimpleNamespace(**named)


def _timing_value(value) -> dict:
    """A panel's timing value with every key, bounded like the sliders it
    replaces (an unknown curve reads as smooth, codes as their curve)."""

    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"a stage timing is an object, got {type(value).__name__}")
    default = _default_timing()
    auto = value.get("auto", default["auto"])
    if isinstance(auto, str):
        auto = auto.strip().lower() not in {"", "false", "0", "no", "off", "none"}
    timing = {"auto": bool(auto)}
    for key, high in (("start", 1.0), ("end", 1.0), ("peak", 1.0), ("strength", 4.0)):
        timing[key] = round(_clamp(_to_float(value.get(key), default[key]), 0.0, high), 4)
    timing["curve"] = _normalize_curve(value.get("curve", default["curve"]))
    return timing


def _default_timing(template_name: str | None = None) -> dict:
    """The timing a panel starts with: auto timing on the default
    template's stage window, as the sliders started."""

    stage = BUILTIN_TEMPLATES.get(template_name or next(iter(BUILTIN_TEMPLATES)), {}).get("stage", STAGE_CHARACTER)
    start, end, peak = STAGE_DEFAULT_TIMINGS[stage]
    return {"auto": True, "start": start, "end": end, "peak": peak, "strength": 1.0, "curve": CURVE_SMOOTH}


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
