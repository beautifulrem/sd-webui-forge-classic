import copy
import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

from lib_artist_mixer.runtime import (
    BASE_CONTEXT_FUSIONS,
    COMBINE_CONCAT,
    COMBINE_OUTPUT_AVG,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    FUSION_QUALITY_DELTA,
    ArtistRuntime,
    MixerState,
    _RUN_STATES,
    _cleanup_runtime,
    _encode_text_batch,
    _finish_run_states,
    _install_model_wrapper,
    _summarize_blocks,
    _validate_anima_unet,
)
from lib_artist_mixer.schedule import (
    CURVE_BACK,
    CURVE_FRONT,
    CURVE_HOLD,
    CURVE_SMOOTH,
    CURVE_TRIANGLE,
    _CURVE_SHAPES,
    parse_inline_weight as _parse_inline_weight,
    parse_blocks as _parse_blocks,
)
from modules import scripts
from modules.anima_feature_conflicts import ARTIST_MIXER, FREEFUSE, REGIONAL, SPATIAL_GROUP, is_feature_active, set_feature_active
from modules.anima_lora_support import model_block_set, stage_window_of
from modules.anima_user_data import TemplateStore, UnreadableDataFile
from modules.anima_support import PasteKey, clamp as _clamp, effective_prompt_batch, process_ref, prompt_schedule, prompt_step, register_negpip_prompts, schedule_steps, script_arguments, to_float as _to_float
from modules.prompt_parser import get_multicond_prompt_list
from remi import ui


logger = logging.getLogger("anima_artist_scheduled_mixer")

EXTENSION_DIR = Path(__file__).resolve().parents[1]
_DATA_NAME = "anima-artist-scheduled-mixer"


# Templates earlier versions kept in the extension folder seed the user copy
_TEMPLATES = TemplateStore(_DATA_NAME, "artist_mixer_templates.json", seed=EXTENSION_DIR / "artist_mixer_templates.json")


REFERENCE_URL = "https://github.com/An1X3R/Anima-Artist-Mixer"


PRESET_CUSTOM = "Custom"
PRESET_COMPOSITION = "Composition"
PRESET_CHARACTER = "Character"
PRESET_STYLE = "Style"

OPT_PERFORMANCE = "Performance"
OPT_BALANCE = "Balance"
OPT_QUALITY = "Quality"

APPLY_BASE = "Base"
APPLY_HIRES = "Hires"
APPLY_BOTH = "Both"

MAX_ARTIST_ROWS = 32
DEFAULT_ARTIST_ROWS = 4

TABLE_HEADERS = [
    "Enabled",
    "Artist",
    "Weight",
    "Blocks",
    "Start",
    "End",
    "Peak",
    "Curve",
    "Stage",
    "Auto Shift",
]
# Values per artist row; a chain (base or hires) has at most MAX_ARTIST_ROWS rows
ROW_WIDTH = len(TABLE_HEADERS)
# The keys of a row of the artist-mix-table widget's value, in TABLE_HEADERS order
ROW_KEYS = tuple(header.lower().replace(" ", "_") for header in TABLE_HEADERS)
# The row chains, in argument order
ROW_CHAINS = ("base", "hires")
# The Gradio panel's argument per row value: "base_artist_0", ...
ROW_ARGS = tuple(f"{chain}_{key}_{index}" for chain in ROW_CHAINS for index in range(MAX_ARTIST_ROWS) for key in ROW_KEYS)
SCRIPT_ARGS_OWNER = "Anima Artist Scheduled Mixer"


@dataclass(frozen=True)
class _MixerArgs:
    """What a run reads from the script args: the HEADER_ARGS controls, then
    the base and the hires row values (``component_values``, the layout of
    the Gradio panel's ROW_ARGS: MAX_ARTIST_ROWS rows of ROW_WIDTH each)."""

    enable: object
    base_row_count: object
    hires_row_count: object
    hires_independent: object
    disable_hires_mixing: object
    runtime_base_shift: object
    runtime_hires_shift: object
    global_strength: object
    optimization: object
    combine_mode: object
    fusion_mode: object
    apply_uncond: object
    enable_cache: object
    component_values: tuple = ()

    @classmethod
    def from_args(cls, args, p=None):
        """The script args, checked by ``script_arguments`` (a wrong count
        raises TypeError): the params() arguments (PARAM_NAMES; each chain's
        rows one table value, the trailing template controls optional), or
        the Gradio panel's whole SCRIPT_ARG_NAMES list (a caller handing
        process() the 653 values itself). The API gives a script only its
        params() slots, so an API request must use the params() layout: a
        Gradio-layout list cut to those slots is refused (TypeError) by its
        table slots. The params() layout has no Shift copies: they are
        ``p``'s Shifts (3.0 without ``p``)."""
        args = tuple(args)
        if len(args) == len(SCRIPT_ARG_NAMES):
            named = script_arguments(SCRIPT_ARGS_OWNER, SCRIPT_ARG_NAMES, args)
            return cls(**{name: getattr(named, name) for name in HEADER_ARGS}, component_values=tuple(getattr(named, name) for name in ROW_ARGS))
        # The template controls may be left out (API callers); nothing else
        defaults = dict.fromkeys(TEMPLATE_ARGS) if len(args) == len(PARAM_NAMES) - len(TEMPLATE_ARGS) else None
        named = script_arguments(SCRIPT_ARGS_OWNER, PARAM_NAMES, args, defaults=defaults)
        for name in TABLE_ARGS:
            value = getattr(named, name)
            if value is not None and not isinstance(value, (list, tuple, dict)):  # dict: a {"data": rows} table
                raise TypeError(
                    f"{SCRIPT_ARGS_OWNER}: argument {PARAM_NAMES.index(name)} ({name}) is a list of artist rows, got {type(value).__name__}; "
                    f"send the {len(PARAM_NAMES)} params() arguments ({', '.join(PARAM_NAMES)}), not the Gradio panel's {len(SCRIPT_ARG_NAMES)}"
                )
        tables = {chain: _rows_of_table(getattr(named, f"{chain}_rows")) for chain in ROW_CHAINS}
        values = []
        for chain in ROW_CHAINS:
            for index in range(MAX_ARTIST_ROWS):
                values.extend(tables[chain][index] if index < len(tables[chain]) else [None] * ROW_WIDTH)
        base_shift = _runtime_shift_from_processing(p, 3.0, False) if p is not None else 3.0
        hires_shift = _runtime_shift_from_processing(p, base_shift, True) if p is not None else base_shift
        return cls(
            **{name: getattr(named, name) for name in COMPACT_HEADER_ARGS},
            base_row_count=len(tables["base"]) or 1,
            hires_row_count=len(tables["hires"]) or 1,
            runtime_base_shift=base_shift,
            runtime_hires_shift=hires_shift,
            component_values=tuple(values),
        )

    def row_values(self, hires):
        """Row values of the base or the ``hires`` rows."""
        size = MAX_ARTIST_ROWS * ROW_WIDTH
        return self.component_values[size : size * 2] if hires else self.component_values[:size]


HEADER_ARGS = tuple(name for name in _MixerArgs.__dataclass_fields__ if name != "component_values")
# Every argument of the Gradio panel, in the order its ui() returned them
SCRIPT_ARG_NAMES = (*HEADER_ARGS, *ROW_ARGS)
# The params() arguments: the header controls that are not derived (no row
# counts, no Shift copies), the two artist tables, then the template
# controls (UI only: the run ignores them)
COMPACT_HEADER_ARGS = ("enable", "hires_independent", "disable_hires_mixing", "global_strength", "optimization", "combine_mode", "fusion_mode", "apply_uncond", "enable_cache")
TABLE_ARGS = ("base_rows", "hires_rows")
TEMPLATE_ARGS = ("template", "template_target", "template_name", "rename_to")
PARAM_NAMES = (*COMPACT_HEADER_ARGS, *TABLE_ARGS, *TEMPLATE_ARGS)


OPTION_LABELS = {
    "optimization": OrderedDict(
        [
            (OPT_PERFORMANCE, {"en": "Performance", "zh": "性能", "aliases": ("Perf", "Speed", "速度")}),
            (OPT_BALANCE, {"en": "Balance", "zh": "平衡", "aliases": ("Balanced", "均衡")}),
            (OPT_QUALITY, {"en": "Quality", "zh": "质量", "aliases": ("High quality", "高质量")}),
        ]
    ),
    "combine": OrderedDict(
        [
            (COMBINE_OUTPUT_AVG, {"en": "Output average", "zh": "输出平均", "aliases": ("output average", "输出平均值")}),
            (COMBINE_CONCAT, {"en": "Token concatenation", "zh": "token拼接", "aliases": ("Token concat", "concat", "拼接", "令牌拼接")}),
        ]
    ),
    "fusion": OrderedDict(
        [
            (FUSION_INTERPOLATE, {"en": "Interpolate", "zh": "插值融合", "aliases": ("interp", "插值")}),
            (
                FUSION_CONCAT_WITH_BASE,
                {"en": "Concat with base", "zh": "拼接底图条件", "aliases": ("concat base", "拼接原始条件")},
            ),
            (
                FUSION_QUALITY_DELTA,
                {
                    "en": "Quality-safe delta",
                    "zh": "保真增量",
                    "aliases": ("delta", "safe delta", "quality delta", "保真差值", "质量安全增量"),
                },
            ),
        ]
    ),
    "curve": OrderedDict(
        [
            (CURVE_SMOOTH, {"en": "Smooth", "zh": "平滑", "aliases": ("平滑曲线",)}),
            (CURVE_HOLD, {"en": "Hold", "zh": "保持", "aliases": ("恒定", "全程保持")}),
            (CURVE_TRIANGLE, {"en": "Triangle", "zh": "三角峰", "aliases": ("Triangle peak", "三角")}),
            (CURVE_FRONT, {"en": "Front loaded", "zh": "前段强化", "aliases": ("Early", "前段")}),
            (CURVE_BACK, {"en": "Back loaded", "zh": "后段强化", "aliases": ("Late", "后段")}),
        ]
    ),
    "stage": OrderedDict(
        [
            (PRESET_CUSTOM, {"en": "Custom", "zh": "自定义", "aliases": ("手动",)}),
            (PRESET_COMPOSITION, {"en": "Composition", "zh": "构图", "aliases": ("Compose", "布局")}),
            (PRESET_CHARACTER, {"en": "Character", "zh": "人物", "aliases": ("Subject", "角色")}),
            (PRESET_STYLE, {"en": "Style", "zh": "画风", "aliases": ("Artist style", "风格")}),
        ]
    ),
    "apply_target": OrderedDict(
        [
            (APPLY_BASE, {"en": "Base", "zh": "底图", "aliases": ("Base image", "底图设置")}),
            (APPLY_HIRES, {"en": "Hires. fix", "zh": "高分修复", "aliases": ("Hires", "Highres", "高分")}),
            (APPLY_BOTH, {"en": "Both", "zh": "两者", "aliases": ("All", "全部")}),
        ]
    ),
}

TABLE_HEADER_LABELS = OrderedDict(
    [
        ("Enabled", {"en": "Enabled", "zh": "启用"}),
        ("Artist", {"en": "Artist", "zh": "画师"}),
        ("Weight", {"en": "Weight", "zh": "权重"}),
        ("Blocks", {"en": "Blocks", "zh": "层数"}),
        ("Start", {"en": "Start", "zh": "开始"}),
        ("End", {"en": "End", "zh": "结束"}),
        ("Peak", {"en": "Peak", "zh": "峰值"}),
        ("Curve", {"en": "Curve", "zh": "曲线"}),
        ("Stage", {"en": "Stage", "zh": "阶段"}),
        ("Auto Shift", {"en": "Auto Shift", "zh": "自动偏移"}),
    ]
)


# The panel's messages (English catalog keys with {{name}} placeholders: the
# SPA translates them, and the labels, through localizations/zh_CN.json)
TEXT = {
    "title": "Anima Artist Scheduled Mixer",
    "empty_template_name": "Template name is empty.",
    "saved_base_template": "Saved base artist-chain template: {{name}}",
    "saved_hires_template": "Saved Hires. fix template: {{name}}",
    "hires_template_disabled": "Enable independent Hires. fix artist chain before saving a Hires. fix template.",
    "no_template_selected": "No template selected.",
    "new_template_name_empty": "New template name is empty.",
    "renamed_template": "Renamed template to: {{name}}",
    "deleted_template": "Deleted template: {{name}}",
    "no_template_deleted": "No template deleted.",
    "template_exists": "A template with that name already exists: {{name}}",
    "template_file_unreadable": "The template file cannot be read; nothing was changed: {{error}}",
    "applied_template_Base": "Applied template `{{name}}` to Base.",
    "applied_template_Hires": "Applied template `{{name}}` to Hires. fix.",
    "applied_template_Both": "Applied template `{{name}}` to Base and Hires. fix.",
    "defaults_restored": "Restored all settings to defaults.",
}


INTRO_EN = f"""
Independent artist encoding and scheduled cross-attention mixing for Anima.
Each artist row is encoded separately as `artist + base prompt`, then mixed inside Anima cross-attention. The default `Output average + Interpolate` path follows the stronger reference behavior from [{REFERENCE_URL}]({REFERENCE_URL}); `Quality-safe delta` is still available when you want a more conservative Forge-friendly blend.

Templates keep stable internal values, so templates saved with English, Chinese, or raw option values all load.

Thanks to **An1X3R/Anima-Artist-Mixer** and **汐浮尘/utowo** for the original split-and-encode/cross-attention design.
""".strip()

COMPATIBILITY_NOTE = "Compatibility: Artist Mixer, FreeFuse, and Regional Conditioning are one-of-three; the active choice locks the others."

SHIFT_HINT = "Stage/Auto Shift reads the current Shift from the generation panel; sampling reads the actual Base/Hires Shift again."

HELP_EN = """
**Columns**

`Artist`: one artist prompt per row. It may include an inline row multiplier such as `(wlop:1.2)`, `[wlop:0.8]`, or `wlop:1.2`; the final row strength is `Weight x inline multiplier`. Keep comma-separated chains for quick migration if needed, but one artist per row gives the cleanest control.

`Weight` controls this artist's relative and absolute contribution. `Blocks` accepts `0-51`, `0,3,5-12`, or negative indices such as `-1`, always in the 28-block (2B) layout: on 2.9B/3.8B models each block also covers its inserted copies, and indices past 27 are ignored, so `0-51` is every block. `Start` and `End` are sampling progress from 0 to 1: a fraction of the sampling steps (Hires. fix and img2img start part-way), not of sigma. `Peak` is the point inside that window where the row reaches full strength; it matters most for Smooth and Triangle curves.

`Curve` shapes the row strength inside the Start/End window. Hold stays at full strength for the whole window. Smooth fades in and out around the peak. Triangle rises and falls linearly. Front loaded starts strong and fades later; Back loaded does the opposite.

`Stage` is a timing preset. Custom keeps your manual values. Composition starts early, Character focuses on the middle, and Style runs late. Choosing a non-Custom stage refreshes Start/End/Peak immediately and enables Auto Shift for that row. Auto Shift keeps the row aligned with the current Forge Shift value; turn it off afterward if you want to fine-tune the numbers by hand.

**Strength**

Global artist strength controls how much of the artist branch is injected into the normal prompt cross-attention output. Row weights choose each artist's share; small row weights also reduce total influence when the active weights sum below 1. A row value like `Artist=(wlop:1.2), Weight=0.5` acts like a final artist row weight of `0.6`.

**Optimization Preset**

Performance limits the default block range and caps the active artist count lower, so it is cheaper. Balance keeps the reference-friendly default while allowing more artists. Quality keeps the broad block range and raises the artist cap for heavier mixes.

**Combine Mode**

Output average runs each artist branch separately, then averages the artist outputs by weight. This is the most predictable mode and matches the recommended reference path. Token concatenation joins the artist contexts first and sends them through attention together; it is useful for compact mixes, but the artists can interact more strongly.

**Fusion Mode**

Interpolate blends between the base prompt output and the artist branch output. It is the strongest and most reference-compatible default. Concat with base lets attention see base tokens and artist tokens together before blending. Quality-safe delta keeps the base output and adds a norm-limited artist difference, which is steadier but can look much weaker.

Templates are saved with stable internal values, so English/Chinese/raw values can be applied.
""".strip()


def _t(key):
    return TEXT.get(key, key)


def _msg(key, **params):
    """A message of TEXT for a toast: translated by the SPA, then ``params`` filled."""
    return ui.message(_t(key), **params) if params else _t(key)


def _bool_label(value, language="en"):
    if language == "zh":
        return "是" if _to_bool(value, True) else "否"
    return "Yes" if _to_bool(value, True) else "No"


def _option_key(group, value, fallback=None):
    mapping = OPTION_LABELS.get(group, {})
    if fallback is None and mapping:
        fallback = next(iter(mapping))
    if value is None:
        return fallback
    text = str(value).strip()
    if text in mapping:
        return text
    folded = text.casefold()
    for key, labels in mapping.items():
        candidates = [key, labels.get("en", ""), labels.get("zh", "")]
        candidates.extend(labels.get("aliases", ()))
        if folded in {str(item).strip().casefold() for item in candidates if item is not None}:
            return key
    return fallback


def _option_label(group, value, language="en"):
    key = _option_key(group, value)
    labels = OPTION_LABELS.get(group, {}).get(key, {})
    return labels.get(language) or labels.get("en") or str(value or "")


def _option_choices(group):
    """``(English label, internal value)`` pairs of an option group."""
    return [(labels["en"], key) for key, labels in OPTION_LABELS.get(group, {}).items()]


def _header_key(value):
    text = str(value or "").strip()
    if text in TABLE_HEADERS:
        return text
    folded = text.casefold()
    for key, labels in TABLE_HEADER_LABELS.items():
        candidates = [key, labels.get("en", ""), labels.get("zh", ""), key.lower().replace(" ", "_")]
        if folded in {str(item).strip().casefold() for item in candidates if item is not None}:
            return key
    return text


def _row_count(value, fallback=1):
    """A row count, clamped to 1..MAX_ARTIST_ROWS."""
    return max(1, min(MAX_ARTIST_ROWS, int(_to_float(value, fallback))))


def _rows_from_components(*values):
    rows = []
    for index in range(0, len(values), ROW_WIDTH):
        chunk = list(values[index : index + ROW_WIDTH])
        if len(chunk) < ROW_WIDTH:
            break
        rows.append(chunk)
    return rows


def _default_blocks_for_optimization(optimization):
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
    if optimization == OPT_PERFORMANCE:
        return "10-18"
    return "0-51"


def _active_rows_from_components(count, shift, optimization, *values, display=False):
    rows = _rows_from_components(*values)
    count = _row_count(count, len(rows) or 1)
    return _normalize_rows(rows, count, shift, optimization, display=display)


def _runtime_shift_from_processing(p, fallback=3.0, hires=None):
    if getattr(p, "is_hr_pass", False) if hires is None else hires:
        return _clamp(_to_float(getattr(p, "hr_distilled_cfg", None), fallback), 1.0, 24.0)
    return _clamp(_to_float(getattr(p, "distilled_cfg_scale", None), fallback), 1.0, 24.0)


def _to_bool(value, fallback=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "启用", "是", "开"}:
        return True
    if text in {"0", "false", "no", "n", "off", "禁用", "否", "关"}:
        return False
    return fallback


def _split_artist_chain(chain):
    source = str(chain or "").replace("，", ",").replace("\r", "\n").replace("\n", ",")
    return [part.strip() for part in source.split(",") if part.strip()]


def _auto_stage_values(stage, shift_value):
    return stage_window_of(_option_key("stage", stage, PRESET_CUSTOM), shift_value, composition=PRESET_COMPOSITION, style=PRESET_STYLE)


def _default_rows(count=DEFAULT_ARTIST_ROWS, shift=3.0, optimization=OPT_BALANCE):
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
    rows = []
    for i in range(max(0, int(count))):
        blocks = _default_blocks_for_optimization(optimization)
        rows.append([True, "", 1.0, blocks, 0.0, 1.0, 0.5, CURVE_HOLD, PRESET_CUSTOM, False])
    return rows


def _coerce_table(value):
    if value is None:
        return []
    if isinstance(value, dict) and "data" in value:
        value = value.get("data")
    rows = []
    for row in value if isinstance(value, (list, tuple)) else []:
        if isinstance(row, dict):
            keyed = {_header_key(k): v for k, v in row.items()}
            rows.append([keyed.get(h) for h in TABLE_HEADERS])
        elif isinstance(row, (list, tuple)):
            padded = list(row)
            if len(padded) == ROW_WIDTH + 1:
                # Older templates had a per-row Shift column before Auto Shift.
                padded.pop(9)
            padded = padded[: ROW_WIDTH]
            padded += [None] * (ROW_WIDTH - len(padded))
            rows.append(padded)
    return rows


def _normalize_rows(value, row_count=None, shift=3.0, optimization=OPT_BALANCE, display=True):
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
    rows = _coerce_table(value)
    if row_count is None:
        row_count = len(rows) or 1
    count = max(0, min(MAX_ARTIST_ROWS, int(_to_float(row_count, len(rows) or 1))))
    defaults = _default_rows(count, shift, optimization)
    normalized = []
    for index in range(count):
        raw = rows[index] if index < len(rows) else defaults[index]
        raw += [None] * (ROW_WIDTH - len(raw))
        enabled = _to_bool(raw[0], True)
        artist = str(raw[1] or "").strip()
        weight = _clamp(_to_float(raw[2], 1.0), -4.0, 4.0)
        blocks = str(raw[3] or defaults[index][3]).strip() or defaults[index][3]
        start = _clamp(_to_float(raw[4], defaults[index][4]), 0.0, 1.0)
        end = _clamp(_to_float(raw[5], defaults[index][5]), 0.0, 1.0)
        peak = _clamp(_to_float(raw[6], defaults[index][6]), 0.0, 1.0)
        curve = _option_key("curve", raw[7], CURVE_SMOOTH)
        stage = _option_key("stage", raw[8], PRESET_CUSTOM)
        auto = _to_bool(raw[9], True)
        if auto and stage != PRESET_CUSTOM:
            start, end, peak = _auto_stage_values(stage, shift)
        enabled_value = _bool_label(enabled) if display else enabled
        curve_value = _option_label("curve", curve) if display else curve
        stage_value = _option_label("stage", stage) if display else stage
        auto_value = _bool_label(auto) if display else auto
        normalized.append(
            [
                enabled_value,
                artist,
                weight,
                blocks,
                round(start, 4),
                round(end, 4),
                round(peak, 4),
                curve_value,
                stage_value,
                auto_value,
            ]
        )
    return normalized


# region Artist tables (the artist-mix-table widget's value)


def _table(rows):
    """Rows (lists in TABLE_HEADERS order) as the widget's value: one object
    per row keyed by ROW_KEYS."""
    return [dict(zip(ROW_KEYS, row)) for row in rows]


def _rows_of_table(value):
    """A table value's rows as lists in TABLE_HEADERS order, at most
    MAX_ARTIST_ROWS (objects keyed by ROW_KEYS or the headers, or lists)."""
    return _coerce_table(value)[:MAX_ARTIST_ROWS]


def _table_value(value):
    """Validate a table param: a list of at most MAX_ARTIST_ROWS rows, each
    an object or a list; the rows come back as objects keyed by ROW_KEYS
    (values as given: a run normalizes them)."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"an artist table is a list of rows, got {type(value).__name__}")
    if len(value) > MAX_ARTIST_ROWS:
        raise ValueError(f"an artist table has at most {MAX_ARTIST_ROWS} rows, got {len(value)}")
    if not all(isinstance(row, (dict, list)) for row in value):
        raise ValueError("each artist row is an object or a list")
    return _table(_coerce_table(value))


def _normalized_table(rows, count=None, shift=3.0, optimization=OPT_BALANCE):
    """``rows`` normalized (internal values, auto rows timed at ``shift``) as a table value."""
    return _table(_normalize_rows(rows, count, shift, optimization, display=False))


def _retimed_blocks_table(table, shift, optimization):
    """What choosing the optimization preset does to a chain: every row on
    its default Blocks, the auto-shifted stage rows re-timed."""
    blocks = _default_blocks_for_optimization(optimization)
    rows = []
    for row in _rows_of_table(table):
        row = list(row)
        row[3] = blocks
        if _to_bool(row[9], True) and _option_key("stage", row[8], PRESET_CUSTOM) != PRESET_CUSTOM:
            row[4], row[5], row[6] = (round(value, 4) for value in _auto_stage_values(row[8], shift))
        rows.append(row)
    return _table(rows)


# endregion

# region Templates


def _template_data():
    """Templates for display; an unreadable file reads as empty and is left alone."""
    return _TEMPLATES.load()


def _template_choices():
    return sorted(_template_data().keys())


def _resolve_template_name(value=None, choices=None):
    choices = choices if choices is not None else _template_choices()
    value = str(value or "").strip()
    if value in choices:
        return value
    return choices[0] if choices else None


class _Refused(Exception):
    """A refused template edit; the store is left as it was."""


def _edit_templates(edit):
    """Apply ``edit(data)``, which returns ``(template to select, status)``
    or raises ``_Refused(status)``, to the saved templates: ``(template to
    select, status, done)``. An unreadable file is never replaced."""
    try:
        value, status = _TEMPLATES.edit(edit)
    except UnreadableDataFile as error:
        return None, _msg("template_file_unreadable", error=str(error)), False
    except _Refused as refused:
        return refused.args[1] if len(refused.args) > 1 else None, refused.args[0], False
    return value, status, True


def _template_record(chain, values, shift):
    """The template of a chain (``base`` / ``hires``) and the global controls,
    from the panel's named ``values``."""
    optimization = _option_key("optimization", values.get("optimization"), OPT_BALANCE)
    rows = _rows_of_table(values.get(f"{chain}_rows"))
    return {
        f"{chain}_rows": _normalize_rows(rows, _row_count(len(rows) or 1), _to_float(shift, 3.0), optimization, display=False),
        "hires_independent": chain == "hires",
        "disable_hires_mixing": bool(values.get("disable_hires_mixing")),
        "global_strength": _to_float(values.get("global_strength"), 0.7),
        "optimization": optimization,
        "combine_mode": _option_key("combine", values.get("combine_mode"), COMBINE_OUTPUT_AVG),
        "fusion_mode": _option_key("fusion", values.get("fusion_mode"), FUSION_INTERPOLATE),
        "apply_uncond": bool(values.get("apply_uncond")),
        "enable_cache": bool(values.get("enable_cache", True)),
    }


def _save_template(name, chain, values, shift):
    """Save ``chain`` of the panel ``values`` as template ``name``:
    ``(template to select, status, saved)``."""
    name = str(name or "").strip()
    if not name:
        return None, _t("empty_template_name"), False
    if chain == "hires" and not _to_bool(values.get("hires_independent"), False):
        return None, _t("hires_template_disabled"), False
    record = _template_record(chain, values, shift)

    def save(data):
        data[name] = record
        return name, _msg(f"saved_{chain}_template", name=name)

    return _edit_templates(save)


def _rename_template(old_name, new_name):
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()

    def rename(data):
        if not old_name or old_name not in data:
            raise _Refused(_t("no_template_selected"))
        if not new_name:
            raise _Refused(_t("new_template_name_empty"), old_name)
        if new_name != old_name and new_name in data:
            raise _Refused(_msg("template_exists", name=new_name), old_name)
        data[new_name] = data.pop(old_name)
        return new_name, _msg("renamed_template", name=new_name)

    return _edit_templates(rename)


def _delete_template(name):
    name = str(name or "").strip()

    def delete(data):
        if data.pop(name, None) is None:
            raise _Refused(_t("no_template_deleted"))
        return None, _msg("deleted_template", name=name)

    return _edit_templates(delete)


def _applied_template(name, target, values, base_shift, hires_shift):
    """The named values applying template ``name`` to ``target`` (Base,
    Hires or Both) gives the panel ``values``; None for no such template."""
    tpl = _template_data().get(_resolve_template_name(name))
    if not isinstance(tpl, dict):
        return None
    optimization_key = _option_key("optimization", tpl.get("optimization"), OPT_BALANCE)
    has_rows = bool(tpl.get("base_rows")) or bool(tpl.get("hires_rows"))
    optimization = optimization_key if has_rows else _option_key("optimization", values.get("optimization"), OPT_BALANCE)
    target = _option_key("apply_target", target, APPLY_BASE)
    applied = {
        "global_strength": _clamp(_to_float(tpl.get("global_strength", 0.7), 0.7), 0.0, 2.0),
        "optimization": optimization_key,
        "combine_mode": _option_key("combine", tpl.get("combine_mode"), COMBINE_OUTPUT_AVG),
        "fusion_mode": _option_key("fusion", tpl.get("fusion_mode"), FUSION_INTERPOLATE),
        "apply_uncond": _to_bool(tpl.get("apply_uncond"), False),
        "enable_cache": _to_bool(tpl.get("enable_cache"), True),
    }
    if target in {APPLY_BASE, APPLY_BOTH}:
        rows = tpl.get("base_rows") or tpl.get("hires_rows") or _rows_of_table(values.get("base_rows"))
        applied["base_rows"] = _normalized_table(rows, None, base_shift, optimization)
    if target in {APPLY_HIRES, APPLY_BOTH}:
        rows = tpl.get("hires_rows") or tpl.get("base_rows") or _rows_of_table(values.get("hires_rows"))
        applied["hires_rows"] = _normalized_table(rows, None, hires_shift, optimization)
        # The Hires switches are the Hires tab's: a Base template applied to
        # the base rows leaves them as they are
        applied["hires_independent"] = _to_bool(tpl.get("hires_independent"), False)
        applied["disable_hires_mixing"] = _to_bool(tpl.get("disable_hires_mixing"), False)
    return applied


# endregion

# region Panel defaults


def _builtin_defaults():
    """The panel's values on a fresh install, by param name."""
    rows = _normalized_table(_default_rows(DEFAULT_ARTIST_ROWS), DEFAULT_ARTIST_ROWS)
    return {
        "enable": False,
        "hires_independent": False,
        "disable_hires_mixing": False,
        "global_strength": 0.7,
        "optimization": OPT_BALANCE,
        "combine_mode": COMBINE_OUTPUT_AVG,
        "fusion_mode": FUSION_INTERPOLATE,
        "apply_uncond": False,
        "enable_cache": True,
        "base_rows": rows,
        "hires_rows": [dict(row) for row in rows],
        "template": _resolve_template_name(None),
        "template_target": APPLY_BASE,
        "template_name": "",
        "rename_to": "",
    }


def _file_stamp(path):
    """What changes when ``path`` is written (atomic writes replace it)."""
    try:
        stat = path.stat()
    except OSError:
        return str(path), None
    return str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino


# (stamp of the templates file, _builtin_defaults()): a schema build resolves
# every param's default from one computation
_DEFAULTS_CACHE = {}


def _panel_defaults():
    """The panel's defaults (static but for the first template's name).
    Not what the last generation used: a job that leaves the mixer out must
    not inherit someone's artists. The browser keeps the user's panel as a
    draft instead."""
    stamp = _file_stamp(_TEMPLATES.path())
    if _DEFAULTS_CACHE.get("stamp") != stamp:
        _DEFAULTS_CACHE.update(stamp=stamp, values=_builtin_defaults())
    return _DEFAULTS_CACHE["values"]


# endregion


def _artist_entries(rows, shift, optimization, num_blocks, report=False):
    """``(row, clean_name, inline_weight, model_blocks)`` of every artist that
    is encoded: the artists of the enabled rows whose Blocks select some of
    the model's ``num_blocks`` blocks, up to the optimization preset's cap.
    ``report`` logs the artists the cap leaves out."""

    optimization = _option_key("optimization", optimization, OPT_BALANCE)
    entries = []
    for row in _normalize_rows(rows, None, shift, optimization, display=False):
        enabled, artist_text = row[0], row[1]
        if not enabled or not artist_text:
            continue
        blocks = None
        for name in _split_artist_chain(artist_text) or ():
            clean_name, inline_weight = _parse_inline_weight(name)
            if not clean_name:
                continue
            if blocks is None:
                blocks = _model_blocks(row[3], num_blocks)
            if not blocks:  # parse_blocks warned that the row is skipped
                break
            entries.append((row, clean_name, inline_weight, blocks))
    max_artists = _MAX_ARTISTS[optimization]
    if len(entries) > max_artists and report:
        logger.warning("Anima artist mixer active artists %d exceed %d for %s preset; truncating.", len(entries), max_artists, optimization)
    return entries[:max_artists]


def _model_blocks(text, num_blocks):
    """Model blocks of a Blocks field, which is written in the 2B (28-block)
    layout like the presets. On the depth-expanded 2.9B/3.8B models a block
    brings its inserted copies along, so a band covers the same depth of any
    model and "0-51" still means every block. Parsed leniently."""
    return model_block_set(text, num_blocks, parse=_parse_blocks)


def _build_artists(p, entries, use_cache):
    """Encode the ``_artist_entries``. Forge's prompt editing in the prompts
    (``[a:b:N]``, ``[a|b]``) resolves like the main prompt's: each distinct
    batch text of the pass's schedule is encoded once per artist and a model
    call uses the entry of its step."""
    schedule = prompt_schedule(effective_prompt_batch(p)[0], *schedule_steps(p))
    artists = []
    for row, clean_name, inline_weight, blocks in entries:
        _enabled, _artist_text, weight, _blocks_text, start, end, peak, curve, _stage, _auto = row
        encoded = {}
        for _step, prompts in schedule:
            if prompts not in encoded:
                texts = [f"{clean_name}\n{prompt}" if prompt.strip() else clean_name for prompt in prompts]
                encoded[prompts] = _encode_text_batch(p, texts, use_cache=use_cache)
        cond, negpip_mask = encoded[schedule[0][1]]
        edited = len(schedule) > 1
        artists.append(
            ArtistRuntime(
                name=clean_name,
                weight=float(weight) * float(inline_weight),
                blocks=blocks,
                start=_clamp(float(start), 0.0, 1.0),
                end=_clamp(float(end), 0.0, 1.0),
                peak=_clamp(float(peak), 0.0, 1.0),
                curve=_option_key("curve", curve, CURVE_SMOOTH),
                cond=cond,
                negpip_mask=negpip_mask,
                ends=tuple(step for step, _prompts in schedule) if edited else (),
                entries=tuple(encoded[prompts] for _step, prompts in schedule) if edited else (),
            )
        )
    return artists


# Artists an optimization preset encodes and mixes at most
_MAX_ARTISTS = {OPT_PERFORMANCE: 6, OPT_BALANCE: 10, OPT_QUALITY: 16}


def _has_and_prompts(p):
    """Whether a positive prompt of the pass is split into AND sub-prompts."""
    try:
        prompts = effective_prompt_batch(p)[0]
    except RuntimeError:  # _build_artists reports the batch mismatch
        return False
    indexes, _flat, _index = get_multicond_prompt_list(prompts)
    return any(len(subprompts) > 1 for subprompts in indexes)


def _resolve_run_rows(p, args, hires, report=True):
    """``(dm, unet, shift, rows)`` the Mixer runs with (``args``: _MixerArgs)
    on the base or the ``hires`` pass, or None when it does not run there:
    disabled, yielding to FreeFuse/Regional, hires mixing off, an ADetailer
    inner run, no Anima model or an AND prompt. ``report`` notes why in the
    infotext and the log."""
    if args is None or not args.enable:
        return None
    owner = next((feature for feature in (FREEFUSE, REGIONAL) if is_feature_active(p, feature)), None)
    if owner is not None:
        if report:
            message = f"disabled: {owner} has priority in headless processing"
            p.extra_generation_params["Anima Artist Mixer"] = message
            logger.warning("Anima Artist Mixer %s", message)
        return None
    if hires and bool(args.disable_hires_mixing):
        return None
    if getattr(p, "_ad_inner", False):
        return None
    unet = getattr(getattr(p, "sd_model", None), "forge_objects", None)
    unet = getattr(unet, "unet", None)
    if unet is None:
        return None
    dm, msg = _validate_anima_unet(unet)
    if dm is None:
        if report:
            logger.warning("Anima artist mixer disabled: %s", msg)
        return None
    # The artist contexts are encoded per image and tiled over the positive
    # rows. Forge runs each AND sub-prompt as its own cond chunk, and a model
    # call does not say which sub-prompt a chunk is (the chunks come in
    # reverse, and split with the free memory), so every sub-prompt's rows
    # would attend to the whole prompt and undo the AND composition.
    if _has_and_prompts(p):
        if report:
            message = "disabled: AND prompts are not supported"
            p.extra_generation_params["Anima Artist Mixer"] = message
            logger.warning("Anima Artist Mixer %s", message)
        return None
    own_rows = hires and args.hires_independent
    shift = _runtime_shift_from_processing(p, args.runtime_hires_shift if own_rows else args.runtime_base_shift, hires)
    row_count = args.hires_row_count if own_rows else args.base_row_count
    rows = _active_rows_from_components(row_count, shift, args.optimization, *args.row_values(own_rows), display=False)
    return dm, unet, shift, rows


# Infotext keys recording the panel beside the "Anima Artist Mixer" summary,
# each one JSON value in internal (not localized) values:
#   options: the global controls that change the output
#   base, hires: the rows that mix, [artist, weight, blocks, start, end, peak,
#     curve, stage, auto_shift] each; hires only when the Hires pass mixes
#     with rows of its own
INFOTEXT_OPTIONS = "Anima Artist Mixer options"
INFOTEXT_BASE = "Anima Artist Mixer base"
INFOTEXT_HIRES = "Anima Artist Mixer hires"
INFOTEXT_KEYS = (INFOTEXT_OPTIONS, INFOTEXT_BASE, INFOTEXT_HIRES)


def _to_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _infotext_rows(rows):
    """The rows of ``rows`` (normalized, display=False) that mix: the enabled
    rows with an artist, without their Enabled column."""
    return [row[1:] for row in rows if row[0] and row[1]]


def generation_params(p, args) -> dict:
    """Infotext of the panel ``args`` (a _MixerArgs) mixes ``p`` with; keys
    match ``INFOTEXT_KEYS``. Each chain's timings are those of its pass's Shift."""
    options = {
        "strength": _to_float(args.global_strength, 0.7),
        "optimization": _option_key("optimization", args.optimization, OPT_BALANCE),
        "combine": _option_key("combine", args.combine_mode, COMBINE_OUTPUT_AVG),
        "fusion": _option_key("fusion", args.fusion_mode, FUSION_INTERPOLATE),
        "apply_uncond": bool(args.apply_uncond),
        "hires_independent": bool(args.hires_independent),
        "disable_hires_mixing": bool(args.disable_hires_mixing),
    }

    def chain(hires):
        shift = _runtime_shift_from_processing(p, args.runtime_hires_shift if hires else args.runtime_base_shift, hires)
        row_count = args.hires_row_count if hires else args.base_row_count
        return _to_json(_infotext_rows(_active_rows_from_components(row_count, shift, args.optimization, *args.row_values(hires), display=False)))

    params = {INFOTEXT_OPTIONS: _to_json(options), INFOTEXT_BASE: chain(False)}
    if getattr(p, "enable_hr", False) and args.hires_independent and not args.disable_hires_mixing:
        params[INFOTEXT_HIRES] = chain(True)
    return params


def _json_value(text, kind):
    """``text`` decoded from JSON when that is a ``kind``; else None."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, kind) else None


@lru_cache(maxsize=4)
def _paste_plan(options_text, base_text, hires_text, base_shift, hires_shift):
    """Param name -> value restoring the panel an infotext records; empty for
    an infotext without it, e.g. the summary alone earlier versions wrote.
    The text-encoding cache is a preference, not part of the image: never pasted."""
    options = _json_value(options_text, dict)
    chains = {"base": _json_value(base_text, list), "hires": _json_value(hires_text, list)}
    if options is None and chains["base"] is None and chains["hires"] is None:
        return {}
    options = options or {}
    optimization = _option_key("optimization", options.get("optimization"), OPT_BALANCE)
    plan = {"enable": True}
    if "strength" in options:
        plan["global_strength"] = _clamp(_to_float(options["strength"], 0.7), 0.0, 2.0)
    for name, group, fallback in (("optimization", "optimization", OPT_BALANCE), ("combine_mode", "combine", COMBINE_OUTPUT_AVG), ("fusion_mode", "fusion", FUSION_INTERPOLATE)):
        key = name.removesuffix("_mode")
        if key in options:
            plan[name] = _option_key(group, options[key], fallback)
    for name in ("apply_uncond", "hires_independent", "disable_hires_mixing"):
        if name in options:
            plan[name] = _to_bool(options[name])
    for prefix, shift in (("base", base_shift), ("hires", hires_shift)):
        if chains[prefix] is None:
            continue
        rows = [[True, *row] for row in chains[prefix] if isinstance(row, (list, tuple))]
        plan[f"{prefix}_rows"] = _normalized_table(rows, _row_count(len(rows)), shift, optimization)
    return plan


def _pasted_plan(params):
    """:func:`_paste_plan` of the parsed infotext ``params``."""
    base_shift = _to_float(params.get("Distilled CFG Scale"), 3.0)
    hires_shift = _to_float(params.get("Hires Distilled CFG Scale"), base_shift)
    texts = (params.get(key) for key in INFOTEXT_KEYS)
    return _paste_plan(*(text if isinstance(text, str) else None for text in texts), base_shift, hires_shift)


def _pasted_value(name, params):
    """The value pasting ``params`` gives param ``name``; None leaves it."""
    value = _pasted_plan(params).get(name)
    return copy.deepcopy(value) if isinstance(value, list) else value


def _paste(name, key=None):
    """``paste=`` of a param: its value in the recorded panel (``key``: the
    infotext key it is bound to, which also names it as a paste source)."""
    function = partial(_pasted_value, name)
    return {"paste": function} if key is None else {"infotext": key, "paste": PasteKey(key, function)}


# region Layout


STAGE_KINDS = {PRESET_COMPOSITION: "composition", PRESET_CHARACTER: "character", PRESET_STYLE: "style"}
SHIFT_PARAMS = {"base": ["core.distilled_cfg_scale"], "hires": ["core.hr_distilled_cfg", "core.distilled_cfg_scale"]}
STEP_PARAMS = {"base": ["sampler.steps"], "hires": ["core.hr_second_pass_steps", "sampler.steps"]}


def _table_props(chain):
    """Props of a chain's artist-mix-table widget."""
    return {
        "chain": chain,
        "shift": SHIFT_PARAMS[chain],
        "steps": STEP_PARAMS[chain],
        "max_rows": MAX_ARTIST_ROWS,
        "optimization_param": "optimization",
        "default_blocks": {key: _default_blocks_for_optimization(key) for key in OPTION_LABELS["optimization"]},
        "max_artists": dict(_MAX_ARTISTS),
        # A curve's shape as lib_artist_mixer.schedule.curve_factor draws it (front / back keep a quarter)
        "curves": [{"value": key, "label": labels["en"], "shape": _CURVE_SHAPES.get(key, "smooth")} for key, labels in OPTION_LABELS["curve"].items()],
        "curve_floor": 0.25,
        "stages": [{"value": key, "label": labels["en"], **({"kind": STAGE_KINDS[key]} if key in STAGE_KINDS else {})} for key, labels in OPTION_LABELS["stage"].items()],
    }


def _params(tab):
    """The panel of ``tab``."""

    def default(name):
        return lambda: copy.deepcopy(_panel_defaults()[name])

    hires = ui.param("hires_independent")
    return ui.Section(
        "Anima Artist Scheduled Mixer",
        ui.Markdown(INTRO_EN),
        ui.Markdown(COMPATIBILITY_NOTE),
        ui.Row(
            ui.Choice("optimization", _option_choices("optimization"), default("optimization"), label="Optimization preset", preset="artist.optimization", on_input="optimization_changed", **_paste("optimization")),
            ui.Slider("global_strength", 0.0, 2.0, default("global_strength"), 0.01, label="Global artist strength", preset="artist.global_strength", **_paste("global_strength")),
            ui.Toggle("enable_cache", default("enable_cache"), label="Enable text-encoding cache", preset="artist.cache"),
        ),
        ui.Row(
            ui.Choice("combine_mode", _option_choices("combine"), default("combine_mode"), label="Combine mode", preset=("artist.combine", COMBINE_OUTPUT_AVG), **_paste("combine_mode")),
            ui.Choice("fusion_mode", _option_choices("fusion"), default("fusion_mode"), label="Fusion mode", preset=("artist.fusion", FUSION_INTERPOLATE), **_paste("fusion_mode")),
            ui.Toggle("apply_uncond", default("apply_uncond"), label="Apply to unconditional rows", preset="artist.apply_uncond", **_paste("apply_uncond")),
        ),
        ui.Tabs(
            ui.Tab(
                "Base",
                ui.Markdown(SHIFT_HINT),
                ui.Custom("base_rows", "artist-mix-table", default("base_rows"), partial(_table_props, "base"), label="Base artists", validate=_table_value, **_paste("base_rows", INFOTEXT_BASE)),
            ),
            ui.Tab(
                "Hires. fix",
                ui.Toggle("hires_independent", default("hires_independent"), label="Use independent Hires. fix artist chain; disabled = inherit base settings", **_paste("hires_independent")),
                ui.Toggle("disable_hires_mixing", default("disable_hires_mixing"), label="Disable artist mixing during Hires. fix", **_paste("disable_hires_mixing")),
                ui.Markdown(SHIFT_HINT, visible_if=hires),
                ui.Custom("hires_rows", "artist-mix-table", default("hires_rows"), partial(_table_props, "hires"), label="Hires. fix artists", validate=_table_value, visible_if=hires, **_paste("hires_rows", INFOTEXT_HIRES)),
            ),
            ui.Tab(
                "Template",
                ui.Row(
                    _TemplateChoice("template", _template_choices, default("template"), label="Template"),
                    ui.Choice("template_target", _option_choices("apply_target"), default("template_target"), label="Apply target"),
                    ui.Button("Apply template", "template_apply", "primary"),
                ),
                ui.Row(
                    ui.Text("template_name", "", label="Template name"),
                    ui.Button("Save base artist-chain settings", "template_save_base"),
                    ui.Button("Save Hires. fix template", "template_save_hires", enabled_if=hires),
                ),
                ui.Row(
                    ui.Text("rename_to", "", label="Rename to"),
                    ui.Button("Rename template", "template_rename"),
                    ui.Button("Delete template", "template_delete", "danger"),
                ),
                ui.Button("Reset all to defaults", "reset_defaults", "danger"),
            ),
            ui.Tab("Guide", ui.Markdown(HELP_EN)),
        ),
        enable=ui.Toggle("enable", default("enable"), label="Enable artist mixing", preset="artist.enabled", **_paste("enable", INFOTEXT_OPTIONS)),
        exclusive=(SPATIAL_GROUP, "artist"),
    )


class _TemplateChoice(ui.Choice):
    """The template dropdown. Templates are shared by both tabs, and a
    delete or rename refreshes the calling form only: a name that is gone
    reads as the first template (None without any) rather than failing the
    job; the run ignores this control."""

    def coerce(self, value, ctx):
        return super().coerce(_resolve_template_name(value, self.values(ctx)), ctx)


def _shifts(ctx):
    """The generation form's base and Hires Shift (an action's ``ctx.core``)."""

    def number(name, fallback):
        value = ctx.core.get(name)
        return _clamp(float(value), 1.0, 24.0) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback

    base = number("distilled_cfg_scale", 3.0)
    return base, number("hr_distilled_cfg", base)


# endregion


class Script(scripts.Script):
    param_order = PARAM_NAMES

    def title(self):
        return _t("title")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        return _params(self.tabname)

    # region Actions

    @ui.action("optimization_changed")
    def optimization_changed(self, ctx, values):
        """Choosing an optimization preset puts every row on its Blocks and
        re-times the auto-shifted stage rows."""
        shifts = dict(zip(ROW_CHAINS, _shifts(ctx)))
        optimization = values.get("optimization")
        return {f"{chain}_rows": ui.update(value=_retimed_blocks_table(values.get(f"{chain}_rows"), shifts[chain], optimization)) for chain in ROW_CHAINS}

    @ui.action("template_apply")
    def template_apply(self, ctx, values):
        target = _option_key("apply_target", values.get("template_target"), APPLY_BASE)
        applied = _applied_template(values.get("template"), target, values, *_shifts(ctx))
        if applied is None:
            return ui.toast(_t("no_template_selected"), "warning")
        message = _msg(f"applied_template_{target}", name=_resolve_template_name(values.get("template")))
        return {**{name: ui.update(value=value) for name, value in applied.items()}, **ui.toast(message, "success")}

    def _template_edit(self, edit):
        selected, status, done = edit
        choices = _template_choices()
        patch = {"template": ui.update(choices=choices, value=_resolve_template_name(selected, choices))}
        if done:
            patch["template_name"] = ui.update(value="")
            patch["rename_to"] = ui.update(value="")
        return {**patch, **ui.toast(status, "success" if done else "warning")}

    @ui.action("template_save_base")
    def template_save_base(self, ctx, values):
        return self._template_edit(_save_template(values.get("template_name"), "base", values, _shifts(ctx)[0]))

    @ui.action("template_save_hires")
    def template_save_hires(self, ctx, values):
        return self._template_edit(_save_template(values.get("template_name"), "hires", values, _shifts(ctx)[1]))

    @ui.action("template_rename")
    def template_rename(self, ctx, values):
        return self._template_edit(_rename_template(values.get("template"), values.get("rename_to")))

    @ui.action("template_delete")
    def template_delete(self, ctx, values):
        return self._template_edit(_delete_template(values.get("template")))

    @ui.action("reset_defaults")
    def reset_defaults(self, ctx, values):
        """Restore the built-in defaults."""
        defaults = _builtin_defaults()
        return {**{name: ui.update(value=value) for name, value in defaults.items()}, **ui.toast(_t("defaults_restored"), "success")}

    # endregion

    def process(self, p, *args, **kwargs):
        # Runs before NegPiP's process_batch: let a negative weight in an
        # artist name that will actually be encoded enable NegPiP. Same
        # gating and rows as process_before_every_sampling, for each pass.
        mixer_args = _MixerArgs.from_args(args, p)  # a wrong count raises before anything changes
        names = []
        passes = [False]
        if getattr(p, "enable_hr", False) and mixer_args.hires_independent:  # hires rows of its own
            passes.append(True)
        for hires in passes:
            run = _resolve_run_rows(p, mixer_args, hires, report=False)
            if run is None:
                continue
            dm, _unet, shift, rows = run
            names.extend(name for _row, name, _weight, _blocks in _artist_entries(rows, shift, mixer_args.optimization, len(dm.blocks)))
        register_negpip_prompts(p, ARTIST_MIXER, names)

    def process_before_every_sampling(self, p, *args, **kwargs):
        mixer_args = _MixerArgs.from_args(args, p)  # a wrong count raises before anything changes

        _finish_run_states()
        set_feature_active(p, ARTIST_MIXER, False)
        run = _resolve_run_rows(p, mixer_args, getattr(p, "is_hr_pass", False))
        if run is None:
            return
        dm, unet, shift, rows = run
        optimization = _option_key("optimization", mixer_args.optimization, OPT_BALANCE)
        combine_mode = _option_key("combine", mixer_args.combine_mode, COMBINE_OUTPUT_AVG)
        fusion_mode = _option_key("fusion", mixer_args.fusion_mode, FUSION_INTERPOLATE)
        try:
            artists = _build_artists(p, _artist_entries(rows, shift, optimization, len(dm.blocks), report=True), bool(mixer_args.enable_cache))
        except Exception as exc:
            logger.exception("Failed to encode Anima artist mixer artists: %s", exc)
            return
        if not artists:
            return
        strength_limit = 1.0 if fusion_mode == FUSION_QUALITY_DELTA else 2.0
        state = MixerState(
            run_id=uuid.uuid4().hex,
            enabled=True,
            global_strength=_clamp(_to_float(mixer_args.global_strength, 0.7), 0.0, strength_limit),
            combine_mode=combine_mode,
            fusion_mode=fusion_mode,
            apply_uncond=bool(mixer_args.apply_uncond),
            artists=artists,
            infotext=p.extra_generation_params,
            step_of=partial(prompt_step, process_ref(p)),
        )
        # postprocess summarizes the latest pass only: an earlier one (the
        # base pass under hires, a previous batch) frees its device tensors now.
        for previous in _RUN_STATES:
            previous.release_device_tensors()
        _RUN_STATES.clear()
        # forge_objects.shallow_copy() shares the long-lived unet patcher; clone
        # so the wrapper and options stay scoped to this pass.
        unet = unet.clone()
        p.sd_model.forge_objects.unet = unet
        _install_model_wrapper(unet, dm, state)
        _RUN_STATES.append(state)
        set_feature_active(p, ARTIST_MIXER)
        p.extra_generation_params["Anima Artist Mixer"] = (
            f"{len(artists)} artists, strength={state.global_strength:.2f}, "
            f"{state.combine_mode}/{state.fusion_mode}, preset={optimization}, shift={shift:.2f}"
        )
        # Keys are the params' ``infotext`` keys, so pasted parameters restore the panel.
        p.extra_generation_params.update(generation_params(p, mixer_args))

    def postprocess_batch(self, p, *args, **kwargs):
        # Sampling (hires included) is over for this batch. The overrides
        # only exist during model calls, and from here on the model wrapper
        # passes through: postprocess_image (e.g. ADetailer's inner
        # generations without this script) is not mixed. The next batch
        # installs a new wrapper in process_before_every_sampling. The run
        # states stay for postprocess.
        _finish_run_states()

    def postprocess(self, p, processed, *args, **kwargs):
        states = _cleanup_runtime()
        for state in states:
            if not state.dispatch_calls:
                logger.warning(
                    "Anima artist mixer did not receive any cross-attn calls for run %s. "
                    "Patched blocks=%s, wrapper checks=%d. If the image is unchanged, reload UI and test again.",
                    state.run_id,
                    _summarize_blocks(state.patched_blocks),
                    state.wrapper_checks,
                )
            else:
                logger.info(
                    "Anima artist mixer run %s finished: dispatch=%d active=%d fallback=%d patched=%s wrapper_checks=%d strength=%.2f mode=%s/%s",
                    state.run_id,
                    state.dispatch_calls,
                    state.active_calls,
                    state.fallback_calls,
                    _summarize_blocks(state.patched_blocks),
                    state.wrapper_checks,
                    state.global_strength,
                    state.combine_mode,
                    state.fusion_mode,
                )
                logger.info(
                    "Anima artist mixer context path: fusion=%s uses_base_concat=%s",
                    state.fusion_mode,
                    state.fusion_mode in BASE_CONTEXT_FUSIONS,
                )
                if state.diff_probe is not None:
                    logger.debug(
                        "Anima artist mixer diff probe: artist_delta_norm=%.4f artist_delta_ratio=%.4f",
                        state.diff_probe[0],
                        state.diff_probe[1],
                    )
                if state.mask_probe is not None:
                    logger.info(
                        "Anima artist mixer mask probe: cond_or_uncond_len=%d context_batch=%d apply_uncond=%s",
                        state.mask_probe[0],
                        state.mask_probe[1],
                        state.apply_uncond,
                    )

    def cleanup(self, p, *args, **kwargs):
        _cleanup_runtime()
        set_feature_active(p, ARTIST_MIXER, False)
