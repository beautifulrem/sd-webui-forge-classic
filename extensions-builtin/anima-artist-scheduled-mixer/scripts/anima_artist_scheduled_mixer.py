import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import gradio as gr

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
    parse_inline_weight as _parse_inline_weight,
    parse_blocks as _parse_blocks,
)
from modules import script_callbacks, scripts, shared
from modules.anima_feature_conflicts import ARTIST_MIXER, FREEFUSE, REGIONAL, is_feature_active, register_exclusive_component, set_feature_active
from modules.anima_lora_support import clamp as _clamp, model_block_set, stage_window_of, to_float as _to_float
from modules.anima_lora_ui import intro_block, register_language_option, ui_language, ui_text
from modules.anima_user_data import TemplateStore, UnreadableDataFile, data_file, read_json_object, write_json_atomic
from modules.anima_presets import register_preset_control
from modules.anima_support import effective_prompt_batch, process_ref, prompt_schedule, prompt_step, register_negpip_prompts, schedule_steps
from modules.infotext_utils import PasteField
from modules.prompt_parser import get_multicond_prompt_list


logger = logging.getLogger("anima_artist_scheduled_mixer")

EXTENSION_DIR = Path(__file__).resolve().parents[1]
_DATA_NAME = "anima-artist-scheduled-mixer"


# Templates earlier versions kept in the extension folder seed the user copy
_TEMPLATES = TemplateStore(_DATA_NAME, "artist_mixer_templates.json", seed=EXTENSION_DIR / "artist_mixer_templates.json")


_SHARED_SETTINGS_NAME = "artist_mixer_current_settings.json"
SETTINGS_TABS = ("txt2img", "img2img")


def _settings_tab(tab):
    """The tab whose remembered panel ``tab`` (a Script's tabname) reads and writes."""
    return tab if tab in SETTINGS_TABS else SETTINGS_TABS[0]


def _current_settings_file(tab):
    """Each tab remembers its own panel. Earlier versions kept one file for
    both: a tab without its own file yet starts from that one. A reset
    empties the tab's file instead of deleting it, so this happens once."""
    shared = data_file(_DATA_NAME, _SHARED_SETTINGS_NAME, seed=EXTENSION_DIR / _SHARED_SETTINGS_NAME)
    return data_file(_DATA_NAME, f"artist_mixer_current_settings_{_settings_tab(tab)}.json", seed=shared)


REFERENCE_URL = "https://github.com/An1X3R/Anima-Artist-Mixer"

LANGUAGE_OPTION = "anima_artist_scheduled_mixer_language"


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
# Components per artist row; a panel (base or hires) has MAX_ARTIST_ROWS rows
ROW_WIDTH = len(TABLE_HEADERS)


@dataclass(frozen=True)
class _MixerArgs:
    """The script args: the HEADER_ARGS controls ui() returns first, then the
    base and the hires row components (``component_values``)."""

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
    def from_args(cls, args):
        """None when ``args`` does not hold the whole header."""
        if len(args) < len(HEADER_ARGS):
            return None
        return cls(*args[: len(HEADER_ARGS)], component_values=tuple(args[len(HEADER_ARGS) :]))

    def row_values(self, hires):
        """Component values of the base or the ``hires`` rows."""
        size = MAX_ARTIST_ROWS * ROW_WIDTH
        return self.component_values[size : size * 2] if hires else self.component_values[:size]


HEADER_ARGS = tuple(name for name in _MixerArgs.__dataclass_fields__ if name != "component_values")


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


LANG = {
    "zh": {
        "title": "Anima 画师串调度混合",
        "accordion": "Anima 画师串调度混合",
        "settings_label": "Anima 画师串调度混合界面语言",
        "enable": "启用画师串混合",
        "base_tab": "底图",
        "hires_tab": "高分修复",
        "base_panel": "底图画师",
        "hires_panel": "高分辨率修复画师",
        "hires_independent": "高分辨率修复使用独立画师串；关闭时继承底图设置",
        "disable_hires_mixing": "高分辨率修复不启用画师串混合",
        "row_count": "画师行数",
        "hr_row_count": "高分画师行数",
        "shift_runtime_hint": "阶段/自动偏移读取生成参数区当前 Shift；生成时会再次读取实际底图/高分 Shift。",
        "artist_table": "画师设置",
        "global_strength": "全局画师强度",
        "optimization": "优化预设",
        "combine": "组合模式",
        "fusion": "融合模式",
        "apply_uncond": "同时作用于负条件",
        "cache": "启用文本编码缓存",
        "template": "模板",
        "template_name": "模板名称",
        "save_base_template": "保存底图画师串设置",
        "save_hires_template": "保存高分辨率修复模板",
        "hires_template_disabled": "请先开启“高分辨率修复使用独立画师串”，再保存高分辨率修复模板。",
        "rename_to": "重命名为",
        "rename_template": "重命名模板",
        "delete_template": "删除模板",
        "reset_defaults": "全部恢复默认",
        "apply_target": "模板应用目标",
        "apply_template": "应用模板",
        "normalize_rows": "刷新阶段预设",
        "help": "说明",
        "status": "状态",
        "empty_template_name": "模板名称为空。",
        "saved_base_template": "已保存底图画师串模板：{name}",
        "saved_hires_template": "已保存高分辨率修复模板：{name}",
        "no_template_selected": "未选择模板。",
        "new_template_name_empty": "新模板名称为空。",
        "renamed_template": "已重命名模板为：{name}",
        "deleted_template": "已删除模板：{name}",
        "no_template_deleted": "没有删除任何模板。",
        "template_exists": "已存在同名模板：{name}",
        "template_file_unreadable": "模板文件无法读取，未做修改：{error}",
        "applied_template": "已将模板 `{name}` 应用到{target}。",
        "defaults_restored": "已恢复全部默认设置。",
    },
    "en": {
        "title": "Anima Artist Scheduled Mixer",
        "accordion": "Anima Artist Scheduled Mixer",
        "settings_label": "Anima Artist Scheduled Mixer UI language",
        "enable": "Enable artist mixing",
        "base_tab": "Base",
        "hires_tab": "Hires. fix",
        "base_panel": "Base artists",
        "hires_panel": "Hires. fix artists",
        "hires_independent": "Use independent Hires. fix artist chain; disabled = inherit base settings",
        "disable_hires_mixing": "Disable artist mixing during Hires. fix",
        "row_count": "Artist rows",
        "hr_row_count": "Hires artist rows",
        "shift_runtime_hint": "Stage/Auto Shift reads the current Shift from the generation panel; sampling reads the actual Base/Hires Shift again.",
        "artist_table": "Artist settings",
        "global_strength": "Global artist strength",
        "optimization": "Optimization preset",
        "combine": "Combine mode",
        "fusion": "Fusion mode",
        "apply_uncond": "Apply to unconditional rows",
        "cache": "Enable text-encoding cache",
        "template": "Template",
        "template_name": "Template name",
        "save_base_template": "Save base artist-chain settings",
        "save_hires_template": "Save Hires. fix template",
        "hires_template_disabled": "Enable independent Hires. fix artist chain before saving a Hires. fix template.",
        "rename_to": "Rename to",
        "rename_template": "Rename template",
        "delete_template": "Delete template",
        "reset_defaults": "Reset all to defaults",
        "apply_target": "Apply target",
        "apply_template": "Apply template",
        "normalize_rows": "Refresh stage presets",
        "help": "Guide",
        "status": "Status",
        "empty_template_name": "Template name is empty.",
        "saved_base_template": "Saved base artist-chain template: {name}",
        "saved_hires_template": "Saved Hires. fix template: {name}",
        "no_template_selected": "No template selected.",
        "new_template_name_empty": "New template name is empty.",
        "renamed_template": "Renamed template to: {name}",
        "deleted_template": "Deleted template: {name}",
        "no_template_deleted": "No template deleted.",
        "template_exists": "A template with that name already exists: {name}",
        "template_file_unreadable": "The template file cannot be read; nothing was changed: {error}",
        "applied_template": "Applied template `{name}` to {target}.",
        "defaults_restored": "Restored all settings to defaults.",
    },
}


INTRO_EN = f"""
Independent artist encoding and scheduled cross-attention mixing for Anima.
Each artist row is encoded separately as `artist + base prompt`, then mixed inside Anima cross-attention. The default `Output average + Interpolate` path follows the stronger reference behavior from [{REFERENCE_URL}]({REFERENCE_URL}); `Quality-safe delta` is still available when you want a more conservative Forge-friendly blend.

The option labels follow the selected UI language, while templates keep stable internal values. Old templates that stored English, Chinese, or raw values should still load correctly after switching languages.

Thanks to **An1X3R/Anima-Artist-Mixer** and **汐浮尘/utowo** for the original split-and-encode/cross-attention design.
"""

INTRO_ZH = f"""
面向 Anima 的独立画师编码与按阶段 cross-attention 混合。每个画师会单独按 `画师 + 主提示词` 编码，再在 Anima 的 cross-attention 内混合。默认的“输出平均 + 插值融合”沿用 [{REFERENCE_URL}]({REFERENCE_URL}) 的强混合路线；需要更保守的 Forge 友好混合时，也可以手动切到“保真增量”。

面板选项会跟随界面语言显示，模板内部使用稳定值保存。旧模板即使保存过英文、中文或原始值，切换语言后也会尽量自动识别并回显。

感谢 **An1X3R/Anima-Artist-Mixer** 与 **汐浮尘/utowo** 的原始独立编码和 cross-attention 混合设计。
"""

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

The UI localizes option labels. Templates are saved with stable internal values, so English/Chinese/raw values can be applied across language modes.
"""

HELP_ZH = """
**列说明**

`画师` 每行建议填一个画师提示词。可以直接写行内倍率，例如 `(wlop:1.2)`、`[wlop:0.8]` 或 `wlop:1.2`；最终强度是 `权重 x 行内倍率`。为了迁移旧画师串，逗号分隔仍可用，但每行一个画师最方便单独控制。

`权重` 控制该画师的相对比例，也会在总权重低于 1 时降低实际介入。`层数` 支持 `0-51`、`0,3,5-12`，也支持 `-1` 这种倒数索引，始终按 28 层（2B）布局填写：在 2.9B/3.8B 模型上每层也覆盖它复制插入的层，大于 27 的索引会被忽略，所以 `0-51` 即全部层。`开始` 和 `结束` 是 0 到 1 的采样进度：按采样步数计的比例（高分修复和图生图从中途开始），不是按 sigma 计。`峰值` 表示这一行在窗口内达到满强度的位置，对平滑和三角峰曲线最明显。

`曲线` 决定窗口内强度怎样变化。保持会在整个窗口内满强度。平滑会围绕峰值淡入淡出。三角峰会线性升到峰值再线性降下。前段强化是一开始更强、后面变弱；后段强化则相反。

`阶段` 是时间预设。自定义会保留你手动填写的值。构图偏前段，人物偏中段，画风偏后段。选择自定义之外的阶段时，界面会立即刷新开始、结束和峰值，并为这一行打开自动偏移。自动偏移会跟随 Forge 生成参数区的 Shift；如果之后想微调数字，再把自动偏移关掉即可。

**强度**

全局画师强度用于控制画师分支往原始提示词 cross-attention 输出里注入多少。单行权重决定画师占比；当活跃权重总和低于 1 时，也会降低整体介入。例如 `画师=(wlop:1.2)，权重=0.5`，最终该行画师权重相当于 `0.6`。

**优化预设**

性能会收窄默认层数范围，并降低可同时参与的画师上限，生成更省。平衡保留更接近参考插件的默认范围，同时允许较多画师。质量保留宽层数范围，并提高画师上限，适合更重的混合。

**组合模式**

输出平均会让每个画师分支分别跑 attention，再按权重平均输出，是最稳定、也最接近参考推荐的方式。token拼接会先把多个画师上下文拼在一起再送入 attention，适合更紧凑的混合，但画师之间会更容易互相影响。

**融合模式**

插值融合会在主提示词输出和画师分支输出之间做插值，是最强、也最接近参考插件默认效果的方式。拼接底图条件会先让 attention 同时看到底图 token 和画师 token，再进行混合。保真增量会保留底图输出，只加入有范数限制的画师差值，更稳但也可能明显偏弱。

面板会按语言汉化选项。模板保存稳定内部值，因此中文、英文或旧 raw 值模板都可以跨语言应用。
"""


def _language():
    return ui_language(LANGUAGE_OPTION)


def _t(key):
    return ui_text(LANG, LANGUAGE_OPTION, key)


def _bool_label(value, language=None):
    language = language or _language()
    return ("是" if _to_bool(value, True) else "否") if language == "zh" else ("Yes" if _to_bool(value, True) else "No")


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


def _option_label(group, value, language=None):
    language = language or _language()
    key = _option_key(group, value)
    labels = OPTION_LABELS.get(group, {}).get(key, {})
    return labels.get(language) or labels.get("en") or str(value or "")


def _option_choices(group, language=None):
    language = language or _language()
    return [_option_label(group, key, language) for key in OPTION_LABELS.get(group, {})]


def _row_component_headers(language=None):
    language = language or _language()
    headers = TABLE_HEADER_LABELS
    return {
        "enabled": headers["Enabled"].get(language, "Enabled"),
        "artist": headers["Artist"].get(language, "Artist"),
        "weight": headers["Weight"].get(language, "Weight"),
        "blocks": headers["Blocks"].get(language, "Blocks"),
        "start": headers["Start"].get(language, "Start"),
        "end": headers["End"].get(language, "End"),
        "peak": headers["Peak"].get(language, "Peak"),
        "curve": headers["Curve"].get(language, "Curve"),
        "stage": headers["Stage"].get(language, "Stage"),
        "auto_shift": headers["Auto Shift"].get(language, "Auto Shift"),
    }


def _header_key(value):
    text = str(value or "").strip()
    if text in TABLE_HEADERS:
        return text
    folded = text.casefold()
    for key, labels in TABLE_HEADER_LABELS.items():
        candidates = [key, labels.get("en", ""), labels.get("zh", "")]
        if folded in {str(item).strip().casefold() for item in candidates if item is not None}:
            return key
    return text


def _row_count(value, fallback=1):
    """A row count from the UI, clamped to 1..MAX_ARTIST_ROWS."""
    return max(1, min(MAX_ARTIST_ROWS, int(_to_float(value, fallback))))


def _rows_from_components(*values):
    rows = []
    for index in range(0, len(values), ROW_WIDTH):
        chunk = list(values[index : index + ROW_WIDTH])
        if len(chunk) < ROW_WIDTH:
            break
        rows.append(chunk)
    return rows


def _component_values_from_rows(rows, shift=3.0, optimization=OPT_BALANCE, count=None, columns=None):
    """Updates of every row component: all columns, showing the first ``count``
    rows and hiding the rest; or only ``columns`` of the shown rows."""
    normalized = _normalize_rows(rows, MAX_ARTIST_ROWS, shift, optimization, display=True)
    if count is None:
        count = len(_coerce_table(rows)) or 1
    count = _row_count(count)
    updates = []
    for index, row in enumerate(normalized):
        if columns is None:
            updates.extend(gr.update(value=value, visible=index < count, interactive=True) for value in row)
        elif index < count:
            updates.extend(gr.update(value=row[column], interactive=True) for column in columns)
        else:
            updates.extend(gr.update() for _ in columns)
    return updates


def _resize_row_components(count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = _row_count(count, len(rows) or 1)
    return _component_values_from_rows(rows, shift, optimization, count)


def _apply_shift_to_timing_components(count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = _row_count(count, len(rows) or 1)
    return _component_values_from_rows(rows, shift, optimization, count, columns=(4, 5, 6))


def _default_blocks_for_optimization(optimization):
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
    if optimization == OPT_PERFORMANCE:
        return "10-18"
    return "0-51"


def _apply_optimization_to_block_timing_components(count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = _row_count(count, len(rows) or 1)
    blocks = _default_blocks_for_optimization(optimization)
    for index in range(min(count, len(rows))):
        row = list(rows[index])
        row += [None] * (ROW_WIDTH - len(row))
        row[3] = blocks
        if _to_bool(row[9], True) and _option_key("stage", row[8], PRESET_CUSTOM) != PRESET_CUSTOM:
            row[4], row[5], row[6] = _auto_stage_values(row[8], shift)
        rows[index] = row
    return _component_values_from_rows(rows, shift, optimization, count, columns=(3, 4, 5, 6))


def _apply_stage_to_row_components(row_index, count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = _row_count(count, len(rows) or 1)
    if 0 <= row_index < len(rows):
        row = list(rows[row_index])
        row += [None] * (ROW_WIDTH - len(row))
        stage = _option_key("stage", row[8], PRESET_CUSTOM)
        if stage == PRESET_CUSTOM:
            row[9] = False
        else:
            start, end, peak = _auto_stage_values(stage, shift)
            row[4], row[5], row[6] = start, end, peak
            row[9] = True
        normalized = _normalize_rows([row], 1, shift, optimization, display=True)[0]
        return [
            gr.update(value=normalized[4], interactive=True),
            gr.update(value=normalized[5], interactive=True),
            gr.update(value=normalized[6], interactive=True),
            gr.update(value=normalized[9], interactive=True),
        ]
    return [gr.update(), gr.update(), gr.update(), gr.update()]


def _apply_auto_shift_to_row_components(row_index, count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = _row_count(count, len(rows) or 1)
    if not (0 <= row_index < min(count, len(rows))):
        return [gr.update(), gr.update(), gr.update()]
    row = list(rows[row_index])
    row += [None] * (ROW_WIDTH - len(row))
    normalized = _normalize_rows([row], 1, shift, optimization, display=True)[0]
    return [
        gr.update(value=normalized[4], interactive=True),
        gr.update(value=normalized[5], interactive=True),
        gr.update(value=normalized[6], interactive=True),
    ]


def _active_rows_from_components(count, shift, optimization, *values, display=False):
    rows = _rows_from_components(*values)
    count = _row_count(count, len(rows) or 1)
    return _normalize_rows(rows, count, shift, optimization, display=display)


def _runtime_shift_from_processing(p, fallback=3.0, hires=None):
    if getattr(p, "is_hr_pass", False) if hires is None else hires:
        return _clamp(_to_float(getattr(p, "hr_distilled_cfg", None), fallback), 1.0, 24.0)
    return _clamp(_to_float(getattr(p, "distilled_cfg_scale", None), fallback), 1.0, 24.0)


def _create_artist_row_controls(prefix, lang, defaults, elem_id_func):
    labels = _row_component_headers(lang)
    components = []
    bool_choices = [_bool_label(True, lang), _bool_label(False, lang)]
    curve_choices = _option_choices("curve", lang)
    stage_choices = _option_choices("stage", lang)
    for index in range(MAX_ARTIST_ROWS):
        row = defaults[index] if index < len(defaults) else _normalize_rows([_default_rows(1, 3.0, OPT_BALANCE)[0]], 1, 3.0, OPT_BALANCE)[0]
        visible = index < len(defaults)
        with gr.Row(elem_id=elem_id_func(f"{prefix}_artist_row_{index}")):
            enabled = gr.Dropdown(label=labels["enabled"], choices=bool_choices, value=row[0], allow_custom_value=False, min_width=82, visible=visible, elem_id=elem_id_func(f"{prefix}_enabled_{index}"))
            artist = gr.Textbox(label=labels["artist"], value=row[1], lines=1, min_width=190, visible=visible, elem_id=elem_id_func(f"{prefix}_artist_{index}"))
            weight = gr.Number(label=labels["weight"], value=row[2], precision=4, min_width=92, visible=visible, elem_id=elem_id_func(f"{prefix}_weight_{index}"))
            blocks = gr.Textbox(label=labels["blocks"], value=row[3], lines=1, min_width=116, visible=visible, elem_id=elem_id_func(f"{prefix}_blocks_{index}"))
            start = gr.Number(label=labels["start"], value=row[4], precision=4, min_width=86, visible=visible, elem_id=elem_id_func(f"{prefix}_start_{index}"))
            end = gr.Number(label=labels["end"], value=row[5], precision=4, min_width=86, visible=visible, elem_id=elem_id_func(f"{prefix}_end_{index}"))
            peak = gr.Number(label=labels["peak"], value=row[6], precision=4, min_width=86, visible=visible, elem_id=elem_id_func(f"{prefix}_peak_{index}"))
            curve = gr.Dropdown(label=labels["curve"], choices=curve_choices, value=row[7], allow_custom_value=False, min_width=130, visible=visible, elem_id=elem_id_func(f"{prefix}_curve_{index}"))
            stage = gr.Dropdown(label=labels["stage"], choices=stage_choices, value=row[8], allow_custom_value=False, min_width=130, visible=visible, elem_id=elem_id_func(f"{prefix}_stage_{index}"))
            auto_shift = gr.Dropdown(label=labels["auto_shift"], choices=bool_choices, value=row[9], allow_custom_value=False, min_width=112, visible=visible, elem_id=elem_id_func(f"{prefix}_auto_shift_{index}"))
        components.extend([enabled, artist, weight, blocks, start, end, peak, curve, stage, auto_shift])
    return components


def _row_outputs(row_components, offsets):
    return [row_components[index + offset] for index in range(0, len(row_components), ROW_WIDTH) for offset in offsets]


def _register_ui_settings():
    register_language_option(
        LANGUAGE_OPTION,
        "Anima Artist Scheduled Mixer language / Anima 画师串调度混合语言",
        ("anima-artist-scheduled-mixer", "Anima Artist Scheduled Mixer"),
    )


script_callbacks.on_ui_settings(_register_ui_settings)


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


def _template_data():
    """Templates for display; an unreadable file reads as empty and is left alone."""
    return _TEMPLATES.load()


def _edit_templates(edit):
    """Apply ``edit(data)``, which returns ``(template to select, status)``, to
    the saved templates; then the dropdown (built from the file as written)
    and the status. An unreadable file is never replaced."""
    try:
        value, status = _TEMPLATES.edit(edit)
    except UnreadableDataFile as error:
        return _template_dropdown_update(), _t("template_file_unreadable").format(error=error)
    return _template_dropdown_update(value), status


def _template_choices():
    return sorted(_template_data().keys())


def _resolve_template_name(value=None, choices=None):
    choices = choices if choices is not None else _template_choices()
    value = str(value or "").strip()
    if value in choices:
        return value
    return choices[0] if choices else None


def _template_dropdown_update(value=None):
    choices = _template_choices()
    value = _resolve_template_name(value, choices)
    return gr.update(choices=choices, value=value)


def _save_template_record(
    name,
    row_key,
    rows,
    row_count,
    shift,
    disable_hires_mixing,
    global_strength,
    optimization,
    combine_mode,
    fusion_mode,
    apply_uncond,
    enable_cache,
    status_key,
):
    name = str(name or "").strip()
    if not name:
        return _template_dropdown_update(), _t("empty_template_name")
    optimization_key = _option_key("optimization", optimization, OPT_BALANCE)
    combine_key = _option_key("combine", combine_mode, COMBINE_OUTPUT_AVG)
    fusion_key = _option_key("fusion", fusion_mode, FUSION_INTERPOLATE)
    row_count = _row_count(row_count, len(_coerce_table(rows)) or 1)
    shift = _to_float(shift, 3.0)
    record = {
        row_key: _normalize_rows(rows, row_count, shift, optimization_key, display=False),
        "hires_independent": row_key == "hires_rows",
        "disable_hires_mixing": bool(disable_hires_mixing),
        "global_strength": _to_float(global_strength, 0.7),
        "optimization": optimization_key,
        "combine_mode": combine_key,
        "fusion_mode": fusion_key,
        "apply_uncond": bool(apply_uncond),
        "enable_cache": bool(enable_cache),
    }

    def save(data):
        data[name] = record
        return name, _t(status_key).format(name=name)

    return _edit_templates(save)


def _save_base_template_ui(name, base_row_count, base_shift, disable_hires_mixing, global_strength, optimization, combine_mode, fusion_mode, apply_uncond, enable_cache, *base_values):
    base_rows = _rows_from_components(*base_values)
    return _save_template_record(
        name,
        "base_rows",
        base_rows,
        base_row_count,
        base_shift,
        disable_hires_mixing,
        global_strength,
        optimization,
        combine_mode,
        fusion_mode,
        apply_uncond,
        enable_cache,
        "saved_base_template",
    )


def _save_hires_template_ui(name, hires_independent, hires_row_count, hires_shift, disable_hires_mixing, global_strength, optimization, combine_mode, fusion_mode, apply_uncond, enable_cache, *hires_values):
    if not _to_bool(hires_independent, False):
        return _template_dropdown_update(), _t("hires_template_disabled")
    hires_rows = _rows_from_components(*hires_values)
    return _save_template_record(
        name,
        "hires_rows",
        hires_rows,
        hires_row_count,
        hires_shift,
        disable_hires_mixing,
        global_strength,
        optimization,
        combine_mode,
        fusion_mode,
        apply_uncond,
        enable_cache,
        "saved_hires_template",
    )


def _rename_template_ui(old_name, new_name):
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()

    def rename(data):
        if not old_name or old_name not in data:
            return None, _t("no_template_selected")
        if not new_name:
            return old_name, _t("new_template_name_empty")
        if new_name != old_name and new_name in data:
            return old_name, _t("template_exists").format(name=new_name)
        data[new_name] = data.pop(old_name)
        return new_name, _t("renamed_template").format(name=new_name)

    return _edit_templates(rename)


def _delete_template_ui(name):
    name = str(name or "").strip()

    def delete(data):
        if data.pop(name, None) is None:
            return None, _t("no_template_deleted")
        return None, _t("deleted_template").format(name=name)

    return _edit_templates(delete)


def _builtin_default_ui_state(lang):
    enable = False
    base_shift = 3.0
    hires_shift = 3.0
    base_count = DEFAULT_ARTIST_ROWS
    hires_count = DEFAULT_ARTIST_ROWS
    optimization_key = OPT_BALANCE
    combine_key = COMBINE_OUTPUT_AVG
    fusion_key = FUSION_INTERPOLATE
    apply_target_key = APPLY_BASE
    global_strength = 0.7
    hires_independent = False
    disable_hires_mixing = False
    apply_uncond = False
    enable_cache = True
    template_name = None

    base_rows = _normalize_rows(_default_rows(base_count, base_shift, optimization_key), base_count, base_shift, optimization_key)
    hires_rows = _normalize_rows(_default_rows(hires_count, hires_shift, optimization_key), hires_count, hires_shift, optimization_key)

    return {
        "enable": enable,
        "template_name": template_name,
        "template_target": _option_label("apply_target", apply_target_key, lang),
        "base_count": base_count,
        "hires_count": hires_count,
        "base_rows": base_rows,
        "hires_rows": hires_rows,
        "hires_independent": hires_independent,
        "disable_hires_mixing": disable_hires_mixing,
        "global_strength": global_strength,
        "optimization": _option_label("optimization", optimization_key, lang),
        "combine": _option_label("combine", combine_key, lang),
        "fusion": _option_label("fusion", fusion_key, lang),
        "apply_uncond": apply_uncond,
        "enable_cache": enable_cache,
    }


def _current_settings_data(tab):
    try:
        return read_json_object(_current_settings_file(tab)) or None
    except UnreadableDataFile:
        logger.exception("Failed to load Anima artist mixer current settings")
        return None


# Per tab, the payload last written to its current-settings file (missing:
# unknown / reset), so generations with an unchanged panel skip the write and its fsync.
_LAST_REMEMBERED_SETTINGS = {}


def _save_current_settings_data(tab, data):
    tab = _settings_tab(tab)
    try:
        # An interrupted or concurrent write must not leave truncated JSON
        # that silently resets the panel.
        write_json_atomic(_current_settings_file(tab), data)
        _LAST_REMEMBERED_SETTINGS[tab] = data
    except Exception:
        logger.exception("Failed to save Anima artist mixer current settings")


def _delete_current_settings_data(tab):
    tab = _settings_tab(tab)
    _LAST_REMEMBERED_SETTINGS.pop(tab, None)
    try:
        # Emptied rather than deleted: a missing file would be seeded again
        # from the settings earlier versions shared between the tabs.
        write_json_atomic(_current_settings_file(tab), {})
    except Exception:
        logger.exception("Failed to delete Anima artist mixer current settings")


def _state_from_current_settings(data, lang):
    if not isinstance(data, dict):
        return None
    optimization_key = _option_key("optimization", data.get("optimization"), OPT_BALANCE)
    combine_key = _option_key("combine", data.get("combine_mode"), COMBINE_OUTPUT_AVG)
    fusion_key = _option_key("fusion", data.get("fusion_mode"), FUSION_INTERPOLATE)
    apply_target_key = _option_key("apply_target", data.get("template_target"), APPLY_BASE)
    base_shift = _to_float(data.get("base_shift"), 3.0)
    hires_shift = _to_float(data.get("hires_shift"), 3.0)
    base_count = _row_count(data.get("base_count"), DEFAULT_ARTIST_ROWS)
    hires_count = _row_count(data.get("hires_count"), DEFAULT_ARTIST_ROWS)
    base_rows = _normalize_rows(data.get("base_rows"), base_count, base_shift, optimization_key)
    hires_rows = _normalize_rows(data.get("hires_rows"), hires_count, hires_shift, optimization_key)
    template_name = str(data.get("template_name") or "").strip()
    if template_name and template_name not in _template_data():
        template_name = None
    return {
        "enable": bool(data.get("enable", False)),
        "template_name": template_name,
        "template_target": _option_label("apply_target", apply_target_key, lang),
        "base_count": base_count,
        "hires_count": hires_count,
        "base_rows": base_rows,
        "hires_rows": hires_rows,
        "hires_independent": bool(data.get("hires_independent", False)),
        "disable_hires_mixing": bool(data.get("disable_hires_mixing", False)),
        "global_strength": _to_float(data.get("global_strength"), 0.7),
        "optimization": _option_label("optimization", optimization_key, lang),
        "combine": _option_label("combine", combine_key, lang),
        "fusion": _option_label("fusion", fusion_key, lang),
        "apply_uncond": bool(data.get("apply_uncond", False)),
        "enable_cache": bool(data.get("enable_cache", True)),
    }


def _initial_ui_defaults(tab, lang):
    current = _state_from_current_settings(_current_settings_data(tab), lang)
    return current if current is not None else _builtin_default_ui_state(lang)


def _current_settings_payload(
    enable,
    template_name,
    template_target,
    base_row_count,
    hires_row_count,
    hires_independent,
    disable_hires_mixing,
    base_shift,
    hires_shift,
    global_strength,
    optimization,
    combine_mode,
    fusion_mode,
    apply_uncond,
    enable_cache,
    *component_values,
):
    optimization_key = _option_key("optimization", optimization, OPT_BALANCE)
    base_values = component_values[: MAX_ARTIST_ROWS * ROW_WIDTH]
    hires_values = component_values[MAX_ARTIST_ROWS * ROW_WIDTH : MAX_ARTIST_ROWS * ROW_WIDTH * 2]
    base_shift = _to_float(base_shift, 3.0)
    hires_shift = _to_float(hires_shift, 3.0)
    base_count = _row_count(base_row_count, DEFAULT_ARTIST_ROWS)
    hires_count = _row_count(hires_row_count, DEFAULT_ARTIST_ROWS)
    return {
        "version": 1,
        "enable": bool(enable),
        "template_name": str(template_name or "").strip(),
        "template_target": _option_key("apply_target", template_target, APPLY_BASE),
        "base_count": base_count,
        "hires_count": hires_count,
        "base_shift": base_shift,
        "hires_shift": hires_shift,
        "base_rows": _normalize_rows(_rows_from_components(*base_values), base_count, base_shift, optimization_key, display=False),
        "hires_rows": _normalize_rows(_rows_from_components(*hires_values), hires_count, hires_shift, optimization_key, display=False),
        "hires_independent": bool(hires_independent),
        "disable_hires_mixing": bool(disable_hires_mixing),
        "global_strength": _to_float(global_strength, 0.7),
        "optimization": optimization_key,
        "combine_mode": _option_key("combine", combine_mode, COMBINE_OUTPUT_AVG),
        "fusion_mode": _option_key("fusion", fusion_mode, FUSION_INTERPOLATE),
        "apply_uncond": bool(apply_uncond),
        "enable_cache": bool(enable_cache),
    }


def _save_current_settings_ui(tab, *values):
    payload = _current_settings_payload(*values)
    if payload != _LAST_REMEMBERED_SETTINGS.get(_settings_tab(tab)):
        _save_current_settings_data(tab, payload)


def _same_value(value):
    return value


def _should_remember_settings(p) -> bool:
    """Remember the panel as the user last generated with: once per
    processing run from the UI, not for API calls, hires passes or later
    X/Y/Z cells (each would overwrite the saved panel with that run's args)."""
    if getattr(p, "is_api", False) or getattr(p, "is_hr_pass", False):
        return False
    if getattr(p, "_anima_artist_settings_remembered", False):
        return False
    p._anima_artist_settings_remembered = True
    state = shared.state
    if int(getattr(state, "job_count", 0) or 0) > 1 and int(getattr(state, "job_no", 0) or 0) > 0:
        return False  # a later job of a multi-job run (X/Y/Z grid cells)
    return True


def _save_runtime_current_settings(tab, args):
    """Remember ``args`` (a generation's _MixerArgs) as ``tab``'s panel."""
    tab = _settings_tab(tab)
    try:
        payload = _current_settings_payload(
            args.enable,
            "",
            APPLY_BASE,
            args.base_row_count,
            args.hires_row_count,
            args.hires_independent,
            args.disable_hires_mixing,
            args.runtime_base_shift,
            args.runtime_hires_shift,
            args.global_strength,
            args.optimization,
            args.combine_mode,
            args.fusion_mode,
            args.apply_uncond,
            args.enable_cache,
            *args.component_values,
        )
        # A generation does not know the template dropdown: keep the one the
        # user saved with "Save current settings".
        last = _LAST_REMEMBERED_SETTINGS.get(tab)
        saved = last or _current_settings_data(tab)
        if isinstance(saved, dict):
            for field_name in ("template_name", "template_target"):
                if field_name in saved:
                    payload[field_name] = saved[field_name]
        # Unchanged since the last successful save: skip the write.
        if payload == last:
            return
        _save_current_settings_data(tab, payload)
    except Exception:
        logger.exception("Failed to save Anima artist mixer runtime settings")


def _do_not_save_to_ui_config(*components):
    for component in components:
        setattr(component, "do_not_save_to_config", True)


def _autosave_triggers(components):
    """Events that save the panel: a textbox's .change fires per keystroke,
    so it saves when its edit is done (blur, Enter). The other controls
    change in discrete steps: a Number field's few characters (Gradio 4.40
    has no Number.blur) and the timings the Shift and Stage handlers write."""
    triggers = []
    for component in components:
        if isinstance(component, gr.Textbox):
            triggers.extend((component.blur, component.submit))
        else:
            triggers.append(component.change)
    return triggers


def _internal_event_kwargs():
    return {"api_name": False, "show_api": False}


def _default_state_updates(lang=None, status_text=None):
    lang = lang or _language()
    state = _builtin_default_ui_state(lang)
    base_updates = _component_values_from_rows(state["base_rows"], 3.0, state["optimization"], state["base_count"])
    hires_updates = _component_values_from_rows(state["hires_rows"], 3.0, state["optimization"], state["hires_count"])
    return (
        gr.update(value=state["enable"], interactive=True),
        _template_dropdown_update(state["template_name"]),
        gr.update(value=state["template_target"]),
        gr.update(value=state["base_count"], interactive=True),
        gr.update(value=state["hires_count"], interactive=True),
        gr.update(value=state["hires_independent"], interactive=True),
        gr.update(value=state["disable_hires_mixing"], interactive=True),
        gr.update(),  # the Shift copies follow the generation panel
        gr.update(),
        gr.update(value=state["global_strength"], interactive=True),
        gr.update(value=state["optimization"], interactive=True),
        gr.update(value=state["combine"], interactive=True),
        gr.update(value=state["fusion"], interactive=True),
        gr.update(value=state["apply_uncond"], interactive=True),
        gr.update(value=state["enable_cache"], interactive=True),
        _t("defaults_restored") if status_text is None else status_text,
        *base_updates,
        *hires_updates,
    )


def _reset_all_defaults_ui(tab):
    _delete_current_settings_data(tab)
    return _default_state_updates()


def _apply_template_updates(name, target, base_row_count, hires_row_count, current_base_shift, current_hires_shift, current_optimization, component_values, status_text=None):
    base_values = component_values[: MAX_ARTIST_ROWS * ROW_WIDTH]
    hires_values = component_values[MAX_ARTIST_ROWS * ROW_WIDTH : MAX_ARTIST_ROWS * ROW_WIDTH * 2]
    base_current = _rows_from_components(*base_values)
    hires_current = _rows_from_components(*hires_values)
    no_selection = (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        _t("no_template_selected"),
        *[gr.update() for _ in range(MAX_ARTIST_ROWS * ROW_WIDTH * 2)],
    )
    data = _template_data()
    name = _resolve_template_name(name, sorted(data.keys()))
    tpl = data.get(name)
    if not isinstance(tpl, dict):
        return no_selection
    optimization_key = _option_key("optimization", tpl.get("optimization"), OPT_BALANCE)
    base_shift = _to_float(current_base_shift, _to_float(tpl.get("base_shift", 3.0), 3.0))
    hires_shift = _to_float(current_hires_shift, _to_float(tpl.get("hires_shift", base_shift), base_shift))
    has_base_rows = bool(tpl.get("base_rows"))
    has_hires_rows = bool(tpl.get("hires_rows"))
    fallback_optimization = _option_key("optimization", current_optimization, OPT_BALANCE)
    base_tpl = _normalize_rows(
        tpl.get("base_rows") or tpl.get("hires_rows") or base_current,
        None,
        base_shift,
        optimization_key if has_base_rows or has_hires_rows else fallback_optimization,
    )
    hires_tpl = _normalize_rows(
        tpl.get("hires_rows") or tpl.get("base_rows") or hires_current,
        None,
        hires_shift,
        optimization_key if has_hires_rows or has_base_rows else fallback_optimization,
    )
    target = _option_key("apply_target", target, APPLY_BASE)
    base_count_value = max(1, len(base_tpl)) if target in {APPLY_BASE, APPLY_BOTH} else max(1, int(_to_float(base_row_count, 1)))
    hires_count_value = max(1, len(hires_tpl)) if target in {APPLY_HIRES, APPLY_BOTH} else max(1, int(_to_float(hires_row_count, 1)))
    base_updates = [gr.update() for _ in range(MAX_ARTIST_ROWS * ROW_WIDTH)]
    hires_updates = [gr.update() for _ in range(MAX_ARTIST_ROWS * ROW_WIDTH)]
    # The Hires switches are the Hires tab's: a Base template (saved with
    # hires_independent off) applied to the base rows leaves them as they are.
    hires_switches = (gr.update(), gr.update())
    if target in {APPLY_HIRES, APPLY_BOTH}:
        hires_switches = (
            gr.update(value=tpl.get("hires_independent", False), interactive=True),
            gr.update(value=tpl.get("disable_hires_mixing", False), interactive=True),
        )
    if target in {APPLY_BASE, APPLY_BOTH}:
        base_updates = _component_values_from_rows(base_tpl, base_shift, optimization_key, base_count_value)
    if target in {APPLY_HIRES, APPLY_BOTH}:
        hires_updates = _component_values_from_rows(hires_tpl, hires_shift, optimization_key, hires_count_value)
    return (
        gr.update(value=base_count_value, interactive=True),
        gr.update(value=hires_count_value, interactive=True),
        *hires_switches,
        gr.update(),
        gr.update(),
        gr.update(value=tpl.get("global_strength", 0.7), interactive=True),
        gr.update(value=_option_label("optimization", optimization_key), interactive=True),
        gr.update(value=_option_label("combine", tpl.get("combine_mode", COMBINE_OUTPUT_AVG)), interactive=True),
        gr.update(value=_option_label("fusion", tpl.get("fusion_mode", FUSION_INTERPOLATE)), interactive=True),
        gr.update(value=tpl.get("apply_uncond", False), interactive=True),
        gr.update(value=tpl.get("enable_cache", True), interactive=True),
        status_text if status_text is not None else _t("applied_template").format(name=name, target=_option_label("apply_target", target)),
        *base_updates,
        *hires_updates,
    )


def _apply_template_ui(name, target, base_row_count, hires_row_count, current_base_shift, current_hires_shift, current_optimization, *component_values):
    return _apply_template_updates(
        name,
        target,
        base_row_count,
        hires_row_count,
        current_base_shift,
        current_hires_shift,
        current_optimization,
        component_values,
    )


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
def _paste_plan(options_text, base_text, hires_text, base_shift, hires_shift, language):
    """Control name -> update (a tuple of updates for "base_rows" and
    "hires_rows") restoring the panel an infotext records; empty for an
    infotext without it, e.g. the summary alone earlier versions wrote.
    ``language`` is the UI language the labels are in (a cache key)."""
    options = _json_value(options_text, dict)
    chains = {"base": _json_value(base_text, list), "hires": _json_value(hires_text, list)}
    if options is None and chains["base"] is None and chains["hires"] is None:
        return {}
    options = options or {}
    optimization = _option_key("optimization", options.get("optimization"), OPT_BALANCE)
    plan = {"enable": gr.update(value=True)}
    if "strength" in options:
        plan["global_strength"] = gr.update(value=_clamp(_to_float(options["strength"], 0.7), 0.0, 2.0))
    for name, group, fallback in (("optimization", "optimization", OPT_BALANCE), ("combine_mode", "combine", COMBINE_OUTPUT_AVG), ("fusion_mode", "fusion", FUSION_INTERPOLATE)):
        key = name.removesuffix("_mode")
        if key in options:
            plan[name] = gr.update(value=_option_label(group, _option_key(group, options[key], fallback)))
    for name in ("apply_uncond", "hires_independent", "disable_hires_mixing"):
        if name in options:
            plan[name] = gr.update(value=_to_bool(options[name]))
    for prefix, shift in (("base", base_shift), ("hires", hires_shift)):
        if chains[prefix] is None:
            continue
        rows = [[True, *row] for row in chains[prefix] if isinstance(row, (list, tuple))]
        count = _row_count(len(rows))
        plan[f"{prefix}_row_count"] = gr.update(value=count)
        plan[f"{prefix}_rows"] = tuple(_component_values_from_rows(rows, shift, optimization, count))
    return plan


def _pasted_plan(params):
    """:func:`_paste_plan` of the parsed infotext ``params``."""
    base_shift = _to_float(params.get("Distilled CFG Scale"), 3.0)
    hires_shift = _to_float(params.get("Hires Distilled CFG Scale"), base_shift)
    texts = (params.get(key) for key in INFOTEXT_KEYS)
    return _paste_plan(*(text if isinstance(text, str) else None for text in texts), base_shift, hires_shift, _language())


def _pasted_update(name, params, index=None):
    """The update pasting ``params`` makes to control ``name`` (to its
    ``index``-th row component for "base_rows"/"hires_rows"); None leaves it."""
    update = _pasted_plan(params).get(name)
    if update is not None and index is not None:
        update = update[index]
    return None if update is None else dict(update)


def _pasted_marker(params):
    """A fresh value when ``params`` restores the panel: its change saves the
    pasted panel like an edit, also when only the textboxes (which save on
    blur) changed."""
    return gr.update(value=uuid.uuid4().hex) if _pasted_plan(params) else None


class _PasteKey(str):
    """An infotext key that is also the paste function of one control. Forge
    pastes through the function, so one recorded value fills many controls;
    being the key, it lets the "Send to" buttons carry the control like any
    field named in ``paste_field_names``."""

    def __new__(cls, key, function):
        self = super().__new__(cls, key)
        self.function = function
        return self

    def __call__(self, params):
        return self.function(params)


def _paste_fields(header, base_rows, hires_rows, marker):
    """``infotext_fields`` of the panel: the ``header`` controls (name ->
    component, as ui() returns them), the base and hires row components and
    the autosave ``marker``."""
    keys = {"base_row_count": INFOTEXT_BASE, "hires_row_count": INFOTEXT_HIRES}
    fields = [
        PasteField(component, _PasteKey(keys.get(name, INFOTEXT_OPTIONS), partial(_pasted_update, name)))
        for name, component in header.items()
        if name in {"enable", "base_row_count", "hires_row_count", "hires_independent", "disable_hires_mixing", "global_strength", "optimization", "combine_mode", "fusion_mode", "apply_uncond"}
    ]
    for name, key, components in (("base_rows", INFOTEXT_BASE, base_rows), ("hires_rows", INFOTEXT_HIRES, hires_rows)):
        fields.extend(PasteField(component, _PasteKey(key, partial(_pasted_update, name, index=index))) for index, component in enumerate(components))
    fields.append(PasteField(marker, _pasted_marker))
    return fields


# Generation-panel Shift sliders: (elem_id, attribute the Script keeps it in)
_SHIFT_SOURCES = (
    ("txt2img_distilled_cfg_scale", "base_shift_component"),
    ("txt2img_hr_distilled_cfg", "hires_shift_component"),
    ("img2img_distilled_cfg_scale", "base_shift_component"),
)


class Script(scripts.Script):
    def __init__(self):
        # Built before the script panels; each tab's runner only reports its own.
        self.base_shift_component = None
        self.hires_shift_component = None
        for elem_id, name in _SHIFT_SOURCES:
            self.on_after_component(partial(self._keep_shift_component, name), elem_id=elem_id)

    def _keep_shift_component(self, name, event):
        setattr(self, name, event.component)

    def title(self):
        return _t("title")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        lang = _language()
        defaults = _initial_ui_defaults(self.tabname, lang)
        table_css = f"""
        <style>
        #{self.elem_id("accordion")} [data-testid="sort-button"],
        #{self.elem_id("accordion")} .sort-button,
        #{self.elem_id("accordion")} button[aria-label*="Sort"],
        #{self.elem_id("accordion")} button[title*="Sort"] {{
            display: none !important;
        }}
        </style>
        """
        with gr.Accordion(_t("accordion"), open=False, elem_id=self.elem_id("accordion")):
            gr.HTML(table_css)
            intro_block(self.elem_id, INTRO_EN, INTRO_ZH, LANGUAGE_OPTION)

            # The generation panel's Shift sliders drive the stage timings; the
            # hidden copies carry their values into the script args.
            base_shift_input = self.base_shift_component
            hires_shift_input = self.hires_shift_component or base_shift_input
            runtime_base_shift = gr.Number(value=getattr(base_shift_input, "value", 3.0), visible=False, elem_id=self.elem_id("runtime_base_shift"))
            runtime_hires_shift = gr.Number(value=getattr(hires_shift_input, "value", 3.0), visible=False, elem_id=self.elem_id("runtime_hires_shift"))
            base_shift_input = base_shift_input or runtime_base_shift
            hires_shift_input = hires_shift_input or runtime_hires_shift
            # Changed by pasting an infotext that records the panel: saves it
            paste_marker = gr.Textbox(value="", visible=False, elem_id=self.elem_id("paste_marker"))
            enable = gr.Checkbox(label=_t("enable"), value=defaults["enable"], elem_id=self.elem_id("enable"))
            gr.Markdown(
                "Compatibility: Artist Mixer, FreeFuse, and Regional Conditioning are "
                "one-of-three; the active choice locks the others."
            )
            register_exclusive_component(
                tab=self.tabname,
                group="anima_spatial_conditioning",
                names=("freefuse", "regional", "artist"),
                name="artist",
                component=enable,
            )
            with gr.Row():
                optimization = gr.Dropdown(label=_t("optimization"), choices=_option_choices("optimization", lang), value=defaults["optimization"], elem_id=self.elem_id("optimization"))
                global_strength = gr.Slider(label=_t("global_strength"), minimum=0.0, maximum=2.0, step=0.01, value=defaults["global_strength"], elem_id=self.elem_id("global_strength"))
                enable_cache = gr.Checkbox(label=_t("cache"), value=defaults["enable_cache"], elem_id=self.elem_id("enable_cache"))
            with gr.Row():
                combine_mode = gr.Dropdown(label=_t("combine"), choices=_option_choices("combine", lang), value=defaults["combine"], elem_id=self.elem_id("combine_mode"))
                fusion_mode = gr.Dropdown(label=_t("fusion"), choices=_option_choices("fusion", lang), value=defaults["fusion"], elem_id=self.elem_id("fusion_mode"))
                apply_uncond = gr.Checkbox(label=_t("apply_uncond"), value=defaults["apply_uncond"], elem_id=self.elem_id("apply_uncond"))

            with gr.Tab(_t("base_tab")):
                with gr.Row():
                    base_row_count = gr.Number(label=_t("row_count"), value=defaults["base_count"], precision=0, elem_id=self.elem_id("base_row_count"))
                    base_apply_presets = gr.Button(_t("normalize_rows"), elem_id=self.elem_id("base_apply_presets"))
                gr.Markdown(value=_t("shift_runtime_hint"))
                gr.Markdown(value=f"**{_t('artist_table')}**")
                base_row_components = _create_artist_row_controls("base", lang, defaults["base_rows"], self.elem_id)

            with gr.Tab(_t("hires_tab")):
                hires_independent = gr.Checkbox(label=_t("hires_independent"), value=defaults["hires_independent"], elem_id=self.elem_id("hires_independent"))
                disable_hires_mixing = gr.Checkbox(label=_t("disable_hires_mixing"), value=defaults["disable_hires_mixing"], elem_id=self.elem_id("disable_hires_mixing"))
                with gr.Row():
                    hires_row_count = gr.Number(label=_t("hr_row_count"), value=defaults["hires_count"], precision=0, elem_id=self.elem_id("hires_row_count"))
                    hires_apply_presets = gr.Button(_t("normalize_rows"), elem_id=self.elem_id("hires_apply_presets"))
                gr.Markdown(value=_t("shift_runtime_hint"))
                gr.Markdown(value=f"**{_t('artist_table')}**")
                hires_row_components = _create_artist_row_controls("hires", lang, defaults["hires_rows"], self.elem_id)

            with gr.Tab(_t("template")):
                with gr.Row():
                    template_choices = _template_choices()
                    template_dropdown = gr.Dropdown(label=_t("template"), choices=template_choices, value=_resolve_template_name(defaults["template_name"], template_choices), allow_custom_value=False, elem_id=self.elem_id("template_dropdown"))
                    template_apply_target = gr.Dropdown(label=_t("apply_target"), choices=_option_choices("apply_target", lang), value=defaults["template_target"], elem_id=self.elem_id("template_apply_target"))
                    template_apply = gr.Button(_t("apply_template"), variant="primary", elem_id=self.elem_id("template_apply"))
                with gr.Row():
                    template_name = gr.Textbox(label=_t("template_name"), value="", elem_id=self.elem_id("template_name"))
                    template_save_base = gr.Button(_t("save_base_template"), elem_id=self.elem_id("template_save_base"))
                    template_save_hires = gr.Button(_t("save_hires_template"), elem_id=self.elem_id("template_save_hires"))
                with gr.Row():
                    rename_to = gr.Textbox(label=_t("rename_to"), value="", elem_id=self.elem_id("rename_to"))
                    rename_button = gr.Button(_t("rename_template"), elem_id=self.elem_id("rename_button"))
                    delete_button = gr.Button(_t("delete_template"), elem_id=self.elem_id("delete_button"))
                    reset_defaults_button = gr.Button(_t("reset_defaults"), elem_id=self.elem_id("reset_defaults_button"))
                template_status = gr.Markdown(value="", elem_id=self.elem_id("template_status"))

            with gr.Tab(_t("help")):
                gr.Markdown(value=HELP_ZH if _language() == "zh" else HELP_EN)

        base_row_count.change(
            fn=_resize_row_components,
            inputs=[base_row_count, base_shift_input, optimization, *base_row_components],
            outputs=base_row_components,
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        hires_row_count.change(
            fn=_resize_row_components,
            inputs=[hires_row_count, hires_shift_input, optimization, *hires_row_components],
            outputs=hires_row_components,
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        for source, runtime_shift in ((self.base_shift_component, runtime_base_shift), (self.hires_shift_component or self.base_shift_component, runtime_hires_shift)):
            if source is not None:
                source.change(fn=_same_value, inputs=[source], outputs=[runtime_shift], queue=False, show_progress=False, **_internal_event_kwargs())
        base_shift_input.change(
            fn=_apply_shift_to_timing_components,
            inputs=[base_row_count, base_shift_input, optimization, *base_row_components],
            outputs=_row_outputs(base_row_components, (4, 5, 6)),
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        hires_shift_input.change(
            fn=_apply_shift_to_timing_components,
            inputs=[hires_row_count, hires_shift_input, optimization, *hires_row_components],
            outputs=_row_outputs(hires_row_components, (4, 5, 6)),
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        base_apply_presets.click(
            fn=_resize_row_components,
            inputs=[base_row_count, base_shift_input, optimization, *base_row_components],
            outputs=base_row_components,
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        hires_apply_presets.click(
            fn=_resize_row_components,
            inputs=[hires_row_count, hires_shift_input, optimization, *hires_row_components],
            outputs=hires_row_components,
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        # User edits only (.input): a template or a reset writing these
        # controls brings its own rows, which .change would rewrite.
        # Handlers writing the Artist/Blocks textboxes: the autosave follows them.
        text_writers = [
            optimization.input(
                fn=_apply_optimization_to_block_timing_components,
                inputs=[row_count, shift_input, optimization, *row_components],
                outputs=_row_outputs(row_components, (3, 4, 5, 6)),
                queue=False,
                show_progress=False,
                **_internal_event_kwargs(),
            )
            for row_count, shift_input, row_components in (
                (base_row_count, base_shift_input, base_row_components),
                (hires_row_count, hires_shift_input, hires_row_components),
            )
        ]
        for row_components, row_count, runtime_shift in (
            (base_row_components, base_row_count, base_shift_input),
            (hires_row_components, hires_row_count, hires_shift_input),
        ):
            for offset in range(0, len(row_components), ROW_WIDTH):
                row_components[offset + 8].input(
                    fn=partial(_apply_stage_to_row_components, offset // ROW_WIDTH),
                    inputs=[row_count, runtime_shift, optimization, *row_components],
                    outputs=[
                        row_components[offset + 4],
                        row_components[offset + 5],
                        row_components[offset + 6],
                        row_components[offset + 9],
                    ],
                    queue=False,
                    show_progress=False,
                    **_internal_event_kwargs(),
                )
                row_components[offset + 9].input(
                    fn=partial(_apply_auto_shift_to_row_components, offset // ROW_WIDTH),
                    inputs=[row_count, runtime_shift, optimization, *row_components],
                    outputs=[
                        row_components[offset + 4],
                        row_components[offset + 5],
                        row_components[offset + 6],
                    ],
                    queue=False,
                    show_progress=False,
                    **_internal_event_kwargs(),
                )
        template_save_base.click(
            fn=_save_base_template_ui,
            inputs=[
                template_name,
                base_row_count,
                base_shift_input,
                disable_hires_mixing,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                *base_row_components,
            ],
            outputs=[template_dropdown, template_status],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        template_save_hires.click(
            fn=_save_hires_template_ui,
            inputs=[
                template_name,
                hires_independent,
                hires_row_count,
                hires_shift_input,
                disable_hires_mixing,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                *hires_row_components,
            ],
            outputs=[template_dropdown, template_status],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        rename_button.click(fn=_rename_template_ui, inputs=[template_dropdown, rename_to], outputs=[template_dropdown, template_status], queue=False, show_progress=False, **_internal_event_kwargs())
        delete_button.click(fn=_delete_template_ui, inputs=[template_dropdown], outputs=[template_dropdown, template_status], queue=False, show_progress=False, **_internal_event_kwargs())
        apply_template = template_apply.click(
            fn=_apply_template_ui,
            inputs=[template_dropdown, template_apply_target, base_row_count, hires_row_count, base_shift_input, hires_shift_input, optimization, *base_row_components, *hires_row_components],
            outputs=[
                base_row_count,
                hires_row_count,
                hires_independent,
                disable_hires_mixing,
                runtime_base_shift,
                runtime_hires_shift,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                template_status,
                *base_row_components,
                *hires_row_components,
            ],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        text_writers.append(apply_template)
        # A reset needs no save: the emptied file already restores the defaults.
        reset_defaults_button.click(
            fn=partial(_reset_all_defaults_ui, self.tabname),
            inputs=[],
            outputs=[
                enable,
                template_dropdown,
                template_apply_target,
                base_row_count,
                hires_row_count,
                hires_independent,
                disable_hires_mixing,
                runtime_base_shift,
                runtime_hires_shift,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                template_status,
                *base_row_components,
                *hires_row_components,
            ],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        current_settings_inputs = [
            enable,
            template_dropdown,
            template_apply_target,
            base_row_count,
            hires_row_count,
            hires_independent,
            disable_hires_mixing,
            runtime_base_shift,
            runtime_hires_shift,
            global_strength,
            optimization,
            combine_mode,
            fusion_mode,
            apply_uncond,
            enable_cache,
            *base_row_components,
            *hires_row_components,
        ]
        _do_not_save_to_ui_config(*current_settings_inputs, paste_marker)
        # Remember this tab's panel as it is edited; "always_last" folds a
        # burst of changes (applying a template, pasting an infotext) into one
        # final save. A textbox saves once its edit is done (blur, Enter), not
        # per keystroke; handlers writing the textboxes save after they finish,
        # and a paste through the marker.
        autosave = {
            "fn": partial(_save_current_settings_ui, self.tabname),
            "inputs": current_settings_inputs,
            "outputs": None,
            "queue": False,
            "show_progress": "hidden",
            **_internal_event_kwargs(),
        }
        gr.on(
            triggers=[*_autosave_triggers(current_settings_inputs), paste_marker.change],
            trigger_mode="always_last",
            **autosave,
        )
        for writer in text_writers:
            writer.then(**autosave)

        register_preset_control(self.tabname, "artist.enabled", enable)
        register_preset_control(self.tabname, "artist.global_strength", global_strength)
        register_preset_control(
            self.tabname,
            "artist.optimization",
            optimization,
            _option_label("optimization", OPT_BALANCE, lang),
        )
        register_preset_control(
            self.tabname,
            "artist.combine",
            combine_mode,
            _option_label("combine", COMBINE_OUTPUT_AVG, lang),
        )
        register_preset_control(
            self.tabname,
            "artist.fusion",
            fusion_mode,
            _option_label("fusion", FUSION_INTERPOLATE, lang),
        )
        register_preset_control(self.tabname, "artist.apply_uncond", apply_uncond)
        register_preset_control(self.tabname, "artist.cache", enable_cache)

        # In the order _MixerArgs.from_args reads them back
        header = {
            "enable": enable,
            "base_row_count": base_row_count,
            "hires_row_count": hires_row_count,
            "hires_independent": hires_independent,
            "disable_hires_mixing": disable_hires_mixing,
            "runtime_base_shift": runtime_base_shift,
            "runtime_hires_shift": runtime_hires_shift,
            "global_strength": global_strength,
            "optimization": optimization,
            "combine_mode": combine_mode,
            "fusion_mode": fusion_mode,
            "apply_uncond": apply_uncond,
            "enable_cache": enable_cache,
        }
        # Pasting an infotext that records the panel restores it (not the
        # cache switch, which does not change the output); the summary alone
        # of earlier versions leaves the panel as it is.
        self.infotext_fields = _paste_fields(header, base_row_components, hires_row_components, paste_marker)
        self.paste_field_names = list(INFOTEXT_KEYS)
        return [*(header[name] for name in HEADER_ARGS), *base_row_components, *hires_row_components]

    def process(self, p, *args, **kwargs):
        # Runs before NegPiP's process_batch: let a negative weight in an
        # artist name that will actually be encoded enable NegPiP. Same
        # gating and rows as process_before_every_sampling, for each pass.
        names = []
        passes = [False]
        mixer_args = _MixerArgs.from_args(args)
        if mixer_args is not None and getattr(p, "enable_hr", False) and mixer_args.hires_independent:  # hires rows of its own
            passes.append(True)
        for hires in passes:
            run = _resolve_run_rows(p, mixer_args, hires, report=False)
            if run is None:
                continue
            dm, _unet, shift, rows = run
            names.extend(name for _row, name, _weight, _blocks in _artist_entries(rows, shift, mixer_args.optimization, len(dm.blocks)))
        register_negpip_prompts(p, ARTIST_MIXER, names)

    def process_before_every_sampling(self, p, *args, **kwargs):
        mixer_args = _MixerArgs.from_args(args)
        if mixer_args is None:
            return
        if _should_remember_settings(p):
            _save_runtime_current_settings(self.tabname, mixer_args)

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
        # Keys match ``self.infotext_fields`` so pasted parameters restore the panel.
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
