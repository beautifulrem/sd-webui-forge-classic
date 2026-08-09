import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from modules import script_callbacks, scripts, shared
from modules.anima_lora_support import merge_consistent_rules


logger = logging.getLogger("anima_lora_layer_weight")
LANGUAGE_OPTION = "anima_lora_layer_weight_language"
LANGUAGE_CHOICES = ("zh", "en")
LANG = {
    "zh": {
        "settings_label": "Anima LoRA 分层权重界面语言",
        "title": "Anima LoRA 分层权重",
        "accordion": "Anima LoRA/LoKR 分层权重",
        "enable_lora": "启用 LoRA 分层权重",
        "enable_lokr": "启用 LoKR 分层权重",
        "target_lora": "目标 LoRA",
        "target_lokr": "目标 LoKR",
        "target_lora_placeholder": "留空=全部 Anima LoRA；填文件名/别名=只控制指定 LoRA",
        "target_lokr_placeholder": "留空=全部 Anima LoKR；填文件名/别名=只控制指定 LoKR",
        "block_enabled": "Anima blocks 层权重",
        "module_enabled": "模块权重",
        "qwen_enabled": "Qwen 文本编码器层权重",
        "block_weights": "Anima blocks 层权重",
        "module_weights": "模块权重",
        "qwen_weights": "Qwen 文本编码器层权重",
        "block_placeholder": "示例: 0-5=0.6, 6-20=1.0, 21-27=0.8",
        "module_placeholder": "self_attn/cross_attn/mlp/adaln/norm",
        "qwen_placeholder": "留空=文本编码器保持原权重；示例: 0-31=1.0",
        "preset_enabled": "启用预设",
        "preset_name": "预设名称",
        "preset_scope": "预设应用范围",
        "scope_all": "提示词里的全部同类型",
        "scope_targets": "仅目标列表",
        "custom_presets": "自定义预设",
        "preset_placeholder": "每行一个预设：名称=blocks=...;modules=...;qwen=...",
        "apply_preset": "应用预设到提示词",
        "lora_usage": "LoRA 使用教程",
        "lokr_usage": "LoKR 使用教程",
    },
    "en": {
        "settings_label": "Anima LoRA Layer Weight UI language",
        "title": "Anima LoRA Layer Weight",
        "accordion": "Anima LoRA/LoKR Layer Weight",
        "enable_lora": "Enable LoRA layer weights",
        "enable_lokr": "Enable LoKR layer weights",
        "target_lora": "Target LoRA",
        "target_lokr": "Target LoKR",
        "target_lora_placeholder": "Empty = all Anima LoRAs; file name or alias = only the listed LoRAs",
        "target_lokr_placeholder": "Empty = all Anima LoKRs; file name or alias = only the listed LoKRs",
        "block_enabled": "Anima block weights",
        "module_enabled": "Module weights",
        "qwen_enabled": "Qwen text-encoder layer weights",
        "block_weights": "Anima block weights",
        "module_weights": "Module weights",
        "qwen_weights": "Qwen text-encoder layer weights",
        "block_placeholder": "Example: 0-5=0.6, 6-20=1.0, 21-27=0.8",
        "module_placeholder": "self_attn/cross_attn/mlp/adaln/norm",
        "qwen_placeholder": "Empty = keep text encoder unchanged; example: 0-31=1.0",
        "preset_enabled": "Enable preset",
        "preset_name": "Preset name",
        "preset_scope": "Preset apply scope",
        "scope_all": "All matching type in prompt",
        "scope_targets": "Target list only",
        "custom_presets": "Custom presets",
        "preset_placeholder": "One preset per line: name=blocks=...;modules=...;qwen=...",
        "apply_preset": "Apply preset to prompt",
        "lora_usage": "LoRA usage guide",
        "lokr_usage": "LoKR usage guide",
    },
}


def _language():
    value = getattr(shared.opts, LANGUAGE_OPTION, "zh")
    return value if value in LANGUAGE_CHOICES else "zh"


def _t(key):
    return LANG.get(_language(), LANG["zh"]).get(key, key)


INTRO_EN = (
    "Control Anima LoRA/LoKR patch strength by block range, module type, and "
    "optional Qwen text-encoder layers. Use it to tune composition, character "
    "features, and style without editing model files."
)
INTRO_ZH = "按 block、模块和可选 Qwen 文本编码器层控制 Anima LoRA/LoKR 强度，用于调节构图、人物特征和画风，不修改模型文件。"


def _intro_text(language):
    return INTRO_ZH if language == "中文" else INTRO_EN


def _intro_block(elem_id):
    with gr.Row():
        intro_language = gr.Radio(
            choices=["English", "中文"],
            value="English",
            label="Description language / 说明语言",
            elem_id=elem_id("intro_language"),
        )
    intro = gr.Markdown(value=INTRO_EN, elem_id=elem_id("intro"))
    intro_language.change(fn=_intro_text, inputs=[intro_language], outputs=[intro], queue=False, show_progress=False)


def _register_ui_settings():
    if LANGUAGE_OPTION in shared.opts.data_labels:
        return
    shared.opts.add_option(
        LANGUAGE_OPTION,
        shared.OptionInfo(
            "zh",
            "Anima LoRA Layer Weight language / Anima LoRA 分层权重语言",
            gr.Radio,
            {"choices": LANGUAGE_CHOICES},
            section=("anima-lora-layer-weight", "Anima LoRA Layer Weight"),
            category_id="system",
        ).needs_reload_ui(),
    )


script_callbacks.on_ui_settings(_register_ui_settings)

ANIMA_BLOCK_RE = re.compile(r"(?:^|\.)(?:diffusion_model\.)?(?:net\.)?blocks\.(\d+)\.")
QWEN_LAYER_RE = re.compile(r"(?:^|\.)(?:text_encoders\.)?qwen3_06b\.model\.layers\.(\d+)\.")
GENERIC_QWEN_LAYER_RE = re.compile(r"(?:^|\.)(?:text_encoders\.)?qwen3_06b\.(?:model\.)?layers\.(\d+)\.")
QWEN_ADAPTER_BLOCK_RE = re.compile(r"(?:^|\.)(?:text_encoders\.)?qwen3_06b\.llm_adapter\.blocks\.(\d+)\.")
LEGACY_QWEN_ADAPTER_BLOCK_RE = re.compile(r"(?:^|\.)(?:diffusion_model\.)?llm_adapter\.blocks\.(\d+)\.")
LORA_MARKERS = (".lora_a.weight", ".lora_b.weight", ".lora_down.weight", ".lora_up.weight")
LOKR_MARKERS = (".lokr_",)
INLINE_LORA_RE = re.compile(r"<lora:([^:>]+):([^>]+)>")
PROMPT_LORA_RE = re.compile(r"<lora:([^:>]+)(?::([^>]*))?>")

DEFAULT_BLOCK_WEIGHTS = "0-27=1.0"
DEFAULT_MODULE_WEIGHTS = "\n".join(
    [
        "self_attn=1.0",
        "cross_attn=1.0",
        "mlp=1.0",
        "adaln=1.0",
        "norm=1.0",
    ]
)
PRESET_SCOPE_ALL = "提示词里的全部同类型"
PRESET_SCOPE_TARGETS = "仅目标列表"
PRESET_SCOPE_ALL_EN = "All matching type in prompt"
PRESET_SCOPE_TARGETS_EN = "Target list only"
DEFAULT_PRESETS_TEXT = "\n".join(
    [
        "平衡=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,cross_attn=1.0,mlp=1.0,adaln=1.0,norm=1.0",
        "保构图=blocks=0-9=0.35,10-18=1.0,19-27=0.7",
        "强画风=blocks=0-9=0.7,10-18=1.0,19-27=0.9",
    ]
)
USAGE_TEXT = """使用方式:
1. LoRA 与 LoKR 分开控制：LoRA 选项卡只控制普通 LoRA，LoKR 选项卡只控制 LoKR。
2. 目标留空时，会自动对所有检测为 Anima 架构且类型匹配的文件启用该选项卡的分层权重。
3. 目标填入文件名、别名或不带后缀文件名时，只对填入的文件生效；多个目标用逗号或换行分隔。
4. 插件命中目标后，提示词里的普通权重会被忽略，例如 <lora:灵魂潮汐:0.6> 里的 0.6 不参与计算，最终权重直接由已开启的分层权重决定。
5. Anima blocks 层权重、模块权重可以同时开启；Qwen 文本编码器层权重独立开启，默认关闭。
6. 同时启用 blocks 和模块权重时，UNet/DiT 最终权重 = block 层权重 * 模块权重；只启用其中一个时，只使用已启用项。
7. Qwen 文本编码器层权重只能在插件面板控制，不支持写在提示词里；文件没有训练文本编码器时会自动跳过文本编码器权重，不会报错。
8. 如果写了某个文件没有训练的 block、模块或 Qwen 文本编码器层，该部分会直接跳过，只对文件实际包含的部分生效。
9. Blocks 0-9 是浅层，主要控制姿态与构图；Blocks 10-18 是中层，主要控制人物外貌、服装等特征；Blocks 19-27 是深层，主要控制画风与光影。
10. 可以直接在提示词权重位置写 block 分层，例如 <lora:灵魂潮汐:0-7=0.6;8-20=1.0;21-27=0.8>。
11. 也可以在提示词里同时写 block 和模块权重，例如 <lora:灵魂潮汐:blocks=0-9=0.8,10-18=1.0,19-27=0.7;modules=self_attn=1.0,mlp=0.8,cross_attn=0.6>。
12. 使用提示词内联分层时，该文件会跳过对应选项卡里的 blocks 和模块设置，只使用提示词内写的 block/模块权重；Qwen 仍只读取插件面板设置。
13. 内联 block 分层也支持直接写 28 个数字，例如 <lora:灵魂潮汐:1,1,0.9,0.9,0.8,...>，顺序对应 blocks 0 到 27。
14. 可以在预设列表里写多行命名预设，例如 平衡=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8;qwen=0-5=1.0。
15. 点击“应用预设到提示词”会把当前页正向提示词中命中的 LoRA/LoKR 权重替换成该预设的 blocks/modules 内联分层语法。
16. 预设应用范围选择“提示词里的全部同类型”时，会替换提示词里所有同类型的 Anima LoRA/LoKR；选择“仅目标列表”时，只替换目标框里填入的 LoRA/LoKR。
"""
USAGE_TEXT_EN = """Usage:
1. LoRA and LoKR are controlled separately. The LoRA tab only affects regular LoRAs; the LoKR tab only affects LoKRs.
2. When the target field is empty, the tab automatically applies to all matching Anima adapters of that type.
3. When targets are filled, only the listed file names, aliases, or names without extension are controlled. Separate multiple targets with commas or new lines.
4. After a target is controlled by this plugin, the normal prompt weight, for example the 0.6 in <lora:name:0.6>, is ignored for the matched adapter.
5. Anima block weights and module weights can be enabled together. Qwen text-encoder layer weights are independent and disabled by default.
6. When block and module weights are both enabled, the final model weight is block weight * module weight.
7. Qwen text-encoder layer weights can only be controlled from this panel. Files without text-encoder training are skipped naturally.
8. Missing blocks, modules, or Qwen layers are skipped; only weights that exist in the selected file are affected.
9. Blocks 0-9 mainly affect pose and composition, 10-18 affect character and outfit features, and 19-27 affect style and lighting.
10. You can write inline block weights in the prompt, for example <lora:name:0-7=0.6;8-20=1.0;21-27=0.8>.
11. Inline syntax can also combine blocks and modules, for example <lora:name:blocks=0-9=0.8,10-18=1.0,19-27=0.7;modules=self_attn=1.0,mlp=0.8,cross_attn=0.6>.
12. Inline block/module weights override the corresponding tab settings for that file. Qwen weights still come from the panel.
13. Inline block weights also support 28 comma-separated numbers in block order from 0 to 27.
14. Presets can be written one per line, for example Balanced=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8;qwen=0-5=1.0.
15. "Apply preset to prompt" replaces matching LoRA/LoKR prompt weights with the preset's inline blocks/modules syntax.
16. The apply scope can affect all matching adapters in the prompt, or only the adapters listed in the target field.
"""


def _usage_text():
    return USAGE_TEXT_EN if _language() == "en" else USAGE_TEXT

_ORIGINAL_ADD_PATCHES = None
_ACTIVE_RULE = None
_LAST_SIGNATURE = None
_SCAN_CACHE = {}
_KIND_CACHE = {}
_QWEN_SCAN_CACHE = {}
_COVERAGE_CACHE = {}


@dataclass
class LayerWeights:
    default: float = 1.0
    values: dict[int, float] = None
    inline: bool = False

    def __post_init__(self):
        if self.values is None:
            self.values = {}

    def factor(self, index: int) -> float:
        return self.values.get(index, self.default)


@dataclass
class PromptRule:
    block_weights: LayerWeights
    module_weights: dict[str, float]


@dataclass
class PresetWeights:
    block_weights: str | None = None
    module_weights: str | None = None
    text_layer_weights: str | None = None


@dataclass
class LoraCoverage:
    blocks: set[int]
    modules: set[str]
    text_layers: set[int]


@dataclass
class LoraDecision:
    apply: bool
    block_weights: LayerWeights
    module_weights: dict[str, float]
    text_layer_weights: LayerWeights
    source: str


class AdapterRule:
    def __init__(
        self,
        adapter_kind: str,
        enabled: bool,
        target_loras: str,
        target_mode: str,
        block_enabled: bool,
        block_weights: str,
        module_enabled: bool,
        module_weights: str,
        text_layer_enabled: bool,
        text_layer_weights: str,
        inline_rules: dict[str, PromptRule],
    ):
        self.adapter_kind = adapter_kind
        self.enabled = bool(enabled)
        self.target_mode = target_mode or "auto"
        self.targets = _parse_targets(target_loras)
        self.block_enabled = bool(block_enabled)
        self.module_enabled = bool(module_enabled)
        self.text_layer_enabled = bool(text_layer_enabled)
        self.block_weights = _parse_layer_weights(block_weights, DEFAULT_BLOCK_WEIGHTS) if self.block_enabled else LayerWeights()
        self.module_weights = _parse_module_weights(module_weights) if self.module_enabled else {}
        self.text_layer_weights = _parse_layer_weights(text_layer_weights, "") if self.text_layer_enabled else LayerWeights()
        self.inline_rules = inline_rules
        self.signature = self._make_signature(target_loras, target_mode, block_weights, module_weights, text_layer_weights)

    def _make_signature(self, target_loras, target_mode, block_weights, module_weights, text_layer_weights):
        inline_sig = ",".join(
            sorted(f"{name}:{rule.block_weights.default}:{rule.block_weights.values}:{rule.module_weights}" for name, rule in self.inline_rules.items())
        )
        return "|".join(
            [
                self.adapter_kind,
                "1" if self.enabled else "0",
                _compact_text(target_loras),
                f"target_mode={target_mode}",
                "block=1" if self.block_enabled else "block=0",
                _compact_text(block_weights),
                "module=1" if self.module_enabled else "module=0",
                _compact_text(module_weights),
                "qwen=1" if self.text_layer_enabled else "qwen=0",
                _compact_text(text_layer_weights),
                inline_sig,
            ]
        )

    def decision_for_lora(self, filename) -> LoraDecision:
        if not self.enabled or not filename:
            return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights, "disabled")

        if _adapter_kind_for_file(filename) != self.adapter_kind:
            return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights, "not-type")

        names = _names_for_lora_file(filename)
        inline_rule = _first_inline_rule(names, self.inline_rules)
        coverage = _lora_coverage(filename, self.adapter_kind)
        text_layer_weights = _filter_layer_weights(self.text_layer_weights, coverage.text_layers) if self.text_layer_enabled else LayerWeights()
        if inline_rule is not None:
            return LoraDecision(True, _filter_layer_weights(inline_rule.block_weights, coverage.blocks), _filter_module_weights(inline_rule.module_weights, coverage.modules), text_layer_weights, "prompt")

        if self.target_mode == "targets":
            if not self.targets or not any(name in self.targets for name in names):
                return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights, "not-target")
        elif self.target_mode == "all":
            if not _is_anima_adapter_file(filename, self.adapter_kind):
                return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights, "not-anima")
        elif self.targets:
            if not any(name in self.targets for name in names):
                return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights, "not-target")
        elif not _is_anima_adapter_file(filename, self.adapter_kind):
            return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights, "not-anima")
        return LoraDecision(True, _filter_layer_weights(self.block_weights, coverage.blocks), _filter_module_weights(self.module_weights, coverage.modules), text_layer_weights, "global")


class ActiveRule:
    def __init__(
        self,
        lora_enabled: bool,
        lora_targets: str,
        lora_target_mode: str,
        lora_block_enabled: bool,
        lora_block_weights: str,
        lora_module_enabled: bool,
        lora_module_weights: str,
        lora_text_layer_enabled: bool,
        lora_text_layer_weights: str,
        lora_inline_rules: dict[str, PromptRule],
        lokr_enabled: bool,
        lokr_targets: str,
        lokr_target_mode: str,
        lokr_block_enabled: bool,
        lokr_block_weights: str,
        lokr_module_enabled: bool,
        lokr_module_weights: str,
        lokr_text_layer_enabled: bool,
        lokr_text_layer_weights: str,
        lokr_inline_rules: dict[str, PromptRule],
    ):
        self.lora_rule = AdapterRule("lora", lora_enabled, lora_targets, lora_target_mode, lora_block_enabled, lora_block_weights, lora_module_enabled, lora_module_weights, lora_text_layer_enabled, lora_text_layer_weights, lora_inline_rules)
        self.lokr_rule = AdapterRule("lokr", lokr_enabled, lokr_targets, lokr_target_mode, lokr_block_enabled, lokr_block_weights, lokr_module_enabled, lokr_module_weights, lokr_text_layer_enabled, lokr_text_layer_weights, lokr_inline_rules)
        self.rules = {
            "lora": self.lora_rule,
            "lokr": self.lokr_rule,
        }
        self.enabled = self.lora_rule.enabled or self.lokr_rule.enabled
        self.signature = f"lora[{self.lora_rule.signature}]|lokr[{self.lokr_rule.signature}]"

    def decision_for_lora(self, filename) -> LoraDecision:
        adapter_kind = _adapter_kind_for_file(filename)
        rule = self.rules.get(adapter_kind)
        if rule is None:
            return LoraDecision(False, LayerWeights(), {}, LayerWeights(), "not-supported-type")
        return rule.decision_for_lora(filename)

    def factor_for_key(self, model_key, decision: LoraDecision) -> float:
        key = _model_key(model_key)
        if not key:
            return 1.0

        factor = 1.0
        text_layer = _qwen_layer_index(key)
        if text_layer is not None:
            factor *= decision.text_layer_weights.factor(text_layer)
            return factor

        block_index = _block_index_from_key(key)
        if block_index is not None:
            factor *= decision.block_weights.factor(block_index)
            factor *= _module_factor(key, decision.module_weights)

        return factor


def _compact_text(text) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split())


def _parse_targets(text) -> set[str]:
    normalized = str(text or "").replace("，", ",").replace("；", ";")
    parts = re.split(r"[,;\n]+", normalized)
    targets = set()
    for part in parts:
        name = part.strip().lower()
        if not name:
            continue
        targets.add(name)
        path = Path(name)
        targets.add(path.name.lower())
        targets.add(path.stem.lower())
    return {target for target in targets if target}


def _parse_float(value, fallback=1.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return fallback


def _is_plain_number(text) -> bool:
    return re.fullmatch(r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*", str(text or "")) is not None


def _looks_like_layer_weights(text) -> bool:
    value = str(text or "").strip()
    if not value or _is_plain_number(value):
        return False

    tokens = [t for t in re.split(r"[\s,，;；]+", value) if t]
    selector_found = False
    numeric_count = 0

    for token in tokens:
        if "=" in token:
            left, _ = token.split("=", 1)
            left = left.strip().lower()
            if left in {"*", "all", "default"} or left.isdigit() or re.fullmatch(r"\d+\s*-\s*\d+", left):
                selector_found = True
                break
            continue

        if _is_plain_number(token):
            numeric_count += 1

    return selector_found or numeric_count >= 2


def _parse_layer_weights(text, fallback_text) -> LayerWeights:
    source = str(text or "").strip() or fallback_text
    source = source.replace("，", ",").replace("；", ";").replace("：", "=")
    tokens = [t for t in re.split(r"[\s,;]+", source) if t]
    values = {}
    default = 1.0
    bare_values = []

    for token in tokens:
        if "=" not in token:
            try:
                bare_values.append(float(token))
            except Exception:
                logger.warning("Ignored invalid layer weight token: %s", token)
            continue

        left, right = token.split("=", 1)
        left = left.strip().lower()
        value = _parse_float(right, 1.0)

        if left in {"*", "all", "default"}:
            default = value
            continue

        range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", left)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start > end:
                start, end = end, start
            for index in range(start, end + 1):
                values[index] = value
            continue

        if left.isdigit():
            values[int(left)] = value
        else:
            logger.warning("Ignored invalid layer selector: %s", left)

    if bare_values and not values:
        values.update({i: v for i, v in enumerate(bare_values)})

    return LayerWeights(default=default, values=values, inline=_looks_like_layer_weights(text))


def _module_key(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "self": "self_attn",
        "selfattn": "self_attn",
        "self_attention": "self_attn",
        "cross": "cross_attn",
        "crossattn": "cross_attn",
        "cross_attention": "cross_attn",
        "ada": "adaln",
        "adaln_modulation": "adaln",
        "modulation": "adaln",
        "norms": "norm",
        "layernorm": "norm",
        "feedforward": "mlp",
        "ffn": "mlp",
    }
    return aliases.get(key, key)


def _parse_module_weights(text) -> dict[str, float]:
    weights = {
        "self_attn": 1.0,
        "cross_attn": 1.0,
        "mlp": 1.0,
        "adaln": 1.0,
        "norm": 1.0,
    }
    source = str(text or DEFAULT_MODULE_WEIGHTS).replace("，", ",").replace("；", ";").replace("：", "=")

    for token in [t for t in re.split(r"[\n,;]+", source) if t.strip()]:
        if "=" not in token:
            continue
        left, right = token.split("=", 1)
        weights[_module_key(left)] = _parse_float(right, 1.0)

    return weights


def _preset_key(name: str) -> str:
    return _compact_text(name).lower()


def _parse_preset_sections(text: str) -> PresetWeights:
    sections: dict[str, str] = {}
    bare_parts = []
    for part in [p.strip() for p in re.split(r"[;；]+", str(text or "")) if p.strip()]:
        if "=" in part or "：" in part:
            if "=" in part:
                left, right = part.split("=", 1)
            else:
                left, right = part.split("：", 1)
            key = left.strip().lower()
            if key in {"blocks", "block", "layers", "layer"}:
                sections["blocks"] = right.strip()
                continue
            if key in {"modules", "module", "mods", "mod"}:
                sections["modules"] = right.strip()
                continue
            if key in {"qwen", "text", "text_encoder", "text_layers", "te"}:
                sections["qwen"] = right.strip()
                continue
        bare_parts.append(part)
    if bare_parts:
        if "blocks" in sections:
            sections["blocks"] = ";".join([sections["blocks"], *bare_parts])
        else:
            sections["blocks"] = ";".join(bare_parts)
    return PresetWeights(
        block_weights=sections.get("blocks") or None,
        module_weights=sections.get("modules") or None,
        text_layer_weights=sections.get("qwen") or None,
    )


def _parse_presets(text: str) -> dict[str, PresetWeights]:
    presets: dict[str, PresetWeights] = {}
    for line in str(text or "").replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            name, body = line.split("=", 1)
        elif "：" in line:
            name, body = line.split("：", 1)
        elif ":" in line:
            name, body = line.split(":", 1)
        else:
            logger.warning("Ignored invalid preset line: %s", line)
            continue

        name = _preset_key(name)
        if not name:
            continue
        presets[name] = _parse_preset_sections(body)
    return presets


def _preset_choices(text: str) -> list[str]:
    presets = _parse_presets(text)
    if not presets:
        return []
    choices = list(presets.keys())
    return choices


def _preset_dropdown_update(text: str, current_value: str = ""):
    choices = _preset_choices(text)
    if current_value and current_value not in choices:
        choices = [current_value] + choices
    value = current_value if current_value in choices else (choices[0] if choices else "")
    return gr.update(choices=choices, value=value)


def _apply_preset_values(
    enabled: bool,
    preset_name: str,
    presets_text: str,
    block_enabled: bool,
    block_weights: str,
    module_enabled: bool,
    module_weights: str,
    text_layer_enabled: bool,
    text_layer_weights: str,
) -> tuple[bool, str, bool, str, bool, str]:
    if not enabled:
        return block_enabled, block_weights, module_enabled, module_weights, text_layer_enabled, text_layer_weights

    presets = _parse_presets(presets_text)
    preset_name_key = _preset_key(preset_name)
    preset = presets.get(preset_name_key) if preset_name_key else next(iter(presets.values()), None)
    if preset is None:
        if preset_name_key:
            logger.warning("Anima layer weight preset not found: %s", preset_name)
        return block_enabled, block_weights, module_enabled, module_weights, text_layer_enabled, text_layer_weights

    return (
        True if preset.block_weights is not None else block_enabled,
        preset.block_weights if preset.block_weights is not None else block_weights,
        True if preset.module_weights is not None else module_enabled,
        preset.module_weights if preset.module_weights is not None else module_weights,
        True if preset.text_layer_weights is not None else text_layer_enabled,
        preset.text_layer_weights if preset.text_layer_weights is not None else text_layer_weights,
    )


def _targets_for_preset_scope(scope: str, target_loras: str) -> str:
    return "" if scope in {PRESET_SCOPE_ALL, PRESET_SCOPE_ALL_EN} else target_loras


def _target_mode_for_preset(enabled: bool, scope: str) -> str:
    if not enabled:
        return "legacy"
    return "all" if scope in {PRESET_SCOPE_ALL, PRESET_SCOPE_ALL_EN} else "targets"


def _select_preset(preset_name: str, presets_text: str) -> PresetWeights | None:
    presets = _parse_presets(presets_text)
    preset_name_key = _preset_key(preset_name)
    preset = presets.get(preset_name_key) if preset_name_key else next(iter(presets.values()), None)
    if preset is None and preset_name_key:
        logger.warning("Anima layer weight preset not found: %s", preset_name)
    return preset


def _preset_prompt_weight_text(preset: PresetWeights | None) -> str:
    if preset is None:
        return ""

    sections = []
    if preset.block_weights:
        sections.append(f"blocks={preset.block_weights}")
    if preset.module_weights:
        sections.append(f"modules={preset.module_weights}")
    return ";".join(sections)


def _apply_preset_to_prompt_text(
    prompt: str,
    target_loras: str,
    adapter_kind: str,
    preset_name: str,
    preset_scope: str,
    presets_text: str,
) -> str:
    prompt_text = str(prompt or "")
    weight_text = _preset_prompt_weight_text(_select_preset(preset_name, presets_text))
    if not weight_text:
        return prompt_text

    targets = _parse_targets(target_loras)
    target_mode = _target_mode_for_preset(True, preset_scope)

    def replace(match):
        name = match.group(1).strip()
        if not name:
            return match.group(0)

        if target_mode == "targets":
            detected_kind = _adapter_kind_for_prompt_name(name)
            controlled = (detected_kind in {None, adapter_kind}) and bool(targets) and bool(_parse_targets(name) & targets)
        else:
            controlled = _prompt_lora_is_controlled(name, targets, adapter_kind, target_mode)

        if not controlled:
            return match.group(0)

        raw_args = match.group(2) or ""
        _, extra_args = _split_lora_args(raw_args) if raw_args else ("", [])
        if extra_args:
            return f"<lora:{name}:{weight_text}:{':'.join(extra_args)}>"
        return f"<lora:{name}:{weight_text}>"

    return PROMPT_LORA_RE.sub(replace, prompt_text)


def _block_index_from_key(key: str) -> int | None:
    key_text = str(key or "")
    match = ANIMA_BLOCK_RE.search(key_text)
    if match:
        return int(match.group(1))

    normalized = key_text.lower().replace(".", "_")
    match = re.search(r"(?:^|_)blocks_(\d+)(?:_|$)", normalized)
    if not match:
        return None
    return int(match.group(1))


def _module_name_from_block_tail(module: str) -> str | None:
    if module.startswith("adaln_modulation"):
        return "adaln"
    if module.startswith("norm"):
        return "norm"
    if module.startswith("self_attn"):
        return "self_attn"
    if module.startswith("cross_attn"):
        return "cross_attn"
    if module.startswith("mlp"):
        return "mlp"
    return None


def _module_from_key(key: str) -> str | None:
    key_text = str(key or "")
    marker = re.search(r"blocks\.\d+\.([^.]+)", key_text)
    if marker:
        module = _module_name_from_block_tail(marker.group(1))
        return module or _module_key(marker.group(1))

    normalized = key_text.lower().replace(".", "_")
    marker = re.search(r"(?:^|_)blocks_\d+_(.+)", normalized)
    if not marker:
        return None

    return _module_name_from_block_tail(marker.group(1))


def _module_factor(key: str, module_weights: dict[str, float]) -> float:
    module = _module_from_key(key)
    if not module:
        return 1.0
    return module_weights.get(module, 1.0)


def _filter_layer_weights(weights: LayerWeights, trained_layers: set[int]) -> LayerWeights:
    if not trained_layers:
        return LayerWeights()
    values = {index: value for index, value in weights.values.items() if index in trained_layers}
    return LayerWeights(default=weights.default, values=values, inline=weights.inline)


def _filter_module_weights(weights: dict[str, float], trained_modules: set[str]) -> dict[str, float]:
    if not weights or not trained_modules:
        return {}
    return {module: value for module, value in weights.items() if module in trained_modules}


def _model_key(key) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple) and key:
        return str(key[0])
    return str(key or "")


def _names_for_lora_file(filename) -> set[str]:
    path = Path(str(filename))
    names = {str(path).lower(), path.name.lower(), path.stem.lower()}

    networks_module = sys.modules.get("networks")
    if networks_module is not None:
        available = getattr(networks_module, "available_networks", {})
        for name, entry in available.items():
            try:
                if os.path.abspath(str(entry.filename)) == os.path.abspath(str(filename)):
                    names.add(str(name).lower())
                    names.add(str(getattr(entry, "alias", "")).lower())
                    names.add(str(entry.get_alias()).lower())
            except Exception:
                continue

    return {name for name in names if name}


def _safe_open_keys(filename: str) -> list[str]:
    from safetensors import safe_open

    with safe_open(filename, framework="pt", device="cpu") as f:
        return list(f.keys())


def _lora_cache_key(filename):
    path = str(filename)
    if Path(path).suffix.lower() != ".safetensors":
        return None

    try:
        stat = os.stat(path)
    except OSError:
        return None

    return (os.path.abspath(path), stat.st_mtime, stat.st_size)


def _is_lora_weight_key(key: str) -> bool:
    key_lower = str(key or "").lower()
    return any(marker in key_lower for marker in LORA_MARKERS)


def _is_lokr_weight_key(key: str) -> bool:
    key_lower = str(key or "").lower()
    return any(marker in key_lower for marker in LOKR_MARKERS)


def _is_adapter_weight_key(key: str, adapter_kind: str) -> bool:
    if adapter_kind == "lokr":
        return _is_lokr_weight_key(key)
    return _is_lora_weight_key(key)


def _adapter_kind_for_file(filename) -> str | None:
    cache_key = _lora_cache_key(filename)
    if cache_key is None:
        return None

    cached = _KIND_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        keys = _safe_open_keys(str(filename))
    except Exception:
        _KIND_CACHE[cache_key] = None
        return None

    has_lokr = any(_is_lokr_weight_key(key) for key in keys)
    has_lora = any(_is_lora_weight_key(key) for key in keys)
    adapter_kind = "lokr" if has_lokr else "lora" if has_lora else None
    _KIND_CACHE[cache_key] = adapter_kind
    return adapter_kind


def _qwen_layer_index(key: str) -> int | None:
    key_text = str(key or "")
    match = QWEN_LAYER_RE.search(key_text) or GENERIC_QWEN_LAYER_RE.search(key_text) or QWEN_ADAPTER_BLOCK_RE.search(key_text) or LEGACY_QWEN_ADAPTER_BLOCK_RE.search(key_text)
    if match:
        return int(match.group(1))

    normalized = key_text.lower().replace(".", "_")
    match = (
        re.search(r"qwen3_06b_(?:model_)?layers_(\d+)", normalized)
        or re.search(r"qwen3_06b_llm_adapter_blocks_(\d+)", normalized)
        or re.search(r"(?:^|_)llm_adapter_blocks_(\d+)", normalized)
    )
    if not match:
        return None
    return int(match.group(1))


def _lora_coverage(filename, adapter_kind: str = "lora") -> LoraCoverage:
    cache_key = _lora_cache_key(filename)
    if cache_key is None:
        return LoraCoverage(blocks=set(), modules=set(), text_layers=set())

    coverage_key = (cache_key, adapter_kind)
    cached = _COVERAGE_CACHE.get(coverage_key)
    if cached is not None:
        return cached

    try:
        keys = _safe_open_keys(str(filename))
    except Exception:
        coverage = LoraCoverage(blocks=set(), modules=set(), text_layers=set())
        _COVERAGE_CACHE[coverage_key] = coverage
        return coverage

    blocks = set()
    modules = set()
    text_layers = set()

    for key in keys:
        if not _is_adapter_weight_key(key, adapter_kind):
            continue

        text_layer = _qwen_layer_index(key)
        if text_layer is not None:
            text_layers.add(text_layer)
            continue

        block_index = _block_index_from_key(key)
        if block_index is not None:
            blocks.add(block_index)
            module = _module_from_key(key)
            if module:
                modules.add(module)

    coverage = LoraCoverage(blocks=blocks, modules=modules, text_layers=text_layers)
    _COVERAGE_CACHE[coverage_key] = coverage
    return coverage


def _is_anima_adapter_file(filename, adapter_kind: str) -> bool:
    path = str(filename)
    cache_key = _lora_cache_key(path)
    if cache_key is None:
        return False

    scan_key = (cache_key, adapter_kind)
    cached = _SCAN_CACHE.get(scan_key)
    if cached is not None:
        return cached

    coverage = _lora_coverage(path, adapter_kind)
    is_anima = bool(coverage.blocks)
    _SCAN_CACHE[scan_key] = is_anima
    return is_anima


def _is_anima_lora_file(filename) -> bool:
    return _is_anima_adapter_file(filename, "lora")


def _is_anima_lokr_file(filename) -> bool:
    return _is_anima_adapter_file(filename, "lokr")


def _has_qwen_text_encoder_file(filename) -> bool:
    path = str(filename)
    cache_key = _lora_cache_key(path)
    if cache_key is None:
        return False

    adapter_kind = _adapter_kind_for_file(path) or "lora"
    scan_key = (cache_key, adapter_kind)
    cached = _QWEN_SCAN_CACHE.get(scan_key)
    if cached is not None:
        return cached

    has_qwen = bool(_lora_coverage(path, adapter_kind).text_layers)
    _QWEN_SCAN_CACHE[scan_key] = has_qwen
    return has_qwen


def _network_on_disk_for_prompt_name(name):
    networks_module = sys.modules.get("networks")
    if networks_module is None:
        try:
            import networks as networks_module  # type: ignore
        except Exception:
            return None

    try:
        available = getattr(networks_module, "available_networks", {})
        aliases = getattr(networks_module, "available_network_aliases", {})
        forbidden = getattr(networks_module, "forbidden_network_aliases", set())

        name_text = str(name)
        name_lower = name_text.lower()
        if name_lower in forbidden:
            entry = _dict_get_case_insensitive(available, name_text)
        else:
            entry = _dict_get_case_insensitive(aliases, name_text) or _dict_get_case_insensitive(available, name_text)

        if entry is None and hasattr(networks_module, "update_available_networks_by_names"):
            networks_module.update_available_networks_by_names([name_text])
            available = getattr(networks_module, "available_networks", {})
            aliases = getattr(networks_module, "available_network_aliases", {})
            entry = _dict_get_case_insensitive(aliases, name_text) or _dict_get_case_insensitive(available, name_text)
        return entry
    except Exception:
        return None


def _dict_get_case_insensitive(mapping, key):
    if key in mapping:
        return mapping[key]

    key_lower = str(key).lower()
    for item_key, value in mapping.items():
        if str(item_key).lower() == key_lower:
            return value

    key_stem = Path(str(key)).stem.lower()
    for item_key, value in mapping.items():
        if Path(str(item_key)).stem.lower() == key_stem:
            return value

    return None


def _adapter_kind_for_prompt_name(name: str) -> str | None:
    entry = _network_on_disk_for_prompt_name(name)
    if entry is None:
        return None
    return _adapter_kind_for_file(getattr(entry, "filename", ""))


def _is_anima_prompt_lora(name: str, adapter_kind: str = "lora") -> bool:
    entry = _network_on_disk_for_prompt_name(name)
    if entry is None:
        return False
    return _is_anima_adapter_file(getattr(entry, "filename", ""), adapter_kind)


def _prompt_lora_is_controlled(name: str, targets: set[str], adapter_kind: str = "lora", target_mode: str = "legacy") -> bool:
    if _adapter_kind_for_prompt_name(name) != adapter_kind:
        return False

    if target_mode == "all":
        return _is_anima_prompt_lora(name, adapter_kind)

    name_keys = _parse_targets(name)
    if target_mode == "targets":
        return bool(targets) and bool(name_keys & targets)

    if targets:
        return bool(name_keys & targets)
    return _is_anima_prompt_lora(name, adapter_kind)


def _first_inline_rule(names: set[str], inline_rules: dict[str, PromptRule]) -> PromptRule | None:
    for name in names:
        rule = inline_rules.get(name)
        if rule is not None:
            return rule
    return None


def _register_inline_rule(rules: dict[str, PromptRule], name: str, rule: PromptRule):
    clean_name = str(name or "").strip().lower()
    if not clean_name:
        return
    path = Path(clean_name)
    aliases = {
        key: rule
        for key in {clean_name, path.name.lower(), path.stem.lower()}
        if key
    }
    merge_consistent_rules(rules, aliases, label="Anima adapter")


def _split_lora_args(text: str) -> tuple[str, list[str]]:
    parts = str(text or "").split(":")
    return parts[0].strip(), parts[1:]


def _split_inline_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    bare_parts = []
    for part in [p.strip() for p in re.split(r"[;；]+", str(text or "")) if p.strip()]:
        if "=" in part:
            left, right = part.split("=", 1)
            key = left.strip().lower()
            if key in {"blocks", "block", "layers", "layer"}:
                sections["blocks"] = right.strip()
                continue
            if key in {"modules", "module", "mods", "mod"}:
                sections["modules"] = right.strip()
                continue
            if key in {"qwen", "text", "text_encoder", "text_layers", "te"}:
                continue
        bare_parts.append(part)
    if bare_parts:
        if "blocks" in sections:
            sections["blocks"] = ";".join([sections["blocks"], *bare_parts])
        else:
            sections["blocks"] = ";".join(bare_parts)
    return sections


def _parse_inline_prompt_rule(text: str) -> PromptRule | None:
    sections = _split_inline_sections(text)
    block_text = sections.get("blocks", "")
    module_text = sections.get("modules", "")
    if not block_text and not module_text:
        return None
    if block_text and not _looks_like_layer_weights(block_text):
        return None

    block_weights = _parse_layer_weights(block_text, "") if block_text else LayerWeights()
    module_weights = _parse_module_weights(module_text) if module_text else {}
    return PromptRule(block_weights=block_weights, module_weights=module_weights)


def _rewrite_lora_prompts(prompts, target_loras: str, adapter_kind: str, target_mode: str) -> tuple[list[str], dict[str, PromptRule]]:
    targets = _parse_targets(target_loras)
    inline_rules: dict[str, PromptRule] = {}

    def rewrite_prompt(prompt):
        def replace(match):
            name = match.group(1).strip()
            weight_text, extra_args = _split_lora_args(match.group(2).strip())
            controlled = _prompt_lora_is_controlled(name, targets, adapter_kind, target_mode)

            rule = _parse_inline_prompt_rule(weight_text)
            if rule is not None and controlled:
                _register_inline_rule(inline_rules, name, rule)
                return f"<lora:{name}:1.0>"

            if controlled and _is_plain_number(weight_text):
                return f"<lora:{name}:1.0>"

            return match.group(0)

        return INLINE_LORA_RE.sub(replace, prompt)

    return [rewrite_prompt(prompt) for prompt in prompts], inline_rules


def install_patch():
    global _ORIGINAL_ADD_PATCHES

    from backend.patcher.base import ModelPatcher
    from backend.patcher.function_chain import function_chain_contains

    if function_chain_contains(ModelPatcher.add_patches, "_anima_layer_weight_patched"):
        return

    _ORIGINAL_ADD_PATCHES = ModelPatcher.add_patches

    def add_patches_with_anima_layers(self, patches: list[dict], strength_patch: float = 1.0, strength_model: float = 1.0, *, filename: str = None, online_mode: bool = None):
        rule = _ACTIVE_RULE
        decision = rule.decision_for_lora(filename) if rule is not None else None
        if decision is None or not decision.apply:
            return _ORIGINAL_ADD_PATCHES(
                self,
                patches,
                strength_patch=strength_patch,
                strength_model=strength_model,
                filename=filename,
                online_mode=online_mode,
            )

        # Delegate tuple construction to the live Forge implementation. This
        # preserves Forge Neo's sixth online-mode field and composes with the
        # stage scheduler regardless of extension load order.
        loaded = set()
        for patch_key, patch_value in patches.items():
            model_key = patch_key if isinstance(patch_key, str) else patch_key[0]
            key_strength = rule.factor_for_key(model_key, decision)
            loaded.update(
                _ORIGINAL_ADD_PATCHES(
                    self,
                    {patch_key: patch_value},
                    strength_patch=key_strength,
                    strength_model=strength_model,
                    filename=filename,
                    online_mode=online_mode,
                )
            )
        return list(loaded)

    add_patches_with_anima_layers._anima_layer_weight_patched = True
    add_patches_with_anima_layers.__wrapped__ = _ORIGINAL_ADD_PATCHES
    ModelPatcher.add_patches = add_patches_with_anima_layers
    logger.info("Installed Anima LoRA layer weight patch")


def _reset_lora_cache_if_needed(p, signature: str):
    global _LAST_SIGNATURE
    if _LAST_SIGNATURE == signature:
        return

    _LAST_SIGNATURE = signature
    sd_model = getattr(p, "sd_model", None)
    if sd_model is not None and hasattr(sd_model, "current_lora_hash"):
        sd_model.current_lora_hash = f"anima-layer-weight:{signature}"


def _apply_prompt_rewrite(p, target_loras: str, adapter_kind: str, target_mode: str) -> dict[str, PromptRule]:
    inline_rules = {}
    start = getattr(p, "iteration", 0) * getattr(p, "batch_size", 1)
    end = start + getattr(p, "batch_size", 1)
    replacements = []

    if hasattr(p, "prompts") and p.prompts:
        rewritten, rules = _rewrite_lora_prompts(p.prompts, target_loras, adapter_kind, target_mode)
        merge_consistent_rules(inline_rules, rules, label="Anima adapter")
        replacements.append(("prompts", None, rewritten))

    if hasattr(p, "all_prompts") and p.all_prompts:
        current, rules = _rewrite_lora_prompts(p.all_prompts[start:end], target_loras, adapter_kind, target_mode)
        merge_consistent_rules(inline_rules, rules, label="Anima adapter")
        replacements.append(("all_prompts", slice(start, end), current))

    if hasattr(p, "hr_prompts") and p.hr_prompts:
        rewritten, rules = _rewrite_lora_prompts(p.hr_prompts, target_loras, adapter_kind, target_mode)
        merge_consistent_rules(inline_rules, rules, label="Anima adapter")
        replacements.append(("hr_prompts", None, rewritten))

    if hasattr(p, "all_hr_prompts") and p.all_hr_prompts:
        current, rules = _rewrite_lora_prompts(p.all_hr_prompts[start:end], target_loras, adapter_kind, target_mode)
        merge_consistent_rules(inline_rules, rules, label="Anima adapter")
        replacements.append(("all_hr_prompts", slice(start, end), current))

    # Validation above is intentionally transactional. Script hook failures are
    # reported and sampling continues, so never leave prompts half rewritten.
    for attribute, selection, values in replacements:
        if selection is None:
            setattr(p, attribute, values)
        else:
            getattr(p, attribute)[selection] = values

    stored_rules = getattr(p, "_anima_lora_inline_rules", {})
    if not isinstance(stored_rules, dict):
        stored_rules = {}
    stored_rules[adapter_kind] = inline_rules
    p._anima_lora_inline_rules = stored_rules
    return inline_rules


def _lora_tab(adapter_kind: str, elem_id, default_target: str = ""):
    target_placeholder = _t("target_lora_placeholder") if adapter_kind == "lora" else _t("target_lokr_placeholder")
    block_placeholder = _t("block_placeholder")
    module_placeholder = _t("module_placeholder")
    qwen_placeholder = _t("qwen_placeholder")
    scope_all = _t("scope_all")
    scope_targets = _t("scope_targets")

    target_loras = gr.Textbox(label=_t("target_lora") if adapter_kind == "lora" else _t("target_lokr"), value=default_target, lines=1, placeholder=target_placeholder, elem_id=elem_id(f"{adapter_kind}_target_loras"))
    with gr.Row():
        block_enabled = gr.Checkbox(label=_t("block_enabled"), value=True, elem_id=elem_id(f"{adapter_kind}_block_enabled"))
        module_enabled = gr.Checkbox(label=_t("module_enabled"), value=False, elem_id=elem_id(f"{adapter_kind}_module_enabled"))
        text_layer_enabled = gr.Checkbox(label=_t("qwen_enabled"), value=False, elem_id=elem_id(f"{adapter_kind}_text_layer_enabled"))

    with gr.Row():
        block_weights = gr.Textbox(label=_t("block_weights"), value=DEFAULT_BLOCK_WEIGHTS, lines=3, placeholder=block_placeholder, elem_id=elem_id(f"{adapter_kind}_block_weights"))
        module_weights = gr.Textbox(label=_t("module_weights"), value=DEFAULT_MODULE_WEIGHTS, lines=5, placeholder=module_placeholder, elem_id=elem_id(f"{adapter_kind}_module_weights"))

    text_layer_weights = gr.Textbox(label=_t("qwen_weights"), value="", lines=3, placeholder=qwen_placeholder, elem_id=elem_id(f"{adapter_kind}_text_layer_weights"))
    preset_enabled = gr.Checkbox(label=_t("preset_enabled"), value=False, visible=False, elem_id=elem_id(f"{adapter_kind}_preset_enabled"))
    preset_name = gr.Dropdown(
        label=_t("preset_name"),
        choices=_preset_choices(DEFAULT_PRESETS_TEXT),
        value="平衡",
        allow_custom_value=True,
        interactive=True,
        elem_id=elem_id(f"{adapter_kind}_preset_name"),
    )
    preset_scope = gr.Dropdown(
        label=_t("preset_scope"),
        choices=[scope_all, scope_targets],
        value=scope_all,
        interactive=True,
        elem_id=elem_id(f"{adapter_kind}_preset_scope"),
    )
    preset_text = gr.Textbox(
        label=_t("custom_presets"),
        value=DEFAULT_PRESETS_TEXT,
        lines=5,
        placeholder=_t("preset_placeholder"),
        elem_id=elem_id(f"{adapter_kind}_preset_text"),
    )
    preset_apply = gr.Button(_t("apply_preset"), variant="primary", elem_id=elem_id(f"{adapter_kind}_preset_apply"))
    preset_text.change(fn=_preset_dropdown_update, inputs=[preset_text, preset_name], outputs=[preset_name])
    return target_loras, block_enabled, block_weights, module_enabled, module_weights, text_layer_enabled, text_layer_weights, preset_enabled, preset_name, preset_scope, preset_text, preset_apply


class Script(scripts.Script):
    def __init__(self):
        self._prompt_component = None
        self._preset_apply_bindings = []
        self._bound_preset_apply_ids = set()

    def title(self):
        return _t("title")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def _bind_preset_apply_button(self, button, target_loras, preset_name, preset_scope, preset_text, adapter_kind: str):
        binding = (button, target_loras, preset_name, preset_scope, preset_text, adapter_kind)
        self._preset_apply_bindings.append(binding)
        self._bind_preset_apply_binding(binding)

    def _bind_preset_apply_binding(self, binding):
        if self._prompt_component is None:
            return

        button, target_loras, preset_name, preset_scope, preset_text, adapter_kind = binding
        binding_id = getattr(button, "elem_id", None) or id(button)
        if binding_id in self._bound_preset_apply_ids:
            return

        def apply_preset(prompt, target_loras_value, preset_name_value, preset_scope_value, preset_text_value):
            return _apply_preset_to_prompt_text(
                prompt,
                target_loras_value,
                adapter_kind,
                preset_name_value,
                preset_scope_value,
                preset_text_value,
            )

        button.click(
            fn=apply_preset,
            inputs=[self._prompt_component, target_loras, preset_name, preset_scope, preset_text],
            outputs=[self._prompt_component],
            queue=False,
            show_progress=False,
        )
        self._bound_preset_apply_ids.add(binding_id)

    def after_component(self, component, **kwargs):
        if getattr(component, "elem_id", None) in {"txt2img_prompt", "img2img_prompt"}:
            self._prompt_component = component
            for binding in self._preset_apply_bindings:
                self._bind_preset_apply_binding(binding)

    def ui(self, is_img2img):
        with gr.Accordion(_t("accordion"), open=False, elem_id=self.elem_id("accordion")):
            _intro_block(self.elem_id)
            with gr.Tabs(elem_id=self.elem_id("tabs")):
                with gr.TabItem("LoRA"):
                    lora_enabled = gr.Checkbox(label=_t("enable_lora"), value=False, elem_id=self.elem_id("lora_enabled"))
                    lora_target_loras, lora_block_enabled, lora_block_weights, lora_module_enabled, lora_module_weights, lora_text_layer_enabled, lora_text_layer_weights, lora_preset_enabled, lora_preset_name, lora_preset_scope, lora_preset_text, lora_preset_apply = _lora_tab("lora", self.elem_id)
                    gr.Textbox(label=_t("lora_usage"), value=_usage_text(), lines=12, interactive=False, elem_id=self.elem_id("lora_usage"))

                with gr.TabItem("LoKR"):
                    lokr_enabled = gr.Checkbox(label=_t("enable_lokr"), value=False, elem_id=self.elem_id("lokr_enabled"))
                    lokr_target_loras, lokr_block_enabled, lokr_block_weights, lokr_module_enabled, lokr_module_weights, lokr_text_layer_enabled, lokr_text_layer_weights, lokr_preset_enabled, lokr_preset_name, lokr_preset_scope, lokr_preset_text, lokr_preset_apply = _lora_tab("lokr", self.elem_id)
                    gr.Textbox(label=_t("lokr_usage"), value=_usage_text(), lines=12, interactive=False, elem_id=self.elem_id("lokr_usage"))

        self._bind_preset_apply_button(lora_preset_apply, lora_target_loras, lora_preset_name, lora_preset_scope, lora_preset_text, "lora")
        self._bind_preset_apply_button(lokr_preset_apply, lokr_target_loras, lokr_preset_name, lokr_preset_scope, lokr_preset_text, "lokr")

        return [
            lora_enabled,
            lora_target_loras,
            lora_block_enabled,
            lora_block_weights,
            lora_module_enabled,
            lora_module_weights,
            lora_text_layer_enabled,
            lora_text_layer_weights,
            lora_preset_enabled,
            lora_preset_name,
            lora_preset_scope,
            lora_preset_text,
            lokr_enabled,
            lokr_target_loras,
            lokr_block_enabled,
            lokr_block_weights,
            lokr_module_enabled,
            lokr_module_weights,
            lokr_text_layer_enabled,
            lokr_text_layer_weights,
            lokr_preset_enabled,
            lokr_preset_name,
            lokr_preset_scope,
            lokr_preset_text,
        ]

    def before_process_batch(
        self,
        p,
        lora_enabled,
        lora_target_loras,
        lora_block_enabled,
        lora_block_weights,
        lora_module_enabled,
        lora_module_weights,
        lora_text_layer_enabled,
        lora_text_layer_weights,
        lora_preset_enabled,
        lora_preset_name,
        lora_preset_scope,
        lora_preset_text,
        lokr_enabled,
        lokr_target_loras,
        lokr_block_enabled,
        lokr_block_weights,
        lokr_module_enabled,
        lokr_module_weights,
        lokr_text_layer_enabled,
        lokr_text_layer_weights,
        lokr_preset_enabled,
        lokr_preset_name,
        lokr_preset_scope,
        lokr_preset_text,
        *args,
        **kwargs,
    ):
        global _ACTIVE_RULE

        install_patch()
        _ACTIVE_RULE = None
        p._anima_lora_inline_rules = {}
        lora_preset_enabled = False
        lokr_preset_enabled = False
        lora_target_mode = _target_mode_for_preset(lora_preset_enabled, lora_preset_scope)
        lokr_target_mode = _target_mode_for_preset(lokr_preset_enabled, lokr_preset_scope)
        lora_block_enabled, lora_block_weights, lora_module_enabled, lora_module_weights, lora_text_layer_enabled, lora_text_layer_weights = _apply_preset_values(
            lora_preset_enabled,
            lora_preset_name,
            lora_preset_text,
            lora_block_enabled,
            lora_block_weights,
            lora_module_enabled,
            lora_module_weights,
            lora_text_layer_enabled,
            lora_text_layer_weights,
        )
        lokr_block_enabled, lokr_block_weights, lokr_module_enabled, lokr_module_weights, lokr_text_layer_enabled, lokr_text_layer_weights = _apply_preset_values(
            lokr_preset_enabled,
            lokr_preset_name,
            lokr_preset_text,
            lokr_block_enabled,
            lokr_block_weights,
            lokr_module_enabled,
            lokr_module_weights,
            lokr_text_layer_enabled,
            lokr_text_layer_weights,
        )
        try:
            lora_inline_rules = _apply_prompt_rewrite(p, lora_target_loras, "lora", lora_target_mode) if lora_enabled else {}
            lokr_inline_rules = _apply_prompt_rewrite(p, lokr_target_loras, "lokr", lokr_target_mode) if lokr_enabled else {}
        except ValueError as error:
            raise scripts.ScriptAbort(str(error)) from error
        _ACTIVE_RULE = ActiveRule(
            lora_enabled,
            lora_target_loras,
            lora_target_mode,
            lora_block_enabled,
            lora_block_weights,
            lora_module_enabled,
            lora_module_weights,
            lora_text_layer_enabled,
            lora_text_layer_weights,
            lora_inline_rules,
            lokr_enabled,
            lokr_target_loras,
            lokr_target_mode,
            lokr_block_enabled,
            lokr_block_weights,
            lokr_module_enabled,
            lokr_module_weights,
            lokr_text_layer_enabled,
            lokr_text_layer_weights,
            lokr_inline_rules,
        )
        _reset_lora_cache_if_needed(p, _ACTIVE_RULE.signature)

        if _ACTIVE_RULE.enabled:
            p.extra_generation_params["Anima LoRA/LoKR Layer Weight"] = _ACTIVE_RULE.signature

    def postprocess(self, p, processed, *args, **kwargs):
        global _ACTIVE_RULE
        _ACTIVE_RULE = None

    def cleanup(self, p, *args, **kwargs):
        global _ACTIVE_RULE
        _ACTIVE_RULE = None
