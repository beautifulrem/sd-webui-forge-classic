import logging
from dataclasses import dataclass

import gradio as gr

from modules import script_callbacks, scripts
from modules.anima_lora_support import (
    LORA_TAG_RE,
    LayerWeights,
    invalidate_cond_caches,
    install_add_patches_hook,
    lora_control_files,
    note_lora_control_yield,
    rewrite_batch_prompts,
    looks_like_layer_weights as _looks_like_layer_weights,
    parse_layer_weights,
    parse_module_weights,
    lora_file_names as _names_for_lora_file,
    name_keys,
    parse_targets as _parse_targets,
    resolve_network,
    split_weight_sections,
    anima_lora_block_count,
    lora_key_factor,
    merge_consistent_rules,
    compact_text as _compact_text,
    is_plain_number as _is_plain_number,
)
from modules.anima_lora_ui import intro_block, register_language_option, ui_language, ui_text
from modules.anima_lora_files import adapter_kind_for_file as _adapter_kind_for_file
from modules.anima_lora_files import is_anima_adapter_file as _is_anima_adapter_file
from modules.anima_lora_files import lora_coverage as _lora_coverage
from modules.anima_presets import register_preset_control


logger = logging.getLogger("anima_lora_layer_weight")
LANGUAGE_OPTION = "anima_lora_layer_weight_language"
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
        "module_placeholder": "self_attn/cross_attn/mlp/adaln",
        "qwen_placeholder": "留空=文本编码器保持原权重；示例: 0-31=1.0",
        "preset_enabled": "启用预设",
        "preset_name": "预设名称",
        "preset_scope": "预设应用范围",
        "scope_all": "提示词里的全部同类型",
        "scope_targets": "仅目标列表",
        "custom_presets": "自定义预设",
        "preset_placeholder": "每行一个预设：名称=blocks=...;modules=...",
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
        "module_placeholder": "self_attn/cross_attn/mlp/adaln",
        "qwen_placeholder": "Empty = keep text encoder unchanged; example: 0-31=1.0",
        "preset_enabled": "Enable preset",
        "preset_name": "Preset name",
        "preset_scope": "Preset apply scope",
        "scope_all": "All matching type in prompt",
        "scope_targets": "Target list only",
        "custom_presets": "Custom presets",
        "preset_placeholder": "One preset per line: name=blocks=...;modules=...",
        "apply_preset": "Apply preset to prompt",
        "lora_usage": "LoRA usage guide",
        "lokr_usage": "LoKR usage guide",
    },
}


def _language():
    return ui_language(LANGUAGE_OPTION)


def _t(key):
    return ui_text(LANG, LANGUAGE_OPTION, key)


INTRO_EN = (
    "Control Anima LoRA/LoKR patch strength by block range, module type, and "
    "optional Qwen text-encoder layers. Use it to tune composition, character "
    "features, and style without editing model files."
)
INTRO_ZH = "按 block、模块和可选 Qwen 文本编码器层控制 Anima LoRA/LoKR 强度，用于调节构图、人物特征和画风，不修改模型文件。"


def _register_ui_settings():
    register_language_option(
        LANGUAGE_OPTION,
        "Anima LoRA Layer Weight language / Anima LoRA 分层权重语言",
        ("anima-lora-layer-weight", "Anima LoRA Layer Weight"),
    )


script_callbacks.on_ui_settings(_register_ui_settings)

DEFAULT_BLOCK_WEIGHTS = "all=1.0"
DEFAULT_MODULE_WEIGHTS = "\n".join(
    [
        "self_attn=1.0",
        "cross_attn=1.0",
        "mlp=1.0",
        "adaln=1.0",
    ]
)
PRESET_SCOPE_ALL = "提示词里的全部同类型"
PRESET_SCOPE_ALL_EN = "All matching type in prompt"
DEFAULT_PRESETS_TEXT = "\n".join(
    [
        "平衡=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,cross_attn=1.0,mlp=1.0,adaln=1.0",
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
13. 内联 block 分层也支持直接写 28 个数字，例如 <lora:灵魂潮汐:1,1,0.9,0.9,0.8,...>，顺序对应 blocks 0 到 27。在 40/52 层的扩展模型（Anima 2.9B/3.8B）上使用 28 层 LoRA 时，block 编号仍按 LoRA 自己的 28 层计算，插入的复制层跟随其来源层。
14. 可以在预设列表里写多行命名预设，例如 平衡=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8。预设只含 blocks 和 modules，Qwen 层权重始终来自面板。
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
13. Inline block weights also support 28 comma-separated numbers in block order from 0 to 27. On depth-expanded models (Anima 2.9B/3.8B) a 28-block LoRA keeps its own block numbering; inserted copies follow the block they were copied from.
14. Presets can be written one per line, for example Balanced=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8. Presets hold blocks and modules; Qwen layer weights always come from the panel.
15. "Apply preset to prompt" replaces matching LoRA/LoKR prompt weights with the preset's inline blocks/modules syntax.
16. The apply scope can affect all matching adapters in the prompt, or only the adapters listed in the target field.
"""


def _usage_text():
    return USAGE_TEXT_EN if _language() == "en" else USAGE_TEXT

_ACTIVE_RULE = None
_LAST_SIGNATURE = None
# ActiveRule.text_encoder_signature of the last batch
_LAST_TEXT_ENCODER_SIGNATURE = None


@dataclass
class PromptRule:
    block_weights: LayerWeights
    module_weights: dict[str, float]


@dataclass
class PresetWeights:
    block_weights: str | None = None
    module_weights: str | None = None


@dataclass
class LoraDecision:
    apply: bool
    block_weights: LayerWeights
    module_weights: dict[str, float]
    text_layer_weights: LayerWeights
    lora_blocks: int | None = None


class AdapterRule:
    def __init__(
        self,
        adapter_kind: str,
        enabled: bool,
        target_loras: str,
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
        if not self.enabled:
            # A disabled tab weights nothing: its fields are not parsed, and
            # editing them must not change the signature (a LoRA reload)
            target_loras = block_weights = module_weights = text_layer_weights = ""
            block_enabled = module_enabled = text_layer_enabled = False
        self.targets = _parse_targets(target_loras)
        self.block_enabled = bool(block_enabled)
        self.module_enabled = bool(module_enabled)
        self.text_layer_enabled = bool(text_layer_enabled)
        self.block_weights = _parse_layer_weights(block_weights, DEFAULT_BLOCK_WEIGHTS) if self.block_enabled else LayerWeights()
        self.module_weights = _parse_module_weights(module_weights) if self.module_enabled else {}
        self.text_layer_weights = _parse_layer_weights(text_layer_weights, "") if self.text_layer_enabled else LayerWeights()
        self.inline_rules = inline_rules
        self.signature = self._make_signature(target_loras, block_weights, module_weights, text_layer_weights)
        # What the text-encoder patches, so the prompt conditioning, depend on
        self.text_encoder_signature = (frozenset(self.targets), _compact_text(text_layer_weights)) if self.enabled and self.text_layer_enabled else None

    def _make_signature(self, target_loras, block_weights, module_weights, text_layer_weights):
        inline_sig = ",".join(
            sorted(f"{name}:{rule.block_weights.default}:{rule.block_weights.values}:{rule.module_weights}" for name, rule in self.inline_rules.items())
        )
        return "|".join(
            [
                self.adapter_kind,
                "1" if self.enabled else "0",
                _compact_text(target_loras),
                "target_mode=legacy",  # the only mode left; kept so infotexts stay comparable
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
            return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights)

        if _adapter_kind_for_file(filename) != self.adapter_kind:
            return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights)

        names = _names_for_lora_file(filename)
        inline_rule = _first_inline_rule(names, self.inline_rules)
        coverage = _lora_coverage(filename, self.adapter_kind)
        text_layer_weights = _filter_layer_weights(self.text_layer_weights, coverage.text_layers) if self.text_layer_enabled else LayerWeights()
        if inline_rule is not None:
            return LoraDecision(True, _filter_layer_weights(inline_rule.block_weights, coverage.blocks), _filter_module_weights(inline_rule.module_weights, coverage.modules), text_layer_weights, anima_lora_block_count(coverage.blocks))

        if self.targets:
            if not any(name in self.targets for name in names):
                return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights)
        elif not _is_anima_adapter_file(filename, self.adapter_kind):
            return LoraDecision(False, self.block_weights, self.module_weights, self.text_layer_weights)
        return LoraDecision(True, _filter_layer_weights(self.block_weights, coverage.blocks), _filter_module_weights(self.module_weights, coverage.modules), text_layer_weights, anima_lora_block_count(coverage.blocks))


class ActiveRule:
    def __init__(
        self,
        lora_enabled: bool,
        lora_targets: str,
        lora_block_enabled: bool,
        lora_block_weights: str,
        lora_module_enabled: bool,
        lora_module_weights: str,
        lora_text_layer_enabled: bool,
        lora_text_layer_weights: str,
        lora_inline_rules: dict[str, PromptRule],
        lokr_enabled: bool,
        lokr_targets: str,
        lokr_block_enabled: bool,
        lokr_block_weights: str,
        lokr_module_enabled: bool,
        lokr_module_weights: str,
        lokr_text_layer_enabled: bool,
        lokr_text_layer_weights: str,
        lokr_inline_rules: dict[str, PromptRule],
        extra_generation_params: dict | None = None,
    ):
        # The run's infotext, and the files left to LoRA Control noted in it
        self.extra_generation_params = extra_generation_params
        self.yielded_to_lora_control: set = set()
        self.lora_rule = AdapterRule("lora", lora_enabled, lora_targets, lora_block_enabled, lora_block_weights, lora_module_enabled, lora_module_weights, lora_text_layer_enabled, lora_text_layer_weights, lora_inline_rules)
        self.lokr_rule = AdapterRule("lokr", lokr_enabled, lokr_targets, lokr_block_enabled, lokr_block_weights, lokr_module_enabled, lokr_module_weights, lokr_text_layer_enabled, lokr_text_layer_weights, lokr_inline_rules)
        self.rules = {
            "lora": self.lora_rule,
            "lokr": self.lokr_rule,
        }
        self.enabled = self.lora_rule.enabled or self.lokr_rule.enabled
        self.signature = f"lora[{self.lora_rule.signature}]|lokr[{self.lokr_rule.signature}]"
        text_encoder = (self.lora_rule.text_encoder_signature, self.lokr_rule.text_encoder_signature)
        self.text_encoder_signature = text_encoder if any(text_encoder) else None

    def decision_for_lora(self, filename) -> LoraDecision:
        adapter_kind = _adapter_kind_for_file(filename)
        rule = self.rules.get(adapter_kind)
        if rule is None:
            return LoraDecision(False, LayerWeights(), {}, LayerWeights())
        return rule.decision_for_lora(filename)

    def factor_for_key(self, model_key, decision: LoraDecision, model_blocks: int | None = None) -> float:
        return lora_key_factor(
            model_key,
            qwen_weights=decision.text_layer_weights,
            block_weights=decision.block_weights,
            module_weights=decision.module_weights,
            lora_blocks=decision.lora_blocks,
            model_blocks=model_blocks,
        )


# Modules a LoRA can train in an Anima block (its LayerNorms have no weights)
_DEFAULT_MODULE_FACTORS = dict.fromkeys(("self_attn", "cross_attn", "mlp", "adaln"), 1.0)


def _parse_layer_weights(text, fallback_text) -> LayerWeights:
    return parse_layer_weights(str(text or "").strip() or fallback_text)


def _parse_module_weights(text) -> dict[str, float]:
    return {**_DEFAULT_MODULE_FACTORS, **parse_module_weights(text or DEFAULT_MODULE_WEIGHTS)}


def _preset_key(name: str) -> str:
    return _compact_text(name).lower()


def _parse_preset_sections(text: str) -> PresetWeights:
    sections = split_weight_sections(text)  # Qwen layer weights come from the panel only
    return PresetWeights(
        block_weights=sections.get("blocks") or None,
        module_weights=sections.get("modules") or None,
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


def _preset_target_mode(scope: str) -> str:
    """Which prompt adapters the "Apply preset to prompt" button rewrites."""
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
    target_mode = _preset_target_mode(preset_scope)

    def replace(match):
        name = match.group(1).strip()
        if not name:
            return match.group(0)

        if target_mode == "targets":
            detected_kind = _adapter_kind_for_prompt_name(name)
            controlled = (detected_kind in {None, adapter_kind}) and bool(targets) and bool(name_keys(name) & targets)
        else:
            controlled = _prompt_lora_is_controlled(name, set(), adapter_kind)

        if not controlled:
            return match.group(0)

        raw_args = match.group(2) or ""
        _, extra_args = _split_lora_args(raw_args) if raw_args else ("", [])
        if extra_args:
            return f"<lora:{name}:{weight_text}:{':'.join(extra_args)}>"
        return f"<lora:{name}:{weight_text}>"

    return LORA_TAG_RE.sub(replace, prompt_text)


def _filter_layer_weights(weights: LayerWeights, trained_layers: set[int]) -> LayerWeights:
    if not trained_layers:
        return LayerWeights()
    values = {index: value for index, value in weights.values.items() if index in trained_layers}
    return LayerWeights(default=weights.default, values=values)


def _filter_module_weights(weights: dict[str, float], trained_modules: set[str]) -> dict[str, float]:
    if not weights or not trained_modules:
        return {}
    return {module: value for module, value in weights.items() if module in trained_modules}


def _adapter_kind_for_prompt_name(name: str) -> str | None:
    entry = resolve_network(name)
    if entry is None:
        return None
    return _adapter_kind_for_file(getattr(entry, "filename", ""))


def _is_anima_prompt_lora(name: str, adapter_kind: str = "lora") -> bool:
    entry = resolve_network(name)
    if entry is None:
        return False
    return _is_anima_adapter_file(getattr(entry, "filename", ""), adapter_kind)


def _prompt_lora_is_controlled(name: str, targets: set[str], adapter_kind: str = "lora") -> bool:
    """A listed target, or with no targets any Anima adapter of the kind."""
    if _adapter_kind_for_prompt_name(name) != adapter_kind:
        return False
    if targets:
        return bool(name_keys(name) & targets)
    return _is_anima_prompt_lora(name, adapter_kind)


def _first_inline_rule(names: set[str], inline_rules: dict[str, PromptRule]) -> PromptRule | None:
    for name in names:
        rule = inline_rules.get(name)
        if rule is not None:
            return rule
    return None


def _register_inline_rule(rules: dict[str, PromptRule], name: str, rule: PromptRule):
    merge_consistent_rules(rules, dict.fromkeys(name_keys(name), rule), label="Anima adapter")


def _split_lora_args(text: str) -> tuple[str, list[str]]:
    parts = str(text or "").split(":")
    return parts[0].strip(), parts[1:]


def _parse_inline_prompt_rule(text: str) -> PromptRule | None:
    sections = split_weight_sections(text)  # a qwen= section is ignored: the panel owns it
    block_text = sections.get("blocks", "")
    module_text = sections.get("modules", "")
    if not block_text and not module_text:
        return None
    if block_text and not _looks_like_layer_weights(block_text):
        return None

    block_weights = _parse_layer_weights(block_text, "") if block_text else LayerWeights()
    module_weights = _parse_module_weights(module_text) if module_text else {}
    return PromptRule(block_weights=block_weights, module_weights=module_weights)


def _rewrite_lora_prompts(prompts, target_loras: str, adapter_kind: str) -> tuple[list[str], dict[str, PromptRule]]:
    targets = _parse_targets(target_loras)
    inline_rules: dict[str, PromptRule] = {}

    def rewrite_prompt(prompt):
        def replace(match):
            if not match.group(2):
                return match.group(0)
            name = match.group(1).strip()
            weight_text, extra_args = _split_lora_args(match.group(2).strip())
            controlled = _prompt_lora_is_controlled(name, targets, adapter_kind)

            rule = _parse_inline_prompt_rule(weight_text)
            if rule is not None and controlled:
                _register_inline_rule(inline_rules, name, rule)
                return f"<lora:{name}:1.0>"

            if controlled and _is_plain_number(weight_text):
                return f"<lora:{name}:1.0>"

            return match.group(0)

        return LORA_TAG_RE.sub(replace, prompt)

    return [rewrite_prompt(prompt) for prompt in prompts], inline_rules


def _add_patches(original, patcher, patches, strength_patch, strength_model, filename, online_mode):
    rule = _ACTIVE_RULE
    decision = rule.decision_for_lora(filename) if rule is not None else None
    if decision is None or not decision.apply:
        return original(patcher, patches, strength_patch=strength_patch, strength_model=strength_model, filename=filename, online_mode=online_mode)
    if filename in lora_control_files():
        # <lora:x:[w@t,...]> keeps its tag, and LoRA Control rewrites each
        # patch's strength every step: the factors stored there would be lost
        _note_lora_control(rule, filename)
        return original(patcher, patches, strength_patch=strength_patch, strength_model=strength_model, filename=filename, online_mode=online_mode)

    # Delegate patch construction to the live Forge implementation. This
    # keeps Forge Neo's offline/online patch storage intact and composes
    # with the stage scheduler regardless of extension load order.
    blocks = getattr(getattr(getattr(patcher, "model", None), "diffusion_model", None), "blocks", None)
    model_blocks = len(blocks) if blocks is not None else None
    # One call per distinct strength, not per key: every call rebuilds the
    # model's state_dict (and runs any wrapper below), and each key keeps
    # its own patch list, so grouping changes nothing else.
    groups: dict[float, dict] = {}
    for patch_key, patch_value in patches.items():
        model_key = patch_key if isinstance(patch_key, str) else patch_key[0]
        groups.setdefault(rule.factor_for_key(model_key, decision, model_blocks), {})[patch_key] = patch_value
    # The factors scale the strength the tag loads at: the run's own tags are
    # rewritten to 1.0, while a nested run (ADetailer) loading an unrewritten
    # <lora:x:0.6> under this rule keeps its 0.6.
    loaded = set()
    for key_strength, group in groups.items():
        loaded.update(original(patcher, group, strength_patch=strength_patch * key_strength, strength_model=strength_model, filename=filename, online_mode=online_mode))
    return list(loaded)


def _note_lora_control(rule, filename):
    note_lora_control_yield(
        getattr(rule, "yielded_to_lora_control", None),
        getattr(rule, "extra_generation_params", None),
        filename,
        feature="Anima LoRA layer weight",
        infotext_key="Anima LoRA/LoKR Layer Weight skipped",
    )


def install_patch():
    # After "Reload UI" the wrapper an earlier copy of this module installed
    # runs this copy's _add_patches, which holds the live rule.
    if install_add_patches_hook("layer_weight.add_patches", _add_patches):
        logger.info("Installed Anima LoRA layer weight patch")


def _reset_lora_cache_if_needed(p, signature: str, text_encoder_signature: str):
    global _LAST_SIGNATURE, _LAST_TEXT_ENCODER_SIGNATURE
    if _LAST_TEXT_ENCODER_SIGNATURE != text_encoder_signature:
        # The prompt conditioning cache only sees the prompts, not these weights
        _LAST_TEXT_ENCODER_SIGNATURE = text_encoder_signature
        invalidate_cond_caches(p)
    if _LAST_SIGNATURE == signature:
        return

    _LAST_SIGNATURE = signature
    sd_model = getattr(p, "sd_model", None)
    if sd_model is not None and hasattr(sd_model, "current_lora_hash"):
        sd_model.current_lora_hash = f"anima-layer-weight:{signature}"


def _apply_prompt_rewrite(p, target_loras: str, adapter_kind: str) -> dict[str, PromptRule]:
    inline_rules = {}

    def rewrite(prompts):
        rewritten, rules = _rewrite_lora_prompts(prompts, target_loras, adapter_kind)
        merge_consistent_rules(inline_rules, rules, label="Anima adapter")
        return rewritten

    # Transactional: script hook failures are reported and sampling
    # continues, so a conflict must not leave prompts half rewritten.
    rewrite_batch_prompts(p, rewrite, rewrite)
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
            intro_block(self.elem_id, INTRO_EN, INTRO_ZH)
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

        register_preset_control(self.tabname, "lora_layer.lora_enabled", lora_enabled)
        register_preset_control(self.tabname, "lora_layer.lokr_enabled", lokr_enabled)

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
        # The hidden preset switches keep their argument slots for API callers,
        # but presets are only applied to the prompt, by their button.
        try:
            lora_inline_rules = _apply_prompt_rewrite(p, lora_target_loras, "lora") if lora_enabled else {}
            lokr_inline_rules = _apply_prompt_rewrite(p, lokr_target_loras, "lokr") if lokr_enabled else {}
        except ValueError as error:
            raise scripts.ScriptAbort(str(error)) from error
        _ACTIVE_RULE = ActiveRule(
            lora_enabled,
            lora_target_loras,
            lora_block_enabled,
            lora_block_weights,
            lora_module_enabled,
            lora_module_weights,
            lora_text_layer_enabled,
            lora_text_layer_weights,
            lora_inline_rules,
            lokr_enabled,
            lokr_target_loras,
            lokr_block_enabled,
            lokr_block_weights,
            lokr_module_enabled,
            lokr_module_weights,
            lokr_text_layer_enabled,
            lokr_text_layer_weights,
            lokr_inline_rules,
            getattr(p, "extra_generation_params", None),
        )
        _reset_lora_cache_if_needed(p, _ACTIVE_RULE.signature, _ACTIVE_RULE.text_encoder_signature)

        if _ACTIVE_RULE.enabled:
            p.extra_generation_params["Anima LoRA/LoKR Layer Weight"] = _ACTIVE_RULE.signature

    def postprocess(self, p, processed, *args, **kwargs):
        global _ACTIVE_RULE
        _ACTIVE_RULE = None

    def cleanup(self, p, *args, **kwargs):
        global _ACTIVE_RULE
        _ACTIVE_RULE = None
