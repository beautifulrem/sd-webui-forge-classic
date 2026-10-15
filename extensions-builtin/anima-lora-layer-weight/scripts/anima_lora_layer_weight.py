import logging
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

from modules import scripts
from modules.anima_lora_support import (
    ANIMA_BLOCK_MAPPINGS,
    LORA_TAG_RE,
    LayerWeights,
    invalidate_cond_caches,
    install_add_patches_hook,
    lora_control_files,
    note_lora_control_yield,
    restore_batch_prompts,
    restore_run_prompts,
    rewrite_batch_prompts,
    looks_like_layer_weights as _looks_like_layer_weights,
    parse_layer_weights,
    parse_module_weights,
    lora_file_names as _names_for_lora_file,
    name_keys,
    network_misses,
    parse_targets as _parse_targets,
    resolve_network,
    split_weight_sections,
    anima_lora_block_count,
    lora_key_factor,
    model_key as _model_key,
    merge_consistent_rules,
    compact_text as _compact_text,
)
from modules.anima_lora_files import adapter_kind_for_file as _adapter_kind_for_file
from modules.anima_lora_files import is_anima_adapter_file as _is_anima_adapter_file
from modules.anima_lora_files import lora_coverage as _lora_coverage
from modules.anima_support import PasteKey, is_plain_number as _is_plain_number, script_arguments, table_params
from remi import ui


logger = logging.getLogger("anima_lora_layer_weight")

TITLE = "Anima LoRA Layer Weight"
INTRO = (
    "Control Anima LoRA/LoKR patch strength by block range, module type, and "
    "optional Qwen text-encoder layers. Use it to tune composition, character "
    "features, and style without editing model files."
)

DEFAULT_BLOCK_WEIGHTS = "all=1.0"
DEFAULT_MODULE_WEIGHTS = "\n".join(
    [
        "self_attn=1.0",
        "cross_attn=1.0",
        "mlp=1.0",
        "adaln=1.0",
    ]
)
# The "Apply preset to prompt" scopes; the legacy Chinese label still reads as "all"
PRESET_SCOPE_ALL = "提示词里的全部同类型"
PRESET_SCOPE_ALL_EN = "All matching type in prompt"
PRESET_SCOPE_TARGETS_EN = "Target list only"
# The values the Gradio panel's zh UI had, read as the English choices
PRESET_SCOPE_ALIASES = {PRESET_SCOPE_ALL: PRESET_SCOPE_ALL_EN, "仅目标列表": PRESET_SCOPE_TARGETS_EN}
DEFAULT_PRESETS_TEXT = "\n".join(
    [
        "平衡=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,cross_attn=1.0,mlp=1.0,adaln=1.0",
        "保构图=blocks=0-9=0.35,10-18=1.0,19-27=0.7",
        "强画风=blocks=0-9=0.7,10-18=1.0,19-27=0.9",
    ]
)
USAGE_TEXT_EN = """1. LoRA and LoKR are controlled separately. The LoRA tab only affects regular LoRAs; the LoKR tab only affects LoKRs.
2. When the target field is empty, the tab automatically applies to all matching Anima adapters of that type.
3. When targets are filled, only the listed file names, aliases, or names without extension are controlled. Separate multiple targets with commas or new lines.
4. After a target is controlled by this plugin, the normal prompt weight, for example the 0.6 in `<lora:name:0.6>`, is ignored for the matched adapter.
5. Anima block weights and module weights can be enabled together. Qwen text-encoder layer weights are independent and disabled by default.
6. When block and module weights are both enabled, the final model weight is block weight * module weight.
7. Qwen text-encoder layer weights can only be controlled from this panel. Files without text-encoder training are skipped naturally.
8. Missing blocks, modules, or Qwen layers are skipped; only weights that exist in the selected file are affected.
9. Blocks 0-9 mainly affect pose and composition, 10-18 affect character and outfit features, and 19-27 affect style and lighting.
10. You can write inline block weights in the prompt, for example `<lora:name:0-7=0.6;8-20=1.0;21-27=0.8>`.
11. Inline syntax can also combine blocks and modules, for example `<lora:name:blocks=0-9=0.8,10-18=1.0,19-27=0.7;modules=self_attn=1.0,mlp=0.8,cross_attn=0.6>`.
12. Inline block/module weights override the corresponding tab settings for that file. Qwen weights still come from the panel.
13. Inline block weights also support 28 comma-separated numbers in block order from 0 to 27. On depth-expanded models (Anima 2.9B/3.8B) a 28-block LoRA keeps its own block numbering; inserted copies follow the block they were copied from.
14. Presets can be written one per line, for example `Balanced=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8`. Presets hold blocks and modules; Qwen layer weights always come from the panel.
15. "Apply preset to prompt" replaces matching LoRA/LoKR prompt weights with the preset's inline blocks/modules syntax.
16. The apply scope can affect all matching adapters in the prompt, or only the adapters listed in the target field."""


# One row per control of a tab, in the order of its script args (the API's
# positional order): (name of the control in _lora_tab, infotext field or
# None, the field of the switch a text only counts under, or None). The LoRA
# tab's args come first, then the LoKR tab's, each named "<kind>_<name>". A
# switch row comes before the text it gates. The preset controls only rewrite
# the prompt, which the infotext keeps as written; their slots stay for API
# callers.
TAB_ARGS = (
    ("enabled", "enabled", None),
    ("target_loras", "targets", "enabled"),
    ("block_enabled", "blocks enabled", None),
    ("block_weights", "blocks", "blocks enabled"),
    ("module_enabled", "modules enabled", None),
    ("module_weights", "modules", "modules enabled"),
    ("text_layer_enabled", "qwen enabled", None),
    ("text_layer_weights", "qwen", "qwen enabled"),
    ("preset_enabled", None, None),
    ("preset_name", None, None),
    ("preset_scope", None, None),
    ("preset_text", None, None),
)
ADAPTER_KINDS = ("lora", "lokr")
TAB_ARG_NAMES = tuple(name for name, _, _ in TAB_ARGS)
SCRIPT_ARG_NAMES = tuple(f"{kind}_{name}" for kind in ADAPTER_KINDS for name in TAB_ARG_NAMES)
# Infotext fields, one per control that changes the run; ``infotext_fields``
# pastes each back into its control
TAB_INFOTEXT_FIELDS = tuple(field for _, field, _ in TAB_ARGS if field is not None)
# The switch of each text field: its text only counts while the switch is on
TAB_TEXT_SWITCHES = {field: switch for _, field, switch in TAB_ARGS if switch is not None}


def _infotext_key(adapter_kind: str, field_name: str) -> str:
    return f"Anima LoRA layer {adapter_kind} {field_name}"


INFOTEXT_KEYS = tuple(_infotext_key(kind, name) for kind in ADAPTER_KINDS for name in TAB_INFOTEXT_FIELDS)


def _arguments(values) -> dict[str, SimpleNamespace]:
    """The script args ``values``, per adapter kind one attribute per
    TAB_ARGS name."""

    arguments = vars(script_arguments("Anima LoRA layer weight", SCRIPT_ARG_NAMES, values))
    return {kind: SimpleNamespace(**{name: arguments[f"{kind}_{name}"] for name in TAB_ARG_NAMES}) for kind in ADAPTER_KINDS}


def _switch(value) -> bool:
    """A switch argument: the checkbox's bool, or a recorded / API-given text
    ("True", "False")."""
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "no", "off", "none"}
    return bool(value)


def _tab_infotext(adapter_kind: str, tab: SimpleNamespace) -> dict:
    """Infotext of one tab (its ``_arguments``), keyed by ``_infotext_key``.
    A disabled tab records its switch alone; an enabled one every control,
    but not the text of a field that is empty or whose switch is off (the
    run ignored it)."""

    if not _switch(tab.enabled):
        return {_infotext_key(adapter_kind, "enabled"): False}
    values = {}
    for name, field, switch in TAB_ARGS:
        if field is None:
            continue
        value = getattr(tab, name)
        values[field] = _switch(value) if switch is None else (str(value or "").strip() if values[switch] else "")
    return {_infotext_key(adapter_kind, field): value for field, value in values.items() if value != ""}


def _pasted_value(adapter_kind: str, field_name: str, params: dict):
    """The value pasting the infotext ``params`` gives one tab control (None
    leaves it alone): a switch's bool, a field's text. A text the recorded
    tab left out while its switch was on pastes empty, which the run reads
    as it read the empty field."""

    value = params.get(_infotext_key(adapter_kind, field_name))
    switch = TAB_TEXT_SWITCHES.get(field_name)
    if switch is None:
        return None if value is None else _switch(value)
    if value is None:
        return "" if _switch(params.get(_infotext_key(adapter_kind, switch), False)) else None
    return str(value)


def _paste_key(adapter_kind: str, field_name: str) -> PasteKey:
    return PasteKey(_infotext_key(adapter_kind, field_name), partial(_pasted_value, adapter_kind, field_name))


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
    def __init__(self, adapter_kind: str, tab: SimpleNamespace, inline_rules: dict[str, PromptRule]):
        """``tab``: the tab's script args, one attribute per TAB_ARGS name."""

        self.adapter_kind = adapter_kind
        # _switch: an API infotext replay may give the recorded "False"
        self.enabled = _switch(tab.enabled)
        if self.enabled:
            target_loras, block_weights, module_weights, text_layer_weights = tab.target_loras, tab.block_weights, tab.module_weights, tab.text_layer_weights
            block_enabled, module_enabled, text_layer_enabled = tab.block_enabled, tab.module_enabled, tab.text_layer_enabled
        else:
            # A disabled tab weights nothing: its fields are not parsed, and
            # editing them must not change the signature (a LoRA reload)
            target_loras = block_weights = module_weights = text_layer_weights = ""
            block_enabled = module_enabled = text_layer_enabled = False
        self.targets = _parse_targets(target_loras)
        self.block_enabled = _switch(block_enabled)
        self.module_enabled = _switch(module_enabled)
        self.text_layer_enabled = _switch(text_layer_enabled)
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
    def __init__(self, tabs: dict[str, SimpleNamespace], inline_rules: dict[str, dict[str, PromptRule]], extra_generation_params: dict | None = None):
        """``tabs``: the script args by adapter kind (``_arguments``);
        ``inline_rules``: the prompt's inline rules by adapter kind."""

        # The run's infotext, and the files left to LoRA Control noted in it
        self.extra_generation_params = extra_generation_params
        self.yielded_to_lora_control: set = set()
        self.rules = {kind: AdapterRule(kind, tabs[kind], inline_rules[kind]) for kind in ADAPTER_KINDS}
        self.lora_rule, self.lokr_rule = self.rules["lora"], self.rules["lokr"]
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
        # The prompt weight is rewritten to 1.0, so the DiT keys outside the
        # blocks (final_layer, x_embedder, t_embedder) run at the blocks'
        # all= weight; the text encoder's own non-layer keys keep 1.0
        fallback = 1.0 if _is_text_encoder_key(model_key) else decision.block_weights.default
        return lora_key_factor(
            model_key,
            fallback,
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


# A preset name ends at its first "=" or ":", ASCII or full-width
_PRESET_SEPARATORS = frozenset("=:＝：")


def _parse_presets(text: str) -> dict[str, PresetWeights]:
    presets: dict[str, PresetWeights] = {}
    for line in str(text or "").replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # The name ends at its first separator: the body holds "=" too
        # (name：blocks=...). The name itself is kept as written
        separator = next((index for index, mark in enumerate(line) if mark in _PRESET_SEPARATORS), -1)
        if separator < 0:
            logger.warning("Ignored invalid preset line: %s", line)
            continue

        name, body = line[:separator], line[separator + 1 :]
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


def _preset_name_patch(text: str, current_value: str = "") -> dict:
    """The preset name choices of a preset text; the selection is kept (a
    name typed in is kept too) or falls back to the first preset."""
    choices = _preset_choices(text)
    if current_value and current_value not in choices:
        choices = [current_value] + choices
    value = current_value if current_value in choices else (choices[0] if choices else "")
    return ui.update(choices=choices, value=value)


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

        # Its text-encoder / UNet arguments would be dropped at generation
        # anyway: the rewrite to 1.0 keeps the weight alone
        return f"<lora:{name}:{weight_text}>"

    return LORA_TAG_RE.sub(replace, prompt_text)


def _is_text_encoder_key(key) -> bool:
    """A Qwen3 text-encoder or LLM-adapter key (qwen3_06b.*), not the DiT's."""
    text = _model_key(key).lower()
    return "qwen" in text or "text_encoder" in text or "llm_adapter" in text


def _filter_layer_weights(weights: LayerWeights, trained_layers: set[int]) -> LayerWeights:
    # Keeps the all= default, which also weights the DiT keys outside the blocks
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


def _prompt_lora_is_controlled(name: str, targets: set[str], adapter_kind: str = "lora", misses: set[str] | None = None) -> bool:
    """A listed target, or with no targets any Anima adapter of the kind.
    ``misses``: see network_misses."""
    entry = resolve_network(name, misses)
    if entry is None:
        return False
    filename = getattr(entry, "filename", "")
    if _adapter_kind_for_file(filename) != adapter_kind:
        return False
    if targets:
        return bool(name_keys(name) & targets)
    return _is_anima_adapter_file(filename, adapter_kind)


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


def _rewrite_lora_prompts(prompts, target_loras: str, adapter_kind: str, misses: set[str] | None = None) -> tuple[list[str], dict[str, PromptRule]]:
    targets = _parse_targets(target_loras)
    inline_rules: dict[str, PromptRule] = {}

    def rewrite_prompt(prompt):
        def replace(match):
            if not match.group(2):
                return match.group(0)
            name = match.group(1).strip()
            weight_text, extra_args = _split_lora_args(match.group(2).strip())
            controlled = _prompt_lora_is_controlled(name, targets, adapter_kind, misses)

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


def _reset_lora_cache_if_needed(p, signature: str, text_encoder_signature: tuple | None):
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
    misses = network_misses(p)

    def rewrite(prompts):
        rewritten, rules = _rewrite_lora_prompts(prompts, target_loras, adapter_kind, misses)
        merge_consistent_rules(inline_rules, rules, label="Anima adapter")
        return rewritten

    # Transactional: script hook failures are reported and sampling
    # continues, so a conflict must not leave prompts half rewritten.
    rewrite_batch_prompts(p, rewrite, rewrite)
    return inline_rules


# Anima's Qwen3 0.6B text encoder
QWEN_LAYERS = 28
# Depth bands of the 28-block (2B) layout the presets are written in
BLOCK_BANDS = (("Composition", 0, 9), ("Character", 10, 18), ("Style", 19, 27))
# The module grid's rows: every module module_name_from_block_tail names
MODULES = ("self_attn", "cross_attn", "mlp", "adaln", "norm")


# The "Apply preset to prompt" toasts (translated by the SPA through localizations/zh_CN.json)
TOAST_NO_MATCH = "No matching adapter in the prompt, or the preset has no weights."
TOAST_APPLIED = "Preset applied to the prompt."


def _grid_presets() -> list[dict]:
    """The block grid's one-click weights: flat, and each named default
    preset's blocks section."""
    presets = [{"label": "Flat 1.0", "value": DEFAULT_BLOCK_WEIGHTS}]
    for line in DEFAULT_PRESETS_TEXT.splitlines():
        name, _, body = line.partition("=")
        blocks = _parse_preset_sections(body).block_weights
        if blocks:
            presets.append({"label": name, "value": blocks})
    return presets


def _text_value(value) -> str:
    """A weight field's value: the weight text (the grid widgets edit it)."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"expected weight text, got {type(value).__name__}")
    return value


class _ScopeChoice(ui.Choice):
    """The preset apply scope; the zh UI's values (callers built for the
    Gradio panel) are accepted as their English choices."""

    def coerce(self, value, ctx):
        return super().coerce(PRESET_SCOPE_ALIASES.get(value, value) if isinstance(value, str) else value, ctx)


def _tab_params(adapter_kind: str) -> ui.Tab:
    """One adapter tab; its inputs are named ``<kind>_<TAB_ARGS name>``."""

    def name(arg):
        return f"{adapter_kind}_{arg}"

    noun = "LoRA" if adapter_kind == "lora" else "LoKR"
    block_props = {
        "mode": "blocks",
        "layouts": [28, 40, 52],
        "mappings": {str(target): list(mapping) for (source, target), mapping in ANIMA_BLOCK_MAPPINGS.items() if source == 28},
        "bands": [{"label": label, "start": start, "end": end} for label, start, end in BLOCK_BANDS],
        "presets": _grid_presets(),
        "placeholder": "Example: 0-5=0.6, 6-20=1.0, 21-27=0.8",
    }
    return ui.Tab(
        noun,
        ui.Toggle(name("enabled"), False, label=f"Enable {noun} layer weights"),
        ui.Text(
            name("target_loras"),
            "",
            label=f"Target {noun}",
            placeholder=f"Empty = all Anima {noun}s; file name or alias = only the listed {noun}s",
        ),
        ui.Row(
            ui.Toggle(name("block_enabled"), True, label="Anima block weights"),
            ui.Toggle(name("module_enabled"), False, label="Module weights"),
            ui.Toggle(name("text_layer_enabled"), False, label="Qwen text-encoder layer weights"),
        ),
        ui.Custom(name("block_weights"), "block-weight-grid", DEFAULT_BLOCK_WEIGHTS, block_props, label="Anima block weights", validate=_text_value, visible_if=ui.param(name("block_enabled"))),
        ui.Custom(
            name("module_weights"),
            "block-weight-grid",
            DEFAULT_MODULE_WEIGHTS,
            {"mode": "modules", "modules": list(MODULES), "placeholder": "self_attn/cross_attn/mlp/adaln/norm"},
            label="Module weights",
            validate=_text_value,
            visible_if=ui.param(name("module_enabled")),
        ),
        ui.Custom(
            name("text_layer_weights"),
            "block-weight-grid",
            "",
            {"mode": "layers", "count": QWEN_LAYERS, "placeholder": "Empty = keep text encoder unchanged; example: 0-31=1.0"},
            label="Qwen text-encoder layer weights",
            validate=_text_value,
            visible_if=ui.param(name("text_layer_enabled")),
        ),
        ui.Group(
            "Prompt presets",
            # Kept for API callers' argument slots; presets only rewrite the prompt
            ui.Toggle(name("preset_enabled"), False, label="Enable preset", hidden=True),
            ui.Row(
                ui.Choice(name("preset_name"), _preset_choices(DEFAULT_PRESETS_TEXT), "平衡", label="Preset name", allow_custom=True),
                _ScopeChoice(name("preset_scope"), [PRESET_SCOPE_ALL_EN, PRESET_SCOPE_TARGETS_EN], PRESET_SCOPE_ALL_EN, label="Preset apply scope"),
            ),
            ui.Text(
                name("preset_text"),
                DEFAULT_PRESETS_TEXT,
                lines=5,
                label="Custom presets",
                placeholder="One preset per line: name=blocks=...;modules=...",
                on_change=f"{adapter_kind}_presets_changed",
            ),
            ui.Button("Apply preset to prompt", f"{adapter_kind}_apply_preset", "primary"),
            open=False,
        ),
        value=adapter_kind,
    )


def _param_rows():
    """``table_params`` rows: SCRIPT_ARG_NAMES with each recorded control's
    infotext key, which also pastes it (``_paste_key``), and the switches'
    preset fields."""
    for kind in ADAPTER_KINDS:
        for arg, field, _switch_field in TAB_ARGS:
            yield f"{kind}_{arg}", None if field is None else _paste_key(kind, field), f"lora_layer.{kind}_enabled" if arg == "enabled" else None


class Script(scripts.Script):
    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        root = ui.Section(
            "Anima LoRA/LoKR Layer Weight",
            ui.Markdown(INTRO),
            ui.Tabs(*(_tab_params(kind) for kind in ADAPTER_KINDS)),
            ui.Group("Usage guide", ui.Markdown(USAGE_TEXT_EN), open=False),
        )
        self.param_order = table_params(_param_rows(), root)
        return root

    def _apply_preset(self, ctx, values, adapter_kind: str):
        def value(arg):
            return values.get(f"{adapter_kind}_{arg}")

        prompt = str(ctx.core.get("prompt") or "")
        rewritten = _apply_preset_to_prompt_text(prompt, value("target_loras"), adapter_kind, value("preset_name"), value("preset_scope"), value("preset_text"))
        if rewritten == prompt:
            return ui.toast(TOAST_NO_MATCH, "info")
        return {"core.prompt": ui.update(value=rewritten), **ui.toast(TOAST_APPLIED, "success")}

    @ui.action("lora_apply_preset")
    def lora_apply_preset(self, ctx, values):
        return self._apply_preset(ctx, values, "lora")

    @ui.action("lokr_apply_preset")
    def lokr_apply_preset(self, ctx, values):
        return self._apply_preset(ctx, values, "lokr")

    @ui.action("lora_presets_changed")
    def lora_presets_changed(self, ctx, values):
        return {"lora_preset_name": _preset_name_patch(values.get("lora_preset_text", ""), values.get("lora_preset_name", ""))}

    @ui.action("lokr_presets_changed")
    def lokr_presets_changed(self, ctx, values):
        return {"lokr_preset_name": _preset_name_patch(values.get("lokr_preset_text", ""), values.get("lokr_preset_name", ""))}

    def before_process_batch(self, p, *args, **kwargs):
        global _ACTIVE_RULE

        tabs = _arguments(args)
        install_patch()
        _ACTIVE_RULE = None
        # The hidden preset switches keep their argument slots for API callers,
        # but presets are only applied to the prompt, by their button.
        try:
            inline_rules = {kind: _apply_prompt_rewrite(p, tab.target_loras, kind) if _switch(tab.enabled) else {} for kind, tab in tabs.items()}
        except ValueError as error:
            raise scripts.ScriptAbort(str(error)) from error
        _ACTIVE_RULE = ActiveRule(tabs, inline_rules, getattr(p, "extra_generation_params", None))
        _reset_lora_cache_if_needed(p, _ACTIVE_RULE.signature, _ACTIVE_RULE.text_encoder_signature)

        if _ACTIVE_RULE.enabled:
            # Keys match ``self.infotext_fields`` so pasted parameters restore the panel
            p.extra_generation_params.update({key: value for kind, tab in tabs.items() for key, value in _tab_infotext(kind, tab).items()})

    def postprocess_batch_list(self, p, pp, *args, **kwargs):
        # The batch's infotexts record the tags as written, <lora:x:0-9=0.5>
        # rather than the <lora:x:1.0> it loaded
        restore_batch_prompts(p)

    def postprocess(self, p, processed, *args, **kwargs):
        global _ACTIVE_RULE
        _ACTIVE_RULE = None
        restore_run_prompts(p)

    def cleanup(self, p, *args, **kwargs):
        global _ACTIVE_RULE
        _ACTIVE_RULE = None
