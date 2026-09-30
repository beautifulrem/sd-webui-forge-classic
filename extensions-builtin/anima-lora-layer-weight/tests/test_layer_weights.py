import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_layer_weight.py"
_MODULE = []

from modules.anima_lora_support import looks_like_layer_weights, parse_layer_weights, parse_module_weights  # noqa: E402


@pytest.fixture
def module():
    if not _MODULE:
        spec = importlib.util.spec_from_file_location("anima_lora_layer_weight_weights", SCRIPT)
        _MODULE.append(importlib.util.module_from_spec(spec))
        spec.loader.exec_module(_MODULE[0])
    return _MODULE[0]


def _block(index, module="mlp.layer1"):
    return f"diffusion_model.blocks.{index}.{module}.weight"


def test_layer_weight_selectors():
    weights = parse_layer_weights("block_lrs=0-2=0.5，9-7=0.25; 12：2.0 all=0.8 bogus=3 x")

    assert weights.values == {0: 0.5, 1: 0.5, 2: 0.5, 7: 0.25, 8: 0.25, 9: 0.25, 12: 2.0}
    assert weights.default == 0.8
    assert weights.factor(1) == 0.5 and weights.factor(27) == 0.8


def test_bare_numbers_are_block_order_only_without_selectors():
    in_order = parse_layer_weights(",".join(str(index / 10) for index in range(28)))
    assert in_order.values == {index: index / 10 for index in range(28)} and in_order.default == 1.0

    assert parse_layer_weights("0.5, 3=0.2").values == {3: 0.2}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0.8", False),
        ("", False),
        ("mlp=0.5", False),
        ("0-9=0.5", True),
        ("all=0.5", True),
        ("7=0.5", True),
        ("1, 0.5", True),
        ("block_lrs=0.5 0.5", True),
    ],
)
def test_looks_like_layer_weights(text, expected):
    assert looks_like_layer_weights(text) is expected


def test_module_weights_normalise_names(module):
    assert parse_module_weights("cross=0.5\nFFN=0.7；self attention=0.9, adaln") == {"cross_attn": 0.5, "mlp": 0.7, "self_attn": 0.9}
    # The panel fills the modules it does not name with 1.0
    assert module._parse_module_weights("mlp=0.2") == {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 0.2, "adaln": 1.0}


def test_inline_prompt_rules(module):
    bare = module._parse_inline_prompt_rule("0-7=0.6;8-20=1.0;21-27=0.8")
    assert bare.block_weights.factor(7) == 0.6 and bare.block_weights.factor(21) == 0.8 and bare.module_weights == {}

    both = module._parse_inline_prompt_rule("blocks=0-9=0.8,10-27=0.7;modules=cross=0.6;qwen=0-3=0")
    assert both.block_weights.factor(10) == 0.7 and both.module_weights["cross_attn"] == 0.6

    assert module._parse_inline_prompt_rule("0.6") is None  # a plain strength
    assert module._parse_inline_prompt_rule("mlp=0.5") is None  # modules need their section name


@pytest.mark.parametrize("text", ["modules＝mlp=0.2", "ｍｏｄｕｌｅｓ＝ｍｌｐ＝０．２", "ＢＬＯＣＫＳ＝０－９＝０．５；ｍｏｄｕｌｅｓ：ｍｌｐ＝０．２"])
def test_inline_prompt_rules_take_full_width_input(module, text):
    rule = module._parse_inline_prompt_rule(text)

    assert rule is not None and rule.module_weights["mlp"] == 0.2 and rule.module_weights["cross_attn"] == 1.0
    assert rule.block_weights.factor(9) == (0.5 if "ＢＬＯＣＫＳ" in text else 1.0)


def test_a_full_width_prompt_strength_is_a_plain_strength(module, monkeypatch):
    monkeypatch.setattr(module, "_prompt_lora_is_controlled", lambda name, *args: name == "cat")

    rewritten, rules = module._rewrite_lora_prompts(["<lora:cat:０．５> <lora:cat:－0.5> <lora:dog:０．５>", "<lora:cat:modules＝mlp=0.2>"], "", "lora")

    assert rewritten == ["<lora:cat:1.0> <lora:cat:1.0> <lora:dog:０．５>", "<lora:cat:1.0>"]
    assert rules["cat"].module_weights["mlp"] == 0.2


def _lora_rule(module, lora):
    """The rule of the LoRA tab controls ``lora`` (TAB_ARGS order, without
    the preset slots) with the LoKR tab off."""

    presets = [False, "", "", ""]
    lokr = [False, "", False, "", False, "", False, ""]
    return module.ActiveRule(module._arguments([*lora, *presets, *lokr, *presets]), {"lora": {}, "lokr": {}})


def _decision(module, lora_blocks=28):
    rule = _lora_rule(module, [True, "", True, "0-9=0.6,10-18=1.0,19-27=0.75", True, "mlp=0.5", True, "0-13=0.3"])
    lora = rule.lora_rule
    return rule, module.LoraDecision(True, lora.block_weights, lora.module_weights, lora.text_layer_weights, lora_blocks)


@pytest.mark.parametrize(
    "key, blocks_28, blocks_52",
    [
        (_block(10), 1.0 * 0.5, 0.6 * 0.5),  # 3.8B block 10 is a copy of 2B block 5
        (_block(10, "self_attn.q_proj"), 1.0, 0.6),
        (_block(27, "cross_attn.k_proj"), 0.75, 1.0),  # 3.8B block 27 copies 2B block 14
        (_block(51, "cross_attn.k_proj"), None, 0.75),
        ("lora_unet_blocks_10_mlp_layer1", 0.5, 0.3),
        ((_block(3), (0, 4)), 0.3, 0.3),  # (key, offset) patch keys
        ("text_encoders.qwen3_06b.model.layers.2.mlp.up_proj.weight", 0.3, 0.3),  # qwen layers ignore blocks/modules
        ("text_encoders.qwen3_06b.model.layers.20.mlp.up_proj.weight", 1.0, 1.0),
        ("diffusion_model.final_layer.linear.weight", 1.0, 1.0),
    ],
)
def test_factor_for_key_on_2b_and_3_8b_models(module, key, blocks_28, blocks_52):
    rule, decision = _decision(module)

    if blocks_28 is not None:
        assert rule.factor_for_key(key, decision, 28) == pytest.approx(blocks_28)
    assert rule.factor_for_key(key, decision, 52) == pytest.approx(blocks_52)


def test_a_3_8b_lora_keeps_its_own_block_numbers(module):
    rule, decision = _decision(module, lora_blocks=52)

    assert rule.factor_for_key(_block(10, "self_attn.q_proj"), decision, 52) == 1.0
    assert rule.factor_for_key(_block(51, "self_attn.q_proj"), decision, 52) == 1.0  # default


def test_all_weights_the_dit_keys_outside_the_blocks(module, monkeypatch):
    # <lora:x:0.6> runs at 1.0, so all=0.6 must carry the DiT's whole strength
    monkeypatch.setattr(module, "_lora_coverage", lambda *args: SimpleNamespace(blocks={0, 1}, modules={"mlp"}, text_layers={2}))
    monkeypatch.setattr(module, "_names_for_lora_file", lambda filename: {"cat"})
    monkeypatch.setattr(module, "_adapter_kind_for_file", lambda filename: "lora")
    rule = _lora_rule(module, [True, "cat", True, "0=0.3,all=0.6", False, "", True, "2=0.5,all=0.8"])
    decision = rule.decision_for_lora("cat.safetensors")

    for key in ("diffusion_model.final_layer.linear.weight", "diffusion_model.x_embedder.proj.1.weight", "diffusion_model.t_embedder.1.linear_1.weight"):
        assert rule.factor_for_key(key, decision, 28) == 0.6
    assert rule.factor_for_key(_block(0), decision, 28) == 0.3
    assert rule.factor_for_key(_block(1), decision, 28) == 0.6
    # The text encoder keeps its own weights, and 1.0 outside its layers
    assert rule.factor_for_key("qwen3_06b.model.layers.2.mlp.up_proj.weight", decision, 28) == 0.5
    assert rule.factor_for_key("qwen3_06b.llm_adapter.out_proj.weight", decision, 28) == 1.0
    assert rule.factor_for_key("qwen3_06b.model.embed_tokens.weight", decision, 28) == 1.0

    # A targeted file with no trained blocks keeps the all= default too
    monkeypatch.setattr(module, "_lora_coverage", lambda *args: SimpleNamespace(blocks=set(), modules=set(), text_layers=set()))
    decision = rule.decision_for_lora("cat.safetensors")
    assert rule.factor_for_key("diffusion_model.final_layer.linear.weight", decision, 28) == 0.6


def test_prompt_rewrite_moves_inline_weights_into_rules(module, monkeypatch):
    monkeypatch.setattr(module, "_prompt_lora_is_controlled", lambda name, *args: name.startswith("sub/cat"))
    prompts = ["<lora:sub/cat.safetensors:0-9=0.5;modules=mlp=0.2> <lora:dog:0-9=0.5>", "<lora:sub/cat.safetensors:0.6>"]

    rewritten, rules = module._rewrite_lora_prompts(prompts, "", "lora")

    assert rewritten == ["<lora:sub/cat.safetensors:1.0> <lora:dog:0-9=0.5>", "<lora:sub/cat.safetensors:1.0>"]
    assert set(rules) == {"sub/cat.safetensors", "cat.safetensors", "cat"}
    assert rules["cat"].block_weights.factor(9) == 0.5 and rules["cat"].module_weights["mlp"] == 0.2


def _fake_index(module, monkeypatch, kinds, looked_up=None):
    """The prompt names in ``kinds`` ({name: adapter kind}) resolve to Anima
    adapters of that kind, any other to nothing."""

    def resolve(name, misses=None):
        if looked_up is not None:
            looked_up.append(name)
        return SimpleNamespace(filename=name) if name in kinds else None

    monkeypatch.setattr(module, "resolve_network", resolve)
    monkeypatch.setattr(module, "_adapter_kind_for_file", kinds.get)
    monkeypatch.setattr(module, "_is_anima_adapter_file", lambda filename, kind: True)


def test_each_prompt_tag_is_looked_up_once(module, monkeypatch):
    looked_up = []
    _fake_index(module, monkeypatch, {"cat": "lora", "fox": "lokr"}, looked_up)

    rewritten, _ = module._rewrite_lora_prompts(["<lora:cat:0.5> <lora:fox:0.5> <lora:typo:0.5>"], "", "lora")

    assert rewritten == ["<lora:cat:1.0> <lora:fox:0.5> <lora:typo:0.5>"]
    assert looked_up == ["cat", "fox", "typo"]


def test_a_misspelt_tag_rescans_the_lora_folders_once_per_run(module, monkeypatch):
    monkeypatch.setattr(module, "install_patch", lambda: None)
    monkeypatch.setattr(module, "_ACTIVE_RULE", None)
    monkeypatch.setattr(module, "_LAST_SIGNATURE", None)
    monkeypatch.setattr(module, "_adapter_kind_for_file", lambda filename: "lora")
    monkeypatch.setattr(module, "_is_anima_adapter_file", lambda filename, kind: True)
    on_disk = {}
    refreshed = []

    def update_available_networks_by_names(names):  # walks every LoRA directory
        refreshed.extend(names)
        networks.available_network_aliases.update((name, on_disk[name]) for name in names if name in on_disk)

    networks = SimpleNamespace(available_networks={}, available_network_aliases={}, forbidden_network_aliases={"none"}, update_available_networks_by_names=update_available_networks_by_names)
    monkeypatch.setitem(sys.modules, "networks", networks)
    written = ["<lora:cta:0-9=0.5>", "<lora:cta:0-9=0.5>"]
    args = [True, "", True, "all=1.0", False, "", False, "", False, "", "", ""] + [True, "", True, "all=1.0", False, "", False, "", False, "", "", ""]
    script = module.Script()

    p = SimpleNamespace(batch_size=1, all_prompts=list(written), all_hr_prompts=list(written), extra_generation_params={})
    for iteration in range(2):  # each batch rewrites its prompts and hires prompts, for both tabs
        p.iteration = iteration
        p.prompts = p.all_prompts[iteration : iteration + 1]
        script.before_process_batch(p, *args)
    assert refreshed == ["cta"]
    assert (p.all_prompts, p.all_hr_prompts) == (written, written)  # no such LoRA: left as written

    on_disk["cta"] = SimpleNamespace(filename="cta.safetensors")  # added before the next generation
    p = SimpleNamespace(iteration=0, batch_size=1, prompts=written[:1], all_prompts=list(written), all_hr_prompts=list(written), extra_generation_params={})
    script.before_process_batch(p, *args)
    assert refreshed == ["cta", "cta"]
    assert p.prompts == ["<lora:cta:1.0>"]


def _run_batch(module, prompts):
    p = SimpleNamespace(prompts=list(prompts), extra_generation_params={})
    lora_args = [True, "", True, "all=1.0", False, "", False, "", False, "", "", ""]
    lokr_args = [False, "", True, "all=1.0", False, "", False, "", False, "", "", ""]
    module.Script().before_process_batch(p, *lora_args, *lokr_args)
    return p


def test_conflicting_inline_rules_abort_the_batch_untouched(module, monkeypatch):
    from modules.scripts import ScriptAbort

    monkeypatch.setattr(module, "install_patch", lambda: None)
    monkeypatch.setattr(module, "_prompt_lora_is_controlled", lambda *args: True)
    monkeypatch.setattr(module, "_ACTIVE_RULE", None)
    monkeypatch.setattr(module, "_LAST_SIGNATURE", None)
    prompts = ["<lora:cat:0-9=0.5>", "<lora:cat:0-9=0.7>"]

    with pytest.raises(ScriptAbort, match="different inline weights"):
        _run_batch(module, prompts)
    assert module._ACTIVE_RULE is None

    p = _run_batch(module, ["<lora:cat:0-9=0.5>", "<lora:cat:0-9=0.5>"])  # the same rule twice is fine
    assert p.prompts == ["<lora:cat:1.0>"] * 2
    assert module._ACTIVE_RULE.lora_rule.inline_rules["cat"].block_weights.factor(9) == 0.5


def test_the_infotext_keeps_the_inline_rules_as_written(module, monkeypatch):
    """The batch loads <lora:cat:1.0>, its rules moved to add_patches; the
    infotext, built from p.prompts and all_hr_prompts after
    postprocess_batch_list, must keep the tags as written, or the image
    recalled from its PNG info renders another one."""
    monkeypatch.setattr(module, "install_patch", lambda: None)
    monkeypatch.setattr(module, "_prompt_lora_is_controlled", lambda *args: True)
    monkeypatch.setattr(module, "_ACTIVE_RULE", None)
    monkeypatch.setattr(module, "_LAST_SIGNATURE", None)
    written = ["a <lora:cat:0-9=0.5;modules=mlp=0.2>", "b <lora:cat:1,1,0.5>"]
    loaded = ["a <lora:cat:1.0>", "b <lora:cat:1.0>"]
    p = SimpleNamespace(iteration=1, batch_size=1, all_prompts=list(written), all_hr_prompts=list(written), extra_generation_params={})
    args = [True, "", True, "all=1.0", False, "", False, "", False, "", "", ""] + [False, "", True, "all=1.0", False, "", False, "", False, "", "", ""]
    script = module.Script()

    # The second batch, in the order of processing.process_images_inner
    p.prompts = p.all_prompts[1:2]
    script.before_process_batch(p, *args)
    assert (p.prompts, p.all_hr_prompts) == (loaded[1:], written[:1] + loaded[1:])
    p.prompts = p.all_prompts[1:2]  # re-sliced after postprocess_batch
    script.postprocess_batch_list(p, SimpleNamespace(images=[]), *args)

    assert p.prompts == written[1:]
    assert p.all_hr_prompts == written  # else a "Hires prompt" of the loaded tags
    # A nested run (ADetailer, in postprocess_image) loads the LoRAs as the batch did
    assert p.all_prompts == written[:1] + loaded[1:]

    script.postprocess(p, SimpleNamespace(), *args)
    assert p.all_prompts == written


def test_apply_preset_writes_the_presets_inline_weights(module, monkeypatch):
    presets = "Balanced=blocks=0-9=0.6,10-27=1.0;modules=mlp=0.8\n# a comment\nno separator\nstyle：blocks=19-27=0.9;qwen=0-3=0"
    parsed = module._parse_presets(presets)
    assert set(parsed) == {"balanced", "style"}
    assert parsed["style"] == module.PresetWeights("19-27=0.9", None)  # Qwen weights come from the panel only
    assert module._select_preset(" BALANCED ", presets) == parsed["balanced"]
    assert module._select_preset("", presets) == parsed["balanced"]  # no name: the first
    assert module._select_preset("missing", presets) is None

    # A full-width "＝" or "：" ends the name too; the name is kept as written
    full_width = module._parse_presets("平衡＝blocks=0-9=0.6;modules=mlp=0.8\nＳｔｙｌｅ：ｂｌｏｃｋｓ＝１９－２７＝０．９；ｍｏｄｕｌｅｓ＝ｍｌｐ＝１")
    assert full_width == {"平衡": module.PresetWeights("0-9=0.6", "mlp=0.8"), "ｓｔｙｌｅ": module.PresetWeights("19-27=0.9", "mlp=1")}
    assert module._select_preset("Ｓｔｙｌｅ", "Ｓｔｙｌｅ＝blocks=0=1") == module.PresetWeights("0=1", None)

    kinds = {"cat": "lora", "dog": "lora", "fox": "lokr"}
    _fake_index(module, monkeypatch, kinds)
    prompt = "<lora:cat:0.6:te=0.2> <lora:dog:1> <lora:fox:0.5> <lora:other:1>"
    inline = "blocks=0-9=0.6,10-27=1.0;modules=mlp=0.8"

    # The tag's other arguments go: generation rewrites a controlled tag to <lora:x:1.0>
    applied = module._apply_preset_to_prompt_text(prompt, "", "lora", "balanced", module.PRESET_SCOPE_ALL_EN, presets)
    assert applied == f"<lora:cat:{inline}> <lora:dog:{inline}> <lora:fox:0.5> <lora:other:1>"
    targeted = module._apply_preset_to_prompt_text(prompt, "dog", "lora", "balanced", "Target list only", presets)
    assert targeted == f"<lora:cat:0.6:te=0.2> <lora:dog:{inline}> <lora:fox:0.5> <lora:other:1>"
    lokr = module._apply_preset_to_prompt_text(prompt, "", "lokr", "style", module.PRESET_SCOPE_ALL, presets)
    assert lokr == "<lora:cat:0.6:te=0.2> <lora:dog:1> <lora:fox:blocks=19-27=0.9> <lora:other:1>"
    assert module._apply_preset_to_prompt_text(prompt, "", "lora", "missing", module.PRESET_SCOPE_ALL, presets) == prompt


def test_prompt_conds_are_recomputed_when_qwen_layer_weights_change(module, monkeypatch):
    monkeypatch.setattr(module, "install_patch", lambda: None)
    monkeypatch.setattr(module, "_prompt_lora_is_controlled", lambda *args: True)
    monkeypatch.setattr(module, "_ACTIVE_RULE", None)
    monkeypatch.setattr(module, "_LAST_SIGNATURE", None)
    monkeypatch.setattr(module, "_LAST_TEXT_ENCODER_SIGNATURE", None, raising=False)
    cache = [None, None, None]  # StableDiffusionProcessing.cached_c

    def run(qwen_enabled, qwen_weights, block_weights="all=1.0"):
        cache[0] = "cond key"
        p = SimpleNamespace(prompts=["<lora:cat:1>"], extra_generation_params={}, cached_c=cache)
        lora_args = [True, "", True, block_weights, False, "", qwen_enabled, qwen_weights, False, "", "", ""]
        lokr_args = [False, "", True, "all=1.0", False, "", False, "", False, "", "", ""]
        module.Script().before_process_batch(p, *lora_args, *lokr_args)
        return cache[0] is None

    assert run(True, "0-13=0.3")
    assert not run(True, "0-13=0.3", block_weights="0-9=0.5")  # the DiT alone
    assert run(True, "0-13=0.5")
    assert run(False, "0-13=0.5")
    assert not run(False, "0-13=0.7")


def test_editing_a_disabled_tab_does_not_reload_the_loras(module, monkeypatch):
    monkeypatch.setattr(module, "install_patch", lambda: None)
    monkeypatch.setattr(module, "_ACTIVE_RULE", None)
    monkeypatch.setattr(module, "_LAST_SIGNATURE", None)
    sd_model = SimpleNamespace(current_lora_hash="compiled")

    def run(block_weights, lokr_targets):
        sd_model.current_lora_hash = "compiled"
        p = SimpleNamespace(prompts=["<lora:cat:1>"], extra_generation_params={}, sd_model=sd_model)
        lora_args = [False, "", True, block_weights, True, "mlp=0.5", False, "", False, "", "", ""]
        lokr_args = [False, lokr_targets, True, block_weights, False, "", False, "", False, "", "", ""]
        module.Script().before_process_batch(p, *lora_args, *lokr_args)
        return sd_model.current_lora_hash != "compiled"

    assert run("0-9=0.5", "")  # the first batch sets the signature
    monkeypatch.setattr(module, "_parse_layer_weights", lambda *args: pytest.fail("parsed a disabled tab"))
    assert not run("0-9=0.7", "dog")
    assert not run("all=0.2", "cat")
