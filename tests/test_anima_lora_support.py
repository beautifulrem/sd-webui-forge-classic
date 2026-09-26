from types import SimpleNamespace

import pytest

from modules.anima_lora_support import (
    ANIMA_BLOCK_MAPPINGS,
    LayerWeights,
    active_template_parts,
    anima_lora_block_count,
    anima_model_block,
    anima_source_block,
    clamp,
    current_sampling_position,
    lora_key_factor,
    merge_consistent_rules,
    model_block_set,
    to_float,
)


def test_manual_timing_ignores_template_parts():
    parts = [{"stage": "style", "mode": "allow"}]

    assert active_template_parts(False, parts) == []
    assert active_template_parts(True, parts) == parts


def test_sampling_position_uses_current_denoiser_step_over_callback_lag():
    params = SimpleNamespace(
        sampling_step=3,
        total_sampling_steps=20,
        denoiser=SimpleNamespace(step=4, total_steps=20),
    )

    assert current_sampling_position(params) == (4, 20)


def test_sampling_position_keeps_larger_callback_total():
    params = SimpleNamespace(
        sampling_step=5,
        total_sampling_steps=30,
        denoiser=SimpleNamespace(step=5, total_steps=20),
    )

    assert current_sampling_position(params) == (5, 30)


def test_same_inline_rule_can_repeat_across_prompt_views():
    rules = {"portrait": (0.5, 1.0)}

    merge_consistent_rules(
        rules,
        {"portrait": (0.5, 1.0)},
        label="Anima adapter",
    )

    assert rules == {"portrait": (0.5, 1.0)}


def test_conflicting_inline_rules_in_one_batch_are_rejected():
    rules = {"portrait": (0.5, 1.0)}

    with pytest.raises(ValueError, match="different inline weights"):
        merge_consistent_rules(
            rules,
            {"portrait": (1.0, 0.5)},
            label="Anima adapter",
        )


def test_block_mappings_match_the_lora_loader():
    import ast
    import pathlib

    source = (pathlib.Path(__file__).parents[1] / "extensions-builtin" / "sd_forge_lora" / "networks.py").read_text(encoding="utf-8")
    tables = {
        node.targets[0].id: tuple(ast.literal_eval(node.value))
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "").startswith("MAPPING_")
    }
    assert tables == {
        "MAPPING_2_TO_29": ANIMA_BLOCK_MAPPINGS[(28, 40)],
        "MAPPING_2_TO_38": ANIMA_BLOCK_MAPPINGS[(28, 52)],
        "MAPPING_29_TO_38": ANIMA_BLOCK_MAPPINGS[(40, 52)],
    }
    for (src, dst), mapping in ANIMA_BLOCK_MAPPINGS.items():
        assert len(mapping) == dst and set(mapping) == set(range(src))


def test_layer_weights_follow_the_lora_layout_on_expanded_models():
    assert anima_lora_block_count({0, 5, 27}) == 28
    assert anima_lora_block_count({39}) == 40
    assert anima_lora_block_count(set()) is None
    # Anima-2.9B inserts copies of blocks 1 and 3 at 2 and 5; the last block
    # of the 2B layout ends up last.
    assert [anima_source_block(i, 28, 40) for i in (2, 3, 5, 39)] == [1, 2, 3, 27]
    assert anima_source_block(30, 40, 40) == 30
    assert anima_source_block(30, 28, None) == 30


def test_reference_blocks_land_on_the_original_block():
    # 2.9B: block 18 of the 2B layout is at 26, its inserted copy at 27.
    assert anima_model_block(18, 28, 40) == 26
    assert anima_model_block(27, 28, 40) == 39
    assert anima_model_block(1, 28, 52) == 1
    assert anima_model_block(18, 28, 28) == 18
    for model_blocks in (40, 52):
        for index in range(28):
            assert anima_source_block(anima_model_block(index, 28, model_blocks), 28, model_blocks) == index


@pytest.mark.parametrize(
    "text, total, expected",
    [
        ("0-3,7", 28, {0, 1, 2, 3, 7}),
        ("1", 40, {1, 2}),  # 2.9B: block 1 and its inserted copy
        ("1", 52, {1, 2, 3}),  # 3.8B: block 1 and its two copies
        ("27", 40, {39}),
        ("0-27", 52, set(range(52))),
        ("26-99", 40, {38, 39}),  # clipped to the 2B layout, then mapped
        ("29", 30, {29}),  # an unknown depth: the indices as written
    ],
)
def test_block_selections_follow_the_2b_layout_onto_the_model(text, total, expected):
    assert model_block_set(text, total, feature="Test") == expected


def test_block_selection_errors_name_the_feature():
    with pytest.raises(ValueError, match="Test block selection has an invalid part 'x'"):
        model_block_set("0,x", 40, feature="Test")
    with pytest.raises(ValueError, match="Test block selection is empty"):
        model_block_set("30-39", 40, feature="Test")  # past the 2B layout


def test_lora_key_factor_weights_each_part_or_keeps_the_fallback():
    block = "diffusion_model.blocks.2.mlp.layer1.weight"
    qwen = "text_encoders.qwen3_06b.model.layers.4.mlp.up_proj.weight"
    other = "diffusion_model.final_layer.linear.weight"
    blocks = LayerWeights(default=1.0, values={1: 0.5, 2: 0.25})
    qwen_weights = LayerWeights(default=1.0, values={4: 0.75})
    weights = {"qwen_weights": qwen_weights, "block_weights": blocks, "module_weights": {"mlp": 0.5}}

    assert lora_key_factor(block, **weights) == 0.25 * 0.5
    assert lora_key_factor((block, 0), **weights) == 0.25 * 0.5  # an offset patch key
    assert lora_key_factor(block, lora_blocks=28, model_blocks=40, **weights) == 0.5 * 0.5  # model block 2 is a copy of 1
    assert lora_key_factor(qwen, **weights) == 0.75
    assert lora_key_factor(other, 0.7, **weights) == 0.7
    assert lora_key_factor("", 0.7, **weights) == 0.7
    # Parts left unset keep the fallback; module weights still scale it
    assert lora_key_factor(block, 0.7) == 0.7
    assert lora_key_factor(qwen, 0.7, block_weights=blocks) == 0.7
    assert lora_key_factor(block, 0.8, module_weights={"mlp": 0.5}) == 0.8 * 0.5


def test_lora_file_names_follow_the_network_index(monkeypatch, tmp_path):
    import sys

    from modules.anima_lora_support import lora_file_names

    path = tmp_path / "Cute_Style.safetensors"
    entry = SimpleNamespace(filename=str(path), alias="cute", get_alias=lambda: "CuteAlias")
    networks = SimpleNamespace(available_networks={"Cute_Style": entry})
    monkeypatch.setitem(sys.modules, "networks", networks)

    names = lora_file_names(path)
    assert {"cute_style.safetensors", "cute_style", "cute", "cutealias"} <= names

    # sd_forge_lora refreshes its index in place
    renamed = SimpleNamespace(filename=str(path), alias="kawaii", get_alias=lambda: "kawaii")
    networks.available_networks.clear()
    networks.available_networks["Cute_Style"] = renamed
    names = lora_file_names(path)
    assert "kawaii" in names and "cute" not in names


def test_an_unparsable_layer_weight_is_skipped_with_a_warning(caplog):
    from modules.anima_lora_support import parse_layer_weights

    weights = parse_layer_weights("all=0.8, 0-9=O.5, 12=0.25")

    assert weights.values == {12: 0.25} and weights.factor(3) == 0.8  # not a silent 1.0
    assert "0-9=O.5" in caplog.text


def test_weight_sections():
    from modules.anima_lora_support import split_weight_sections

    assert split_weight_sections("blocks=0-9=0.8,10-27=1;modules=mlp=0.5；qwen=0-3=0") == {"blocks": "0-9=0.8,10-27=1", "modules": "mlp=0.5", "qwen": "0-3=0"}
    assert split_weight_sections("0-7=0.6;8-20=1.0") == {"blocks": "0-7=0.6;8-20=1.0"}
    assert split_weight_sections("block：0.5 0.6;modules=cross=1;20-27=0.7") == {"blocks": "0.5 0.6;20-27=0.7", "modules": "cross=1"}
    assert split_weight_sections("block_lrs=0.5 0.5") == {"blocks": "block_lrs=0.5 0.5"}
    assert split_weight_sections("") == {}


def test_lora_name_keys_and_targets():
    from modules.anima_lora_support import name_keys, parse_targets

    assert name_keys(" Sub/Cat.safetensors ") == {"sub/cat.safetensors", "cat.safetensors", "cat"}
    assert name_keys("") == set()
    assert parse_targets("Cat，sub/dog.safetensors；\nbird") == {"cat", "sub/dog.safetensors", "dog.safetensors", "dog", "bird"}


def test_prompt_names_resolve_like_the_lora_loader(monkeypatch):
    import sys

    from modules.anima_lora_support import resolve_network

    cat, alias_owner, stem_owner = SimpleNamespace(filename="cat"), SimpleNamespace(filename="a"), SimpleNamespace(filename="b")
    refreshed = []

    def update_available_networks_by_names(names):
        refreshed.extend(names)
        if names == ["late"]:
            networks.available_network_aliases["late"] = cat

    networks = SimpleNamespace(
        available_networks={"cat": cat, "style": stem_owner},
        available_network_aliases={"cat": cat, "style": alias_owner},
        forbidden_network_aliases={"none", "style"},
        update_available_networks_by_names=update_available_networks_by_names,
    )
    monkeypatch.setitem(sys.modules, "networks", networks)

    assert resolve_network(" cat ") is cat
    assert resolve_network("style") is stem_owner  # an alias shared by two files: the network name wins
    assert resolve_network("Cat") is None  # load_networks does not match case-insensitively either
    assert resolve_network("late") is cat and refreshed == ["Cat", "late"]


def test_a_section_name_ends_at_its_first_separator():
    from modules.anima_lora_support import split_weight_sections

    assert split_weight_sections("modules：mlp=0.5") == {"modules": "mlp=0.5"}
    assert split_weight_sections("blocks：0-9=0.5；qwen：0-3=0") == {"blocks": "0-9=0.5", "qwen": "0-3=0"}


def test_layer_weights_from_a_prompt_are_bounded_and_finite(caplog):
    from modules.anima_lora_support import MAX_LAYER_INDEX, parse_layer_weights

    huge = parse_layer_weights("0-2000000000=0.5")
    assert len(huge.values) == MAX_LAYER_INDEX + 1 and huge.factor(MAX_LAYER_INDEX) == 0.5

    weights = parse_layer_weights(f"300-400=0.5, 999999=0.5, {'9' * 5000}=0.5, 1=nan, 2=inf, 3-4=-inf, all=nan, 5=0.25")
    assert weights.values == {5: 0.25} and weights.default == 1.0
    assert "300-400" in caplog.text and "1=nan" in caplog.text
    assert parse_layer_weights("0.5, inf, nan").values == {0: 0.5}


def test_non_finite_module_weights_are_skipped():
    from modules.anima_lora_support import parse_module_weights

    assert parse_module_weights("mlp=nan, cross=inf, self=0.5") == {"self_attn": 0.5}


def test_unparsable_module_weights_are_skipped_with_a_warning(caplog):
    from modules.anima_lora_support import parse_module_weights

    with caplog.at_level("WARNING"):
        weights = parse_module_weights("mlp=0.8x, cross=, self=0.5")

    assert weights == {"self_attn": 0.5}  # not silently full strength
    assert "mlp=0.8x" in caplog.text and "cross=" in caplog.text


def test_manual_timing_keeps_a_ban_only_templates_action():
    from modules.anima_lora_support import manual_window_bans

    assert manual_window_bans([{"stage": "style", "mode": "ban"}])
    assert not manual_window_bans([{"stage": "style", "mode": "ban"}, {"stage": "character", "mode": "allow"}])
    assert not manual_window_bans([])


def test_sigma_progress_is_the_position_in_the_full_schedule():
    import torch

    from modules.anima_lora_support import sigma_progress

    sigmas = torch.tensor([14.6, 7.0, 3.0, 1.0, 0.0])
    assert sigma_progress(torch.tensor([3.0, 3.0]), sigmas) == 0.5
    assert sigma_progress(2.9, [14.6, 7.0, 3.0, 1.0, 0.0]) == 0.5  # nearest schedule sigma
    assert sigma_progress(torch.tensor([3.0]), torch.tensor([1.0])) is None
    assert sigma_progress(None, sigmas) is None


def test_sigma_progress_reads_a_schedule_off_the_device_once():
    import torch

    from modules.anima_lora_support import sigma_progress

    class Schedule:  # a device tensor: every tolist() is a device-to-host copy
        def __init__(self, values):
            self.values, self.copies = values, 0

        def flatten(self):
            return self

        def tolist(self):
            self.copies += 1
            return list(self.values)

    schedule = Schedule([14.6, 7.0, 3.0, 1.0, 0.0])
    progress = [sigma_progress(torch.tensor([sigma]), schedule) for sigma in (14.6, 7.0, 3.0, 1.0)]
    assert progress == [0.0, 0.25, 0.5, 0.75] and schedule.copies == 1

    hires = Schedule([3.0, 0.0])  # the next pass's schedule
    assert sigma_progress(torch.tensor([0.0]), hires) == 1.0 and hires.copies == 1


def test_ui_text_falls_back_to_chinese_and_to_the_key(monkeypatch):
    from modules import shared
    from modules.anima_lora_support import intro_text, ui_text

    strings = {"zh": {"title": "标题"}, "en": {"title": "Title"}}
    monkeypatch.setattr(shared, "opts", type("Opts", (), {"lang": "en"})(), raising=False)
    assert ui_text(strings, "lang", "title") == "Title" and ui_text(strings, "lang", "other") == "other"
    assert ui_text(strings, "unset_option", "title") == "标题"
    assert intro_text("中文", "en", "zh") == "zh" and intro_text("English", "en", "zh") == "en"


def test_cond_caches_are_invalidated_where_the_class_shares_them():
    from modules.anima_lora_support import invalidate_cond_caches

    class Processing:
        cached_c = ["key", "conds"]
        cached_hr_uc = ["key", "conds"]

    p = Processing()
    invalidate_cond_caches(p)

    assert Processing.cached_c == [None, "conds"] and Processing.cached_hr_uc == [None, "conds"]


def test_batch_prompt_rewrite_is_all_or_nothing():
    from modules.anima_lora_support import rewrite_batch_prompts

    p = SimpleNamespace(iteration=1, batch_size=1, prompts=["b"], all_prompts=["a", "b"], all_hr_prompts=["ha", "hb"])
    rewrite_batch_prompts(p, lambda prompts: [x.upper() for x in prompts], lambda prompts: [x + "!" for x in prompts])
    assert (p.prompts, p.all_prompts, p.all_hr_prompts) == (["B"], ["a", "B"], ["ha", "hb!"])

    def conflict(prompts):
        raise ValueError("conflict")

    with pytest.raises(ValueError):
        rewrite_batch_prompts(p, lambda prompts: ["x"] * len(prompts), conflict)
    assert (p.prompts, p.all_prompts, p.all_hr_prompts) == (["B"], ["a", "B"], ["ha", "hb!"])


def test_to_float_falls_back_on_what_is_not_a_number():
    assert [to_float(value, 3.0) for value in (None, "", "x", [1], "0.5", 2, True)] == [3.0, 3.0, 3.0, 3.0, 0.5, 2.0, 1.0]
    assert clamp(5, 0, 1) == 1 and clamp(-1, 0, 1) == 0 and clamp(0.5, 0, 1) == 0.5
