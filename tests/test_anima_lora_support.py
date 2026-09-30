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


def test_block_selection_parser_can_be_supplied():
    calls = []

    def lenient(text, total):
        calls.append((text, total))
        return {total - 1}

    assert model_block_set("junk", 40, parse=lenient) == {39}  # the 2B layout's last block
    assert model_block_set("junk", 30, parse=lenient) == {29}  # an unknown depth: the model's own
    assert calls == [("junk", 28), ("junk", 30)]


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


def test_an_unknown_name_is_rescanned_for_once_per_run(monkeypatch):
    import sys

    from modules.anima_lora_support import network_misses, resolve_network

    late = SimpleNamespace(filename="late")
    on_disk = {}
    refreshed = []

    def update_available_networks_by_names(names):  # walks every LoRA directory
        refreshed.extend(names)
        networks.available_network_aliases.update((name, on_disk[name]) for name in names if name in on_disk)

    networks = SimpleNamespace(available_networks={}, available_network_aliases={}, forbidden_network_aliases={"none"}, update_available_networks_by_names=update_available_networks_by_names)
    monkeypatch.setitem(sys.modules, "networks", networks)

    run = SimpleNamespace()
    for _ in range(3):  # batches, passes, tabs
        assert resolve_network("late", network_misses(run)) is None
    assert refreshed == ["late"]
    assert network_misses(SimpleNamespace(**vars(run))) is network_misses(run)  # a nested run on a shallow copy

    on_disk["late"] = late  # added to disk before the next generation
    assert resolve_network("late", network_misses(SimpleNamespace())) is late
    assert refreshed == ["late", "late"]
    assert resolve_network("late") is late and refreshed == ["late", "late"]  # found without a rescan now


def test_a_section_name_ends_at_its_first_separator():
    from modules.anima_lora_support import split_weight_sections

    assert split_weight_sections("modules：mlp=0.5") == {"modules": "mlp=0.5"}
    assert split_weight_sections("blocks：0-9=0.5；qwen：0-3=0") == {"blocks": "0-9=0.5", "qwen": "0-3=0"}
    assert split_weight_sections("modules＝mlp=0.5") == {"modules": "mlp=0.5"}
    assert split_weight_sections("ＢＬＯＣＫＳ＝０－９＝０．５；ｍｏｄｕｌｅｓ＝ｍｌｐ＝０．５") == {"blocks": "0-9=0.5", "modules": "mlp=0.5"}


def test_weight_text_folds_full_width_input_to_ascii():
    from modules.anima_lora_support import fold_weight_text, is_plain_number

    assert fold_weight_text("ｍｏｄｕｌｅｓ：ｍｌｐ＝０．５，ｃｒｏｓｓ＝－１；　ａｌｌ＝1") == "modules=mlp=0.5,cross=-1; all=1"
    assert fold_weight_text("0-9=0.5;modules=mlp=0.8") == "0-9=0.5;modules=mlp=0.8"
    assert fold_weight_text(None) == ""
    assert is_plain_number("０．５") and is_plain_number("－0.5") and is_plain_number(" ＋.５ ")
    assert not is_plain_number("0：5") and not is_plain_number("０．５ｘ")


@pytest.mark.parametrize("dash", ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212", "\ufe63", "\uff0d"])
def test_dash_variants_read_as_ascii_hyphens(dash):
    # Ranges pasted from prose ("0–9=0.7", as the generation guide once wrote them)
    from modules.anima_lora_support import parse_layer_weights, plain_number

    assert parse_layer_weights(f"0{dash}9=0.7, 10{dash}18=1").values == {**dict.fromkeys(range(10), 0.7), **dict.fromkeys(range(10, 19), 1.0)}
    assert plain_number(f"{dash}0.5") == -0.5


def test_compatibility_digits_are_not_read_as_other_numbers(caplog):
    # Superscripts, circled digits and ligatures are not decimal digits: they are
    # rejected with a warning, never silently read as another value
    from modules.anima_lora_support import parse_layer_weights, plain_number

    for text in ("0.5\u00b2", "\u2460", "\u2474", "1\u2044" "2"):
        assert plain_number(text) is None, text
    assert parse_layer_weights("0-9=0.5\u00b2").values == {}
    assert "0-9=0.5\u00b2" in caplog.text
    assert plain_number("\u0665") == 5.0  # a decimal digit of another script, as float() reads it
    assert plain_number("0\u30025") == 0.5  # the ideographic full stop as a decimal point


def _seconds(parse, text, repeats=3):
    import time

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        parse(text)
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("parser", ["parse_layer_weights", "looks_like_layer_weights", "parse_module_weights", "split_weight_sections"])
@pytest.mark.parametrize(
    "shape",
    [
        lambda spaces: f"0=1{spaces}x",  # spaces around "=" are dropped
        lambda spaces: f"0{spaces}-{spaces}9=1",  # and around a range's dash
        lambda spaces: f"block_lrs{spaces}x",
        lambda spaces: f"1{spaces}x",
    ],
    ids=["spaced-equals", "spaced-range-dash", "block-lrs-prefix", "bare-tokens"],
)
def test_weight_text_parses_in_linear_time_on_long_whitespace_runs(parser, shape):
    # <lora:x:0=1      x> reaches the tokenizer from every prompt: time must
    # grow with the text (x4 for four times the text), not with its square (x16)
    import logging

    import modules.anima_lora_support as support

    parse = getattr(support, parser)
    logging.disable(logging.WARNING)
    try:
        short, long = (_seconds(parse, shape(" " * n)) for n in (10000, 40000))
    finally:
        logging.disable(logging.NOTSET)
    assert long < 8 * short + 1e-3, (short, long)


def test_layer_weights_from_a_prompt_are_bounded_and_finite(caplog):
    from modules.anima_lora_support import MAX_LAYER_INDEX, parse_layer_weights

    huge = parse_layer_weights("0-2000000000=0.5")
    assert len(huge.values) == MAX_LAYER_INDEX + 1 and huge.factor(MAX_LAYER_INDEX) == 0.5

    weights = parse_layer_weights(f"300-400=0.5, 999999=0.5, {'9' * 5000}=0.5, 1=nan, 2=inf, 3-4=-inf, all=nan, 5=0.25")
    assert weights.values == {5: 0.25} and weights.default == 1.0
    assert "300-400" in caplog.text and "1=nan" in caplog.text
    assert parse_layer_weights("0.5, inf, nan").values == {0: 0.5}


@pytest.mark.parametrize(
    "text, values, ignored",
    [
        ("0-9=0 5", dict.fromkeys(range(10), 0.0), "5"),  # not a value "0 5", not a silent 0-9=0
        ("1 0=0.5", {0: 0.5}, "1"),  # not block 10
        ("0.8, 3=0.2", {3: 0.2}, "0.8"),
        ("0.5 0.6 3=0.2", {3: 0.2}, "0.5 0.6"),
    ],
)
def test_bare_numbers_next_to_selectors_are_ignored_with_a_warning(text, values, ignored, caplog):
    from modules.anima_lora_support import parse_layer_weights

    with caplog.at_level("WARNING"):
        assert parse_layer_weights(text).values == values
    assert f"Ignored Anima layer weights without a selector: {ignored}" in caplog.text


def test_layer_indices_may_have_leading_zeros(caplog):
    from modules.anima_lora_support import parse_layer_weights

    weights = parse_layer_weights(f"05=0.5, 0010-0012=0.3, {'0' * 5000}7=0.2")
    assert weights.values == {5: 0.5, 7: 0.2, 10: 0.3, 11: 0.3, 12: 0.3}
    assert "past index" not in caplog.text


_BLOCKS_0_9 = dict.fromkeys(range(10), 0.5)


# (text, default, values, looks like layer weights): forms (and rejections) from before the
# shared weight tokenizer, which must parse the same
@pytest.mark.parametrize(
    "text, default, values, looks",
    [
        ("0.5 0.8 1", 1.0, {0: 0.5, 1: 0.8, 2: 1.0}, True),
        ("0.5, 0.8", 1.0, {0: 0.5, 1: 0.8}, True),
        ("0.5;0.8\n1", 1.0, {0: 0.5, 1: 0.8, 2: 1.0}, True),
        ("0.5 -0.3", 1.0, {0: 0.5, 1: -0.3}, True),
        ("0.5 -0.3 all=0.8", 0.8, {0: 0.5, 1: -0.3}, True),  # bare values count when only the default is set
        ("0.5 0.6 3=0.2", 1.0, {3: 0.2}, True),
        ("0-9=0.5,10-27=1", 1.0, {**_BLOCKS_0_9, **dict.fromkeys(range(10, 28), 1.0)}, True),
        ("0-9=0.5 10-12=1", 1.0, {**_BLOCKS_0_9, 10: 1.0, 11: 1.0, 12: 1.0}, True),
        ("0-9=0.5\n10=1", 1.0, {**_BLOCKS_0_9, 10: 1.0}, True),
        ("\t0-3=0.5\r\n4=1\r\n", 1.0, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 1.0}, True),
        ("0-3=0.5 all=0 12=1", 0.0, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 12: 1.0}, True),
        ("*=0.8", 0.8, {}, True),
        ("DEFAULT=0.8", 0.8, {}, True),
        ("block_lr=0.5,0.6", 1.0, {0: 0.5, 1: 0.6}, True),
        ("0-9=0.5；10=1", 1.0, {**_BLOCKS_0_9, 10: 1.0}, True),
        ("0.5，0.6", 1.0, {0: 0.5, 1: 0.6}, True),
        ("3=０.５", 1.0, {3: 0.5}, True),
        ("0-2=1e-1, 3=+0.5, 4=.5, 5=-0.5", 1.0, {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.5, 4: 0.5, 5: -0.5}, True),
        ("3=0.2 x", 1.0, {3: 0.2}, True),
        ("0-9-12=0.5", 1.0, {}, False),
        ("mlp=0.5", 1.0, {}, False),
        ("0.5 x", 1.0, {0: 0.5}, False),
        ("0.5", 1.0, {0: 0.5}, False),
        ("", 1.0, {}, False),
    ],
)
def test_layer_weight_forms_parse_as_before_the_shared_tokenizer(text, default, values, looks):
    from modules.anima_lora_support import looks_like_layer_weights, parse_layer_weights

    weights = parse_layer_weights(text)
    assert (weights.default, weights.values, looks_like_layer_weights(text)) == (default, values, looks)


@pytest.mark.parametrize(
    "text, default, values",
    [
        ("0-9 = 0.5", 1.0, _BLOCKS_0_9),
        ("0 - 9=0.5", 1.0, _BLOCKS_0_9),
        ("0 - 9 = 0.5, 10 = 1", 1.0, {**_BLOCKS_0_9, 10: 1.0}),
        ("all = 0.8", 0.8, {}),
        ("0-9 =0.5, 10-27 = 1", 1.0, {**_BLOCKS_0_9, **dict.fromkeys(range(10, 28), 1.0)}),
        ("12 = 2 all = 0.8", 0.8, {12: 2.0}),
        ("０００５=0.5", 1.0, {5: 0.5}),
        ("０００５-０００７=0.3", 1.0, {5: 0.3, 6: 0.3, 7: 0.3}),
        ("０－９＝０．５，１０＝１；ａｌｌ＝0.8", 0.8, {**_BLOCKS_0_9, 10: 1.0}),
        ("0-9=0.5　10=1", 1.0, {**_BLOCKS_0_9, 10: 1.0}),  # ideographic space
        ("0-9：0.5", 1.0, _BLOCKS_0_9),
        ("ｂｌｏｃｋ＿ｌｒｓ＝０－９＝０．５", 1.0, _BLOCKS_0_9),
    ],
)
def test_layer_weights_allow_spaces_and_full_width_input(text, default, values, caplog):
    from modules.anima_lora_support import looks_like_layer_weights, parse_layer_weights

    weights = parse_layer_weights(text)
    assert (weights.default, weights.values) == (default, values)
    assert looks_like_layer_weights(text)
    assert "Ignored" not in caplog.text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mlp = 0.5", {"mlp": 0.5}),
        (" self attention = 0.9 , cross = 0.3 ", {"self_attn": 0.9, "cross_attn": 0.3}),
        ("ｍｌｐ＝０．５", {"mlp": 0.5}),
        ("mlp：0.5，cross：0.3", {"mlp": 0.5, "cross_attn": 0.3}),
    ],
)
def test_module_weights_allow_spaces_and_full_width_input(text, expected):
    from modules.anima_lora_support import parse_module_weights

    assert parse_module_weights(text) == expected


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


def test_rewritten_prompts_are_restored_as_written():
    import copy

    from modules.anima_lora_support import restore_batch_prompts, restore_run_prompts, rewrite_batch_prompts

    p = SimpleNamespace(iteration=1, batch_size=2, all_prompts=["a", "b", "c", "d"], all_hr_prompts=["ha", "hb", "hc", "hd"])
    p.prompts = p.all_prompts[2:4]
    # Two rewriters of the batch (the layer weight's two tabs, the stage scheduler)
    rewrite_batch_prompts(p, lambda prompts: [x + "1" for x in prompts], lambda prompts: [x + "1" for x in prompts])
    rewrite_batch_prompts(p, lambda prompts: [x + "2" for x in prompts], None)
    assert p.all_prompts == ["a", "b", "c12", "d12"] and p.all_hr_prompts == ["ha", "hb", "hc1", "hd1"]

    p.all_prompts[3] = "changed by another script"
    p.prompts = p.all_prompts[2:4]  # processing re-slices it after postprocess_batch
    # A nested run on a shallow copy of p has nothing of its own to restore
    nested = copy.copy(p)
    nested.prompts = list(p.prompts)
    restore_batch_prompts(nested)
    assert nested.prompts == ["c12", "changed by another script"]

    restore_batch_prompts(p)
    assert p.prompts == ["c", "changed by another script"] and p.all_hr_prompts == ["ha", "hb", "hc", "hd"]
    assert p.all_prompts == ["a", "b", "c12", "changed by another script"]  # a nested run still reads what was loaded

    restore_run_prompts(p)
    assert p.all_prompts == ["a", "b", "c", "changed by another script"]
    p.all_prompts[2] = "c12"
    restore_run_prompts(p)  # forgotten after the run
    assert p.all_prompts[2] == "c12"


def test_to_float_falls_back_on_what_is_not_a_number():
    assert [to_float(value, 3.0) for value in (None, "", "x", [1], "0.5", 2, True)] == [3.0, 3.0, 3.0, 3.0, 0.5, 2.0, 1.0]
    assert clamp(5, 0, 1) == 1 and clamp(-1, 0, 1) == 0 and clamp(0.5, 0, 1) == 0.5


# The tag pattern before it kept "<" out of a tag, for comparison
_UNBOUNDED_LORA_TAG_RE = r"<lora:([^:>]+)(?::([^>]*))?>"


def _tag_matches(pattern, text):
    import re

    return [(match.span(), match.groups()) for match in re.finditer(pattern, text)]


@pytest.mark.parametrize(
    "prompt",
    [
        "masterpiece, 1girl, <lora:cat:0.8>, <lora:sub/dog-v2:1:0.5> smile",
        "<lora:x:[0.5@0,1@0.5]> <lora:y> <lora:z:blocks=0-9=0.6,10-27=1;modules=mlp=0.5;qwen=0-3=0>",
        "<lora:a:te=0:unet=1.2>BREAK<lora:b:0.3>\n<lora:c d:1>",
        "<lora:> <lora::1> <lora:x:> a <lora:foo:1 b",
        "<hypernet:h:1> <lora:q:1> （<lora:中文:0.7>） <lora:x:1>> <<lora:y:1>",
    ],
)
def test_lora_tags_match_as_before(prompt):
    from modules.anima_lora_support import LORA_TAG_RE

    assert _tag_matches(LORA_TAG_RE, prompt) == _tag_matches(_UNBOUNDED_LORA_TAG_RE, prompt)


def test_lora_tags_match_as_before_unless_a_tag_would_hold_another_tags_start():
    import random

    from modules.anima_lora_support import LORA_TAG_RE

    pieces = ["<lora:", "<", ">", ":", "a", "b", "0.5", "@", "[", "]", ",", " ", "=", "\n", "lora"]
    rng = random.Random(0)
    compared = 0
    for _ in range(20000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 14)))
        before = _tag_matches(_UNBOUNDED_LORA_TAG_RE, text)
        if any("<" in text[start + 1 : end] for (start, end), _ in before):
            continue  # "<lora:<lora:x:1>" now reads as the inner tag
        assert _tag_matches(LORA_TAG_RE, text) == before, text
        compared += 1
    assert compared > 15000


def test_unterminated_lora_tags_are_scanned_in_linear_time():
    import time

    from modules.anima_lora_support import LORA_TAG_RE

    start = time.perf_counter()
    assert LORA_TAG_RE.findall("<lora:" * 4000) == []
    assert LORA_TAG_RE.findall("<lora:a:" * 4000 + ">") == [("a", "")]
    assert time.perf_counter() - start < 0.2  # about 1 s with the unbounded pattern


def test_weight_numbers_share_one_grammar(caplog):
    # A lone strength and a value inside a weight text read the same way: an
    # exponent is fine, a "_" separator is refused ("0_5" is not 5), as are
    # inf / nan and overflow
    from modules.anima_lora_support import parse_layer_weights, parse_module_weights, plain_number

    assert plain_number("1e-1") == 0.1 and parse_layer_weights("0=1e-1").values == {0: 0.1}
    for text in ("0_5", "1_000", "inf", "nan", "1e999"):
        assert plain_number(text) is None, text
    assert parse_layer_weights("0-9=0_5, all=0_5").values == {} and parse_layer_weights("all=0_5").default == 1.0
    assert parse_module_weights("mlp=0_5") == {}
    assert "0-9=0_5" in caplog.text and "mlp=0_5" in caplog.text
