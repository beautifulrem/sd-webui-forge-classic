from types import SimpleNamespace

import pytest

from modules.anima_lora_support import (
    ANIMA_BLOCK_MAPPINGS,
    active_template_parts,
    anima_lora_block_count,
    anima_model_block,
    anima_source_block,
    current_sampling_position,
    merge_consistent_rules,
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

    source = pathlib.Path("extensions-builtin/sd_forge_lora/networks.py").read_text(encoding="utf-8")
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
