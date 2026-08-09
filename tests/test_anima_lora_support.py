from types import SimpleNamespace

import pytest

from modules.anima_lora_support import (
    active_template_parts,
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
