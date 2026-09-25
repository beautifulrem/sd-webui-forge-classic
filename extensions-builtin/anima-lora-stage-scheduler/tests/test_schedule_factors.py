import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_stage_scheduler.py"
_MODULE = []


@pytest.fixture
def module(tmp_path, monkeypatch):
    from modules import paths_internal

    # _template_file() seeds <data dir>/extension-data with the bundled templates
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    if not _MODULE:
        spec = importlib.util.spec_from_file_location("anima_lora_stage_scheduler_factors", SCRIPT)
        _MODULE.append(importlib.util.module_from_spec(spec))
        spec.loader.exec_module(_MODULE[0])
    return _MODULE[0]


def _block(index, module="mlp.layer1"):
    return f"diffusion_model.blocks.{index}.{module}.weight"


def _config(module, template, *, panel="", mode=None, targets="", shift=3.0):
    return module._build_pass_config(
        mode or module.MODE_PROMPT_ALL, targets, panel, template, "", True, None, 0.2, 0.8, 0.5, 1.0, module.CURVE_SMOOTH, shift
    )


BLOCKS = "blocks=0-9=0.2,10-18=1.0,19-27=0.5"


@pytest.mark.parametrize(
    "model_blocks, index, expected",
    [
        (28, 10, 1.0),
        (28, 27, 0.5),
        (40, 13, 0.2),  # 2.9B block 13 is a copy of 2B block 9
        (40, 27, 1.0),  # ... and 27 of block 18
        (40, 39, 0.5),
        (52, 10, 0.2),  # 3.8B block 10 is a copy of 2B block 5
        (52, 27, 1.0),
        (52, 51, 0.5),
    ],
)
def test_block_weights_follow_a_2b_lora_onto_expanded_models(module, model_blocks, index, expected):
    spec = module._parse_panel_weight(BLOCKS)

    assert spec.strength_for_key(_block(index), 0.7, 28, model_blocks) == expected


def test_block_weights_keep_model_indices_without_a_known_lora_layout(module):
    spec = module._parse_panel_weight(BLOCKS + ";modules=mlp=0.5")

    assert spec.strength_for_key(_block(27), 0.7, None, 52) == 0.5 * 0.5
    assert spec.strength_for_key(_block(51), 0.7, None, 52) == 1.0 * 0.5  # past the listed blocks: default
    assert spec.strength_for_key(_block(51, "self_attn.q_proj"), 0.7, 52, 52) == 1.0
    assert spec.strength_for_key(_block(20, "self_attn.q_proj"), 0.7, 52, 52) == 0.5


def test_added_patches_get_scheduled_per_key_strengths(module):
    config = _config(module, module.PRESET_CHARACTER_ALLOW, panel=BLOCKS)
    state = module.RuntimeState(SimpleNamespace(), "run", True, config, config, module.DISABLE_NONE, [], False)
    qwen = "text_encoders.qwen3_06b.model.layers.2.mlp.up_proj.weight"
    keys = [_block(10), _block(51), qwen]
    destination = {key: [SimpleNamespace(patch=[(0.8, object(), 1.0, None, None)])] for key in keys}

    module._patch_added_strengths(destination, {}, state, config, True, 28, 52)

    strengths = {key: destination[key][0].patch[0][0] for key in keys}
    assert isinstance(strengths[_block(10)], module.ScheduledStrength)
    assert (strengths[_block(10)].base_strength, strengths[_block(51)].base_strength) == (0.2, 0.5)
    assert strengths[qwen] == 0.8 and not isinstance(strengths[qwen], module.ScheduledStrength)
    assert len(state.live_wrappers()) == 2
    state.set_factor(0.5)
    assert float(strengths[_block(51)]) == 0.25


def _factors(module, config, steps):
    return [module._stage_curve_factor(config, step, 101) for step in steps]  # progress = step / 100


def test_character_window_fades_in_and_out(module):
    config = _config(module, module.PRESET_CHARACTER_ALLOW)  # window 0.20-0.74, peak 0.47 at shift 3
    before, start, rising, peak, end, after = _factors(module, config, (10, 20, 30, 47, 74, 90))

    assert before == start == end == after == 0.0
    assert 0.0 < rising < 0.5
    assert peak == pytest.approx(1.0)


def test_composition_window_is_strongest_at_the_start(module):
    config = _config(module, module.PRESET_COMPOSITION_ALLOW)  # window 0.00-0.32, front-loaded

    assert _factors(module, config, (0, 16, 32, 50)) == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_style_ban_zeroes_only_its_window_and_moves_with_shift(module):
    config = _config(module, module.PRESET_STYLE_BAN)  # style window from 0.62
    assert _factors(module, config, (0, 50, 61, 62, 70, 100)) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    late = _config(module, module.PRESET_STYLE_BAN, shift=24.0)  # style window from 0.74
    assert _factors(module, late, (70, 74)) == [1.0, 0.0]


def test_bundled_multi_part_template(module):
    config = _config(module, "防画风过拟合")  # composition + character allowed, style banned

    start, middle, late, end = _factors(module, config, (0, 47, 70, 100))
    assert start == pytest.approx(1.0) and middle == pytest.approx(1.0)
    assert late == end == 0.0  # still inside the character window, but style is banned


def test_prompt_lora_tags_get_the_panel_multiplier(module):
    config = _config(module, module.PRESET_CHARACTER_ALLOW, panel="0.6", mode=module.MODE_TARGETS, targets="cat")
    prompts = ["a <lora:cat:1.2> b <lora:dog:0.9>", "<lora:sub/cat:0.3>"]

    assert module._rewrite_prompt_loras(prompts, config) == ["a <lora:cat:0.6> b <lora:dog:0.9>", "<lora:sub/cat:0.6>"]

    layered = _config(module, module.PRESET_CHARACTER_ALLOW, panel=BLOCKS, mode=module.MODE_TARGETS, targets="cat")
    assert module._rewrite_prompt_loras(["<lora:cat:1.2>"], layered) == ["<lora:cat:1>"]  # weights applied per key
    unset = _config(module, module.PRESET_CHARACTER_ALLOW, panel="0", mode=module.MODE_TARGETS, targets="cat")
    assert module._rewrite_prompt_loras(["<lora:cat:1.2>"], unset) == ["<lora:cat:1.2>"]


def test_batch_rewrite_touches_the_current_batch_and_hires_tags_can_be_dropped(module):
    config = _config(module, module.PRESET_CHARACTER_ALLOW, panel="0.5")
    state = module.RuntimeState(None, "run", True, config, config, module.DISABLE_SELECTED, ["<lora:dog:1>"], False)
    p = SimpleNamespace(
        iteration=1, batch_size=1, enable_hr=True,
        prompts=["<lora:cat:1>"], all_prompts=["<lora:a:1>", "<lora:cat:1>"],
        hr_prompts=[], all_hr_prompts=["<lora:cat:1> <lora:dog:1>", "<lora:dog:1>"],
    )  # fmt: skip

    module._remove_hires_lora_tags(p, state)
    module._rewrite_batch_prompts(p, state)

    assert p.prompts == ["<lora:cat:0.5>"]
    assert p.all_prompts == ["<lora:a:1>", "<lora:cat:0.5>"]  # other batches untouched
    assert p.all_hr_prompts == ["<lora:cat:1> ", ""]
    assert {"cat"} <= config.prompt_names and "a" not in config.prompt_names

    state.hr_disable_mode = module.DISABLE_ALL
    p.all_hr_prompts = ["x <lora:cat:1>"]
    module._remove_hires_lora_tags(p, state)
    assert p.all_hr_prompts == ["x "]
