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

    # _TEMPLATES.path() seeds <data dir>/extension-data with the bundled templates
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
    state = module.RuntimeState(SimpleNamespace(), True, config, config, module.DISABLE_NONE, [], False)
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
    return [module._stage_curve_factor(config, step / 100) for step in steps]


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
    state = module.RuntimeState(None, True, config, config, module.DISABLE_SELECTED, ["<lora:dog:1>"], False)
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


@pytest.mark.parametrize(
    "hires_panel, loads_per_batch",
    [
        (BLOCKS, [1, 0]),  # same panel (other template and shift): one set of patches serves both passes
        ("0.5", [4, 4]),  # other panel: the patches differ, so each pass change rebuilds them
    ],
)
def test_hires_fix_reloads_loras_only_when_the_passes_differ(module, monkeypatch, hires_panel, loads_per_batch):
    from backend.patcher.base import ModelPatcher
    from modules import extra_networks

    sd_model = SimpleNamespace(current_lora_hash=None)
    loads = []

    def load_networks(p, extra_network_data):  # an unchanged LoRA set: rebuilt only after a hash reset
        if sd_model.current_lora_hash != "compiled":
            loads[-1] += 1
            sd_model.current_lora_hash = "compiled"

    monkeypatch.setattr(ModelPatcher, "add_patches", ModelPatcher.add_patches)
    monkeypatch.setattr(extra_networks, "activate", load_networks)
    module.install_patch()
    base_data, hires_data = object(), object()
    p = SimpleNamespace(sd_model=sd_model, enable_hr=True, extra_network_data=base_data, hr_extra_network_data=hires_data)
    base = _config(module, module.PRESET_CHARACTER_ALLOW, panel=BLOCKS)
    hires = _config(module, module.PRESET_STYLE_ALLOW, panel=hires_panel, shift=5.0)
    p._anima_lora_stage_scheduler_state = module.RuntimeState(p, True, base, hires, module.DISABLE_NONE, [], False)

    for _batch in range(2):
        loads.append(0)
        # processing.py: the batch, setup_conds (hires conds, then base again), sample_hr_pass
        for data in (base_data, hires_data, base_data, hires_data):
            extra_networks.activate(p, data)

    assert loads == loads_per_batch


def test_shared_patches_follow_the_timing_of_the_current_pass(module, monkeypatch):
    p = SimpleNamespace(is_hr_pass=False)
    base = _config(module, module.PRESET_CHARACTER_ALLOW, panel=BLOCKS)  # window 0.20-0.74
    hires = _config(module, module.PRESET_STYLE_ALLOW, panel=BLOCKS)  # window 0.62-1.00
    state = module.RuntimeState(p, True, base, hires, module.DISABLE_NONE, [], False)
    strength = module.ScheduledStrength(1.0, "x.safetensors", state.patch_pass("base"))
    state.register(strength)
    monkeypatch.setattr(module, "_ACTIVE_STATE", state)

    def factor_at(step):
        denoiser = SimpleNamespace(p=p, step=step, total_steps=101)
        module._denoiser_callback(SimpleNamespace(denoiser=denoiser, sampling_step=step, total_sampling_steps=101))
        return strength.factor

    assert strength.pass_name == module.BOTH_PASSES
    assert factor_at(47) == pytest.approx(1.0) and factor_at(90) == 0.0
    p.is_hr_pass = True
    assert factor_at(47) == 0.0 and factor_at(90) > 0.0


def _manual(module, template, start, end, peak, strength=1.0):
    return module._build_pass_config(module.MODE_PROMPT_ALL, "", "", template, "", False, None, start, end, peak, strength, module.CURVE_HOLD, 3.0)


def test_a_ban_template_keeps_blocking_under_manual_timing(module):
    ban = _manual(module, module.PRESET_STYLE_BAN, 0.62, 1.0, 0.86)  # its window, as the template loads it
    assert _factors(module, ban, (30, 61, 62, 90)) == [1.0, 1.0, 0.0, 0.0]

    allow = _manual(module, module.PRESET_STYLE_ALLOW, 0.62, 1.0, 0.86)
    assert _factors(module, allow, (30, 61, 62, 90)) == [0.0, 0.0, 1.0, 1.0]


def test_stage_strength_scales_auto_timed_templates(module):
    half = module._build_pass_config(module.MODE_PROMPT_ALL, "", "", module.PRESET_CHARACTER_ALLOW, "", True, None, 0.2, 0.8, 0.5, 0.5, module.CURVE_SMOOTH, 3.0)
    assert max(_factors(module, half, range(101))) == pytest.approx(0.5)
    ban = module._build_pass_config(module.MODE_PROMPT_ALL, "", "", module.PRESET_STYLE_BAN, "", True, None, 0.2, 0.8, 0.5, 0.5, module.CURVE_SMOOTH, 3.0)
    assert _factors(module, ban, (0, 90)) == [0.5, 0.0]

    # Moving the timing with Shift keeps the slider's multiplier
    *_, strength = module._auto_timing_ui(module.PRESET_CHARACTER_ALLOW, 5.0, True, 0.2, 0.74, 0.47, 0.5)
    assert strength["value"] == 0.5


def test_progress_is_the_position_in_the_full_sigma_schedule(module):
    import torch

    sigmas = torch.linspace(1.0, 0.0, 11)
    unet = SimpleNamespace(model_options={"transformer_options": {"sampling_sigmas": sigmas}})
    p = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))
    # Hires. fix at denoise 0.5: its first step is halfway down the schedule
    params = SimpleNamespace(sigma=sigmas[5:6].repeat(2), denoiser=SimpleNamespace(p=p, step=0, total_steps=5), sampling_step=0, total_sampling_steps=5)
    assert module._sampling_progress(params) == 0.5

    unet.model_options = {}
    params.denoiser.step = 2
    assert module._sampling_progress(params) == 0.5  # by steps without a schedule


def test_scheduled_strengths_work_in_boft_arithmetic(module):
    import torch

    from modules_forge.packages.comfy.weight_adapter.boft import BOFTAdapter

    torch.manual_seed(0)
    adapter = BOFTAdapter(set(), (torch.randn(1, 2, 2, 2), None, 0.0, None))
    weight = torch.randn(4, 3)
    strength = module.ScheduledStrength(0.5, "x.safetensors", "base")

    expected = adapter.calculate_weight(weight.clone(), "k", 0.5, 1.0, None, lambda a: a)
    scheduled = adapter.calculate_weight(weight.clone(), "k", strength, 1.0, None, lambda a: a)

    assert not torch.allclose(expected, weight)
    assert torch.allclose(scheduled, expected)  # not left unpatched by a TypeError


def test_prompt_conds_are_recomputed_when_the_panels_text_encoder_weights_change(module, monkeypatch):
    from backend.args import dynamic_args

    monkeypatch.setattr(dynamic_args, "online_lora", False)
    monkeypatch.setattr(module, "_TEXT_ENCODER_SIGNATURE", None, raising=False)
    cache = [None, None, None]  # StableDiffusionProcessing.cached_c

    def run(panel, enabled=True):
        cache[0] = "cond key"
        p = SimpleNamespace(cached_c=cache, extra_generation_params={})
        args = [module.MODE_PROMPT_ALL, "", panel, module.PRESET_CHARACTER_ALLOW, "", True, 0.2, 0.8, 0.5, 1.0, module.CURVE_SMOOTH]
        module.Script().process(p, enabled, *args, False, *args, module.DISABLE_NONE, "")
        module._cleanup_runtime(p)
        return cache[0] is None

    assert run("qwen=0-27=0.5")
    assert not run("qwen=0-27=0.5")
    assert run("qwen=0-27=0.2")
    assert run("blocks=0-9=0.5")  # the text encoder is back to the prompt's weights
    assert not run("blocks=0-9=0.2") and not run("0", enabled=False)
    assert run("qwen=0-27=0.2") and run("qwen=0-27=0.2", enabled=False)


def test_auto_detect_schedules_anima_lokrs_too(module, tmp_path):
    import torch
    from safetensors.torch import save_file

    lokr = tmp_path / "subject.safetensors"
    save_file({"lycoris_blocks_3_mlp_layer1.lokr_w1": torch.zeros(1)}, str(lokr))
    sdxl = tmp_path / "sdxl.safetensors"
    save_file({"lora_unet_mid_block_attentions_0_proj_in.lora_down.weight": torch.zeros(1)}, str(sdxl))
    config = _config(module, module.PRESET_CHARACTER_ALLOW, mode=module.MODE_AUTO_ANIMA)

    assert config.file_matches(str(lokr), frozenset())
    assert not config.file_matches(str(sdxl), frozenset())


def test_template_dropdowns_show_the_templates_as_saved(module):
    builtins = list(module.BUILTIN_TEMPLATES)

    *dropdowns, status = module._save_template_ui("mine", True, builtins[0], 0.0, 1.0, 0.5, 1.0, module.CURVE_SMOOTH)
    assert status == "已保存模板：mine" and len(dropdowns) == 3
    assert all("mine" in dropdown["choices"] and dropdown["value"] == "mine" for dropdown in dropdowns)

    *dropdowns, status = module._rename_template_ui("mine", "renamed")
    assert all("renamed" in dropdown["choices"] and "mine" not in dropdown["choices"] and dropdown["value"] == "renamed" for dropdown in dropdowns)

    *dropdowns, status = module._delete_template_ui("renamed")
    assert status == "已删除模板：renamed"
    assert all("renamed" not in dropdown["choices"] for dropdown in dropdowns)
    assert "renamed" not in module._load_custom_templates()


def test_template_messages_follow_the_ui_language(module, monkeypatch):
    from modules import shared

    builtin = next(iter(module.BUILTIN_TEMPLATES))
    assert set(module.LANG["zh"]) == set(module.LANG["en"])

    monkeypatch.setattr(shared, "opts", SimpleNamespace(**{module.LANGUAGE_OPTION: "zh"}), raising=False)
    assert module._save_template_ui("", True, builtin, 0.0, 1.0, 0.5, 1.0, module.CURVE_SMOOTH)[-1] == "模板名称不能为空。"
    assert module._parts_from_presets(["x"])[1] == "未知基础预设：x"
    assert module._rename_template_ui("a", "a")[-1] == "新旧名称相同，无需重命名。"

    monkeypatch.setattr(shared, "opts", SimpleNamespace(**{module.LANGUAGE_OPTION: "en"}), raising=False)
    assert module._save_template_ui("", True, builtin, 0.0, 1.0, 0.5, 1.0, module.CURVE_SMOOTH)[-1] == "The template name cannot be empty."
    assert module._parts_from_presets(["x"])[1] == "Unknown base preset: x"
    assert module._delete_template_ui(builtin)[-1] == "Built-in templates cannot be deleted."
    conflict = [module.PRESET_STYLE_ALLOW, module.PRESET_STYLE_BAN]
    assert module._parts_from_presets(conflict)[1] == f"Conflicting combo: {module.STAGE_STYLE} is selected as both Apply and Block; not saved."
    help_text = module._template_help_ui(builtin, 3.0)["value"]
    assert "Templates are built from base presets" in help_text and "auto window" in help_text
