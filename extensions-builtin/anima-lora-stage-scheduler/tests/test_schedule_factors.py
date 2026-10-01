import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from lib_stage_scheduler import options, runtime, templates, ui

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_stage_scheduler.py"
_MODULE = []


@pytest.fixture
def module():
    if not _MODULE:
        spec = importlib.util.spec_from_file_location("anima_lora_stage_scheduler_factors", SCRIPT)
        _MODULE.append(importlib.util.module_from_spec(spec))
        spec.loader.exec_module(_MODULE[0])
    return _MODULE[0]


def _block(index, module="mlp.layer1"):
    return f"diffusion_model.blocks.{index}.{module}.weight"


def _config(template, *, panel="", mode=None, targets="", shift=3.0):
    return runtime._build_pass_config(
        mode or options.MODE_PROMPT_ALL, targets, panel, template, "", True, None, 0.2, 0.8, 0.5, 1.0, options.CURVE_SMOOTH, shift
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
def test_block_weights_follow_a_2b_lora_onto_expanded_models(model_blocks, index, expected):
    spec = options._parse_panel_weight(BLOCKS)

    assert spec.strength_for_key(_block(index), 0.7, 28, model_blocks) == expected


def test_block_weights_keep_model_indices_without_a_known_lora_layout():
    spec = options._parse_panel_weight(BLOCKS + ";modules=mlp=0.5")

    assert spec.strength_for_key(_block(27), 0.7, None, 52) == 0.5 * 0.5
    assert spec.strength_for_key(_block(51), 0.7, None, 52) == 1.0 * 0.5  # past the listed blocks: default
    assert spec.strength_for_key(_block(51, "self_attn.q_proj"), 0.7, 52, 52) == 1.0
    assert spec.strength_for_key(_block(20, "self_attn.q_proj"), 0.7, 52, 52) == 0.5


@pytest.mark.parametrize("text, numeric", [("０．５", 0.5), ("－0.5", -0.5), ("＋．５", 0.5)])
def test_a_full_width_panel_strength_is_a_number(text, numeric, caplog):
    spec = options._parse_panel_weight(text)

    assert (spec.numeric, spec.raw) == (numeric, text)
    assert "Ignored" not in caplog.text


@pytest.mark.parametrize("text", ["ｍｏｄｕｌｅｓ＝ｍｌｐ＝０．５", "modules＝mlp=0.5", "modules：mlp=0.5", "ＢＬＯＣＫＳ＝０－９＝０．２；ｍｏｄｕｌｅｓ＝ｍｌｐ＝０．５"])
def test_full_width_panel_sections_are_recognised(text, caplog):
    spec = options._parse_panel_weight(text)

    assert spec.has_module and spec.module_weights == {"mlp": 0.5} and spec.raw == text
    if spec.has_block:
        assert spec.block_weights.factor(9) == 0.2 and spec.block_weights.factor(10) == 1.0
    assert "Ignored" not in caplog.text


def test_added_patches_get_scheduled_per_key_strengths():
    config = _config(options.PRESET_CHARACTER_ALLOW, panel=BLOCKS)
    state = runtime.RuntimeState(SimpleNamespace(), True, config, config, options.DISABLE_NONE, [], False)
    qwen = "text_encoders.qwen3_06b.model.layers.2.mlp.up_proj.weight"
    keys = [_block(10), _block(51), qwen]
    destination = {key: [SimpleNamespace(patch=[(0.8, object(), 1.0, None, None)])] for key in keys}

    runtime._patch_added_strengths(destination, {}, state, config, True, 28, 52)

    strengths = {key: destination[key][0].patch[0][0] for key in keys}
    assert isinstance(strengths[_block(10)], runtime.ScheduledStrength)
    assert (strengths[_block(10)].base_strength, strengths[_block(51)].base_strength) == (0.2, 0.5)
    assert strengths[qwen] == 0.8 and not isinstance(strengths[qwen], runtime.ScheduledStrength)
    assert len(state.live_wrappers()) == 2
    state.set_factor(0.5)
    assert float(strengths[_block(51)]) == 0.25


def _factors(config, steps):
    return [runtime._stage_curve_factor(config, step / 100) for step in steps]


def test_character_window_fades_in_and_out():
    config = _config(options.PRESET_CHARACTER_ALLOW)  # window 0.20-0.74, peak 0.47 at shift 3
    before, start, rising, peak, end, after = _factors(config, (10, 20, 30, 47, 74, 90))

    assert before == start == end == after == 0.0
    assert 0.0 < rising < 0.5
    assert peak == pytest.approx(1.0)


def test_composition_window_is_strongest_at_the_start():
    config = _config(options.PRESET_COMPOSITION_ALLOW)  # window 0.00-0.32, front-loaded

    assert _factors(config, (0, 16, 32, 50)) == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_style_ban_zeroes_only_its_window_and_moves_with_shift():
    config = _config(options.PRESET_STYLE_BAN)  # style window from 0.62
    assert _factors(config, (0, 50, 61, 62, 70, 100)) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    late = _config(options.PRESET_STYLE_BAN, shift=24.0)  # style window from 0.74
    assert _factors(late, (70, 74)) == [1.0, 0.0]


def test_bundled_multi_part_template():
    config = _config("防画风过拟合")  # composition + character allowed, style banned

    start, middle, late, end = _factors(config, (0, 47, 70, 100))
    assert start == pytest.approx(1.0) and middle == pytest.approx(1.0)
    assert late == end == 0.0  # still inside the character window, but style is banned


def test_prompt_lora_tags_get_the_panel_multiplier():
    config = _config(options.PRESET_CHARACTER_ALLOW, panel="0.6", mode=options.MODE_TARGETS, targets="cat")
    prompts = ["a <lora:cat:1.2> b <lora:dog:0.9>", "<lora:sub/cat:0.3>"]

    assert runtime._rewrite_prompt_loras(prompts, config) == ["a <lora:cat:0.6> b <lora:dog:0.9>", "<lora:sub/cat:0.6>"]

    layered = _config(options.PRESET_CHARACTER_ALLOW, panel=BLOCKS, mode=options.MODE_TARGETS, targets="cat")
    assert runtime._rewrite_prompt_loras(["<lora:cat:1.2>"], layered) == ["<lora:cat:1>"]  # weights applied per key
    unset = _config(options.PRESET_CHARACTER_ALLOW, panel="0", mode=options.MODE_TARGETS, targets="cat")
    assert runtime._rewrite_prompt_loras(["<lora:cat:1.2>"], unset) == ["<lora:cat:1.2>"]
    # The panel's weight replaces the tag's text-encoder / UNet arguments too
    assert runtime._rewrite_prompt_loras(["<lora:cat:1:0.5>", "<lora:cat:1:te=0>"], config) == ["<lora:cat:0.6>", "<lora:cat:0.6>"]


def test_lora_control_tags_keep_their_schedule_and_text_encoder_off():
    # LoRA Control schedules <lora:x:[w@t,...]>, and Forge loads its text
    # encoder side at 0 only while the "@" is in the tag
    config = _config(options.PRESET_CHARACTER_ALLOW, panel="0.8", mode=options.MODE_PROMPT_ALL)
    prompts = ["a <lora:cat:[0.5@0,1@0.5]> b <lora:dog:1>", "<lora:cat:[0@0,1@1]:unet=1>"]

    assert runtime._rewrite_prompt_loras(prompts, config) == ["a <lora:cat:[0.5@0,1@0.5]> b <lora:dog:0.8>", "<lora:cat:[0@0,1@1]:unet=1>"]


def test_batch_rewrite_touches_the_current_batch_and_hires_tags_can_be_dropped():
    config = _config(options.PRESET_CHARACTER_ALLOW, panel="0.5")
    state = runtime.RuntimeState(None, True, config, config, options.DISABLE_SELECTED, ["<lora:dog:1>"], False)
    p = SimpleNamespace(
        iteration=1, batch_size=1, enable_hr=True,
        prompts=["<lora:cat:1>"], all_prompts=["<lora:a:1>", "<lora:cat:1>"],
        hr_prompts=[], all_hr_prompts=["<lora:cat:1> <lora:dog:1>", "<lora:dog:1>"],
    )  # fmt: skip

    runtime._remove_hires_lora_tags(p, state)
    runtime._rewrite_batch_prompts(p, state)

    assert p.prompts == ["<lora:cat:0.5>"]
    assert p.all_prompts == ["<lora:a:1>", "<lora:cat:0.5>"]  # other batches untouched
    assert p.all_hr_prompts == ["<lora:cat:1> ", ""]
    assert {"cat"} <= config.prompt_names and "a" not in config.prompt_names

    state.hr_disable_mode = options.DISABLE_ALL
    p.all_hr_prompts = ["x <lora:cat:1>"]
    runtime._remove_hires_lora_tags(p, state)
    assert p.all_hr_prompts == ["x "]


@pytest.mark.parametrize(
    "hires_panel, loads_per_batch",
    [
        (BLOCKS, [1, 0]),  # same panel (other template and shift): one set of patches serves both passes
        ("0.5", [4, 4]),  # other panel: the patches differ, so each pass change rebuilds them
    ],
)
def test_hires_fix_reloads_loras_only_when_the_passes_differ(monkeypatch, hires_panel, loads_per_batch):
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
    runtime.install_patch()
    base_data, hires_data = object(), object()
    p = SimpleNamespace(sd_model=sd_model, enable_hr=True, extra_network_data=base_data, hr_extra_network_data=hires_data)
    base = _config(options.PRESET_CHARACTER_ALLOW, panel=BLOCKS)
    hires = _config(options.PRESET_STYLE_ALLOW, panel=hires_panel, shift=5.0)
    p._anima_lora_stage_scheduler_state = runtime.RuntimeState(p, True, base, hires, options.DISABLE_NONE, [], False)

    for _batch in range(2):
        loads.append(0)
        # processing.py: the batch, setup_conds (hires conds, then base again), sample_hr_pass
        for data in (base_data, hires_data, base_data, hires_data):
            extra_networks.activate(p, data)

    assert loads == loads_per_batch


def test_shared_patches_follow_the_timing_of_the_current_pass(monkeypatch):
    p = SimpleNamespace(is_hr_pass=False)
    base = _config(options.PRESET_CHARACTER_ALLOW, panel=BLOCKS)  # window 0.20-0.74
    hires = _config(options.PRESET_STYLE_ALLOW, panel=BLOCKS)  # window 0.62-1.00
    state = runtime.RuntimeState(p, True, base, hires, options.DISABLE_NONE, [], False)
    strength = runtime.ScheduledStrength(1.0, "x.safetensors", state.patch_pass("base"))
    state.register(strength)
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", state)

    def factor_at(step):
        denoiser = SimpleNamespace(p=p, step=step, total_steps=101)
        runtime._denoiser_callback(SimpleNamespace(denoiser=denoiser, sampling_step=step, total_sampling_steps=101))
        return strength.factor

    assert strength.pass_name == runtime.BOTH_PASSES
    assert factor_at(47) == pytest.approx(1.0) and factor_at(90) == 0.0
    p.is_hr_pass = True
    assert factor_at(47) == 0.0 and factor_at(90) > 0.0


def _manual(template, start, end, peak, strength=1.0):
    return runtime._build_pass_config(options.MODE_PROMPT_ALL, "", "", template, "", False, None, start, end, peak, strength, options.CURVE_HOLD, 3.0)


def test_a_ban_template_keeps_blocking_under_manual_timing():
    ban = _manual(options.PRESET_STYLE_BAN, 0.62, 1.0, 0.86)  # its window, as the template loads it
    assert _factors(ban, (30, 61, 62, 90)) == [1.0, 1.0, 0.0, 0.0]

    allow = _manual(options.PRESET_STYLE_ALLOW, 0.62, 1.0, 0.86)
    assert _factors(allow, (30, 61, 62, 90)) == [0.0, 0.0, 1.0, 1.0]


def test_stage_strength_scales_auto_timed_templates():
    half = runtime._build_pass_config(options.MODE_PROMPT_ALL, "", "", options.PRESET_CHARACTER_ALLOW, "", True, None, 0.2, 0.8, 0.5, 0.5, options.CURVE_SMOOTH, 3.0)
    assert max(_factors(half, range(101))) == pytest.approx(0.5)
    ban = runtime._build_pass_config(options.MODE_PROMPT_ALL, "", "", options.PRESET_STYLE_BAN, "", True, None, 0.2, 0.8, 0.5, 0.5, options.CURVE_SMOOTH, 3.0)
    assert _factors(ban, (0, 90)) == [0.5, 0.0]

    # Moving the timing with Shift keeps the slider's multiplier
    *_, strength = ui._auto_timing_ui(options.PRESET_CHARACTER_ALLOW, 5.0, True, 0.2, 0.74, 0.47, 0.5)
    assert strength["value"] == 0.5


def test_progress_is_the_position_in_the_full_sigma_schedule():
    import torch

    sigmas = torch.linspace(1.0, 0.0, 11)
    unet = SimpleNamespace(model_options={"transformer_options": {"sampling_sigmas": sigmas}})
    p = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))
    # Hires. fix at denoise 0.5: its first step is halfway down the schedule
    params = SimpleNamespace(sigma=sigmas[5:6].repeat(2), denoiser=SimpleNamespace(p=p, step=0, total_steps=5), sampling_step=0, total_sampling_steps=5)
    assert runtime._sampling_progress(params) == 0.5

    unet.model_options = {}
    params.denoiser.step = 2
    assert runtime._sampling_progress(params) == 0.5  # by steps without a schedule


def test_scheduled_strengths_work_in_boft_arithmetic():
    import torch

    from modules_forge.packages.comfy.weight_adapter.boft import BOFTAdapter

    torch.manual_seed(0)
    adapter = BOFTAdapter(set(), (torch.randn(1, 2, 2, 2), None, 0.0, None))
    weight = torch.randn(4, 3)
    strength = runtime.ScheduledStrength(0.5, "x.safetensors", "base")

    expected = adapter.calculate_weight(weight.clone(), "k", 0.5, 1.0, None, lambda a: a)
    scheduled = adapter.calculate_weight(weight.clone(), "k", strength, 1.0, None, lambda a: a)

    assert not torch.allclose(expected, weight)
    assert torch.allclose(scheduled, expected)  # not left unpatched by a TypeError


def test_prompt_conds_are_recomputed_when_the_panels_text_encoder_weights_change(module, monkeypatch):
    from backend.args import dynamic_args

    monkeypatch.setattr(dynamic_args, "online_lora", False)
    monkeypatch.setattr(runtime, "_TEXT_ENCODER_SIGNATURE", None, raising=False)
    cache = [None, None, None]  # StableDiffusionProcessing.cached_c

    def run(panel, enabled=True):
        cache[0] = "cond key"
        p = SimpleNamespace(cached_c=cache, extra_generation_params={})
        args = [options.MODE_PROMPT_ALL, "", panel, options.PRESET_CHARACTER_ALLOW, "", True, 0.2, 0.8, 0.5, 1.0, options.CURVE_SMOOTH]
        module.Script().process(p, enabled, *args, False, *args, options.DISABLE_NONE, "")
        runtime._cleanup_runtime(p)
        return cache[0] is None

    assert run("qwen=0-27=0.5")
    assert not run("qwen=0-27=0.5")
    assert run("qwen=0-27=0.2")
    assert run("blocks=0-9=0.5")  # the text encoder is back to the prompt's weights
    assert not run("blocks=0-9=0.2") and not run("0", enabled=False)
    assert run("qwen=0-27=0.2") and run("qwen=0-27=0.2", enabled=False)


def test_process_resolves_each_panels_stage_in_finalize_timing_only(module, monkeypatch):
    from backend.args import dynamic_args

    monkeypatch.setattr(dynamic_args, "online_lora", False)
    reads = []
    template_config = templates._template_config
    monkeypatch.setattr(runtime, "_template_config", lambda name: reads.append(name) or template_config(name))
    p = SimpleNamespace(extra_generation_params={})
    base = [options.MODE_PROMPT_ALL, "", "", options.PRESET_STYLE_ALLOW, "", True, 0.2, 0.8, 0.5, 1.0, options.CURVE_SMOOTH]
    hires = [options.MODE_PROMPT_ALL, "", "", options.PRESET_COMPOSITION_ALLOW, "", True, 0.2, 0.8, 0.5, 1.0, options.CURVE_SMOOTH]

    module.Script().process(p, True, *base, True, *hires, options.DISABLE_NONE, "")
    state = p._anima_lora_stage_scheduler_state
    runtime._cleanup_runtime(p)

    assert reads == [options.PRESET_STYLE_ALLOW, options.PRESET_COMPOSITION_ALLOW]
    assert state.base.stage == options.STAGE_STYLE and state.hires.stage == options.STAGE_COMPOSITION


def test_a_manual_template_without_a_stage_keeps_its_first_applying_part(monkeypatch):
    template = {"auto": False, "parts": [{"stage": options.STAGE_STYLE, "mode": options.PART_BAN}, {"stage": options.STAGE_COMPOSITION, "mode": options.PART_ALLOW}]}
    monkeypatch.setattr(runtime, "_template_config", lambda name: dict(template))

    assert _manual("hand-written", 0.0, 0.3, 0.1).stage == options.STAGE_COMPOSITION


def test_auto_detect_schedules_anima_lokrs_too(tmp_path):
    import torch
    from safetensors.torch import save_file

    lokr = tmp_path / "subject.safetensors"
    save_file({"lycoris_blocks_3_mlp_layer1.lokr_w1": torch.zeros(1)}, str(lokr))
    sdxl = tmp_path / "sdxl.safetensors"
    save_file({"lora_unet_mid_block_attentions_0_proj_in.lora_down.weight": torch.zeros(1)}, str(sdxl))
    config = _config(options.PRESET_CHARACTER_ALLOW, mode=options.MODE_AUTO_ANIMA)

    assert config.file_matches(str(lokr), frozenset())
    assert not config.file_matches(str(sdxl), frozenset())


def test_template_dropdowns_show_the_templates_as_saved():
    builtins = list(options.BUILTIN_TEMPLATES)

    *dropdowns, status = ui._save_template_ui("mine", True, builtins[0], 0.0, 1.0, 0.5, 1.0, options.CURVE_SMOOTH)
    assert status == "已保存模板：mine" and len(dropdowns) == 3
    assert all("mine" in dropdown["choices"] for dropdown in dropdowns) and dropdowns[2]["value"] == "mine"

    *dropdowns, status = ui._rename_template_ui("mine", "renamed")
    assert all("renamed" in dropdown["choices"] and "mine" not in dropdown["choices"] for dropdown in dropdowns)
    assert dropdowns[2]["value"] == "renamed"

    *dropdowns, status = ui._delete_template_ui("renamed")
    assert status == "已删除模板：renamed"
    assert all("renamed" not in dropdown["choices"] for dropdown in dropdowns)
    assert "renamed" not in templates._load_custom_templates()


def test_template_management_leaves_the_panels_selections_alone():
    # Selecting a panel template re-applies its timing and strength to that
    # panel: only the manager's dropdown follows the managed template.
    builtin = list(options.BUILTIN_TEMPLATES)[1]

    base, hires, manager, _ = ui._save_template_ui("mine", True, builtin, 0.0, 1.0, 0.5, 1.0, options.CURVE_SMOOTH, builtin, builtin)
    assert "value" not in base and "value" not in hires and "mine" in base["choices"] and "mine" in hires["choices"]
    assert manager["value"] == "mine"
    base, hires, manager, _ = ui._save_preset_combo_template_ui("combo", [options.PRESET_STYLE_ALLOW], 3.0, "mine", builtin)
    assert "value" not in base and "value" not in hires and manager["value"] == "combo"

    # A panel on the renamed template follows it; the other keeps its own
    base, hires, manager, _ = ui._rename_template_ui("mine", "renamed", "mine", builtin)
    assert base["value"] == "renamed" and "value" not in hires and manager["value"] == "renamed"

    # A panel on the deleted template falls back; the other keeps its own
    base, hires, manager, _ = ui._delete_template_ui("renamed", builtin, "renamed")
    assert "value" not in base and hires["value"] == base["choices"][0]
    assert "renamed" not in hires["choices"]


def test_combo_templates_take_their_window_from_shift_when_applied():
    # Auto timing recomputes the window from Shift: a combo stores none
    ui._save_preset_combo_template_ui("combo", [options.PRESET_STYLE_ALLOW], 5.0)
    saved = templates._load_custom_templates()["combo"]
    assert saved["auto"] is True and not {"start", "end", "peak"} & set(saved)

    # Templates saved with a window (older versions) still apply from Shift
    templates._TEMPLATES.edit(lambda data: (data.update(old={**saved, "start": 0.9, "end": 0.95, "peak": 0.9}), ("old", "")))
    for name in ("combo", "old"):
        _, start, end, peak, strength, _ = ui._apply_template_ui(name, 5.0)
        window = templates._auto_stage_values(options.STAGE_STYLE, 5.0)[:3]
        assert (start["value"], end["value"], peak["value"]) == tuple(round(value, 4) for value in window), name
        assert strength["value"] == 1.0
        config = runtime._build_pass_config(options.MODE_PROMPT_ALL, "", "", name, "", True, None, 0.2, 0.8, 0.5, 1.0, options.CURVE_SMOOTH, 5.0)
        assert (config.start, config.end, config.peak) == window, name


def test_template_messages_follow_the_ui_language(monkeypatch):
    from modules import shared

    builtin = next(iter(options.BUILTIN_TEMPLATES))
    assert set(options.LANG["zh"]) == set(options.LANG["en"])

    monkeypatch.setattr(shared, "opts", SimpleNamespace(**{options.LANGUAGE_OPTION: "zh"}), raising=False)
    assert ui._save_template_ui("", True, builtin, 0.0, 1.0, 0.5, 1.0, options.CURVE_SMOOTH)[-1] == "模板名称不能为空。"
    assert templates._parts_from_presets(["x"])[1] == "未知基础预设：x"
    assert ui._rename_template_ui("a", "a")[-1] == "新旧名称相同，无需重命名。"

    monkeypatch.setattr(shared, "opts", SimpleNamespace(**{options.LANGUAGE_OPTION: "en"}), raising=False)
    assert ui._save_template_ui("", True, builtin, 0.0, 1.0, 0.5, 1.0, options.CURVE_SMOOTH)[-1] == "The template name cannot be empty."
    assert templates._parts_from_presets(["x"])[1] == "Unknown base preset: x"
    assert ui._delete_template_ui(builtin)[-1] == "Built-in templates cannot be deleted."
    conflict = [options.PRESET_STYLE_ALLOW, options.PRESET_STYLE_BAN]
    assert templates._parts_from_presets(conflict)[1] == f"Conflicting combo: {options.STAGE_STYLE} is selected as both Apply and Block; not saved."
    help_text = ui._template_help_ui(builtin, 3.0)["value"]
    assert "Templates are built from base presets" in help_text and "auto window" in help_text


def test_template_help_shows_each_parts_own_strength(monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared, "opts", SimpleNamespace(**{options.LANGUAGE_OPTION: "en"}), raising=False)
    parts = [
        {"stage": options.STAGE_CHARACTER, "mode": options.PART_ALLOW, "strength": 0.5},
        {"stage": options.STAGE_STYLE, "mode": options.PART_BAN, "strength": 1.0},
    ]
    templates._TEMPLATES.edit(lambda data: (data.update(halved={"auto": True, "parts": parts}), ("halved", "")))

    help_text = ui._template_help_ui("halved", 3.0)["value"]
    assert "strength 0.5" in help_text and "strength 0 in window" in help_text and "strength 1" not in help_text


def test_per_lora_templates_resolve_their_stage_and_curve_like_the_panel():
    manual = {"auto": False, "stage": options.STAGE_STYLE, "start": 0.1, "end": 0.4, "peak": 0.2, "strength": 2.0, "curve": options.CURVE_HOLD}
    unknown_curve = {"auto": True, "stage": options.STAGE_COMPOSITION, "curve": "no such curve"}
    templates._TEMPLATES.edit(lambda data: (data.update(manual=manual, unknown_curve=unknown_curve), ("manual", "")))
    base = runtime._build_pass_config(options.MODE_PROMPT_ALL, "", "", options.PRESET_CHARACTER_ALLOW, "cat=manual\ndog=unknown_curve\nfox=" + options.PRESET_STYLE_ALLOW, True, options.STAGE_CHARACTER, 0.2, 0.8, 0.5, 1.0, options.CURVE_TRIANGLE, 3.0)

    cat = base.config_for_file("cat.safetensors", {"cat"})
    assert (cat.stage, cat.curve, cat.parts, cat.stage_strength) == (options.STAGE_STYLE, options.CURVE_HOLD, [], 2.0)
    assert (cat.start, cat.end, cat.peak) == (0.1, 0.4, 0.2)

    dog = base.config_for_file("dog.safetensors", {"dog"})
    assert dog.stage == options.STAGE_COMPOSITION and dog.curve == options.CURVE_SMOOTH  # an invalid curve reads as smooth
    assert (dog.start, dog.end, dog.peak) == templates._auto_stage_values(options.STAGE_COMPOSITION, 3.0)[:3]

    fox = base.config_for_file("fox.safetensors", {"fox"})
    assert fox.stage == options.STAGE_STYLE and fox.parts == templates._template_parts(options.BUILTIN_TEMPLATES[options.PRESET_STYLE_ALLOW])
    assert base.config_for_file("other.safetensors", {"other"}) is base


def test_per_lora_templates_named_apart_by_case_stay_apart():
    late = {"auto": False, "stage": options.STAGE_STYLE, "start": 0.6, "end": 0.9, "peak": 0.7}
    lower = {"auto": False, "stage": options.STAGE_STYLE, "start": 0.1, "end": 0.3, "peak": 0.2}
    templates._TEMPLATES.edit(lambda data: (data.update(Late=late, late=lower), ("Late", "")))
    base = runtime._build_pass_config(options.MODE_PROMPT_ALL, "", "", options.PRESET_CHARACTER_ALLOW, "cat=Late\ndog=late", True, options.STAGE_CHARACTER, 0.2, 0.8, 0.5, 1.0, options.CURVE_SMOOTH, 3.0)

    cat = base.config_for_file("cat.safetensors", {"cat"})
    dog = base.config_for_file("dog.safetensors", {"dog"})

    assert (cat.template_name, cat.start, cat.end) == ("Late", 0.6, 0.9)
    assert (dog.template_name, dog.start, dog.end) == ("late", 0.1, 0.3)
    assert base.config_for_file("cat.safetensors", {"cat"}) is cat  # cached


def test_the_infotext_keeps_the_lora_weights_as_written(module):
    """The batch loads its LoRAs at the panel's multiplier; the infotext, built
    from p.prompts and all_hr_prompts after postprocess_batch_list, keeps the
    tags as written."""
    config = _config(options.PRESET_CHARACTER_ALLOW, panel="0.5")
    state = runtime.RuntimeState(None, True, config, config, options.DISABLE_NONE, [], False)
    written = ["a <lora:cat:1.2:te=0>", "b <lora:cat:0.8>"]
    p = SimpleNamespace(iteration=1, batch_size=1, enable_hr=True, all_prompts=list(written), all_hr_prompts=list(written), hr_prompts=[])
    script = module.Script()

    p.prompts = p.all_prompts[1:2]
    runtime._rewrite_batch_prompts(p, state)  # before_process_batch
    assert (p.prompts, p.all_hr_prompts[1]) == (["b <lora:cat:0.5>"], "b <lora:cat:0.5>")
    p.prompts = p.all_prompts[1:2]  # re-sliced after postprocess_batch
    script.postprocess_batch_list(p, SimpleNamespace(images=[]))

    assert p.prompts == written[1:] and p.all_hr_prompts == written
    assert p.all_prompts[1] == "b <lora:cat:0.5>"  # what a nested run (ADetailer) reads
    script.postprocess(p, SimpleNamespace())
    assert p.all_prompts == written
