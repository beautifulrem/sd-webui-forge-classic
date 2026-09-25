import gc
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_stage_scheduler.py"


def _load():
    spec = importlib.util.spec_from_file_location("anima_lora_stage_scheduler_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(module, process):
    return module.RuntimeState(
        p=process,
        run_id="run",
        enabled=True,
        base=SimpleNamespace(),
        hires=SimpleNamespace(),
        hr_disable_mode="",
        hr_disable_tags=[],
        original_online_lora=False,
    )


def test_strengths_of_reloaded_patches_stop_being_tracked():
    module = _load()
    state = _state(module, SimpleNamespace())
    config = SimpleNamespace()
    kept = module.ScheduledStrength(1.0, config, "base")
    state.register(kept)
    state.register(module.ScheduledStrength(1.0, config, "base"))  # its patch is already gone
    gc.collect()

    assert state.live_wrappers() == [kept]
    assert len(state.wrappers) == 1


def test_step_factors_are_computed_once_per_pass_config(monkeypatch):
    module = _load()
    process = SimpleNamespace(is_hr_pass=False)
    state = _state(module, process)
    base_config, other_config = SimpleNamespace(name="a"), SimpleNamespace(name="b")
    strengths = [module.ScheduledStrength(1.0, base_config, "base") for _ in range(5)]
    strengths.append(module.ScheduledStrength(1.0, other_config, "base"))
    hires = module.ScheduledStrength(1.0, base_config, "hires")
    for strength in (*strengths, hires):
        state.register(strength)

    calls = []
    monkeypatch.setattr(module, "_stage_curve_factor", lambda config, step, total: calls.append(config) or 0.25)
    monkeypatch.setattr(module, "_ACTIVE_STATE", state)
    params = SimpleNamespace(denoiser=SimpleNamespace(p=process, step=3, total_steps=10), sampling_step=3, total_sampling_steps=10)

    module._denoiser_callback(params)

    assert len(calls) == 2  # one per distinct config of the active pass
    assert all(strength.factor == 0.25 for strength in strengths)
    assert hires.factor == 0.0


def test_panel_weight_keeps_the_incoming_strength_where_it_sets_nothing():
    module = _load()
    block_key = "diffusion_model.blocks.3.mlp.layer1.weight"
    qwen_key = "text_encoders.qwen3_06b.model.layers.2.mlp.up_proj.weight"
    other_key = "diffusion_model.final_layer.linear.weight"

    blocks_only = module._parse_panel_weight("blocks=3=0.5")
    assert blocks_only.strength_for_key(block_key, 0.7) == 0.5  # its own setting
    assert blocks_only.strength_for_key(qwen_key, 0.7) == 0.7  # no qwen= section
    assert blocks_only.strength_for_key(other_key, 0.7) == 0.7  # not a block key

    modules_only = module._parse_panel_weight("modules=mlp=0.5")
    assert modules_only.strength_for_key(block_key, 0.8) == 0.8 * 0.5

    assert module._parse_panel_weight("0.25").strength_for_key(qwen_key, 0.7) == 0.25
