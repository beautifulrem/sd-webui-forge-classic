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
