import copy
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


def _state(module, process, base=None, hires=None, outer_state=None):
    return module.RuntimeState(
        p=process,
        run_id="run",
        enabled=True,
        base=base or SimpleNamespace(patch_signature="same"),
        hires=hires or SimpleNamespace(patch_signature="same"),
        hr_disable_mode="",
        hr_disable_tags=[],
        original_online_lora=False,
        outer_state=outer_state,
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


def test_step_factors_are_computed_once_per_file_config(monkeypatch):
    module = _load()
    process = SimpleNamespace(is_hr_pass=False)
    shared_config, other_config = SimpleNamespace(name="a"), SimpleNamespace(name="b")
    per_file = {"a1.safetensors": shared_config, "a2.safetensors": shared_config, "b.safetensors": other_config}
    lookups = []
    base = SimpleNamespace(patch_signature="base", config_for_file=lambda filename, names: lookups.append(filename) or per_file[filename])
    state = _state(module, process, base=base, hires=SimpleNamespace(patch_signature="hires"))
    strengths = [module.ScheduledStrength(1.0, filename, "base") for filename in per_file for _ in range(2)]
    strengths.append(module.ScheduledStrength(1.0, "b.safetensors", module.BOTH_PASSES))
    hires = module.ScheduledStrength(1.0, "a1.safetensors", "hires")
    for strength in (*strengths, hires):
        state.register(strength)

    calls = []
    monkeypatch.setattr(module, "_stage_curve_factor", lambda config, step, total: calls.append(config) or 0.25)
    monkeypatch.setattr(module, "_ACTIVE_STATE", state)
    params = SimpleNamespace(denoiser=SimpleNamespace(p=process, step=3, total_steps=10), sampling_step=3, total_sampling_steps=10)

    module._denoiser_callback(params)
    module._denoiser_callback(params)

    assert len(calls) == 4  # each step: one per distinct config of the active pass
    assert sorted(lookups) == sorted(per_file)  # per-file configs are looked up once per run
    assert all(strength.factor == 0.25 for strength in strengths)
    assert hires.factor == 0.0  # built for the other pass only


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


def test_loras_reload_only_when_the_passes_load_different_patches():
    module = _load()
    base_data, hires_data = object(), object()
    sd_model = SimpleNamespace(current_lora_hash="compiled")
    process = SimpleNamespace(sd_model=sd_model, enable_hr=True, extra_network_data=base_data, hr_extra_network_data=hires_data)
    state = _state(module, process, hires=SimpleNamespace(patch_signature="other panel"))
    process._anima_lora_stage_scheduler_state = state

    def activate(data):
        sd_model.current_lora_hash = "compiled"  # what load_networks leaves behind
        module._prepare_for_extra_network_activation(process, data)
        return sd_model.current_lora_hash != "compiled"

    assert activate(base_data)  # first activation of a run: rebuild
    assert not activate(base_data)  # next batch, same pass: keep the patches
    assert activate(hires_data) and activate(base_data)  # other panel: rebuild

    state.hires = SimpleNamespace(patch_signature="same")
    assert activate(hires_data)  # loaded for base only: rebuild once
    assert not activate(base_data) and not activate(hires_data)  # shared by both passes

    module._cleanup_runtime(process)
    assert sd_model.current_lora_hash == "anima-stage-scheduler:released"


def test_cleanup_only_undoes_its_own_run(monkeypatch):
    from backend.args import dynamic_args

    module = _load()
    monkeypatch.setattr(dynamic_args, "online_lora", True)
    sd_model = SimpleNamespace(current_lora_hash="compiled")
    outer = SimpleNamespace(sd_model=sd_model)
    state = _state(module, outer)
    state.loaded_pass = module.BOTH_PASSES
    outer._anima_lora_stage_scheduler_state = state
    monkeypatch.setattr(module, "_ACTIVE_STATE", state)

    # Nested process_images (ADetailer): a copied p carries the outer reference, a new one none
    nested, fresh = copy.copy(outer), SimpleNamespace(sd_model=sd_model)
    module._cleanup_runtime(nested)
    module._cleanup_runtime(fresh)
    module.Script().process(fresh, False, *[None] * 25)  # disabled there
    assert module._ACTIVE_STATE is state and outer._anima_lora_stage_scheduler_state is state
    assert sd_model.current_lora_hash == "compiled"

    # A nested run of its own hands the schedule back when it ends
    inner = _state(module, nested, outer_state=state)
    nested._anima_lora_stage_scheduler_state = inner
    module._ACTIVE_STATE = inner
    module._cleanup_runtime(nested)
    assert module._ACTIVE_STATE is state and outer._anima_lora_stage_scheduler_state is state

    module._cleanup_runtime(outer)
    assert module._ACTIVE_STATE is None and outer._anima_lora_stage_scheduler_state is None
    assert sd_model.current_lora_hash == "anima-stage-scheduler:released"


def test_install_patch_is_idempotent_under_outer_wrappers(monkeypatch):
    from backend.patcher.base import ModelPatcher
    from modules import extra_networks

    module = _load()
    monkeypatch.setattr(ModelPatcher, "add_patches", ModelPatcher.add_patches)
    monkeypatch.setattr(extra_networks, "activate", lambda p, extra_network_data: None)
    module.install_patch()
    inner = extra_networks.activate

    def outer(p, extra_network_data):  # another extension's wrapper, installed later
        return inner(p, extra_network_data)

    outer.__wrapped__ = inner
    extra_networks.activate = outer
    add_patches = ModelPatcher.add_patches
    module.install_patch()  # once per batch
    module.install_patch()

    assert extra_networks.activate is outer
    assert ModelPatcher.add_patches is add_patches


def test_an_unreadable_template_file_is_never_overwritten(tmp_path, monkeypatch):
    from modules import paths_internal

    module = _load()
    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    user_file = module._template_file()
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text('{"mine": ', encoding="utf-8")  # e.g. an interrupted older write

    *_, message = module._save_template_ui("new", True, "", 0.0, 1.0, 0.5, 1.0, "")

    assert "未做修改" in message
    assert user_file.read_text(encoding="utf-8") == '{"mine": '
