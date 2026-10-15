import copy
import gc
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from lib_stage_scheduler import options, runtime, templates, ui

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_stage_scheduler.py"


def _load():
    spec = importlib.util.spec_from_file_location("anima_lora_stage_scheduler_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(process, base=None, hires=None, outer_state=None, activation_pass=None):
    """``activation_pass``: the pass of the run's own extra-network activation
    in progress (see _activate), during which it loads its LoRAs."""
    return runtime.RuntimeState(
        p=process,
        enabled=True,
        base=base or SimpleNamespace(patch_signature="same"),
        hires=hires or SimpleNamespace(patch_signature="same"),
        hr_disable_mode="",
        hr_disable_tags=[],
        original_online_lora=False,
        current_activation_pass=activation_pass,
        outer_state=outer_state,
    )


def test_strengths_of_reloaded_patches_stop_being_tracked():
    state = _state(SimpleNamespace())
    config = SimpleNamespace()
    kept = runtime.ScheduledStrength(1.0, config, "base")
    state.register(kept)
    state.register(runtime.ScheduledStrength(1.0, config, "base"))  # its patch is already gone
    gc.collect()

    assert state.live_wrappers() == [kept]
    assert len(state.wrappers) == 1


def test_step_factors_are_computed_once_per_file_config(monkeypatch):
    process = SimpleNamespace(is_hr_pass=False)
    shared_config, other_config = SimpleNamespace(name="a"), SimpleNamespace(name="b")
    per_file = {"a1.safetensors": shared_config, "a2.safetensors": shared_config, "b.safetensors": other_config}
    lookups = []
    base = SimpleNamespace(patch_signature="base", config_for_file=lambda filename, names: lookups.append(filename) or per_file[filename])
    state = _state(process, base=base, hires=SimpleNamespace(patch_signature="hires"))
    strengths = [runtime.ScheduledStrength(1.0, filename, "base") for filename in per_file for _ in range(2)]
    strengths.append(runtime.ScheduledStrength(1.0, "b.safetensors", runtime.BOTH_PASSES))
    hires = runtime.ScheduledStrength(1.0, "a1.safetensors", "hires")
    for strength in (*strengths, hires):
        state.register(strength)

    calls = []
    monkeypatch.setattr(runtime, "_stage_curve_factor", lambda config, progress: calls.append(config) or 0.25)
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", state)
    params = SimpleNamespace(denoiser=SimpleNamespace(p=process, step=3, total_steps=10), sampling_step=3, total_sampling_steps=10)

    runtime._denoiser_callback(params)
    runtime._denoiser_callback(params)

    assert len(calls) == 4  # each step: one per distinct config of the active pass
    assert sorted(lookups) == sorted(per_file)  # per-file configs are looked up once per run
    assert all(strength.factor == 0.25 for strength in strengths)
    assert hires.factor == 0.0  # built for the other pass only


def test_panel_weight_keeps_the_incoming_strength_where_it_sets_nothing():
    block_key = "diffusion_model.blocks.3.mlp.layer1.weight"
    qwen_key = "text_encoders.qwen3_06b.model.layers.2.mlp.up_proj.weight"
    other_key = "diffusion_model.final_layer.linear.weight"

    blocks_only = options._parse_panel_weight("blocks=3=0.5")
    assert blocks_only.strength_for_key(block_key, 0.7) == 0.5  # its own setting
    assert blocks_only.strength_for_key(qwen_key, 0.7) == 0.7  # no qwen= section
    assert blocks_only.strength_for_key(other_key, 0.7) == 0.7  # not a block key

    modules_only = options._parse_panel_weight("modules=mlp=0.5")
    assert modules_only.strength_for_key(block_key, 0.8) == 0.8 * 0.5

    assert options._parse_panel_weight("0.25").strength_for_key(qwen_key, 0.7) == 0.25


def test_loras_reload_only_when_the_passes_load_different_patches():
    base_data, hires_data = object(), object()
    sd_model = SimpleNamespace(current_lora_hash="compiled")
    process = SimpleNamespace(sd_model=sd_model, enable_hr=True, extra_network_data=base_data, hr_extra_network_data=hires_data)
    state = _state(process, hires=SimpleNamespace(patch_signature="other panel"))
    process._anima_lora_stage_scheduler_state = state

    def activate(data):
        reloaded = []

        def load_networks(p, extra_network_data):  # an unchanged LoRA set: rebuilt only after a hash reset
            reloaded.append(sd_model.current_lora_hash != "compiled")
            sd_model.current_lora_hash = "compiled"

        runtime._activate(load_networks, process, data)
        return reloaded[0]

    assert activate(base_data)  # first activation of a run: rebuild
    assert not activate(base_data)  # next batch, same pass: keep the patches
    assert activate(hires_data) and activate(base_data)  # other panel: rebuild

    state.hires = SimpleNamespace(patch_signature="same")
    assert activate(hires_data)  # loaded for base only: rebuild once
    assert not activate(base_data) and not activate(hires_data)  # shared by both passes

    runtime._cleanup_runtime(process)
    assert sd_model.current_lora_hash.startswith("anima-stage-scheduler:released")


def test_cleanup_only_undoes_its_own_run(monkeypatch):
    from backend.args import dynamic_args

    module = _load()
    monkeypatch.setattr(dynamic_args, "online_lora", True)
    sd_model = SimpleNamespace(current_lora_hash="compiled")
    outer = SimpleNamespace(sd_model=sd_model)
    state = _state(outer)
    state.loaded_pass, state.loaded_hash = runtime.BOTH_PASSES, "compiled"
    outer._anima_lora_stage_scheduler_state = state
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", state)

    # Nested process_images (ADetailer): a copied p carries the outer reference, a new one none
    nested, fresh = copy.copy(outer), SimpleNamespace(sd_model=sd_model)
    runtime._cleanup_runtime(nested)
    runtime._cleanup_runtime(fresh)
    module.Script().process(fresh, False, *[None] * 25)  # disabled there
    assert runtime._ACTIVE_STATE is state and outer._anima_lora_stage_scheduler_state is state
    assert sd_model.current_lora_hash == "compiled"

    # A nested run of its own hands the schedule back when it ends
    inner = _state(nested, outer_state=state)
    nested._anima_lora_stage_scheduler_state = inner
    runtime._ACTIVE_STATE = inner
    runtime._cleanup_runtime(nested)
    assert runtime._ACTIVE_STATE is state and outer._anima_lora_stage_scheduler_state is state

    runtime._cleanup_runtime(outer)
    assert runtime._ACTIVE_STATE is None and outer._anima_lora_stage_scheduler_state is None
    assert sd_model.current_lora_hash.startswith("anima-stage-scheduler:released")


def test_install_patch_is_idempotent_under_outer_wrappers(monkeypatch):
    from backend.patcher.base import ModelPatcher
    from modules import extra_networks

    monkeypatch.setattr(ModelPatcher, "add_patches", ModelPatcher.add_patches)
    monkeypatch.setattr(extra_networks, "activate", lambda p, extra_network_data: None)
    runtime.install_patch()
    inner = extra_networks.activate

    def outer(p, extra_network_data):  # another extension's wrapper, installed later
        return inner(p, extra_network_data)

    outer.__wrapped__ = inner
    extra_networks.activate = outer
    add_patches = ModelPatcher.add_patches
    runtime.install_patch()  # once per batch
    runtime.install_patch()

    assert extra_networks.activate is outer
    assert ModelPatcher.add_patches is add_patches


def test_an_unreadable_template_file_is_never_overwritten(tmp_path, monkeypatch):
    from modules import paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    user_file = templates._TEMPLATES.path()
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text('{"mine": ', encoding="utf-8")  # e.g. an interrupted older write

    edit = ui.save_template("new", True, "", None)

    assert not edit.ok and edit.status["text"] == options._t("templates_unreadable") and edit.status["params"]["error"]
    assert user_file.read_text(encoding="utf-8") == '{"mine": '


def test_a_reloaded_script_keeps_driving_the_installed_wrappers(monkeypatch):
    from backend.patcher.base import ModelPatcher
    from modules import extra_networks

    monkeypatch.setattr(ModelPatcher, "add_patches", ModelPatcher.add_patches)
    monkeypatch.setattr(extra_networks, "activate", lambda p, extra_network_data: None)
    _load()
    runtime.install_patch()
    _load()  # "Reload UI" executes the script again; its batches install again
    runtime.install_patch()

    key = "diffusion_model.blocks.0.mlp.layer1.weight"
    panel = SimpleNamespace(has_block=False, strength_for_key=lambda *args: 0.25)
    config = SimpleNamespace(patch_signature="same", file_matches=lambda filename, names: True, panel_weight=panel, panel_weight_nonzero=True)
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", _state(SimpleNamespace(), base=config, activation_pass="base"))
    patcher = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {key: None}), patches={})

    ModelPatcher.add_patches(patcher, {key: object()}, filename="x.safetensors", online_mode=False)

    assert patcher.patches[key][0][0] == 0.25  # the panel of the run the live module holds


def test_a_reload_ui_registers_each_callback_again_exactly_once(monkeypatch):
    # Reload UI clears the callbacks and re-runs the script, but the lib stays
    # imported: a registration (or its once-flag) living in the lib would be
    # lost, one living in the script must not be doubled
    from modules import script_callbacks

    for key, callbacks in script_callbacks.callback_map.items():
        monkeypatch.setitem(script_callbacks.callback_map, key, list(callbacks))
    monkeypatch.setattr(script_callbacks, "ordered_callbacks_map", {})

    def registered(key):
        return [entry.callback for entry in script_callbacks.callback_map[key] if "/anima_lora_stage_scheduler" in entry.name]  # "<extension>/<file>/<category>/<name>[-n]"

    for _ in range(2):  # startup, then Reload UI
        script_callbacks.clear_callbacks()
        script = _load()
        script.Script(), script.Script()  # txt2img and img2img each build one
        assert registered("callbacks_cfg_denoiser") == [runtime._denoiser_callback]
        # The paste reads the recorded codes itself (lib_stage_scheduler.params): no callback
        assert registered("callbacks_infotext_pasted") == []


def test_only_scheduled_loras_are_patched_online(monkeypatch):
    from backend.patcher.base import ModelPatcher
    from modules import extra_networks

    calls = []

    def add_patches(patcher, patches, strength_patch=1.0, strength_model=1.0, *, filename=None, online_mode=None):
        calls.append((filename, online_mode))
        return []

    monkeypatch.setattr(ModelPatcher, "add_patches", add_patches)
    monkeypatch.setattr(extra_networks, "activate", lambda p, extra_network_data: None)
    runtime.install_patch()
    config = SimpleNamespace(patch_signature="same", file_matches=lambda filename, names: filename == "scheduled.safetensors", panel_weight=SimpleNamespace(has_block=False))
    state = _state(SimpleNamespace(), base=config, activation_pass="base")  # the user runs LoRAs offline
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", state)
    patcher = SimpleNamespace(patches={}, weight_wrapper_patches={})
    for filename in ("scheduled.safetensors", "other.safetensors"):
        ModelPatcher.add_patches(patcher, {}, filename=filename, online_mode=True)  # the run switched online on
    assert calls == [("scheduled.safetensors", True), ("other.safetensors", False)]

    state.original_online_lora = True  # the user's own online setting stays
    ModelPatcher.add_patches(patcher, {}, filename="other.safetensors", online_mode=True)
    assert calls[-1] == ("other.safetensors", True)


def test_a_next_run_with_the_same_schedule_keeps_the_loaded_loras():
    sd_model = SimpleNamespace(current_lora_hash="compiled")
    data = object()
    strength = runtime.ScheduledStrength(1.0, "x.safetensors", runtime.BOTH_PASSES)

    def run(signature="panel", enabled=True):
        p = SimpleNamespace(sd_model=sd_model, enable_hr=False, extra_network_data=data)
        state, reloads = None, []

        def load_networks(p, extra_network_data):  # rebuilds the patches (and their strengths) after a hash reset
            if sd_model.current_lora_hash != "compiled":
                reloads.append(True)
                sd_model.current_lora_hash = "compiled"
                if state is not None:
                    state.register(strength)

        if enabled:
            state = _state(p, base=SimpleNamespace(patch_signature=signature), hires=SimpleNamespace(patch_signature=signature))
            p._anima_lora_stage_scheduler_state = state
            runtime._activate(load_networks, p, data)
        else:
            load_networks(p, data)
        reloaded = bool(reloads)
        scheduled = state is not None and any(wrapper is strength for wrapper in state.live_wrappers())
        runtime._cleanup_runtime(p)
        return reloaded, scheduled

    assert run() == (True, True)
    assert run() == (False, True)  # same schedule: its patches, and their strengths, are kept
    assert run("other panel") == (True, True)
    assert run(enabled=False) == (True, False)  # never the scheduled patches without the scheduler
    assert run() == (True, True)


def test_a_next_run_never_takes_a_nested_runs_patches_for_its_own():
    """ADetailer's detailing run, with a LoRA set of its own, reloads its LoRAs
    at their plain strength in the outer run's postprocess_image: a next run
    with the same panels and that LoRA set must rebuild its scheduled patches."""
    sd_model = SimpleNamespace(current_lora_hash=str([]))
    loads = []

    def load_networks(p, extra_network_data):  # extra_network_data: the prompt's LoRA set
        if sd_model.current_lora_hash != str(extra_network_data):
            loads.append(p)
            sd_model.current_lora_hash = str(extra_network_data)

    def run(loras, nested_loras=None):
        p = SimpleNamespace(sd_model=sd_model, enable_hr=False, extra_network_data=loras)
        p._anima_lora_stage_scheduler_state = _state(p)
        runtime._activate(load_networks, p, loras)
        if nested_loras is not None:  # the scheduler not among the detailing run's scripts
            nested = SimpleNamespace(sd_model=sd_model, enable_hr=False, extra_network_data=nested_loras)
            runtime._activate(load_networks, nested, nested_loras)
        runtime._cleanup_runtime(p)
        return p

    run(["a.safetensors"], nested_loras=["b.safetensors"])
    second = run(["b.safetensors"])
    assert loads[-1] is second  # not the detailing run's unscheduled patches

    third = run(["b.safetensors"])
    assert loads[-1] is second and third is not second  # its own scheduled patches are kept


def test_loras_the_scheduler_leaves_stay_online_while_freefuse_needs_them(monkeypatch):
    from backend.patcher.base import ModelPatcher
    from modules import anima_online_lora, extra_networks

    calls = []

    def add_patches(patcher, patches, strength_patch=1.0, strength_model=1.0, *, filename=None, online_mode=None):
        calls.append((filename, online_mode))
        return []

    monkeypatch.setattr(ModelPatcher, "add_patches", add_patches)
    monkeypatch.setattr(extra_networks, "activate", lambda p, extra_network_data: None)
    monkeypatch.setattr(anima_online_lora, "_leases", set())
    monkeypatch.setattr(anima_online_lora, "_user_setting", None)
    runtime.install_patch()
    # The layer-weight wrapper, installed after the scheduler, weights every file
    spec = importlib.util.spec_from_file_location("anima_lora_layer_weight_under_test", Path(__file__).parents[2] / "anima-lora-layer-weight" / "scripts" / "anima_lora_layer_weight.py")
    layer_weight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layer_weight)
    layer_weight.install_patch()
    decision = layer_weight.LoraDecision(True, None, {}, None)
    monkeypatch.setattr(layer_weight, "_ACTIVE_RULE", SimpleNamespace(decision_for_lora=lambda filename: decision, factor_for_key=lambda *args: 0.5))

    # Target list only: FreeFuse's subjects are a listed LoRA, an off-list one and a LoKr
    config = SimpleNamespace(patch_signature="same", file_matches=lambda filename, names: filename == "listed.safetensors", panel_weight=SimpleNamespace(has_block=False))
    process = SimpleNamespace()
    state = _state(process, base=config, activation_pass="base")  # the user runs LoRAs offline
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", state)
    anima_online_lora.acquire_online_lora(process, runtime.ONLINE_LORA_OWNER)
    anima_online_lora.acquire_online_lora(process, "anima_freefuse")
    patcher = SimpleNamespace(patches={}, weight_wrapper_patches={}, model=SimpleNamespace())
    subjects = ("listed.safetensors", "offlist.safetensors", "subject_lokr.safetensors")
    for filename in subjects:
        ModelPatcher.add_patches(patcher, {"diffusion_model.blocks.0.mlp.weight": object()}, filename=filename, online_mode=True)
    assert calls == [(filename, True) for filename in subjects]  # FreeFuse's validate_patches needs them online

    anima_online_lora.release_online_lora(process, "anima_freefuse")
    ModelPatcher.add_patches(patcher, {"diffusion_model.blocks.0.mlp.weight": object()}, filename="offlist.safetensors", online_mode=True)
    assert calls[-1] == ("offlist.safetensors", False)  # the scheduler's own lease alone: merged weights
    anima_online_lora.release_online_lora(process, runtime.ONLINE_LORA_OWNER)


def test_a_nested_run_samples_the_loaded_patches_at_their_plain_strength(monkeypatch):
    outer = SimpleNamespace(is_hr_pass=False)
    state = _state(outer, base=SimpleNamespace(patch_signature="same"))
    strength = runtime.ScheduledStrength(0.8, "x.safetensors", runtime.BOTH_PASSES)
    state.register(strength)
    strength.factor = 0.0  # the outer run's last step, outside the window
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", state)

    # ADetailer's detailing run, in the outer run's postprocess_image
    nested = SimpleNamespace(is_hr_pass=False)
    runtime._denoiser_callback(SimpleNamespace(denoiser=SimpleNamespace(p=nested, step=0, total_steps=10), sampling_step=0, total_sampling_steps=10))

    assert float(strength) == 0.8


def test_a_nested_run_loads_its_loras_at_their_own_strength(monkeypatch):
    from backend.patcher.base import ModelPatcher
    from modules import extra_networks

    key = "diffusion_model.blocks.0.mlp.layer1.weight"
    patcher = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {key: None}), patches={})

    def load_networks(p, extra_network_data):  # every run's prompt: <lora:x:0.6>
        ModelPatcher.add_patches(patcher, {key: object()}, 0.6, filename="x.safetensors", online_mode=False)

    monkeypatch.setattr(ModelPatcher, "add_patches", ModelPatcher.add_patches)
    monkeypatch.setattr(extra_networks, "activate", load_networks)
    runtime.install_patch()
    panel = SimpleNamespace(has_block=False, strength_for_key=lambda *args: 0.25)
    config = SimpleNamespace(patch_signature="same", file_matches=lambda filename, names: True, panel_weight=panel, panel_weight_nonzero=True)
    outer = SimpleNamespace(sd_model=SimpleNamespace(current_lora_hash=None), enable_hr=False, extra_network_data=object())
    state = _state(outer, base=config)
    outer._anima_lora_stage_scheduler_state = state
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", state)

    extra_networks.activate(outer, outer.extra_network_data)
    # ADetailer's detailing runs, the scheduler not among their scripts: a new
    # p, and a copied one still carrying the outer run's state
    fresh = SimpleNamespace(sd_model=outer.sd_model, enable_hr=False, extra_network_data=object())
    copied = copy.copy(outer)
    copied.extra_network_data = object()
    for nested in (fresh, copied):
        extra_networks.activate(nested, nested.extra_network_data)

    assert [patch[0] for patch in patcher.patches[key]] == [0.25, 0.6, 0.6]


def test_loras_scheduled_by_lora_control_are_left_to_it(monkeypatch):
    from backend.patcher.base import ModelPatcher
    from modules import extra_networks, scripts

    calls = []

    def add_patches(patcher, patches, strength_patch=1.0, strength_model=1.0, *, filename=None, online_mode=None):
        calls.append((filename, online_mode))
        return []

    monkeypatch.setattr(ModelPatcher, "add_patches", add_patches)
    monkeypatch.setattr(extra_networks, "activate", lambda p, extra_network_data: None)
    # LoRA Control fills its class-level mapping per run
    lora_control = type("LoRAControl", (), {"mapping": {"controlled.safetensors": [(0.0, 0.0), (1.0, 1.0)]}})
    control_path = "/webui/extensions-builtin/sd_forge_lora_schedule/scripts/control.py"
    monkeypatch.setattr(scripts, "scripts_data", [scripts.ScriptClassData(lora_control, control_path, "/webui", None)])
    runtime.install_patch()
    config = SimpleNamespace(patch_signature="same", file_matches=lambda filename, names: True, panel_weight=SimpleNamespace(has_block=False))
    process = SimpleNamespace(extra_generation_params={})
    state = _state(process, base=config, activation_pass="base")
    monkeypatch.setattr(runtime, "_ACTIVE_STATE", state)
    patcher = SimpleNamespace(patches={}, weight_wrapper_patches={})
    for filename in ("scheduled.safetensors", "controlled.safetensors"):
        ModelPatcher.add_patches(patcher, {}, filename=filename, online_mode=True)
    assert calls == [("scheduled.safetensors", True), ("controlled.safetensors", False)]
    assert process.extra_generation_params["Anima stage scheduler skipped"] == "controlled"
