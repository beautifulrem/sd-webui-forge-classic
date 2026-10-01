import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.argv = sys.argv[:1]
EXTENSION_PATH = Path(__file__).parents[1]


def _load(monkeypatch):
    monkeypatch.syspath_prepend(str(EXTENSION_PATH))
    # modules.infotext_utils: set up by conftest.py for the whole session
    spec = importlib.util.spec_from_file_location("anima_freefuse_under_test", EXTENSION_PATH / "scripts" / "anima_freefuse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enabled(module):
    """Script args of an enabled panel: process() reads only the switch."""
    return (True, *[None] * (len(module.SCRIPT_ARG_NAMES) - 1))


def test_online_lora_stays_on_until_freefuse_and_the_stage_scheduler_both_end(monkeypatch):
    from backend.args import dynamic_args
    from modules import anima_online_lora

    module = _load(monkeypatch)
    monkeypatch.setattr(anima_online_lora, "_leases", set())
    monkeypatch.setattr(anima_online_lora, "_user_setting", None)
    monkeypatch.setattr(dynamic_args, "online_lora", False)  # the user runs LoRAs offline
    p = SimpleNamespace(sd_model=None, sampler_name="Euler", enable_hr=False, extra_generation_params={})
    script = module.AnimaFreeFuseScript()

    anima_online_lora.acquire_online_lora(p, "anima_lora_stage_scheduler")  # its process() ran first
    script.process(p, *_enabled(module))
    assert dynamic_args.online_lora is True

    anima_online_lora.release_online_lora(p, "anima_lora_stage_scheduler")  # its postprocess() ran first
    assert dynamic_args.online_lora is True  # FreeFuse's subjects still load online
    script.postprocess(p, None)
    script.cleanup(p)
    assert dynamic_args.online_lora is False


def _run_process(monkeypatch, module, *, hook_installed, tagged):
    from backend.args import dynamic_args
    from lib_anima_freefuse import runtime
    from modules import anima_online_lora

    monkeypatch.setattr(anima_online_lora, "_leases", set())
    monkeypatch.setattr(anima_online_lora, "_user_setting", None)
    monkeypatch.setattr(dynamic_args, "online_lora", True)
    monkeypatch.setattr(module, "is_anima_engine", lambda model: True)
    monkeypatch.setattr(module, "install_patch_metadata_hook", lambda: hook_installed)
    value = torch.zeros(1)
    if tagged:
        runtime._tag_patch_value(value, "/loras/cat.safetensors")
    unet = SimpleNamespace(patches={"layer.weight": [(1.0, value, 1.0)]})
    sd_model = SimpleNamespace(current_lora_hash="loaded", forge_objects_after_applying_lora=SimpleNamespace(unet=unet))
    p = SimpleNamespace(sd_model=sd_model, sampler_name="Euler", enable_hr=False, extra_generation_params={})
    script = module.AnimaFreeFuseScript()
    script.process(p, *_enabled(module))
    script.postprocess(p, None)
    script.cleanup(p)
    return sd_model.current_lora_hash


def test_tagged_online_loras_are_not_reloaded_from_disk(monkeypatch):
    module = _load(monkeypatch)
    # The LoRA hash already carries the online lease: a matching hash means
    # the loaded online patches can be reused as they are.
    assert _run_process(monkeypatch, module, hook_installed=False, tagged=True) == "loaded"


def test_loras_reload_when_the_patches_predate_the_filename_tags(monkeypatch):
    module = _load(monkeypatch)
    assert _run_process(monkeypatch, module, hook_installed=True, tagged=True) != "loaded"
    assert _run_process(monkeypatch, module, hook_installed=False, tagged=False) != "loaded"


def test_a_hires_run_drops_the_refiner_and_samples_both_passes_with_freefuse(monkeypatch):
    # README: the resolver clears Refiner checkpoint switching (FreeFuse needs
    # one checkpoint) and the Hires. fix pass routes LoRAs too
    from backend.args import dynamic_args
    from modules import anima_online_lora
    from modules.anima_feature_conflicts import CONFLICT_METADATA_KEY

    module = _load(monkeypatch)
    monkeypatch.setattr(anima_online_lora, "_leases", set())
    monkeypatch.setattr(anima_online_lora, "_user_setting", None)
    monkeypatch.setattr(dynamic_args, "online_lora", True)
    monkeypatch.setattr(module, "install_patch_metadata_hook", lambda: False)
    p = SimpleNamespace(
        sd_model=None,
        sampler_name="Euler",
        enable_hr=True,
        hr_sampler_name="DPM++ 2M",
        refiner_checkpoint="refiner.safetensors",
        refiner_checkpoint_info=object(),
        refiner_switch_at=0.8,
        extra_generation_params={},
    )
    script = module.AnimaFreeFuseScript()
    script.process(p, *_enabled(module))
    try:
        assert (p.refiner_checkpoint, p.refiner_checkpoint_info, p.refiner_switch_at) == (None, None, None)
        assert p.extra_generation_params[CONFLICT_METADATA_KEY] == "Refiner cleared: FreeFuse requires a single checkpoint"
        assert p.sampler_name == p.hr_sampler_name == "Anima FreeFuse Euler"
    finally:
        script.postprocess(p, None)
        script.cleanup(p)
