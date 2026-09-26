import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_layer_weight.py"


def _load():
    spec = importlib.util.spec_from_file_location("anima_lora_layer_weight_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patches_are_added_once_per_distinct_strength(monkeypatch):
    from backend.patcher import base

    calls = []

    def recording_add_patches(self, patches, strength_patch=1.0, strength_model=1.0, *, filename=None, online_mode=None):
        calls.append((dict(patches), strength_patch))
        return list(patches)

    monkeypatch.setattr(base.ModelPatcher, "add_patches", recording_add_patches)
    module = _load()
    module.install_patch()
    strengths = {"a.weight": 0.5, "b.weight": 1.0, "c.weight": 0.5}
    module._ACTIVE_RULE = SimpleNamespace(
        decision_for_lora=lambda filename: SimpleNamespace(apply=True),
        factor_for_key=lambda key, decision, blocks: strengths[key],
    )
    try:
        patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=None))
        loaded = base.ModelPatcher.add_patches(patcher, {key: object() for key in strengths}, filename="x.safetensors")
    finally:
        module._ACTIVE_RULE = None

    assert sorted(loaded) == sorted(strengths)
    assert len(calls) == 2
    for patches, strength in calls:
        assert all(strengths[key] == strength for key in patches)


def test_layer_factors_scale_the_strength_the_tag_loads_at(monkeypatch):
    from backend.patcher import base

    calls = []

    def recording_add_patches(self, patches, strength_patch=1.0, strength_model=1.0, *, filename=None, online_mode=None):
        calls.append(strength_patch)
        return list(patches)

    monkeypatch.setattr(base.ModelPatcher, "add_patches", recording_add_patches)
    module = _load()
    module.install_patch()
    # The rule lives until postprocess: a nested run (ADetailer) loads its
    # own, unrewritten <lora:x:0.6> under it
    monkeypatch.setattr(
        module,
        "_ACTIVE_RULE",
        SimpleNamespace(decision_for_lora=lambda filename: SimpleNamespace(apply=True), factor_for_key=lambda key, decision, blocks: 0.5),
    )

    base.ModelPatcher.add_patches(SimpleNamespace(model=None), {"a.weight": object()}, 0.6, filename="x.safetensors")
    base.ModelPatcher.add_patches(SimpleNamespace(model=None), {"a.weight": object()}, 1.0, filename="x.safetensors")  # the run's own tag

    assert calls == [0.6 * 0.5, 0.5]


def test_a_reloaded_module_drives_the_wrapper_an_earlier_copy_installed(monkeypatch):
    from backend.patcher import base

    calls = []

    def recording_add_patches(self, patches, strength_patch=1.0, strength_model=1.0, *, filename=None, online_mode=None):
        calls.append(strength_patch)
        return list(patches)

    monkeypatch.setattr(base.ModelPatcher, "add_patches", recording_add_patches)
    _load().install_patch()
    module = _load()  # "Reload UI" executes the script again; its batches install again
    module.install_patch()
    monkeypatch.setattr(
        module,
        "_ACTIVE_RULE",
        SimpleNamespace(decision_for_lora=lambda filename: SimpleNamespace(apply=True), factor_for_key=lambda key, decision, blocks: 0.5),
    )

    base.ModelPatcher.add_patches(SimpleNamespace(model=None), {"a.weight": object()}, filename="x.safetensors")

    assert calls == [0.5]  # the rule of the live module's batch
