import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from backend.patcher import base
from backend.patcher.lora import merge_lora_to_weight
from modules_forge.packages.comfy.weight_adapter import LoRAAdapter

EXTENSIONS = Path(__file__).parents[1] / "extensions-builtin"


def _lora(out_features=4, reshape=None):
    return LoRAAdapter(set(), (torch.ones(out_features, 2), torch.ones(2, 4), None, None, None, reshape))


def test_zero_strength_adapters_are_skipped(monkeypatch):
    strengths = []
    calculate_weight = LoRAAdapter.calculate_weight
    monkeypatch.setattr(LoRAAdapter, "calculate_weight", lambda self, *args, **kwargs: strengths.append(args[2]) or calculate_weight(self, *args, **kwargs))

    weight = merge_lora_to_weight([(0.0, _lora(), 1.0, None, None), (0.5, _lora(), 1.0, None, None)], torch.zeros(4, 4), "k")
    assert strengths == [0.5]
    assert torch.equal(weight, torch.ones(4, 4))  # 0.5 * (ones(4, 2) @ ones(2, 4))

    narrowed = merge_lora_to_weight([(0.0, _lora(), 1.0, (0, 0, 2), None)], torch.zeros(4, 4), "k")
    assert narrowed.shape == (4, 4)  # the whole weight comes back, not the offset slice

    padded = merge_lora_to_weight([(0.0, _lora(6, reshape=[6, 4]), 1.0, None, None)], torch.zeros(4, 4), "k")
    assert padded.shape == (6, 4)  # a LoRA that pads the weight still pads it at strength 0


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, EXTENSIONS / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(index, module):
    return f"diffusion_model.blocks.{index}.{module}.weight"


LAYER_FACTORS = {
    _block(3, "mlp.layer1"): 0.5,
    _block(3, "self_attn.q_proj"): 1.0,
    _block(20, "mlp.layer2"): 0.25,
    "text_encoders.qwen3_06b.model.layers.2.mlp.up_proj.weight": 0.3,
}


@pytest.mark.parametrize("stage_scheduler_outermost", [False, True])
def test_layer_weight_and_stage_scheduler_compose_in_either_order(tmp_path, monkeypatch, stage_scheduler_outermost):
    from modules import extra_networks, paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))  # stage templates

    def forge_add_patches(self, patches, strength_patch=1.0, strength_model=1.0, *, filename=None, online_mode=None):
        for key, value in patches.items():
            self.weight_wrapper_patches.setdefault(key, []).append(SimpleNamespace(patch=[(strength_patch, value, strength_model, None, None)]))
        return list(patches)

    monkeypatch.setattr(base.ModelPatcher, "add_patches", forge_add_patches)
    monkeypatch.setattr(extra_networks, "activate", extra_networks.activate)
    layer = _load("anima_lora_layer_weight_composed", "anima-lora-layer-weight/scripts/anima_lora_layer_weight.py")
    stage = _load("anima_lora_stage_scheduler_composed", "anima-lora-stage-scheduler/scripts/anima_lora_stage_scheduler.py")
    for module in (layer, stage) if stage_scheduler_outermost else (stage, layer):
        module.install_patch()

    monkeypatch.setattr(layer, "_ACTIVE_RULE", SimpleNamespace(decision_for_lora=lambda filename: SimpleNamespace(apply=True), factor_for_key=lambda key, decision, blocks: LAYER_FACTORS[key]))
    config = stage._build_pass_config(stage.MODE_PROMPT_ALL, "", "modules=mlp=0.5", stage.PRESET_CHARACTER_ALLOW, "", True, None, 0.2, 0.8, 0.5, 1.0, stage.CURVE_SMOOTH, 3.0)
    monkeypatch.setattr(stage, "_ACTIVE_STATE", stage.RuntimeState(SimpleNamespace(), True, config, config, stage.DISABLE_NONE, [], False))
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=None), weight_wrapper_patches={}, patches={})

    base.ModelPatcher.add_patches(patcher, {key: object() for key in LAYER_FACTORS}, strength_patch=0.8, filename="x.safetensors", online_mode=True)

    entries = {key: entry.patch[0][0] for key, (entry,) in patcher.weight_wrapper_patches.items()}
    assert {key: float(strength) for key, strength in entries.items()} == {
        _block(3, "mlp.layer1"): 0.5 * 0.5,  # layer factor, times the stage panel's module weight
        _block(3, "self_attn.q_proj"): 1.0,
        _block(20, "mlp.layer2"): 0.25 * 0.5,
        "text_encoders.qwen3_06b.model.layers.2.mlp.up_proj.weight": 0.3,  # the panel sets no qwen weights
    }
    assert all(isinstance(entries[key], stage.ScheduledStrength) for key in LAYER_FACTORS if ".blocks." in key)
