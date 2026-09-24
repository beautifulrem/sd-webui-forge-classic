import sys
from pathlib import Path

import pytest
import torch


EXTENSION_PATH = Path(__file__).parents[1]


def _load():
    try:
        from backend.patcher.base import LowVramPatch, ModelPatcher
    except Exception as error:  # pragma: no cover - needs the WebUI backend
        pytest.skip(f"Forge backend is not importable: {error}")
    if str(EXTENSION_PATH) not in sys.path:
        sys.path.insert(0, str(EXTENSION_PATH))
    from lib_anima_freefuse import runtime

    runtime.install_patch_metadata_hook()
    return runtime, ModelPatcher, LowVramPatch


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2, bias=False)


def _patcher(ModelPatcher, *, online):
    patcher = ModelPatcher(_Model(), torch.device("cpu"), torch.device("cpu"))
    diff = torch.ones(2, 2)
    patcher.add_patches(
        {"layer.weight": ("diff", (diff,))},
        filename="/loras/cat.safetensors",
        online_mode=online,
    )
    return patcher


def _state(runtime):
    return runtime.AnimaFreeFuseState(
        adapters=[runtime.AdapterSpec("A", "cat", "cat")],
        token_positions={},
        background_positions=[],
        collect_step=0,
        collect_block=0,
        top_k_ratio=0.1,
        temperature=1.0,
        bg_scale=1.0,
        balance_iterations=1,
        feather=0,
        routing_strength=1.0,
        routing_end=1.0,
        bias_scale=0.0,
        positive_bias=0.0,
        bias_blocks=set(),
    )


def test_online_lora_wrappers_are_validated_and_routed():
    runtime, ModelPatcher, _ = _load()
    patcher = _patcher(ModelPatcher, online=True)
    state = _state(runtime)

    state.validate_patches(patcher)

    assert state.matched_files["A"] == {"/loras/cat.safetensors"}
    (function,) = patcher.weight_wrapper_patches["layer.weight"]
    key, patches = runtime._lora_function_patches(function)
    assert key == "layer.weight"
    assert [state.adapter_for_patch(patch) for patch in patches] == ["A"]


def test_offline_lora_patches_are_rejected():
    runtime, ModelPatcher, _ = _load()
    patcher = _patcher(ModelPatcher, online=False)

    with pytest.raises(ValueError, match="requires online LoRA patches"):
        _state(runtime).validate_patches(patcher)


def test_lowvram_patches_expose_their_patch_list():
    runtime, ModelPatcher, LowVramPatch = _load()
    patcher = _patcher(ModelPatcher, online=False)

    key, patches = runtime._lora_function_patches(
        LowVramPatch("layer.weight", patcher.patches)
    )

    assert key == "layer.weight"
    assert patches is patcher.patches["layer.weight"]
    assert runtime._lora_function_patches(lambda weight: weight) is None
