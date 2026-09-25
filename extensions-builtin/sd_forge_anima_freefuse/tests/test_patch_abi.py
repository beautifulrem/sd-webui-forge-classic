import sys
from pathlib import Path
from types import SimpleNamespace

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
    assert state.routed_weight_keys == {"layer.weight"}
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


class _WrapperPatcher:
    def __init__(self, dit, options=None):
        self.model = SimpleNamespace(diffusion_model=dit)
        self.model_options = dict(options or {})

    def clone(self):
        return _WrapperPatcher(self.model.diffusion_model, self.model_options)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def set_transformer_option(self, key, value):
        # The real clone() deep-copies model_options; copy the nested dict here
        options = dict(self.model_options.get("transformer_options", {}))
        options[key] = value
        self.model_options["transformer_options"] = options


def test_model_wrapper_runs_and_restores_every_forward():
    runtime, _, _ = _load()
    block = torch.nn.Module()
    block.cross_attn = torch.nn.Linear(2, 2)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([block])
    state = SimpleNamespace(
        phase="collect",
        validate_patches=lambda patcher: None,
        routed_weight_keys=set(),
        update_grid=lambda input_x, model: None,
    )
    calls = []

    def previous(apply_model, args):
        calls.append("previous")
        return apply_model(args["input"], args["timestep"], **args["c"])

    def model_function(x, t, **c):
        calls.append("model")
        assert "forward" in block.cross_attn.__dict__
        return x

    for options in ({}, {"model_function_wrapper": previous}):
        patched = runtime.apply_freefuse_patch(_WrapperPatcher(dit, options), state)
        patched.model_options["model_function_wrapper"](
            model_function,
            {"input": torch.zeros(1), "timestep": torch.ones(1), "c": {}},
        )

    assert calls == ["model", "previous", "model"]
    assert all("forward" not in module.__dict__ for module in dit.modules())

    def failing(x, t, **c):
        raise RuntimeError("model call failed")

    patched = runtime.apply_freefuse_patch(_WrapperPatcher(dit), state)
    with pytest.raises(RuntimeError, match="model call failed"):
        patched.model_options["model_function_wrapper"](
            failing,
            {"input": torch.zeros(1), "timestep": torch.ones(1), "c": {}},
        )
    assert all("forward" not in module.__dict__ for module in dit.modules())


def test_cross_attention_applies_negpip_to_values_only():
    runtime, _, _ = _load()
    from backend.nn.anima import SelfCrossAttention

    torch.manual_seed(0)
    attention = SelfCrossAttention(8, context_dim=6, n_heads=2, head_dim=4)
    x = torch.randn(1, 5, 8)
    context = torch.randn(1, 3, 6)
    mask = torch.tensor([[[1.0], [-1.0], [1.0]]])
    captured = {}

    def compute_attention(q, k, v, transformer_options=None, mask=None):
        captured.update(k=k, v=v)
        return q

    attention.compute_attention = compute_attention
    state = SimpleNamespace(collect_block=-1, bias_blocks=set())
    forward = runtime._make_cross_attention_forward(attention.forward, attention, state, 0)
    forward(x, context=context, transformer_options={"negpip_mask": mask})

    _, expected_k, _ = attention.compute_qkv(x, context)
    _, _, expected_v = attention.compute_qkv(x, context * mask)
    assert torch.allclose(captured["k"], expected_k)
    assert torch.allclose(captured["v"], expected_v)


def test_adapter_lookup_is_cached_per_patch_object():
    runtime, _, _ = _load()
    state = _state(runtime)
    patch = (1.0, torch.zeros(1), 1.0)
    runtime._tag_patch_value(patch[1], "cat.safetensors")
    calls = []
    original = runtime._selector_matches
    runtime._selector_matches = lambda *args: calls.append(args) or original(*args)
    try:
        assert state.adapter_for_patch(patch) == "A"
        assert state.adapter_for_patch(patch) == "A"
        other = (1.0, torch.zeros(1), 1.0)  # untagged: no adapter
        assert state.adapter_for_patch(other) is None
    finally:
        runtime._selector_matches = original
    assert len(calls) == 2  # one match per distinct patch


def test_attention_bias_is_reused_until_the_mask_set_changes():
    runtime, _, _ = _load()
    state = _state(runtime)
    state.token_positions = {"A": [1]}
    state.bias_scale = 1.0
    state.grid_height = state.grid_width = 2
    state.masks = {"A": torch.tensor([[1.0, 0.0, 0.0, 1.0]])}
    state.phase = "generate"
    q = torch.zeros(2, 4, 1, 1)
    options = {"cond_or_uncond": [0, 1]}

    first = state.attention_bias(q, 3, options)
    assert state.attention_bias(q, 3, options) is first
    assert torch.equal(first[0, 0, :, 1], torch.tensor([0.0, -1.0, -1.0, 0.0]))
    assert torch.equal(first[1], torch.zeros_like(first[1]))  # uncond row

    state.abort()
    state.masks = {"A": torch.tensor([[0.0, 0.0, 0.0, 0.0]])}
    state.phase = "generate"
    assert state.attention_bias(q, 3, options) is not first


def _routed_linear(runtime, ModelPatcher, state):
    patcher = _patcher(ModelPatcher, online=True)
    state.validate_patches(patcher)
    layer = patcher.model.layer
    layer.weight_function = list(patcher.weight_wrapper_patches["layer.weight"])
    layer.bias_function = []
    forward = runtime._make_linear_forward(layer.forward, layer, state, "blocks.0.mlp.layer1")
    return layer, forward


def test_linear_routing_scales_the_adapter_delta_by_its_mask():
    runtime, ModelPatcher, _ = _load()
    state = _state(runtime)
    state.routing_strength = 0.5
    layer, forward = _routed_linear(runtime, ModelPatcher, state)
    state.grid_height, state.grid_width = 1, 2
    state.masks = {"A": torch.tensor([[1.0, 0.0]])}
    state.phase = "generate"
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # two spatial tokens

    output = forward(x)

    base = torch.nn.functional.linear(x, layer.weight)
    delta = torch.nn.functional.linear(x, torch.ones(2, 2))  # the "diff" patch
    factors = torch.tensor([1.0, 0.5]).view(1, 2, 1)  # (1 - s) + mask * s
    assert torch.allclose(output, base + delta * factors)
    assert len(layer.weight_function) == 1 and layer.bias_function == []  # restored


def test_linear_routing_leaves_adapters_out_while_collecting():
    runtime, ModelPatcher, _ = _load()
    state = _state(runtime)
    layer, forward = _routed_linear(runtime, ModelPatcher, state)
    state.phase = "collect"
    weight_before = layer.weight.detach().clone()
    x = torch.randn(1, 2, 2)

    assert torch.allclose(forward(x), torch.nn.functional.linear(x, layer.weight))
    assert torch.equal(layer.weight, weight_before)  # merged into a copy only
