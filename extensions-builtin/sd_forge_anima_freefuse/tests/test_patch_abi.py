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


def _lora_adapter(*, rank=3, out_features=6, in_features=8, alpha=5.0, mid=None, dora_scale=None):
    from modules_forge.packages.comfy.weight_adapter import LoRAAdapter

    up = torch.randn(out_features, rank)
    down = torch.randn(rank, in_features)
    return LoRAAdapter(set(), (up, down, alpha, mid, dora_scale, None))


def _lora_forward(runtime, ModelPatcher, adapter, *, dtype=torch.float32, strength=0.7):
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(8, 6, bias=True)
    model.to(dtype)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patcher.add_patches(
        {"layer.weight": adapter},
        strength_patch=strength,
        filename="/loras/cat.safetensors",
        online_mode=True,
    )
    state = _state(runtime)
    state.routing_strength = 0.5
    state.validate_patches(patcher)
    layer = patcher.model.layer
    layer.weight_function = list(patcher.weight_wrapper_patches["layer.weight"])
    layer.bias_function = []
    state.grid_height, state.grid_width = 1, 3
    state.masks = {"A": torch.tensor([[1.0, 0.0, 0.5]])}
    state.phase = "generate"
    return layer, runtime._make_linear_forward(layer.forward, layer, state, "blocks.0.mlp.layer1")


def _dense_only(runtime, forward, x):
    original = runtime._plain_lora_factors
    runtime._plain_lora_factors = lambda patch, weight: None
    try:
        return forward(x)
    finally:
        runtime._plain_lora_factors = original


@pytest.mark.parametrize("dtype, tolerance", [(torch.float32, 1e-5), (torch.bfloat16, 5e-2)])
def test_plain_lora_runs_low_rank_and_matches_the_dense_merge(dtype, tolerance):
    runtime, ModelPatcher, _ = _load()
    torch.manual_seed(0)
    adapter = _lora_adapter()
    layer, forward = _lora_forward(runtime, ModelPatcher, adapter, dtype=dtype)
    x = torch.randn(2, 3, 8).to(dtype)
    merges = []
    original_merge = runtime.merge_lora_to_weight
    runtime.merge_lora_to_weight = lambda *args, **kwargs: merges.append(args) or original_merge(*args, **kwargs)
    try:
        low_rank = forward(x)
    finally:
        runtime.merge_lora_to_weight = original_merge
    dense = _dense_only(runtime, forward, x)

    assert merges == []  # no dense delta was built
    assert low_rank.dtype == dense.dtype == dtype
    assert torch.allclose(low_rank.float(), dense.float(), rtol=tolerance, atol=tolerance)
    # Reference: base + 0.7 * (5 / 3) * up @ down, scaled by (1 - s) + mask * s
    up, down = adapter.weights[0], adapter.weights[1]
    base = torch.nn.functional.linear(x.float(), layer.weight.float(), layer.bias.float())
    delta = torch.nn.functional.linear(x.float(), 0.7 * (5.0 / 3.0) * up @ down)
    factors = torch.tensor([1.0, 0.5, 0.75]).view(1, 3, 1)
    assert torch.allclose(low_rank.float(), base + delta * factors, rtol=tolerance, atol=tolerance)


def test_non_plain_adapters_keep_the_dense_merge():
    runtime, ModelPatcher, _ = _load()
    from modules_forge.packages.comfy.weight_adapter import LoKrAdapter

    torch.manual_seed(0)
    weight = torch.randn(6, 8)
    plain = _lora_adapter()
    dora = _lora_adapter(dora_scale=torch.ones(6, 1))
    locon = _lora_adapter(mid=torch.randn(3, 3, 1, 1))
    lokr = LoKrAdapter(set(), (None,) * 9)

    assert runtime._plain_lora_factors([0.7, plain, 1.0, None, None], weight) is not None
    assert runtime._plain_lora_factors([0.7, dora, 1.0, None, None], weight) is None
    assert runtime._plain_lora_factors([0.7, locon, 1.0, None, None], weight) is None
    assert runtime._plain_lora_factors([0.7, lokr, 1.0, None, None], weight) is None
    assert runtime._plain_lora_factors([0.7, plain, 1.0, (0, 0, 3), None], weight) is None
    assert runtime._plain_lora_factors([0.7, plain, 0.5, None, None], weight) is None
    assert runtime._plain_lora_factors([0.7, plain, 1.0, None, lambda a: a], weight) is None
    assert runtime._plain_lora_factors([0.7, plain, 1.0, None, None], torch.randn(8, 6)) is None

    # A DoRA patch still goes through merge_lora_to_weight in the forward.
    layer, forward = _lora_forward(runtime, ModelPatcher, dora)
    merges = []
    original_merge = runtime.merge_lora_to_weight
    runtime.merge_lora_to_weight = lambda *args, **kwargs: merges.append(args) or original_merge(*args, **kwargs)
    try:
        forward(torch.randn(1, 3, 8))
    finally:
        runtime.merge_lora_to_weight = original_merge
    assert len(merges) == 1


def _bias_forward(runtime, ModelPatcher, *, weight_patch):
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(2, 2, bias=True)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patches = {"layer.bias": ("diff", (torch.tensor([0.5, -1.0]),))}  # the LoRA's diff_b
    if weight_patch:
        patches["layer.weight"] = ("diff", (torch.ones(2, 2),))
    patcher.add_patches(patches, filename="/loras/cat.safetensors", online_mode=True)
    # Another LoRA's bias patch is not routed
    patcher.add_patches({"layer.bias": ("diff", (torch.tensor([0.25, 0.25]),))}, filename="/loras/dog.safetensors", online_mode=True)
    state = _state(runtime)
    state.routing_strength = 0.5
    state.validate_patches(patcher)
    layer = patcher.model.layer
    layer.weight_function = list(patcher.weight_wrapper_patches.get("layer.weight", []))
    layer.bias_function = list(patcher.weight_wrapper_patches["layer.bias"])
    state.grid_height, state.grid_width = 1, 2
    state.masks = {"A": torch.tensor([[1.0, 0.0]])}
    return state, layer, runtime._make_linear_forward(layer.forward, layer, state, "blocks.0.mlp.layer1")


@pytest.mark.parametrize("weight_patch", [True, False])
def test_linear_routing_masks_the_adapter_bias_patch(weight_patch):
    runtime, ModelPatcher, _ = _load()
    state, layer, forward = _bias_forward(runtime, ModelPatcher, weight_patch=weight_patch)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # two spatial tokens
    base = torch.nn.functional.linear(x, layer.weight, layer.bias + torch.tensor([0.25, 0.25]))
    delta = torch.tensor([0.5, -1.0]).expand(1, 2, 2)
    if weight_patch:
        delta = delta + torch.nn.functional.linear(x, torch.ones(2, 2))

    state.phase = "collect"  # the subject LoRA is left out, bias included
    assert torch.allclose(forward(x), base)

    state.phase = "generate"
    factors = torch.tensor([1.0, 0.5]).view(1, 2, 1)  # (1 - s) + mask * s
    assert torch.allclose(forward(x), base + delta * factors)
    assert len(layer.bias_function) == 2  # restored


def test_layers_with_only_an_adapter_bias_patch_are_routed():
    runtime, _, _ = _load()
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList()
    dit.proj = torch.nn.Linear(2, 2)
    state = SimpleNamespace(
        validate_patches=lambda patcher: None,
        routed_weight_keys={"diffusion_model.proj.bias"},
        update_grid=lambda input_x, model: None,
    )
    routed = []

    def model_function(x, t, **c):
        routed.append("forward" in dit.proj.__dict__)
        return x

    patched = runtime.apply_freefuse_patch(_WrapperPatcher(dit), state)
    patched.model_options["model_function_wrapper"](model_function, {"input": torch.zeros(1), "timestep": torch.ones(1), "c": {}})
    assert routed == [True]


def _quantized_linear(kind):
    """A Forge Linear of the GGUF or mixed-precision ops (the Anima scaled FP8
    and INT8 ConvRot modes), and the dense weight it computes with."""
    import backend.operations as operations
    from backend.nn.anima_quant import quantize_int8_convrot_state_dict, scale_fp8_state_dict

    torch.manual_seed(0)
    weight, bias = torch.randn(4, 256), torch.randn(4)
    if kind == "gguf":
        from backend.operations_gguf import ParameterGGUF, dequantize_tensor
        from modules_forge.packages import gguf

        packed = ParameterGGUF(
            torch.from_numpy(gguf.quantize(weight.numpy(), gguf.GGMLQuantizationType.Q8_0)),
            tensor_type=gguf.GGMLQuantizationType.Q8_0,
            tensor_shape=weight.shape,
        )
        packed.gguf_cls.bake(packed)
        f32_bias = ParameterGGUF(bias, tensor_type=gguf.GGMLQuantizationType.F32, tensor_shape=bias.shape)
        operations.current_device, operations.current_dtype = torch.device("cpu"), torch.float32
        layer = operations.ForgeOperationsGGUF.Linear()
        layer._load_from_state_dict({"weight": packed, "bias": f32_bias}, "", {}, True, [], [], [])
        return layer, dequantize_tensor(packed).float()
    from backend.operations_mixed_precision import mixed_precision_ops

    state_dict = {"blocks.1.mlp.layer1.weight": weight, "blocks.1.mlp.layer1.bias": bias}
    if kind == "fp8":
        assert scale_fp8_state_dict(state_dict, fp8_matmul=False) == 1
    else:
        assert quantize_int8_convrot_state_dict(state_dict) == 1
    layer = mixed_precision_ops({}, compute_dtype=torch.float32).Linear(256, 4)
    layer.load_state_dict({key.removeprefix("blocks.1.mlp.layer1."): value for key, value in state_dict.items()}, strict=False)
    assert not isinstance(layer, torch.nn.Linear)
    return layer, layer.weight.dequantize().float()


@pytest.mark.parametrize("kind", ["fp8", "int8", "gguf"])
def test_quantized_linears_are_routed(kind):
    runtime, ModelPatcher, _ = _load()
    layer, dense = _quantized_linear(kind)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList()
    dit.proj = layer
    state = _state(runtime)
    delta_weight = torch.randn(4, 256) * 0.1
    state.validate_patches = lambda patcher: state.routed_weight_keys.add("diffusion_model.proj.weight")
    state.update_grid = lambda input_x, model: None
    installed = []

    def model_function(x, t, **c):
        installed.append("forward" in layer.__dict__)
        return x

    patched = runtime.apply_freefuse_patch(_WrapperPatcher(dit), state)
    patched.model_options["model_function_wrapper"](model_function, {"input": torch.zeros(1), "timestep": torch.ones(1), "c": {}})
    assert installed == [True]

    # The routed forward dequantizes the layer and masks the adapter's delta
    model = torch.nn.Module()
    model.layer = layer
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patcher.add_patches({"layer.weight": ("diff", (delta_weight,))}, filename="/loras/cat.safetensors", online_mode=True)
    state = _state(runtime)
    state.routing_strength = 0.5
    state.validate_patches(patcher)
    layer.weight_function = list(patcher.weight_wrapper_patches["layer.weight"])
    layer.bias_function = []
    state.grid_height, state.grid_width = 1, 2
    state.masks = {"A": torch.tensor([[1.0, 0.0]])}
    state.phase = "generate"
    forward = runtime._make_linear_forward(layer.forward, layer, state, "blocks.0.mlp.layer1")
    x = torch.randn(1, 2, 256)

    output = forward(x)

    base = torch.nn.functional.linear(x, dense, layer.bias.float())
    delta = torch.nn.functional.linear(x, delta_weight)
    factors = torch.tensor([1.0, 0.5]).view(1, 2, 1)
    assert output.shape == (1, 2, 4)
    assert torch.allclose(output, base + delta * factors, atol=1e-3)


def test_unroutable_adapter_layers_abort_instead_of_applying_globally():
    runtime, _, _ = _load()
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList()
    dit.embedding = torch.nn.Embedding(4, 2)
    dit.norm = torch.nn.LayerNorm(2)
    state = SimpleNamespace(validate_patches=lambda patcher: None, update_grid=lambda input_x, model: None)

    # A norm's scale is left global, as FreeFuse routes Linear layers only
    state.routed_weight_keys = {"diffusion_model.norm.weight", "diffusion_model.norm.bias"}
    runtime.apply_freefuse_patch(_WrapperPatcher(dit), state)

    for key in ("diffusion_model.embedding.weight", "diffusion_model.missing.weight"):
        state.routed_weight_keys = {key, "diffusion_model.norm.weight"}
        with pytest.raises(ValueError, match="cannot route") as error:
            runtime.apply_freefuse_patch(_WrapperPatcher(dit), state)
        assert key.split(".")[1] in str(error.value) and "norm" not in str(error.value)


def test_patch_metadata_hook_uses_the_shared_add_patches_hook():
    runtime, ModelPatcher, _ = _load()
    from backend.patcher.function_chain import function_chain_contains
    from modules import anima_lora_support

    assert runtime.install_patch_metadata_hook() is False  # installed by _load
    assert anima_lora_support.script_hooks["freefuse_metadata"] is runtime._add_patches_with_metadata
    assert function_chain_contains(ModelPatcher.add_patches, "_anima_hook_freefuse_metadata")


def test_collect_similarity_matches_a_direct_computation():
    runtime, _, _ = _load()
    state = _state(runtime)
    state.adapters = [runtime.AdapterSpec("A", "cat", "cat"), runtime.AdapterSpec("B", "dog", "dog")]
    state.token_positions = {"A": [0, 1], "B": [3]}
    state.background_positions = [2, 9]  # 9 lies past the text: dropped
    state.top_k_ratio, state.temperature = 0.25, 0.5
    state.grid_height, state.grid_width = 16, 17  # 272 image tokens: two query chunks
    state.begin_collect()
    torch.manual_seed(0)
    spatial, dim = 272, 4
    q = torch.randn(4, spatial + 3, 2, dim)  # tokens past the image grid are ignored
    k = torch.randn(4, 5, 2, dim)
    x = torch.randn(4, spatial + 3, 6)

    state.collect_similarity(x, q, k, {"cond_or_uncond": [1, 0]})  # positive rows 2 and 3

    rows = [2, 3]
    attention = torch.softmax(torch.einsum("bshd,blhd->bhsl", q[rows, :spatial], k[rows]) / dim**0.5, dim=-1)
    hidden = x[rows, :spatial]

    def score(indices):  # mean attention of each image token to the phrase, over heads
        return attention[..., indices].mean(dim=(1, 3))

    def similarity(values):  # to the mean hidden state of the top 25% tokens
        top = values.topk(spatial // 4, dim=-1).indices
        core = torch.stack([hidden[row, top[row]].mean(dim=0) for row in range(len(rows))])
        return torch.softmax(torch.einsum("bsc,bc->bs", hidden, core) / 0.5, dim=-1)

    a, b = score([0, 1]), score([3])
    expected = {"A": similarity(2 * a - b), "B": similarity(2 * b - a)}
    for name, value in expected.items():
        (sample,) = state.similarity_samples[name]
        assert sample.shape == (2, spatial) and torch.allclose(sample, value, atol=1e-6)
    (background,) = state.background_samples
    assert torch.allclose(background, similarity(score([2])), atol=1e-6)

    state.current_step = 1  # other steps are not collected
    state.collect_similarity(x, q, k, {"cond_or_uncond": [1, 0]})
    assert len(state.background_samples) == 1


def test_complete_releases_the_cached_tensors():
    runtime, ModelPatcher, _ = _load()
    state = _state(runtime)
    patcher = _patcher(ModelPatcher, online=True)
    state.validate_patches(patcher)
    (function,) = patcher.weight_wrapper_patches["layer.weight"]
    _, (patch,) = runtime._lora_function_patches(function)
    assert state.adapter_for_patch(patch) == "A"
    state.similarity_samples = {"A": [torch.ones(1, 4)]}
    state.background_samples = [torch.ones(1, 4)]
    state.grid_height = state.grid_width = 2
    state.masks = {"A": torch.tensor([[1.0, 0.0, 0.0, 1.0]])}
    state.phase = "generate"
    state.mask_for_output("A", torch.zeros(1, 4, 3), spatial_capable=True)
    assert state._fitted and state._adapter_by_patch

    state.complete()

    assert state.phase == "complete" and not state.routing_active()
    assert state.similarity_samples == {} and state.background_samples == []
    assert state.masks == {} and state._fitted == {} and state._adapter_by_patch == {}


def test_untagged_patches_are_detected_offline_and_online():
    runtime, ModelPatcher, _ = _load()
    for online in (False, True):
        assert not runtime.has_untagged_patches(_patcher(ModelPatcher, online=online))
        patcher = ModelPatcher(_Model(), torch.device("cpu"), torch.device("cpu"))
        patcher.add_patches({"layer.weight": ("diff", (torch.ones(2, 2),))}, online_mode=online)
        assert runtime.has_untagged_patches(patcher)
