"""``should_quantize_on_load`` / ``should_keep_sensitive_weights_precise``: which
checkpoints the loader quantizes on load, and which keep their sensitive parts precise."""

from types import SimpleNamespace

import pytest
import torch

from backend.nn.anima_quant import should_keep_sensitive_weights_precise, should_quantize_on_load


def _quantizes(mode="int8_convrot", cls_name="CosmosTransformer3DModel", quant_config=None, nunchaku=False, state_dict_dtype=torch.bfloat16):
    return should_quantize_on_load(mode, cls_name, quant_config, nunchaku, state_dict_dtype)


@pytest.mark.parametrize("mode", ["int8_convrot", "scaled_fp8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_full_precision_anima_checkpoint_is_quantized(mode, dtype):
    assert _quantizes(mode=mode, state_dict_dtype=dtype) is True


@pytest.mark.parametrize(
    "overrides",
    [
        dict(mode=None),
        dict(mode=""),
        dict(cls_name="Flux"),
        dict(quant_config={"format": "float8_e4m3fn"}),
        dict(nunchaku=True),
        dict(state_dict_dtype=torch.float8_e4m3fn),
        dict(state_dict_dtype=torch.float8_e5m2),
        dict(state_dict_dtype="gguf"),
    ],
)
def test_other_models_and_prequantized_checkpoints_are_left_alone(overrides):
    assert _quantizes(**overrides) is False


def _keeps_precise(cls_name="CosmosTransformer3DModel", storage_dtype=torch.float8_e4m3fn, state_dict_dtype=torch.bfloat16):
    return should_keep_sensitive_weights_precise(cls_name, storage_dtype, state_dict_dtype)


@pytest.mark.parametrize("storage_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_on_the_fly_fp8_anima_keeps_its_sensitive_weights_precise(storage_dtype, dtype):
    assert _keeps_precise(storage_dtype=storage_dtype, state_dict_dtype=dtype) is True


@pytest.mark.parametrize(
    "overrides",
    [
        dict(cls_name="Flux"),
        dict(storage_dtype=torch.bfloat16),
        dict(storage_dtype="gguf", state_dict_dtype="gguf"),
        dict(state_dict_dtype=torch.float8_e4m3fn),
        dict(state_dict_dtype=torch.float8_e5m2),
        dict(state_dict_dtype="gguf"),
    ],
)
def test_other_models_and_fp8_checkpoints_keep_their_storage(overrides):
    assert _keeps_precise(**overrides) is False


_ANIMA = dict(in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1, model_channels=256, crossattn_emb_channels=64, adaln_lora_dim=32, num_blocks=3, num_heads=2)
FP8 = torch.float8_e4m3fn


def _load_anima(monkeypatch, *, anima_quant=None, storage_override=None):
    """Drive ``load_huggingface_component``'s DiT branch on the CPU with a tiny
    full-precision (bf16) Anima checkpoint. Returns the model, the state dict
    it loaded (quantized in place when a mode applied) and the parameter count
    the loader sized the model's initial device with."""

    import backend.args
    from backend import loader, memory_management, operations
    from backend.nn.anima import Anima

    cpu = torch.device("cpu")
    sized = []
    monkeypatch.setattr(memory_management, "get_torch_device", lambda: cpu)
    monkeypatch.setattr(memory_management, "unet_offload_device", lambda: cpu)
    monkeypatch.setattr(memory_management, "unet_initial_load_device", lambda parameters, dtype: sized.append(parameters) or cpu)
    monkeypatch.setattr(memory_management, "should_use_bf16", lambda *args, **kwargs: True)
    monkeypatch.setattr(backend.args.dynamic_args, "anima_quant", anima_quant)
    monkeypatch.setattr(backend.args.dynamic_args, "forge_unet_storage_dtype", storage_override)
    monkeypatch.setattr(backend.args.dynamic_args, "ops", None)

    generator = torch.Generator().manual_seed(3)
    with operations.using_forge_operations(device=cpu, dtype=torch.float32, manual_cast_enabled=True):
        shapes = {key: value.shape for key, value in Anima(**_ANIMA).state_dict().items()}
    state_dict = {key: (torch.randn(shape, generator=generator) * 0.02).to(torch.bfloat16) for key, shape in shapes.items()}
    guess = SimpleNamespace(unet_config=dict(_ANIMA), nunchaku=False, supported_inference_dtypes=[torch.bfloat16, torch.float32])

    model = loader.load_huggingface_component(guess, "transformer", "diffusers", "CosmosTransformer3DModel", "/unused", state_dict)
    return model, state_dict, sized


def test_loader_quantizes_a_full_precision_anima_into_mixed_precision_ops(monkeypatch):
    from backend import operations, utils

    model, state_dict, sized = _load_anima(monkeypatch, anima_quant="scaled_fp8")

    assert any(key.endswith(".comfy_quant") for key in state_dict)  # quantized, so detect_quantization re-ran
    assert model.storage_dtype == torch.bfloat16  # the MixedPrecision switch
    layer = model.blocks[2].mlp.layer1
    assert "MixedPrecision" in type(layer).__qualname__
    assert isinstance(layer.weight, operations.QuantizedTensor) and layer.weight._qdata.dtype == FP8
    sensitive = model.blocks[0].mlp.layer1.weight
    assert not isinstance(sensitive, operations.QuantizedTensor) and sensitive.dtype == torch.bfloat16
    # sized from the quantized state dict (scales and configs included), as a pre-quantized checkpoint is
    assert sized == [utils.calculate_parameters(state_dict)]


def test_loader_leaves_anima_alone_without_a_mode(monkeypatch):
    from backend import operations

    model, state_dict, _ = _load_anima(monkeypatch)

    assert not any(key.endswith(".comfy_quant") for key in state_dict)
    layer = model.blocks[2].mlp.layer1
    assert "MixedPrecision" not in type(layer).__qualname__
    assert not isinstance(layer.weight, operations.QuantizedTensor)


def test_loader_keeps_sensitive_weights_precise_for_on_the_fly_fp8(monkeypatch):
    model, _, _ = _load_anima(monkeypatch, storage_override=FP8)

    assert model.storage_dtype == FP8
    assert model.blocks[2].mlp.layer1.weight.dtype == FP8
    for weight in (model.blocks[0].mlp.layer1.weight, model.final_layer.linear.weight):
        assert weight.dtype == model.computation_dtype != FP8
