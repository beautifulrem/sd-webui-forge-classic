import json

import torch

from backend import operations
from backend.nn.anima import Anima
from backend.nn.anima_quant import int8_matmul_available, keep_sensitive_weights_precise, quantize_int8_convrot_state_dict, scale_fp8_state_dict
from backend.state_dict import detect_quantization

FP8 = torch.float8_e4m3fn


def _config(state_dict, key):
    return json.loads(state_dict[key].numpy().tobytes())


def _anima(**ops):
    with operations.using_forge_operations(device=torch.device("cpu"), manual_cast_enabled=True, **ops):
        return Anima(4, 4, 2, 1, model_channels=256, crossattn_emb_channels=64, adaln_lora_dim=32, num_blocks=3, num_heads=2).eval()


def _checkpoint():
    generator = torch.Generator().manual_seed(7)
    model = _anima(dtype=torch.float32)
    # DiT-like weights: small, with rare outliers
    state = {}
    for k, v in model.state_dict().items():
        w = torch.randn(v.shape, generator=generator) * 0.01
        if w.ndim == 2:
            w.view(-1)[:: max(1, w.numel() // 4)] = 0.2
        state[k] = w.to(torch.bfloat16)
    return state


def test_only_non_sensitive_block_linears_are_quantized():
    state = _checkpoint()
    original = {k: v.clone() for k, v in state.items()}

    quantized = scale_fp8_state_dict(state, fp8_matmul=False)

    fp8_keys = sorted(k for k, v in state.items() if v.dtype == FP8)
    assert quantized == len(fp8_keys) > 0
    for key in fp8_keys:
        prefix = key[: -len("weight")]
        assert key.startswith(("blocks.1.", "blocks.2.")) and not key.startswith("blocks.1.adaln_modulation_")
        assert state[f"{prefix}weight_scale"].dtype == torch.float32
        assert _config(state, f"{prefix}comfy_quant") == {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}
    for key, value in original.items():
        if key not in fp8_keys:
            assert torch.equal(state[key], value), key
    assert detect_quantization(state, is_unet=True) is not None


def test_fp8_matmul_keeps_the_fast_path_except_for_the_ffn_down_projection_and_modulation():
    state = _checkpoint()
    scale_fp8_state_dict(state, fp8_matmul=True)
    configs = {k: _config(state, k) for k in state if k.endswith("comfy_quant")}
    for key, config in configs.items():
        # The AdaLN modulation GEMMs have one row per image: FP8 there only
        # adds input-quantization kernels and rounding.
        if key.endswith("mlp.layer2.comfy_quant") or ".adaln_modulation_" in key:
            assert config == {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}, key
        else:
            assert config == {"format": "float8_e4m3fn"}, key
    assert any(key.endswith("mlp.layer2.comfy_quant") for key in configs)
    assert any(".adaln_modulation_" in key and state[key[: -len("comfy_quant")] + "weight"].dtype == FP8 for key in configs)


def test_scaled_weights_are_closer_than_a_plain_cast():
    state = _checkpoint()
    reference = {k: v.float() for k, v in state.items()}
    scale_fp8_state_dict(state, fp8_matmul=False)

    for key, value in state.items():
        if value.dtype != FP8:
            continue
        ref = reference[key]
        scaled = value.float() * state[key[: -len("weight")] + "weight_scale"]
        plain = ref.to(FP8).float()
        assert (scaled - ref).norm() < (plain - ref).norm(), key


def test_scaled_model_loads_and_tracks_full_precision_better_than_plain_fp8():
    checkpoint = _checkpoint()
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, 4, 1, 8, 8, generator=generator)
    t = torch.tensor([0.5])
    context = torch.randn(1, 5, 64, generator=generator)

    def run(model):
        with torch.inference_mode():
            return model(x, t, context, transformer_options={}).float()

    full = _anima(dtype=torch.float32)
    full.load_state_dict({k: v.float() for k, v in checkpoint.items()})

    plain = _anima(dtype=FP8)
    keep_sensitive_weights_precise(plain, torch.float32)
    plain.load_state_dict({k: v.clone() for k, v in checkpoint.items()})

    scaled_state = dict(checkpoint)
    scale_fp8_state_dict(scaled_state, fp8_matmul=False)
    quant_config = detect_quantization(scaled_state, is_unet=True)
    scaled = _anima(dtype=torch.float32, extra_dtype=quant_config)
    scaled.load_state_dict(scaled_state)

    assert isinstance(scaled.blocks[2].mlp.layer1.weight, operations.QuantizedTensor)
    reference = run(full)
    assert (run(scaled) - reference).norm() < (run(plain) - reference).norm()


def _int8_checkpoint():
    # 256-wide context so the cross-attention K/V projections are rotatable too
    generator = torch.Generator().manual_seed(11)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        model = Anima(4, 4, 2, 1, model_channels=256, crossattn_emb_channels=256, adaln_lora_dim=32, num_blocks=3, num_heads=2).eval()
    return model, {k: (torch.randn(v.shape, generator=generator) * 0.02).to(torch.bfloat16) for k, v in model.state_dict().items()}


def test_int8_convrot_quantizes_only_the_large_block_gemms():
    _, state = _int8_checkpoint()
    original = {k: v.clone() for k, v in state.items()}

    quantized = quantize_int8_convrot_state_dict(state)

    int8_keys = sorted(k for k, v in state.items() if v.dtype == torch.int8)
    assert quantized == len(int8_keys) > 0
    for key in int8_keys:
        prefix = key[: -len("weight")]
        assert key.startswith(("blocks.1.", "blocks.2.")), key
        assert key.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "output_proj.weight", "mlp.layer1.weight")), key
        assert state[f"{prefix}weight_scale"].shape == (state[key].shape[0], 1)
        assert _config(state, f"{prefix}comfy_quant") == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    for key, value in original.items():
        if key not in int8_keys:
            assert torch.equal(state[key], value), key  # AdaLN, FFN down, sensitive parts


def test_int8_convrot_model_loads_and_stays_close_to_full_precision():
    full, checkpoint = _int8_checkpoint()
    full.load_state_dict({k: v.float() for k, v in checkpoint.items()})
    state = dict(checkpoint)
    quantize_int8_convrot_state_dict(state)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True, extra_dtype=detect_quantization(state, is_unet=True)):
        quantized = Anima(4, 4, 2, 1, model_channels=256, crossattn_emb_channels=256, adaln_lora_dim=32, num_blocks=3, num_heads=2).eval()
    quantized.load_state_dict(state)
    assert isinstance(quantized.blocks[2].self_attn.q_proj.weight, operations.QuantizedTensor)

    generator = torch.Generator().manual_seed(5)
    x = torch.randn(1, 4, 1, 8, 8, generator=generator)
    context = torch.randn(1, 5, 256, generator=generator)
    with torch.inference_mode():
        reference = full(x, torch.tensor([0.5]), context, transformer_options={})
        output = quantized(x, torch.tensor([0.5]), context, transformer_options={}).float()
    assert (output - reference).norm() / reference.norm() < 0.05


def test_int8_needs_a_real_int8_gemm():
    assert not int8_matmul_available(torch.device("cpu"))


def test_quantize_on_load_falls_back_where_the_mode_cannot_run(monkeypatch):
    from backend import memory_management
    from backend.nn import anima_quant

    state = _checkpoint()
    original = {k: v.clone() for k, v in state.items()}

    monkeypatch.setattr(memory_management, "should_use_bf16", lambda device: False)
    anima_quant.quantize_on_load(state, anima_quant.ANIMA_QUANT_SCALED_FP8, torch.device("cpu"))
    assert state.keys() == original.keys()  # no bf16: untouched

    monkeypatch.setattr(memory_management, "should_use_bf16", lambda device: True)
    anima_quant.quantize_on_load(state, anima_quant.ANIMA_QUANT_INT8_CONVROT, torch.device("cpu"))
    assert state.keys() == original.keys()  # no INT8 GEMM on the CPU: untouched

    anima_quant.quantize_on_load(state, anima_quant.ANIMA_QUANT_SCALED_FP8, torch.device("cpu"))
    assert detect_quantization(state, is_unet=True) is not None


def test_scaling_on_the_load_device_keeps_the_results_and_stores_them_on_the_cpu():
    """quantize_on_load scales on the load device (CPU scaling cost ~8 s per 2B
    params per load); the stored tensors match the CPU formula bit for bit."""
    state = _checkpoint()
    expected = {}
    for key, value in state.items():
        if key.startswith(("blocks.1.", "blocks.2.")) and value.ndim == 2:
            scale = value.abs().amax().float().clamp_min(1e-12) / 448.0
            expected[key] = ((value.float() / scale).clamp_(-448.0, 448.0).to(FP8), scale)

    default, on_device = dict(state), dict(state)
    assert scale_fp8_state_dict(default, fp8_matmul=False) == scale_fp8_state_dict(on_device, fp8_matmul=False, device=torch.device("cpu"))
    for result in (default, on_device):
        for key, value in result.items():
            assert value.device.type == "cpu", key
            if value.dtype == FP8:
                prefix = key[: -len("weight")]
                weight, scale = expected[key]
                assert torch.equal(value.view(torch.uint8), weight.view(torch.uint8)), key
                assert torch.equal(result[f"{prefix}weight_scale"], scale), key
    assert default.keys() == on_device.keys()


def test_quantize_on_load_scales_fp8_on_the_load_device(monkeypatch):
    from backend import memory_management
    from backend.nn import anima_quant

    calls = []
    monkeypatch.setattr(memory_management, "should_use_bf16", lambda device: True)
    monkeypatch.setattr(anima_quant, "scale_fp8_state_dict", lambda state, **kwargs: calls.append(kwargs) or 0)
    anima_quant.quantize_on_load({}, anima_quant.ANIMA_QUANT_SCALED_FP8, torch.device("cpu"))
    assert calls and calls[0]["device"] == torch.device("cpu")
