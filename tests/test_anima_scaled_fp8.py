import json

import torch

from backend import operations
from backend.nn.anima import Anima, keep_sensitive_weights_precise, scale_fp8_state_dict
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


def test_fp8_matmul_keeps_the_fast_path():
    state = _checkpoint()
    scale_fp8_state_dict(state, fp8_matmul=True)
    key = next(k for k in state if k.endswith("comfy_quant"))
    assert _config(state, key) == {"format": "float8_e4m3fn"}


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
