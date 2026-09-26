"""Load-time storage of the Anima DiT in low bits: the precision-sensitive
parameters kept out of FP8 storage, and the "Diffusion in Low Bits" modes
that quantize a full-precision checkpoint while loading (see backend/loader.py).
Kept apart from the model file (upstream's backend/nn/anima.py)."""

import torch
from torch import nn


# Parts of the DiT that low-bit storage hurts most (the layout comfy-quants
# uses for its Anima FP8/NVFP4 checkpoints): patch/timestep embedders, the
# final projection, block 0, block 1's modulation, and every norm.
_PRECISION_SENSITIVE_PREFIXES = (
    "x_embedder.",
    "t_embedder.",
    "t_embedding_norm.",
    "final_layer.",
    "blocks.0.",
    "blocks.1.adaln_modulation_",
)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def should_keep_sensitive_weights_precise(cls_name: str, storage_dtype, state_dict_dtype) -> bool:
    """Whether the loader applies ``keep_sensitive_weights_precise``: only to an
    Anima DiT stored in FP8 on the fly, not to a checkpoint already FP8 or GGUF
    (its sensitive parts have no higher precision to keep)."""

    return cls_name == "CosmosTransformer3DModel" and storage_dtype in _FP8_DTYPES and state_dict_dtype not in (*_FP8_DTYPES, "gguf")


def keep_sensitive_weights_precise(model: nn.Module, dtype: torch.dtype) -> int:
    """With on-the-fly FP8 storage, store the precision-sensitive parameters
    (see above) in ``dtype`` instead. Call before loading the state dict, so
    they receive the checkpoint's original precision. Returns how many
    parameters were kept."""

    kept = 0
    for module_name, module in model.named_modules():
        name = module_name + "."
        sensitive = name.startswith(_PRECISION_SENSITIVE_PREFIXES) or isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm))
        if not sensitive:
            continue
        converted = False
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param.dtype in _FP8_DTYPES:
                setattr(module, param_name, nn.Parameter(param.detach().to(dtype), requires_grad=False))
                converted = True
                kept += 1
        if converted and hasattr(module, "parameters_manual_cast"):
            module.parameters_manual_cast = True  # cast to the input's dtype
    return kept


# "Diffusion in Low Bits" modes that quantize a full-precision Anima DiT while
# loading (see backend/loader.py); the precision-sensitive parts stay as they are.
ANIMA_QUANT_SCALED_FP8 = "scaled_fp8"
ANIMA_QUANT_INT8_CONVROT = "int8_convrot"

_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_INT8_CONVROT_GROUP = 256
# Block Linears whose GEMMs have enough rows for INT8 to pay off. The AdaLN
# modulation Linears only see one row per image, and the FFN down projection
# carries the outlier activations (Ideogram INT8 study, arXiv:2606.12280).
_INT8_LINEAR_SUFFIXES = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.output_proj.weight",
    "mlp.layer1.weight",
)
_FFN_DOWN_SUFFIX = "mlp.layer2.weight"


def _comfy_quant_conf(conf: dict) -> torch.Tensor:
    import json

    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _block_linear_weights(state_dict: dict[str, torch.Tensor]):
    """``(key, prefix, weight)`` for full-precision block Linear weights outside
    the precision-sensitive parts. Lazily, over a fixed key list: callers
    replace each weight as they go, so the originals are freed one by one."""

    for key in [k for k in state_dict if k.startswith("blocks.") and k.endswith(".weight")]:
        weight = state_dict[key]
        if weight.ndim != 2 or not weight.is_floating_point() or weight.dtype in _FP8_DTYPES or key.startswith(_PRECISION_SENSITIVE_PREFIXES):
            continue
        yield key, key[: -len("weight")], weight


def scale_fp8_state_dict(state_dict: dict[str, torch.Tensor], *, fp8_matmul: bool, device: torch.device = None) -> int:
    """Quantize the block Linear weights of a full-precision state dict to
    per-tensor scaled ``float8_e4m3fn`` in the ``comfy_quant`` layout, in place.

    A plain ``.to(float8)`` cast leaves most of FP8's range unused for small DiT
    weights (many fall below e4m3's smallest normal); scaling each tensor by
    ``absmax / 448`` keeps them at full relative precision for the same VRAM.
    ``fp8_matmul`` keeps FP8 GEMMs (``--fast-fp8``) except for the FFN down
    projection, whose input activations exceed FP8's range most; otherwise
    every weight is dequantized for a regular matmul. Scaling runs on ``device``
    (the load device) and the results are stored on the CPU, as for INT8.
    Returns how many weights were quantized."""

    fp8_gemm = _comfy_quant_conf({"format": "float8_e4m3fn"})
    full_gemm = _comfy_quant_conf({"format": "float8_e4m3fn", "full_precision_matrix_mult": True})

    quantized = 0
    for key, prefix, weight in _block_linear_weights(state_dict):
        weight = weight.to(device=device, dtype=torch.float32)
        scale = weight.abs().amax().clamp_min(1e-12) / _FP8_E4M3_MAX
        state_dict[key] = (weight / scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).cpu()
        state_dict[f"{prefix}weight_scale"] = scale.cpu()
        state_dict[f"{prefix}comfy_quant"] = fp8_gemm if fp8_matmul and not key.endswith(_FFN_DOWN_SUFFIX) else full_gemm
        quantized += 1
    return quantized


def quantize_int8_convrot_state_dict(state_dict: dict[str, torch.Tensor], *, device: torch.device = None) -> int:
    """Quantize the block attention projections and FFN up projections of a
    full-precision state dict to INT8 with ConvRot (group-wise Hadamard
    rotation, arXiv:2512.03673) and per-row scales, in the ``comfy_quant``
    ``int8_tensorwise`` layout, in place. Activations are quantized per row at
    run time by comfy-kitchen's ``int8_linear``.

    Rotating runs on ``device`` (the load device: much faster than the CPU);
    the results are stored on the CPU like the rest of the state dict. Returns
    how many weights were quantized."""

    from backend.quant_ops import TensorWiseINT8Layout

    conf = _comfy_quant_conf({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": _INT8_CONVROT_GROUP})
    quantized = 0
    for key, prefix, weight in _block_linear_weights(state_dict):
        if not key.endswith(_INT8_LINEAR_SUFFIXES) or weight.shape[1] % _INT8_CONVROT_GROUP:
            continue
        qdata, params = TensorWiseINT8Layout.quantize(
            weight.to(device=device, dtype=torch.float32),
            is_weight=True,
            per_channel=True,
            convrot=True,
            convrot_groupsize=_INT8_CONVROT_GROUP,
        )
        state_dict[key] = qdata.cpu()
        state_dict[f"{prefix}weight_scale"] = params.scale.float().cpu()
        state_dict[f"{prefix}comfy_quant"] = conf
        quantized += 1
    return quantized


def should_quantize_on_load(mode, cls_name: str, quant_config, nunchaku: bool, state_dict_dtype) -> bool:
    """Whether the loader applies ``quantize_on_load`` (``mode`` is the chosen
    "Diffusion in Low Bits" Anima mode, empty for none): only to an Anima DiT
    stored at full precision, never to one already mixed-precision, Nunchaku,
    FP8 or GGUF."""

    return bool(mode) and cls_name == "CosmosTransformer3DModel" and quant_config is None and not nunchaku and state_dict_dtype in (torch.float16, torch.bfloat16, torch.float32)


def quantize_on_load(state_dict: dict[str, torch.Tensor], mode: str, load_device: torch.device) -> None:
    """Apply a "Diffusion in Low Bits" Anima mode to a full-precision state dict
    before the model is built; the mixed-precision loader takes it from there.
    Unsupported devices keep the option's plain storage dtype, with a warning."""

    import logging

    from backend import memory_management
    from backend.args import args

    logger = logging.getLogger("loader")
    # The mixed-precision ops compute in bf16, or in fp32 where bf16 is unsupported
    if not memory_management.should_use_bf16(load_device):
        logger.warning(f"Anima {mode} needs a bf16-capable device ; loading without it")
        return
    if mode == ANIMA_QUANT_SCALED_FP8:
        fp8_matmul = args.fast_fp8 and memory_management.supports_fp8_compute(load_device)
        quantized = scale_fp8_state_dict(state_dict, fp8_matmul=fp8_matmul, device=load_device)
        logger.info(f"Anima: stored {quantized} weights as scaled FP8" + (" (FP8 matmul)" if fp8_matmul else ""))
    elif mode == ANIMA_QUANT_INT8_CONVROT:
        if not int8_matmul_available(load_device):
            logger.warning("Anima INT8 needs comfy-kitchen's INT8 GEMM (CUDA 13 build or --enable-triton-backend, SM 8.0+) ; loading at full precision")
            return
        quantized = quantize_int8_convrot_state_dict(state_dict, device=load_device)
        logger.info(f"Anima: stored {quantized} weights as INT8 (ConvRot)")
    else:
        logger.warning(f"Unknown Anima quantization mode {mode!r}")


def int8_matmul_available(device: torch.device) -> bool:
    """Whether comfy-kitchen has a real INT8 GEMM (CUDA / Triton, SM 8.0+) for
    ``device``; its eager fallback would make INT8 weights slower than bf16."""

    from backend.quant_ops import ck

    device = torch.device(device)
    if device.type != "cuda":
        return False
    try:
        return ck.registry.get_capable_backend(
            "int8_linear",
            {
                "x": torch.zeros(1, _INT8_CONVROT_GROUP, device=device, dtype=torch.bfloat16),
                "weight": torch.zeros(1, _INT8_CONVROT_GROUP, device=device, dtype=torch.int8),
                "weight_scale": torch.ones(1, 1, device=device),
            },
        ) != "eager"
    except Exception:
        return False
