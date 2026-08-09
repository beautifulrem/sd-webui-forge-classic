import importlib.metadata

import torch
from comfy_kitchen.tensor import AsymW4A8Int8Layout, get_layout_class

from backend.quant_ops import QUANT_ALGOS


def test_comfy_kitchen_version_exposes_asymmetric_w4a8_layout():
    assert importlib.metadata.version("comfy-kitchen") == "0.2.27"
    assert get_layout_class("AsymW4A8Int8Layout") is AsymW4A8Int8Layout


def test_quantization_registry_contains_atomic_upstream_parameters():
    assert QUANT_ALGOS["asym_w4a8_int8"] == {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "AsymW4A8Int8Layout",
        "quantize_input": False,
    }
    assert "pre_quant_scale" in QUANT_ALGOS["nvfp4"]["parameters"]
