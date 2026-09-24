import importlib.metadata
from pathlib import Path

import torch
from comfy_kitchen.tensor import AsymW4A8Int8Layout, get_layout_class

from backend.quant_ops import QUANT_ALGOS


REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def _pinned_version(package: str) -> str:
    for line in REQUIREMENTS.read_text().splitlines():
        name, sep, version = line.partition("==")
        if sep and name.strip() == package:
            return version.strip()
    raise AssertionError(f"{package} is not pinned in requirements.txt")


def test_comfy_kitchen_version_exposes_asymmetric_w4a8_layout():
    assert importlib.metadata.version("comfy-kitchen") == _pinned_version("comfy-kitchen")
    assert get_layout_class("AsymW4A8Int8Layout") is AsymW4A8Int8Layout


def test_quantization_registry_contains_atomic_upstream_parameters():
    assert QUANT_ALGOS["asym_w4a8_int8"] == {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "AsymW4A8Int8Layout",
        "quantize_input": False,
    }
    assert "pre_quant_scale" in QUANT_ALGOS["nvfp4"]["parameters"]
