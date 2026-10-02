from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_quant_platform_policy import load_function


ROOT = Path(__file__).resolve().parents[1]


def quantized_loader():
    # Loading the application modules would initialize a GPU. Extract the actual
    # loader and registry instead; these error paths only need CPU tensors.
    namespace = {"torch": torch, "json": json, "get_layout_class": lambda name: object}
    for relative, name in (("backend/quant_ops.py", "QUANT_ALGOS"), ("backend/operations_mixed_precision.py", "_GROUPED_INT8_FORMATS")):
        tree = ast.parse((ROOT / relative).read_text())
        node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return load_function("backend/operations_mixed_precision.py", "_load_quantized_module", namespace)


def test_weight_dtype_ignores_metadata_and_counts_tensor_elements():
    weight_dtype = load_function("backend/utils.py", "weight_dtype", {"torch": torch})
    state = {
        "model.metadata": {"format": "fp8"},
        "model.small": torch.ones(1, dtype=torch.float16),
        "model.large": torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16)),
        "elsewhere": torch.ones(20, dtype=torch.float32),
    }
    assert weight_dtype(state, "model.") == torch.bfloat16
    assert weight_dtype({"metadata": "only metadata"}) is None


@pytest.mark.parametrize("quant_format,width", [("asym_w4a8_int8", 7), ("w6a8_int8", 8)])
def test_grouped_int8_rejects_invalid_packed_width(quant_format, width):
    module = SimpleNamespace(factory_kwargs={"device": "cpu", "dtype": torch.float32}, _disabled_formats=set(), _full_precision_mm=False, _orig_shape=(4, 16))
    conf = torch.tensor(list(json.dumps({"format": quant_format}).encode()), dtype=torch.uint8)
    state = {"layer.weight": torch.zeros(4, width, dtype=torch.int8), "layer.comfy_quant": conf}
    with pytest.raises(ValueError, match="packed weight width"):
        quantized_loader()(module, lambda *args: None, state, "layer.", {}, False, [], [], [])


@pytest.mark.parametrize("quant_format,width", [("asym_w4a8_int8", 8), ("w6a8_int8", 12)])
def test_grouped_int8_valid_width_requires_scales(quant_format, width):
    module = SimpleNamespace(factory_kwargs={"device": "cpu", "dtype": torch.float32}, _disabled_formats=set(), _full_precision_mm=False, _orig_shape=(4, 16))
    conf = torch.tensor(list(json.dumps({"format": quant_format}).encode()), dtype=torch.uint8)
    state = {"layer.weight": torch.zeros(4, width, dtype=torch.int8), "layer.comfy_quant": conf}
    with pytest.raises(ValueError, match=f"Missing {quant_format} group scale"):
        quantized_loader()(module, lambda *args: None, state, "layer.", {}, False, [], [], [])
