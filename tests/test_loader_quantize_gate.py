"""``should_quantize_on_load``: which checkpoints the loader quantizes on load."""

import ast
from pathlib import Path

import pytest
import torch

from backend.nn.anima import should_quantize_on_load


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


def test_loader_gates_quantize_on_load_with_the_helper():
    # The loader cannot be driven on CPU without a checkpoint: check that the
    # ``if`` around its quantize_on_load call is the helper
    source = Path(__file__).resolve().parents[1] / "backend" / "loader.py"
    is_call = lambda node, name: isinstance(node, ast.Call) and getattr(node.func, "id", None) == name  # noqa: E731
    gates = [node for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))) if isinstance(node, ast.If) and any(isinstance(statement, ast.Expr) and is_call(statement.value, "quantize_on_load") for statement in node.body)]
    assert len(gates) == 1 and is_call(gates[0].test, "should_quantize_on_load")
