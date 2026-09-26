"""``should_quantize_on_load`` / ``should_keep_sensitive_weights_precise``: which
checkpoints the loader quantizes on load, and which keep their sensitive parts precise."""

import ast
from pathlib import Path

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


def _loader_gates(call):
    # The loader cannot be driven on CPU without a checkpoint: find the ``if``
    # statements whose body calls ``call``
    source = Path(__file__).resolve().parents[1] / "backend" / "loader.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.If) and any(_is_call(getattr(statement, "value", None), call) for statement in node.body)]


def _is_call(node, name):
    return isinstance(node, ast.Call) and getattr(node.func, "id", None) == name


@pytest.mark.parametrize(
    ("call", "gate"),
    [("quantize_on_load", "should_quantize_on_load"), ("keep_sensitive_weights_precise", "should_keep_sensitive_weights_precise")],
)
def test_loader_gates_each_step_with_its_helper(call, gate):
    gates = _loader_gates(call)
    assert len(gates) == 1 and _is_call(gates[0].test, gate)
