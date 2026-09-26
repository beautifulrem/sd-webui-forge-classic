"""The ``quantize_on_load`` gate in ``backend/loader.py`` (evaluated from its AST:
the loader itself cannot be driven on CPU without a checkpoint)."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[1] / "backend" / "loader.py"


def _calls_quantize_on_load(statement):
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and getattr(statement.value.func, "id", None) == "quantize_on_load"


# The ``if`` whose own body calls ``quantize_on_load``.
_gate = next(
    node
    for node in ast.walk(ast.parse(_SOURCE.read_text(encoding="utf-8")))
    if isinstance(node, ast.If) and any(_calls_quantize_on_load(statement) for statement in node.body)
)
_condition = compile(ast.Expression(_gate.test), str(_SOURCE), "eval")


def _quantizes(anima_quant="int8_convrot", cls_name="CosmosTransformer3DModel", quant_config=None, nunchaku=False, state_dict_dtype=torch.bfloat16):
    names = {
        "anima_quant": anima_quant,
        "cls_name": cls_name,
        "quant_config": quant_config,
        "guess": SimpleNamespace(nunchaku=nunchaku),
        "state_dict_dtype": state_dict_dtype,
        "torch": torch,
    }
    return bool(eval(_condition, names))  # noqa: S307 - our own source file


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_full_precision_anima_checkpoint_is_quantized(dtype):
    assert _quantizes(state_dict_dtype=dtype)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(anima_quant=None),
        dict(anima_quant=""),
        dict(cls_name="Flux"),
        dict(quant_config={"format": "float8_e4m3fn"}),
        dict(nunchaku=True),
        dict(state_dict_dtype=torch.float8_e4m3fn),
        dict(state_dict_dtype=torch.float8_e5m2),
        dict(state_dict_dtype="gguf"),
    ],
)
def test_other_models_and_prequantized_checkpoints_are_left_alone(overrides):
    assert not _quantizes(**overrides)
