import pytest
import torch

from backend import operations
from backend.operations import ForgeOperations, ForgeOperationsFP8, fp8_linear


def _fp8_layer(in_features=8, out_features=4) -> ForgeOperationsFP8.Linear:
    layer = ForgeOperationsFP8.Linear(in_features, out_features, bias=False)
    layer.weight = torch.nn.Parameter(torch.randn(out_features, in_features).to(torch.float8_e4m3fn), requires_grad=False)
    return layer


@pytest.fixture
def recorded_linear(monkeypatch):
    calls = []

    def fake_linear(input, weight, bias=None):
        calls.append(tuple(input.shape))
        return torch.zeros(input.shape[0], weight.shape[0])

    monkeypatch.setattr(torch.nn.functional, "linear", fake_linear)
    return calls


@pytest.mark.parametrize("shape", [(6, 8), (2, 3, 8), (1, 2, 3, 4, 8)])
def test_fp8_linear_flattens_any_leading_dims(recorded_linear, shape):
    layer = _fp8_layer()
    x = torch.randn(shape, dtype=torch.bfloat16)

    out = fp8_linear(layer, x)

    assert out is not None
    assert tuple(out.shape) == (*shape[:-1], 4)
    assert len(recorded_linear) == 1 and len(recorded_linear[0]) == 2


def test_fp8_linear_does_not_clamp_a_caller_nd_tensor(recorded_linear):
    layer = _fp8_layer()
    x = torch.full((1, 2, 2, 2, 8), 1000.0, dtype=torch.bfloat16)

    fp8_linear(layer, x)

    assert torch.all(x == 1000.0)


def test_fp8_linear_skips_layers_with_weight_functions(monkeypatch):
    layer = _fp8_layer()
    layer.weight_function = [lambda w: w]

    def boom(*args, **kwargs):
        raise AssertionError("FP8 compute must not run on a patched weight")

    monkeypatch.setattr(operations, "weights_manual_cast", boom)

    assert fp8_linear(layer, torch.randn(2, 8, dtype=torch.bfloat16)) is None


def test_fp8_forward_logs_a_repeated_error_once(monkeypatch):
    layer = _fp8_layer()
    errors = []

    def fail(self, x):
        raise RuntimeError("no fp8 kernel")

    monkeypatch.setattr(operations, "fp8_linear", fail)
    monkeypatch.setattr(operations, "_fp8_linear_errors", set())
    monkeypatch.setattr(operations.memory_management.logger, "error", errors.append)
    monkeypatch.setattr(ForgeOperations.Linear, "forward", lambda self, x: x)

    for _ in range(3):
        layer(torch.zeros(1))

    assert len(errors) == 1
