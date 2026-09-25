import contextlib

import pytest
import torch

from backend import memory_management, operations, stream
from backend.operations import ForgeOperations, weights_manual_cast


class _FakeStream:
    def wait_stream(self, other):
        pass


@pytest.fixture
def offload_calls(monkeypatch):
    calls = []

    def get_offload_stream(device):
        calls.append(device)
        return _FakeStream()

    monkeypatch.setattr(stream, "should_use_stream", lambda: True)
    monkeypatch.setattr(stream, "stream_context", lambda: lambda s: contextlib.nullcontext())
    monkeypatch.setattr(memory_management, "get_offload_stream", get_offload_stream)
    monkeypatch.setattr(memory_management, "sync_stream", lambda device, s: None)
    return calls


def test_resident_weights_skip_the_offload_stream(offload_calls):
    layer = ForgeOperations.Linear(4, 4)

    weight, bias, (offload_stream, _, _) = weights_manual_cast(layer, torch.randn(2, 4))

    assert offload_calls == []
    assert offload_stream is None
    assert weight.dtype == torch.float32 and bias is not None


def test_moving_weights_use_the_offload_stream(offload_calls, monkeypatch):
    layer = ForgeOperations.Linear(4, 4)
    target = torch.device("meta")
    # Pretend the input lives on another device than the (CPU) weights.
    monkeypatch.setattr(memory_management, "cast_to", lambda weight, **kwargs: weight)
    monkeypatch.setattr(memory_management, "device_supports_non_blocking", lambda device: False)

    _, _, (offload_stream, _, _) = weights_manual_cast(layer, None, dtype=torch.float32, device=target)

    assert offload_calls == [target]
    assert isinstance(offload_stream, _FakeStream)


def test_main_stream_worker_is_a_no_op_without_offload_stream(offload_calls):
    layer = ForgeOperations.Linear(4, 4)
    weight, bias, signal = weights_manual_cast(layer, torch.randn(2, 4))

    with operations.main_stream_worker(weight, bias, signal):
        pass

    assert offload_calls == []
