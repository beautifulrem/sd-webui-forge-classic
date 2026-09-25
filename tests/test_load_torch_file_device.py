import contextlib

import pytest
import torch
from safetensors.torch import save_file

from backend import utils


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"w": torch.arange(4.0)}, str(path))
    return str(path)


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_safetensors_accepts_device_strings(checkpoint, device):
    sd = utils.load_torch_file(checkpoint, device=device)
    assert torch.equal(sd["w"], torch.arange(4.0))


def test_safetensors_keeps_the_device_index(checkpoint, monkeypatch):
    seen = []

    @contextlib.contextmanager
    def fake_open(path, framework, device):
        seen.append(device)
        yield type("F", (), {"keys": lambda self: [], "metadata": lambda self: None})()

    monkeypatch.setattr(utils.safetensors, "safe_open", fake_open)

    utils.load_torch_file(checkpoint, device=torch.device("cuda", 1))

    assert seen == ["cuda:1"]
