"""``<lora:…>`` activation: a network that is not there becomes a job warning."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lora(monkeypatch):
    from modules import shared

    fake = types.ModuleType("networks")
    fake.available_networks = {}
    fake.loaded_networks = []
    fake.calls = []

    def load_networks(names, te, unet):
        fake.calls.append(names)
        return [name for name in names if name.startswith("missing")]

    fake.load_networks = load_networks
    monkeypatch.setitem(sys.modules, "networks", fake)
    monkeypatch.setattr(shared, "opts", types.SimpleNamespace(sd_lora="None", lora_add_hashes_to_infotext=False, extra_networks_default_multiplier=1.0))
    spec = importlib.util.spec_from_file_location("extra_networks_lora_under_test", ROOT / "extra_networks_lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExtraNetworkLora(), fake


class Processing:
    def __init__(self):
        self.all_prompts = ["a girl"]
        self.comments = {}
        self.extra_generation_params = {}

    def comment(self, text):
        self.comments[text] = 1


def test_a_missing_lora_is_reported_as_a_comment(lora):
    from modules.extra_networks import ExtraNetworkParams

    network, fake = lora
    p = Processing()
    network.activate(p, [ExtraNetworkParams(["missing_one", "1"]), ExtraNetworkParams(["here", "0.5"]), ExtraNetworkParams(["missing_one", "0.3"])])
    assert fake.calls == [["missing_one", "here", "missing_one"]]
    assert list(p.comments) == ["LoRA not found or failed to load: missing_one"]


def test_found_loras_add_no_comment(lora):
    from modules.extra_networks import ExtraNetworkParams

    network, _ = lora
    p = Processing()
    network.activate(p, [ExtraNetworkParams(["here", "1"])])
    assert p.comments == {}
