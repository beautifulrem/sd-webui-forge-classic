"""tools/tiny_anima.py: the tiny Anima checkpoint the extension end-to-end
runs use (tests/test_remi_extensions_e2e.py loads and generates with it)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


tiny = _tool("tiny_anima")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    return tiny.write(tmp_path_factory.mktemp("tiny") / tiny.FILENAME, seed=0)


def test_the_checkpoint_has_a_real_anima_layout(checkpoint):
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt") as file:
        keys = set(file.keys())
        assert tiny.METADATA_KEY in file.metadata()
        assert file.get_slice("net.x_embedder.proj.1.weight").get_shape() == [tiny.DIT["model_channels"], (tiny.DIT["in_channels"] + 1) * 4]
    # 28 DiT blocks, the LLM adapter kept with the DiT (as Anima does), the text encoder and the VAE
    assert {f"net.blocks.{index}.cross_attn.output_proj.weight" for index in range(28)} <= keys
    assert any(key.startswith("net.llm_adapter.") for key in keys)
    assert any(key.startswith("text_encoders.qwen3_06b.transformer.model.layers.") for key in keys)
    assert any(key.startswith("vae.decoder.") for key in keys)
    assert tiny.is_tiny(checkpoint) and not tiny.is_tiny(ROOT / "README.md") and not tiny.is_tiny(None)


def test_the_weights_follow_the_seed(checkpoint):
    import torch

    first, again, other = tiny.state_dict(0), tiny.state_dict(0), tiny.state_dict(1)
    key = "net.blocks.3.mlp.layer1.weight"
    assert torch.equal(first[key], again[key]) and not torch.equal(first[key], other[key])
    # the cross-attention gain: sharper attention on the prompt tokens
    assert torch.all(first["net.blocks.0.cross_attn.q_norm.weight"] == tiny.CROSS_ATTENTION_GAIN)
    assert torch.all(first["net.blocks.0.self_attn.q_norm.weight"] == 1.0)


def test_the_test_loras_fit_it(checkpoint):
    loras = _tool("anima_test_loras")
    layers = loras.target_layers(loras.read_header(checkpoint))
    assert len(layers) == 28 * len(loras.MODULES)
    assert layers["blocks.0.mlp.layer2"] == (tiny.DIT["model_channels"], tiny.DIT["model_channels"] * 4)
