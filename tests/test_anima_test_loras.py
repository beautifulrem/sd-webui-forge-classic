"""tools/anima_test_loras.py: the test LoRAs tools/smoke_real.py --features
exercises the LoRA features with fit the checkpoint and read as Anima LoRAs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from modules.anima_lora_files import adapter_kind_for_file, is_anima_adapter_file, lora_coverage

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "anima_test_loras.py"
_spec = importlib.util.spec_from_file_location("anima_test_loras", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _checkpoint(path, blocks=3, dim=16, hidden=48):
    """A checkpoint with Anima's DiT key layout (tiny shapes)."""
    tensors = {"model.diffusion_model.x_embedder.proj.1.weight": torch.zeros(dim, 4)}
    for block in range(blocks):
        prefix = f"model.diffusion_model.blocks.{block}"
        tensors[f"{prefix}.self_attn.output_proj.weight"] = torch.zeros(dim, dim)
        tensors[f"{prefix}.self_attn.q_norm.weight"] = torch.zeros(dim)
        tensors[f"{prefix}.cross_attn.output_proj.weight"] = torch.zeros(dim, dim)
        tensors[f"{prefix}.mlp.layer1.weight"] = torch.zeros(hidden, dim)
        tensors[f"{prefix}.mlp.layer2.weight"] = torch.zeros(dim, hidden)
        tensors[f"model.diffusion_model.llm_adapter.blocks.{block}.self_attn.o_proj.weight"] = torch.zeros(8, 8)
    save_file(tensors, str(path))
    return path


def test_test_loras_fit_the_checkpoint_and_read_as_anima_loras(tmp_path):
    checkpoint = _checkpoint(tmp_path / "anima.safetensors")
    paths = tool.write_test_loras(checkpoint, tmp_path / "Lora")
    assert [path.name for path in paths] == ["remi-test-a.safetensors", "remi-test-b.safetensors"]

    layers = tool.target_layers(tool.read_header(checkpoint))
    assert set(layers) == {f"blocks.{block}.{module}" for block in range(3) for module in tool.MODULES}
    assert layers["blocks.0.mlp.layer2"] == (16, 48)

    a, b = (load_file(str(path)) for path in paths)
    for layer, (out_features, in_features) in layers.items():
        down, up = a[f"diffusion_model.{layer}.lora_down.weight"], a[f"diffusion_model.{layer}.lora_up.weight"]
        assert down.shape == (tool.RANK, in_features) and up.shape == (out_features, tool.RANK)
        assert float(a[f"diffusion_model.{layer}.alpha"]) == tool.RANK
    assert not torch.equal(a["diffusion_model.blocks.0.mlp.layer2.lora_up.weight"], b["diffusion_model.blocks.0.mlp.layer2.lora_up.weight"])

    for path in paths:
        assert adapter_kind_for_file(path) == "lora" and is_anima_adapter_file(path)
        coverage = lora_coverage(path)
        assert coverage.blocks == {0, 1, 2} and coverage.modules == {"self_attn", "cross_attn", "mlp"}


def test_deterministic(tmp_path):
    checkpoint = _checkpoint(tmp_path / "anima.safetensors")
    first = load_file(str(tool.write_test_loras(checkpoint, tmp_path / "one")[0]))
    second = load_file(str(tool.write_test_loras(checkpoint, tmp_path / "two")[0]))
    assert first.keys() == second.keys() and all(torch.equal(first[key], second[key]) for key in first)


def test_lora_keys_map_onto_the_anima_model():
    from modules_forge.packages.comfy.lora import model_lora_keys_unet

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = torch.nn.Module()
            self.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.diffusion_model.blocks[0].mlp = torch.nn.Module()
            self.diffusion_model.blocks[0].mlp.layer2 = torch.nn.Linear(4, 4, bias=False)
            self.diffusion_model.config = {}
            self.config = SimpleNamespace(huggingface_repo="circlestone-labs/Anima")

    key_map = model_lora_keys_unet(Model(), {})
    assert key_map.get("diffusion_model.blocks.0.mlp.layer2") == "diffusion_model.blocks.0.mlp.layer2.weight"


def test_refiner_checkpoint_differs_only_in_the_patched_layers(tmp_path):
    checkpoint = _checkpoint(tmp_path / "anima.safetensors")
    path = tool.write_refiner_checkpoint(checkpoint, tmp_path / "refiner" / "remi-test-refiner.safetensors")
    original, refiner = load_file(str(checkpoint)), load_file(str(path))
    assert original.keys() == refiner.keys()
    layers = tool.target_layers(tool.read_header(checkpoint))
    for key, tensor in original.items():
        assert refiner[key].dtype == tensor.dtype and refiner[key].shape == tensor.shape
        patched = any(key == f"model.diffusion_model.{layer}.weight" for layer in layers)
        assert torch.equal(refiner[key], tensor) is not patched, key
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as file:
        assert file.metadata()["remi_test_refiner"] == "1"
    again = load_file(str(tool.write_refiner_checkpoint(checkpoint, tmp_path / "again.safetensors")))
    assert all(torch.equal(again[key], refiner[key]) for key in refiner)


def test_test_lokr_fits_the_checkpoint_and_reads_as_an_anima_lokr(tmp_path):
    checkpoint = _checkpoint(tmp_path / "anima.safetensors")
    path = tool.write_test_lokr(checkpoint, tmp_path / "Lora")
    assert path.name == f"{tool.LOKR_NAME}.safetensors"
    tensors = load_file(str(path))
    for layer, (out_features, in_features) in tool.target_layers(tool.read_header(checkpoint)).items():
        w1, w2_a, w2_b = (tensors[f"diffusion_model.{layer}.lokr_{name}"] for name in ("w1", "w2_a", "w2_b"))
        assert torch.kron(w1.float(), w2_a.float() @ w2_b.float()).shape == (out_features, in_features)
        assert float(tensors[f"diffusion_model.{layer}.alpha"]) == tool.RANK
    assert adapter_kind_for_file(path) == "lokr" and is_anima_adapter_file(path, "lokr")
    assert lora_coverage(path, "lokr").blocks == {0, 1, 2}
    again = load_file(str(tool.write_test_lokr(checkpoint, tmp_path / "again")))
    assert again.keys() == tensors.keys() and all(torch.equal(again[key], tensors[key]) for key in tensors)
