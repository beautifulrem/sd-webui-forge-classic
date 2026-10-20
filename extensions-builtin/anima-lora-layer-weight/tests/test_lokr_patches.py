"""The LoKR tab end to end on a real LoKr file, without a model download.

``tools/anima_test_loras.py`` writes the LoKr ``tools/smoke_real.py
--features`` loads into the real model (``lokr_w1`` and a low-rank
``lokr_w2_a`` / ``lokr_w2_b`` per layer). Here it is written for a tiny
checkpoint with Anima's DiT layout and goes the way a generation takes it:
the prompt tag through ``before_process_batch`` (rewrite, rule, infotext),
the file through Forge's LoRA loader (``load_lora`` builds comfy's
``LoKrAdapter`` patches), the patches through ``ModelPatcher.add_patches``
(the panel's hook) and ``merge_lora_to_weight`` (the weights the model
runs with). Block weight ``all=1`` must give the weights of the plain tag at
1, ``all=0`` those of the tag at 0, a block or module at 0 must leave
exactly that part untouched, and the LoRA tab must not touch a LoKr."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_lora_layer_weight.py"
TOOL = Path(__file__).resolve().parents[3] / "tools" / "anima_test_loras.py"
BLOCKS, DIM, HIDDEN = 3, 16, 48
LAYERS = ("self_attn.output_proj", "cross_attn.output_proj", "mlp.layer2")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load(TOOL, "anima_test_loras_for_lokr")


class _Anima(torch.nn.Module):
    """The part of Anima's DiT the test adapters patch (tiny shapes)."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList()
        for _ in range(BLOCKS):
            block = torch.nn.Module()
            for name, (out_features, in_features) in (("self_attn", (DIM, DIM)), ("cross_attn", (DIM, DIM)), ("mlp", (DIM, HIDDEN))):
                part = torch.nn.Module()
                layer = torch.nn.Linear(in_features, out_features, bias=False)
                layer.weight.data = torch.randn(out_features, in_features, generator=generator)
                if name == "mlp":
                    part.layer2 = layer
                else:
                    part.output_proj = layer
                setattr(block, name, part)
            self.diffusion_model.blocks.append(block)
        self.diffusion_model.config = {}
        self.config = SimpleNamespace(huggingface_repo="circlestone-labs/Anima")

    def checkpoint_tensors(self) -> dict:
        return {f"model.{key}": value.clone() for key, value in self.state_dict().items()}


@pytest.fixture
def adapters(tmp_path):
    model = _Anima()
    checkpoint = tmp_path / "anima.safetensors"
    save_file(model.checkpoint_tensors(), str(checkpoint))
    lokr = tool.write_test_lokr(checkpoint, tmp_path / "Lora")
    lora = tool.write_test_loras(checkpoint, tmp_path / "Lora")[0]
    return SimpleNamespace(model=model, lokr=lokr, lora=lora)


@pytest.fixture
def module(monkeypatch, adapters):
    from backend.patcher import base

    # The hook wraps ModelPatcher.add_patches; restored after the test
    monkeypatch.setattr(base.ModelPatcher, "add_patches", base.ModelPatcher.add_patches)
    module = _load(SCRIPT, "anima_lora_layer_weight_lokr")
    # The sd_forge_lora index: the prompt names resolve to the files
    entries = {path.stem: SimpleNamespace(filename=str(path)) for path in (adapters.lokr, adapters.lora)}
    index = SimpleNamespace(available_networks=entries, available_network_aliases=entries, forbidden_network_aliases={"none"}, update_available_networks_by_names=lambda names: None)
    monkeypatch.setitem(sys.modules, "networks", index)
    monkeypatch.setattr(module, "_LAST_SIGNATURE", None)
    monkeypatch.setattr(module, "_LAST_TEXT_ENCODER_SIGNATURE", None)
    yield module
    module._ACTIVE_RULE = None


def _tab(enabled=False, block_weights="all=1.0", module_weights=None):
    """One tab's 12 script args (TAB_ARGS order)."""
    return [enabled, "", True, block_weights, module_weights is not None, module_weights or "", False, "", False, "", "", ""]


def _weights(module, adapters, prompt: str, *, lora_tab=None, lokr_tab=None):
    """The patched weights a run of ``prompt`` (one tag) loads, and its
    infotext params, as a generation computes them."""
    from backend.patcher.base import ModelPatcher, merge_lora_to_weight
    from modules_forge.packages.comfy.lora import load_lora, model_lora_keys_unet

    p = SimpleNamespace(iteration=0, batch_size=1, prompts=[prompt], all_prompts=[prompt], all_hr_prompts=[prompt], extra_generation_params={})
    module.Script().before_process_batch(p, *(lora_tab or _tab()), *(lokr_tab or _tab()))
    # extra_networks parses the rewritten tag: <lora:name:weight>
    name, weight = p.prompts[0].removeprefix("<lora:").removesuffix(">").split(":")
    path = adapters.lokr if name == adapters.lokr.stem else adapters.lora
    patches, unmatched = load_lora(load_file(str(path)), model_lora_keys_unet(adapters.model))
    assert not unmatched
    patcher = ModelPatcher(adapters.model, torch.device("cpu"), torch.device("cpu"))
    loaded = patcher.add_patches(patches=patches, strength_patch=float(weight), filename=str(path))
    assert sorted(loaded) == sorted(patches)
    state = adapters.model.state_dict()
    weights = {key: merge_lora_to_weight(patcher.patches[key], state[key].float().clone(), key) for key in sorted(patcher.patches)}
    module.Script().postprocess(p, SimpleNamespace(), *(lora_tab or _tab()), *(lokr_tab or _tab()))
    return weights, p.extra_generation_params


def _key(block, layer):
    return f"diffusion_model.blocks.{block}.{layer}.weight"


def test_the_fixture_is_a_lokr_the_loader_reads_as_one(adapters):
    from modules.anima_lora_files import adapter_kind_for_file, is_anima_adapter_file, lora_coverage
    from modules_forge.packages.comfy.lora import load_lora, model_lora_keys_unet
    from modules_forge.packages.comfy.weight_adapter.lokr import LoKrAdapter

    tensors = load_file(str(adapters.lokr))
    assert {key.rsplit(".", 1)[1] for key in tensors} == {"lokr_w1", "lokr_w2_a", "lokr_w2_b", "alpha"}
    assert adapter_kind_for_file(adapters.lokr) == "lokr" and is_anima_adapter_file(adapters.lokr, "lokr")
    assert not is_anima_adapter_file(adapters.lokr, "lora")
    coverage = lora_coverage(adapters.lokr, "lokr")
    assert coverage.blocks == set(range(BLOCKS)) and coverage.modules == {"self_attn", "cross_attn", "mlp"}

    patches, unmatched = load_lora(tensors, model_lora_keys_unet(adapters.model))
    assert not unmatched
    assert sorted(patches) == sorted(_key(block, layer) for block in range(BLOCKS) for layer in LAYERS)
    assert all(isinstance(patch, LoKrAdapter) for patch in patches.values())
    # The delta is kron(w1, w2_a @ w2_b) * alpha / rank, the layer's shape
    prefix = "diffusion_model.blocks.1.mlp.layer2"
    expected = torch.kron(tensors[f"{prefix}.lokr_w1"].float(), tensors[f"{prefix}.lokr_w2_a"].float() @ tensors[f"{prefix}.lokr_w2_b"].float())
    weight = adapters.model.state_dict()[f"{prefix}.weight"].float()
    from backend.patcher.base import merge_lora_to_weight

    merged = merge_lora_to_weight([(1.0, patches[f"{prefix}.weight"], 1.0, None, None)], weight.clone(), f"{prefix}.weight")
    assert expected.shape == weight.shape
    torch.testing.assert_close(merged - weight, expected, rtol=1e-4, atol=1e-6)


def test_block_weights_all_reproduce_the_plain_tag(module, adapters):
    original = {key: value.float() for key, value in adapters.model.state_dict().items()}
    at_one, plain_info = _weights(module, adapters, "<lora:remi-test-lokr:1>")
    at_zero, _ = _weights(module, adapters, "<lora:remi-test-lokr:0>")
    assert plain_info == {}
    assert all(not torch.equal(at_one[key], original[key]) for key in at_one)  # the LoKr changes every layer
    assert all(torch.equal(at_zero[key], original[key]) for key in at_zero)

    # The tab rewrites the tag to 1.0, so its own weight is ignored: all=1 is the tag at 1 ...
    full, info = _weights(module, adapters, "<lora:remi-test-lokr:0.3>", lokr_tab=_tab(True, "all=1.0"))
    assert all(torch.equal(full[key], at_one[key]) for key in at_one)
    assert info == {
        "Anima LoRA layer lora enabled": False,
        "Anima LoRA layer lokr enabled": True,
        "Anima LoRA layer lokr blocks enabled": True,
        "Anima LoRA layer lokr blocks": "all=1.0",
        "Anima LoRA layer lokr modules enabled": False,
        "Anima LoRA layer lokr qwen enabled": False,
    }

    # ... and all=0 the tag at 0
    zeroed, info = _weights(module, adapters, "<lora:remi-test-lokr:1>", lokr_tab=_tab(True, "all=0"))
    assert all(torch.equal(zeroed[key], at_zero[key]) for key in at_zero)
    assert info["Anima LoRA layer lokr blocks"] == "all=0"


def test_a_block_or_module_at_zero_leaves_exactly_that_part(module, adapters):
    original = {key: value.float() for key, value in adapters.model.state_dict().items()}
    at_one, _ = _weights(module, adapters, "<lora:remi-test-lokr:1>")

    blocks, info = _weights(module, adapters, "<lora:remi-test-lokr:1>", lokr_tab=_tab(True, "0=0, 2=0.5"))
    assert info["Anima LoRA layer lokr blocks"] == "0=0, 2=0.5"
    for layer in LAYERS:
        assert torch.equal(blocks[_key(0, layer)], original[_key(0, layer)])
        assert torch.equal(blocks[_key(1, layer)], at_one[_key(1, layer)])
        half = original[_key(2, layer)] + (at_one[_key(2, layer)] - original[_key(2, layer)]) / 2
        torch.testing.assert_close(blocks[_key(2, layer)], half, rtol=1e-4, atol=1e-5)

    modules, info = _weights(module, adapters, "<lora:remi-test-lokr:1>", lokr_tab=_tab(True, "all=1.0", "mlp=0"))
    assert info["Anima LoRA layer lokr modules"] == "mlp=0"
    for block in range(BLOCKS):
        assert torch.equal(modules[_key(block, "mlp.layer2")], original[_key(block, "mlp.layer2")])
        assert torch.equal(modules[_key(block, "self_attn.output_proj")], at_one[_key(block, "self_attn.output_proj")])


def test_each_tab_controls_only_its_adapter_kind(module, adapters):
    lokr_at_one, _ = _weights(module, adapters, "<lora:remi-test-lokr:1>")
    lora_at_one, _ = _weights(module, adapters, "<lora:remi-test-a:1>")

    # The LoRA tab at 0 leaves a LoKr alone (and its tag at the written weight) ...
    untouched, info = _weights(module, adapters, "<lora:remi-test-lokr:1>", lora_tab=_tab(True, "all=0"))
    assert all(torch.equal(untouched[key], lokr_at_one[key]) for key in lokr_at_one)
    assert info["Anima LoRA layer lora blocks"] == "all=0" and info["Anima LoRA layer lokr enabled"] is False

    # ... and the LoKR tab at 0 a LoRA
    untouched, _ = _weights(module, adapters, "<lora:remi-test-a:1>", lokr_tab=_tab(True, "all=0"))
    assert all(torch.equal(untouched[key], lora_at_one[key]) for key in lora_at_one)
