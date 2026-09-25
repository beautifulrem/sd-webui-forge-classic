import torch
from safetensors.torch import save_file

from modules.anima_lora_files import adapter_kind_for_file, is_anima_adapter_file, lora_coverage
from modules.anima_lora_support import anima_lora_block_count, looks_like_layer_weights, parse_layer_weights, parse_module_weights


def _save(path, keys):
    save_file({key: torch.zeros(1) for key in keys}, str(path))
    return path


def test_coverage_lists_trained_blocks_modules_and_text_layers(tmp_path):
    lora = _save(
        tmp_path / "style.safetensors",
        [
            "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
            "diffusion_model.blocks.27.mlp.layer1.lora_up.weight",
            "text_encoders.qwen3_06b.model.layers.3.mlp.down_proj.lora_up.weight",
            "diffusion_model.blocks.5.cross_attn.alpha",  # not a weight: ignored
        ],
    )
    coverage = lora_coverage(lora)

    assert coverage.blocks == {0, 27} and coverage.modules == {"self_attn", "mlp"} and coverage.text_layers == {3}
    assert adapter_kind_for_file(lora) == "lora" and is_anima_adapter_file(lora)
    assert anima_lora_block_count(coverage.blocks) == 28


def test_lokr_and_text_encoder_only_files(tmp_path):
    lokr = _save(tmp_path / "a.safetensors", ["lycoris_blocks_3_mlp_layer1.lokr_w1"])
    text_only = _save(tmp_path / "b.safetensors", ["text_encoders.qwen3_06b.model.layers.1.mlp.lora_up.weight"])

    assert adapter_kind_for_file(lokr) == "lokr" and lora_coverage(lokr, "lokr").blocks == {3}
    assert not is_anima_adapter_file(text_only)  # no DiT blocks
    assert adapter_kind_for_file(tmp_path / "missing.safetensors") is None
    assert lora_coverage(tmp_path / "notes.txt").blocks == frozenset()


def test_layer_weight_syntax():
    weights = parse_layer_weights("block_lrs=0-2=0.5, 9=2, all=0.8")
    assert weights.values == {0: 0.5, 1: 0.5, 2: 0.5, 9: 2.0} and weights.default == 0.8
    assert parse_layer_weights("0.1 0.2 0.3").values == {0: 0.1, 1: 0.2, 2: 0.3}
    assert parse_layer_weights("5-3=0.4").values == {3: 0.4, 4: 0.4, 5: 0.4}
    assert looks_like_layer_weights("3=0.2") and looks_like_layer_weights("0.5,0.6")
    assert not looks_like_layer_weights("0.5") and not looks_like_layer_weights("mlp=0.8")
    assert parse_module_weights("self=0.5；ffn=2，ada=bad") == {"self_attn": 0.5, "mlp": 2.0, "adaln": 1.0}
