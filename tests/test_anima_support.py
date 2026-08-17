from types import SimpleNamespace

import torch

from modules.anima_support import anima_auxiliary_denoiser, is_anima_engine
from modules_forge.packages.huggingface_guess.detection import detect_unet_config


def test_anima_capability_marker_accepts_wrapped_engines():
    assert is_anima_engine(SimpleNamespace(is_anima_engine=True))


def test_models_without_anima_capability_are_rejected():
    assert not is_anima_engine(SimpleNamespace())
    assert not is_anima_engine(None)


def test_auxiliary_denoiser_restores_existing_marker():
    process = SimpleNamespace(_anima_auxiliary_denoiser=False)
    model = SimpleNamespace(p=process)

    with anima_auxiliary_denoiser(model):
        assert process._anima_auxiliary_denoiser is True

    assert process._anima_auxiliary_denoiser is False


def test_auxiliary_denoiser_removes_temporary_marker():
    process = SimpleNamespace()
    model = SimpleNamespace(p=process)

    with anima_auxiliary_denoiser(model):
        assert process._anima_auxiliary_denoiser is True

    assert not hasattr(process, "_anima_auxiliary_denoiser")


def test_anima_model_detection_uses_checkpoint_block_count():
    prefix = "model.diffusion_model."
    state_dict = {
        f"{prefix}blocks.{index}.mlp.layer1.weight": torch.empty(1)
        for index in range(40)
    }
    state_dict[f"{prefix}llm_adapter.blocks.0.cross_attn.q_proj.weight"] = torch.empty(1)
    state_dict[f"{prefix}x_embedder.proj.1.weight"] = torch.empty(2048, 68)

    config = detect_unet_config(state_dict, prefix)

    assert config["image_model"] == "anima"
    assert config["num_blocks"] == 40
