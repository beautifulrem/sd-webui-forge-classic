from types import SimpleNamespace

import torch

from modules.anima_support import (
    anima_auxiliary_denoiser,
    install_forward_override,
    is_anima_engine,
    restore_forward_override,
)
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


def test_forward_override_restore_keeps_later_class_patches_visible():
    class Module(torch.nn.Module):
        def forward(self, x):
            return x + 1

    module = Module()
    wrapper = lambda x: x * 10
    token = install_forward_override(module, wrapper)
    assert module(torch.tensor(1.0)) == 10

    assert restore_forward_override(module, wrapper, token)
    assert "forward" not in module.__dict__

    Module.forward = lambda self, x: x + 2
    assert module(torch.tensor(1.0)) == 3


def test_forward_override_restore_leaves_foreign_overrides_alone():
    module = torch.nn.Identity()
    outer = lambda x: x
    existing = install_forward_override(module, outer)
    inner = lambda x: x
    token = install_forward_override(module, inner)

    assert restore_forward_override(module, inner, token)
    assert module.__dict__["forward"] is outer
    assert not restore_forward_override(module, inner, token)
    assert restore_forward_override(module, outer, existing)
    assert "forward" not in module.__dict__
