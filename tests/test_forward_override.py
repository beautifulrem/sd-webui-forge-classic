import torch

from modules.forward_override import install_forward_override, restore_forward_override


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


def test_named_override_restores_other_methods():
    class Engine:
        def get_learned_conditioning(self, prompt):
            return prompt

    engine = Engine()
    wrapper = lambda prompt: ["wrapped", *prompt]
    token = install_forward_override(engine, wrapper, name="get_learned_conditioning")
    assert engine.get_learned_conditioning(["a"]) == ["wrapped", "a"]

    assert restore_forward_override(engine, wrapper, token, name="get_learned_conditioning")
    assert "get_learned_conditioning" not in engine.__dict__
