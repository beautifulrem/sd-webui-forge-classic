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


def test_active_overrides_are_tracked_until_restored():
    from modules.forward_override import overridden_modules

    module = torch.nn.Linear(1, 1)
    first = lambda x: x
    second = lambda x: x
    first_token = install_forward_override(module, first)
    second_token = install_forward_override(module, second)
    assert module in overridden_modules()

    assert restore_forward_override(module, second, second_token)
    assert module in overridden_modules()
    assert restore_forward_override(module, first, first_token)
    assert module not in overridden_modules()
