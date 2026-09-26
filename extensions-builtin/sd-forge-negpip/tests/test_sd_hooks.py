import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.argv = sys.argv[:1]  # modules.shared parses sys.argv on import
EXTENSION_PATH = Path(__file__).parents[1]
for path in (str(EXTENSION_PATH.parents[1]), str(EXTENSION_PATH)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.nn.unet import CrossAttention  # noqa: E402
from lib_negpip import sd as negpip_sd  # noqa: E402
from modules.forward_override import install_forward_override, overridden_modules  # noqa: E402


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = CrossAttention(8, heads=2, dim_head=4)  # self-attention: never hooked
        self.attn2 = CrossAttention(8, context_dim=6, heads=2, dim_head=4)


class _UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])


class _NegPiP:
    _patched = [False, False]


def _instance():
    # No negative conditioning: the hooked forward computes plain cross-attention
    return SimpleNamespace(is_xl=False, batch_size=1, rev=True, active=True, c_len=1, uc_len=1, conds=[], c_tokens=[], unconds=[], uc_tokens=[])


@pytest.fixture
def unet(monkeypatch):
    torch.manual_seed(0)
    unet = _UNet()
    model = SimpleNamespace(forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=unet))))
    monkeypatch.setattr(negpip_sd, "shared", SimpleNamespace(sd_model=model))  # the real one loads a checkpoint
    monkeypatch.setattr(_NegPiP, "_patched", [False, False])
    yield unet
    negpip_sd.patch_sd_negpip(None, _NegPiP, unpatch=True)


def _inputs():
    return torch.randn(1, 5, 8), torch.randn(1, 3, 6)


def test_patch_hooks_cross_attention_and_unpatch_restores_forward(unet):
    x, context = _inputs()
    expected = [block.attn2(x, context) for block in unet.blocks]

    negpip_sd.patch_sd_negpip(_instance(), _NegPiP)
    assert _NegPiP._patched[0]
    for block in unet.blocks:
        assert "forward" in block.attn2.__dict__ and "_negpip_restore" in block.attn2.__dict__
        assert "forward" not in block.attn1.__dict__
    assert len(negpip_sd._PATCHED_MODULES) == 2
    with torch.inference_mode():
        for block, want in zip(unet.blocks, expected):
            assert torch.allclose(block.attn2(x, context), want, atol=1e-6)

    negpip_sd.patch_sd_negpip(_instance(), _NegPiP)  # already patched: no second layer
    assert len(negpip_sd._PATCHED_MODULES) == 2

    negpip_sd.patch_sd_negpip(None, _NegPiP, unpatch=True)
    assert not _NegPiP._patched[0] and negpip_sd._PATCHED_MODULES == []
    for block in unet.blocks:
        attention = block.attn2
        # the override is removed, not pinned: class-level patches apply again
        assert "forward" not in attention.__dict__ and "_negpip_restore" not in attention.__dict__
        assert attention.forward.__func__ is CrossAttention.forward
        assert attention not in overridden_modules()


def test_unpatch_under_a_stacked_override_leaves_a_pass_through(unet):
    x, context = _inputs()
    attention = unet.blocks[0].attn2
    expected = attention(x, context)

    instance = _instance()
    negpip_sd.patch_sd_negpip(instance, _NegPiP)
    ours = attention.forward
    calls = []

    def outer(*args, **kwargs):
        calls.append(1)
        return ours(*args, **kwargs)

    install_forward_override(attention, outer)  # another extension wrapped our hook
    negpip_sd.patch_sd_negpip(None, _NegPiP, unpatch=True)

    assert attention.__dict__["forward"] is outer  # its chain is kept
    assert "_negpip_restore" not in attention.__dict__
    # ours no longer touches the NegPiP state: it would fail on an emptied instance
    instance.__dict__.clear()
    with torch.inference_mode():
        assert torch.allclose(attention(x, context), expected, atol=1e-6)
    assert calls == [1]
    assert unet.blocks[1].attn2.forward.__func__ is CrossAttention.forward  # the others were restored


def test_unpatch_after_the_model_is_gone_does_not_raise(unet, monkeypatch):
    negpip_sd.patch_sd_negpip(_instance(), _NegPiP)
    refs = [weakref.ref(block.attn2) for block in unet.blocks]
    # the checkpoint was unloaded or replaced meanwhile
    monkeypatch.setattr(negpip_sd, "shared", SimpleNamespace(sd_model=None))
    unet.blocks = torch.nn.ModuleList()
    gc.collect()
    assert all(ref() is None for ref in refs)  # the patch does not keep the modules alive

    negpip_sd.patch_sd_negpip(None, _NegPiP, unpatch=True)
    assert not _NegPiP._patched[0] and negpip_sd._PATCHED_MODULES == []
