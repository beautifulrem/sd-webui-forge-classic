"""FreeFuse's cross-attention override composes with the Cross-Attention
Boost (both wrapper orders) and Regional, through the real wrappers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

EXTENSIONS = Path(__file__).resolve().parents[2]
for name in ("sd_forge_anima_freefuse", "sd_forge_anima_xattn_boost", "sd_forge_anima_regional"):
    if str(EXTENSIONS / name) not in sys.path:
        sys.path.insert(0, str(EXTENSIONS / name))

from backend import operations  # noqa: E402
from backend.nn.anima import Block, _fn  # noqa: E402
from lib_anima_freefuse import runtime  # noqa: E402
from lib_anima_regional.regional import Region, RegionalState, apply_regional_patch  # noqa: E402
from lib_anima_xattn_boost.boost import BoostState, apply_xattn_boost_patch  # noqa: E402

B, T, H, W, D, CTX, TEXT = 2, 1, 3, 4, 32, 16, 5
MARKERS = [0, 1]  # row 0 positive, row 1 negative


class _Patcher:
    def __init__(self, dit, options=None):
        self.model = SimpleNamespace(diffusion_model=dit)
        self.model_options = dict(options or {})

    def clone(self):
        return _Patcher(self.model.diffusion_model, self.model_options)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def set_transformer_option(self, key, value):
        options = dict(self.model_options.get("transformer_options", {}))
        options[key] = value
        self.model_options["transformer_options"] = options


def _dit():
    generator = torch.Generator().manual_seed(0)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        block = Block(D, CTX, num_heads=2, adaln_lora_dim=8)
    block.load_state_dict({k: torch.randn(v.shape, generator=generator) * 0.2 for k, v in block.state_dict().items()})
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([block.eval()])
    dit.patch_spatial = dit.patch_temporal = 1
    return dit


def _inputs():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(B, T, H, W, D, generator=generator) * 3.0
    emb = torch.randn(B, T, D, generator=generator)
    context = torch.randn(B, TEXT, CTX, generator=generator)
    lora = torch.randn(B, T, 3 * D, generator=generator) * 0.1
    return x, emb, context, lora


def _state(**overrides):
    state = runtime.AnimaFreeFuseState(
        adapters=[runtime.AdapterSpec("A", "cat", "cat"), runtime.AdapterSpec("B", "dog", "dog")],
        token_positions={"A": [1], "B": [3]},
        background_positions=[4],
        collect_step=0,
        collect_block=-1,
        top_k_ratio=0.25,
        temperature=1.0,
        bg_scale=1.0,
        balance_iterations=1,
        feather=0,
        routing_strength=1.0,
        routing_end=1.0,
        bias_scale=6.0,
        positive_bias=1.0,
        bias_blocks={0},
        **overrides,
    )
    # No LoRA is loaded here: only the cross-attention part runs
    state.validate_patches = lambda patcher: None
    left = torch.tensor([[1.0, 1.0, 0.0, 0.0] * H])
    state.masks = {"A": left, "B": 1.0 - left}
    state.begin_generate()
    return state


def _args():
    options = {"cond_or_uncond": list(MARKERS)}
    return {"input": torch.zeros(B, 16, T, H, W), "timestep": torch.tensor([0.9] * B), "c": {"transformer_options": options}, "cond_or_uncond": list(MARKERS)}


def _model_function(dit, x, emb, context, lora):
    def run(_input, _t, transformer_options=None, **c):
        h = x
        for block in dit.blocks:
            h = block(h, emb, context, adaln_lora_B_T_3D=lora, transformer_options=transformer_options, silu_emb_B_T_D=torch.nn.functional.silu(emb))
        return h

    return run


def _reference(block, x, emb, context, lora, gain, options):
    """The block with its cross-attention residual scaled by ``gain`` on the
    positive row (the boost without norm matching); ``block.cross_attn`` runs
    whatever override is installed on it."""

    silu = torch.nn.functional.silu(emb)

    def modulation(mlp):
        return [chunk[:, :, None, None, :] for chunk in (mlp[2](mlp[1](silu)) + lora).chunk(3, dim=-1)]

    shift_sa, scale_sa, gate_sa = modulation(block.adaln_modulation_self_attn)
    shift_ca, scale_ca, gate_ca = modulation(block.adaln_modulation_cross_attn)
    shift_mlp, scale_mlp, gate_mlp = modulation(block.adaln_modulation_mlp)
    shape = x.shape[1:4]
    x = x + gate_sa * block.self_attn(_fn(x, block.layer_norm_self_attn, scale_sa, shift_sa).flatten(1, 3), None).unflatten(1, shape)
    cross = block.cross_attn(_fn(x, block.layer_norm_cross_attn, scale_ca, shift_ca).flatten(1, 3), context, transformer_options=options).unflatten(1, shape)
    x = x + cross * gate_ca * torch.tensor([gain, 1.0]).view(-1, 1, 1, 1, 1)
    return x + gate_mlp * block.mlp(_fn(x, block.layer_norm_mlp, scale_mlp, shift_mlp))


def _run(wrap):
    dit = _dit()
    inputs = _inputs()
    wrapper = wrap(_Patcher(dit)).model_options.get("model_function_wrapper", lambda function, args: function(None, None, **args["c"]))
    with torch.no_grad():
        return dit, inputs, wrapper(_model_function(dit, *inputs), _args())


def _freefuse_reference(gain):
    """FreeFuse alone installed around the reference block math."""

    dit = _dit()
    x, emb, context, lora = _inputs()
    wrapper = runtime.apply_freefuse_patch(_Patcher(dit), _state()).model_options["model_function_wrapper"]
    options = {"cond_or_uncond": list(MARKERS)}
    with torch.no_grad():
        return wrapper(lambda *a, **k: _reference(dit.blocks[0], x, emb, context, lora, gain, options), _args())


def _boost(model):
    return apply_xattn_boost_patch(model, BoostState(gain=2.0, band=0.85, renorm=0.0))


def _freefuse(model):
    return runtime.apply_freefuse_patch(model, _state())


def test_freefuse_bias_changes_the_positive_row_only():
    _, _, plain = _run(lambda model: model)
    _, _, fused = _run(_freefuse)
    assert torch.equal(fused[1], plain[1])  # negative row: no bias
    assert not torch.allclose(fused[0], plain[0])
    torch.testing.assert_close(fused, _freefuse_reference(1.0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("order", ["boost outer", "freefuse outer"])
def test_freefuse_and_the_boost_compose_in_either_wrapper_order(order):
    # The outer wrapper installs its override first; the inner one's goes on
    # top and is called first.
    if order == "boost outer":
        wrap = lambda model: _boost(_freefuse(model))  # noqa: E731
    else:
        wrap = lambda model: _freefuse(_boost(model))  # noqa: E731
    dit, _, composed = _run(wrap)
    _, _, boost_only = _run(_boost)
    _, _, freefuse_only = _run(_freefuse)

    torch.testing.assert_close(composed, _freefuse_reference(2.0), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(composed[0], freefuse_only[0])  # the boost ran
    assert not torch.allclose(composed[0], boost_only[0])  # the bias ran
    assert torch.equal(composed[1], freefuse_only[1])
    for module in dit.modules():
        assert not {"forward", "compute_qkv", "compute_attention"} & set(module.__dict__)


def test_freefuse_leaves_a_regional_context_alone():
    # A region covering the whole image at strength 1 replaces the positive
    # row's cross-attention with the region context's; FreeFuse's bias and
    # collection describe the base prompt's tokens, so only the base call
    # gets them (whichever wrapper installs first).
    region_context = torch.randn(1, TEXT, CTX, generator=torch.Generator().manual_seed(5))

    def regional(model):
        state = RegionalState(regions=[Region(region_context, 0.0, 0.0, 1.0, 1.0, 1.0)], blocks={0}, feather=0.0, base_preserve=0.0, start_sigma=100.0, end_sigma=0.0)
        return apply_regional_patch(model, state)

    _, _, regional_only = _run(regional)
    for wrap in (lambda model: _freefuse(regional(model)), lambda model: regional(_freefuse(model))):
        _, _, composed = _run(wrap)
        # The positive row is the region's output, which carries no bias
        torch.testing.assert_close(composed[0], regional_only[0], rtol=1e-5, atol=1e-5)
