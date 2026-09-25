"""Numerical guard for Anima DiT speed work: a tiny seeded model must keep
producing the same output (float32, CPU)."""

import torch

from backend import operations
from backend.nn.anima import Anima


def _model():
    generator = torch.Generator().manual_seed(1234)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        model = Anima(4, 4, 2, 1, model_channels=256, crossattn_emb_channels=64, adaln_lora_dim=32, num_blocks=2, num_heads=2)
    state = {k: torch.randn(v.shape, generator=generator) * 0.05 for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), generator


def run_reference():
    model, generator = _model()
    x = torch.randn(2, 4, 1, 16, 12, generator=generator)
    t = torch.tensor([0.3, 0.7])
    context = torch.randn(2, 7, 64, generator=generator)
    with torch.inference_mode():
        return model(x, t, context, transformer_options={})


# Recorded from the reference implementation.
EXPECTED = [0.12767835, 0.78065872, 2.86163592, 0.0336951]


def test_anima_output_matches_reference():
    out = run_reference()
    stats = torch.stack([out.mean(), out.std(), out.abs().max(), out.flatten()[::97].sum()])
    assert torch.allclose(stats, torch.tensor(EXPECTED), rtol=1e-4, atol=1e-5), stats.tolist()


def test_cross_attention_projects_other_contexts_like_its_own():
    model, generator = _model()
    attention = model.blocks[0].cross_attn
    x = torch.randn(2, 10, 256, generator=generator)
    context = torch.randn(2, 7, 64, generator=generator)
    seen = {}

    def modifier(next_attention, q, k, v, *, mask=None, transformer_options, is_self_attention):
        seen["project"] = transformer_options.get("anima_project_kv")
        return next_attention(q, k, v, mask=mask)

    with torch.inference_mode():
        _, k, v = attention.compute_qkv(x, context)
        k2, v2 = attention.compute_kv(context)
        attention(x, context, transformer_options={"anima_attention_modifiers": (modifier,)})

    assert torch.equal(k, k2) and torch.equal(v, v2)
    assert seen["project"] is not None


def test_one_reference_latent_serves_a_cfg_batch_of_several_images():
    from backend.args import dynamic_args

    model, generator = _model()
    x = torch.randn(4, 4, 1, 16, 12, generator=generator)  # 2 images x cond/uncond
    ref = torch.randn(1, 4, 1, 16, 12, generator=generator)
    t = torch.full((4,), 0.5)
    context = torch.randn(4, 7, 64, generator=generator)
    saved = list(dynamic_args.ref_latents)
    dynamic_args.ref_latents[:] = [ref]
    try:
        with torch.inference_mode():
            batched = model(x, t, context, transformer_options={})
            single = model(x[1:2], t[1:2], context[1:2], transformer_options={})
    finally:
        dynamic_args.ref_latents[:] = saved
    assert batched.shape == x.shape
    assert torch.allclose(batched[1:2], single, atol=1e-5)
