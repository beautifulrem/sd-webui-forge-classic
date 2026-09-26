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


def test_value_context_projects_only_v_from_the_other_context():
    model, generator = _model()
    attention = model.blocks[0].cross_attn
    x = torch.randn(2, 10, 256, generator=generator)
    context = torch.randn(2, 7, 64, generator=generator)
    mask = torch.where(torch.rand(2, 7, 1, generator=generator) < 0.3, -1.0, 1.0)
    heads = (attention.n_heads, attention.head_dim)

    with torch.inference_mode():
        q, k, v = attention.compute_qkv(x, context, value_context=context * mask)
        # The hand-written NegPiP / FreeFuse projection this replaces
        expected_q = attention.q_norm(attention.q_proj(x).unflatten(-1, heads))
        expected_k = attention.k_norm(attention.k_proj(context).unflatten(-1, heads))
        expected_v = attention.v_norm(attention.v_proj(context * mask).unflatten(-1, heads))

    assert torch.equal(q, expected_q) and torch.equal(k, expected_k) and torch.equal(v, expected_v)


def test_block_computes_silu_emb_itself_when_the_model_does_not_pass_it():
    """Callers other than Anima.forward (external code driving a Block) omit
    ``silu_emb_B_T_D``: the block's own SiLU must give the same output."""
    model, generator = _model()
    captured = {}
    model.blocks[0].register_forward_pre_hook(lambda _module, args, kwargs: captured.update(args=args, kwargs=dict(kwargs)), with_kwargs=True)
    x = torch.randn(2, 4, 1, 16, 12, generator=generator)
    context = torch.randn(2, 7, 64, generator=generator)
    with torch.inference_mode():
        model(x, torch.tensor([0.3, 0.7]), context, transformer_options={})
        args, kwargs = captured["args"], captured["kwargs"]
        passed = model.blocks[0](*args, **kwargs)
        kwargs.pop("silu_emb_B_T_D")
        computed = model.blocks[0](*args, **kwargs)
    assert torch.equal(computed, passed)
