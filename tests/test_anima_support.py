from types import SimpleNamespace

import torch

from modules.anima_support import (
    NEGPIP_CONTEXT_MASK_KEY,
    NEGPIP_MASK_KEY,
    anima_auxiliary_denoiser,
    is_anima_engine,
    negpip_mask_for,
    register_negpip_prompts,
    split_conditioning,
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




def test_split_conditioning_keeps_the_negpip_mask_separate():
    crossattn = torch.ones(1, 3, 2)
    mask = torch.tensor([[[1.0], [-1.0], [1.0]]])

    split = split_conditioning({"crossattn": crossattn, "c_negpip_mask": mask})
    assert split[0] is crossattn and split[1] is mask
    plain = torch.zeros(1, 3, 2)
    assert split_conditioning(plain) == (plain, None)


def test_negpip_mask_only_applies_to_matching_contexts():
    mask = torch.tensor([[[1.0], [-1.0], [1.0]]])
    options = {NEGPIP_MASK_KEY: mask}

    tiled = negpip_mask_for(torch.zeros(4, 3, 2), options)
    assert tiled.shape == (4, 3, 1)
    assert negpip_mask_for(torch.zeros(4, 5, 2), options) is None
    assert negpip_mask_for(torch.zeros(4, 3, 2), {NEGPIP_MASK_KEY: None}) is None


def test_foreign_contexts_never_inherit_the_base_mask():
    base = torch.tensor([[[1.0], [-1.0], [1.0]]])
    context = torch.zeros(1, 3, 2)

    assert negpip_mask_for(context, {NEGPIP_MASK_KEY: base, NEGPIP_CONTEXT_MASK_KEY: None}) is None
    assert negpip_mask_for(context, {NEGPIP_MASK_KEY: base, "anima_nag_skip": "x"}) is not None  # base pass
    own = negpip_mask_for(context, {NEGPIP_MASK_KEY: base, NEGPIP_CONTEXT_MASK_KEY: -base, "anima_nag_skip": "x"})
    assert own[0, :, 0].tolist() == [-1.0, 1.0, -1.0]


def test_all_positive_masks_are_dropped_and_extra_prompts_registered():
    ones = torch.ones(1, 3, 1)
    assert split_conditioning({"crossattn": torch.zeros(1, 3, 2), "c_negpip_mask": ones})[1] is None

    process = SimpleNamespace()
    register_negpip_prompts(process, "regional", ["(hat:-1)", None, ""])
    register_negpip_prompts(process, "regional", ["(cap:-1)"])  # replaced, not appended
    assert process.negpip_extra_prompts == {"regional": ["(cap:-1)"]}


def test_process_ref_does_not_keep_the_process_alive():
    import gc

    from modules.anima_support import is_anima_auxiliary_denoiser, process_ref

    class Process:
        _anima_auxiliary_denoiser = True

    process = Process()
    proxy = process_ref(process)
    assert is_anima_auxiliary_denoiser(proxy)

    del process
    gc.collect()
    # The hook outlived its generation: no error, just "not a probe".
    assert is_anima_auxiliary_denoiser(proxy) is False


def test_process_ref_falls_back_for_objects_without_weakrefs():
    from modules.anima_support import process_ref

    process = SimpleNamespace(x=1)
    assert process_ref(process) is process
    assert process_ref(None) is None


def test_conditioning_context_unwraps_negpip_dicts():
    from modules.anima_support import conditioning_context

    tensor = torch.zeros(1, 2, 3)
    assert conditioning_context(tensor) is tensor
    assert conditioning_context({"crossattn": tensor, "c_negpip_mask": tensor}) is tensor
    assert conditioning_context({"c_crossattn": tensor}) is tensor


def test_split_conditioning_batches_prompts_of_different_lengths():
    short, long = torch.randn(2, 3), torch.randn(3, 3)
    masks = [torch.tensor([[1.0], [-1.0]]), torch.ones(3, 1)]

    context, mask = split_conditioning({"crossattn": [short, long], "c_negpip_mask": masks})

    assert context.shape == (2, 6, 3) and mask.shape == (2, 6, 1)
    assert torch.equal(context[0], short.repeat(3, 1)) and torch.equal(context[1], long.repeat(2, 1))
    assert mask[0, :, 0].tolist() == [1.0, -1.0] * 3
    # Tiling a whole key/value sequence leaves attention unchanged
    query = torch.randn(1, 4, 3)
    plain = torch.nn.functional.scaled_dot_product_attention(query, short[None], short[None])
    tiled = torch.nn.functional.scaled_dot_product_attention(query, context[:1], context[:1])
    assert torch.allclose(plain, tiled, atol=1e-6)


def test_extra_prompt_encoding_keeps_the_reference_latents(monkeypatch):
    from types import SimpleNamespace

    from backend.args import dynamic_args
    from modules.anima_support import preserved_reference_latents

    reference, initial = torch.ones(1), torch.zeros(1)
    monkeypatch.setattr(dynamic_args, "ref_latents", [reference])
    sd_model = SimpleNamespace(ini_latent=initial)

    with preserved_reference_latents(sd_model):
        dynamic_args.ref_latents.clear()  # what a positive-prompt encode does
        sd_model.ini_latent = None

    assert dynamic_args.ref_latents == [reference] and sd_model.ini_latent is initial
