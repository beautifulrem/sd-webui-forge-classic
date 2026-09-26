from types import SimpleNamespace

import pytest
import torch

from modules.anima_support import (
    NEGPIP_CONTEXT_MASK_KEY,
    NEGPIP_MASK_KEY,
    anima_auxiliary_denoiser,
    is_anima_engine,
    negpip_mask_for,
    parse_block_list,
    positive_rows,
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


class _EncodingModel:
    """An Anima engine stand-in: ``(1, tokens, dim)`` per prompt, padded to
    different lengths, resetting the reference latents like a positive encode."""

    current_lora_hash = "[]"

    def __init__(self):
        self.calls = []
        self.ini_latent = torch.zeros(1)
        self.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=SimpleNamespace(patches_uuid="te-1")))

    def get_learned_conditioning(self, prompt):
        from backend.args import dynamic_args

        self.calls.append(list(prompt))
        dynamic_args.ref_latents.clear()
        self.ini_latent = None
        return [torch.full((1, 2 * (index + 1), 3), float(index)) for index in range(len(prompt))]


def test_extra_prompts_are_encoded_as_one_cpu_batch(monkeypatch):
    from backend.args import dynamic_args
    from modules.anima_support import encode_extra_prompts

    reference = torch.ones(1)
    monkeypatch.setattr(dynamic_args, "ref_latents", [reference])
    model = _EncodingModel()
    initial = model.ini_latent

    context, mask = encode_extra_prompts(SimpleNamespace(sd_model=model, width=64, height=64), ["artist\na cat", "artist\na dog"])

    assert model.calls == [["artist\na cat", "artist\na dog"]]
    # Different padded lengths are tiled to a common one
    assert context.shape == (2, 4, 3) and context.device.type == "cpu" and mask is None
    assert context[0].unique().tolist() == [0.0] and context[1].unique().tolist() == [1.0]
    assert dynamic_args.ref_latents == [reference] and model.ini_latent is initial

    negpip = {"crossattn": torch.zeros(1, 2, 3), "c_negpip_mask": torch.tensor([[[1.0], [-1.0]]])}
    model.get_learned_conditioning = lambda prompt: negpip
    context, mask = encode_extra_prompts(SimpleNamespace(sd_model=model), ["(hat:-1)"])
    assert context.shape == (1, 2, 3) and mask[0, :, 0].tolist() == [1.0, -1.0]


def test_conditioning_cache_key_follows_what_the_encoding_depends_on(monkeypatch):
    import modules
    from modules.anima_support import conditioning_cache_key

    model_data = SimpleNamespace(forge_hash="checkpoint+te")
    monkeypatch.setattr(modules, "sd_models", SimpleNamespace(model_data=model_data), raising=False)
    model = _EncodingModel()
    p = SimpleNamespace(sd_model=model)
    first = conditioning_cache_key(p)
    assert conditioning_cache_key(p) == first

    # Qwen layer weights re-apply the TE LoRA patches without changing its hash
    model.forge_objects.clip.patcher.patches_uuid = "te-2"
    second = conditioning_cache_key(p)
    model_data.forge_hash = "checkpoint+other-te"
    third = conditioning_cache_key(p)
    model.get_learned_conditioning = model.get_learned_conditioning  # NegPiP's hook
    assert len({first, second, third, conditioning_cache_key(p)}) == 4


def test_negpip_ignores_extra_prompts_of_a_feature_that_yields():
    from modules.anima_feature_conflicts import ARTIST_MIXER, FREEFUSE, REGIONAL, set_feature_active
    from modules.anima_support import negpip_extra_prompts

    process = SimpleNamespace()
    register_negpip_prompts(process, ARTIST_MIXER, ["artist"])
    register_negpip_prompts(process, "other", ["other"])
    assert negpip_extra_prompts(process) == ["artist", "other"]  # nothing marked yet

    register_negpip_prompts(process, REGIONAL, ["region"])
    set_feature_active(process, REGIONAL)
    assert negpip_extra_prompts(process) == ["other", "region"]

    set_feature_active(process, REGIONAL, False)
    set_feature_active(process, FREEFUSE)
    assert negpip_extra_prompts(process) == ["other"]


def test_block_list_accepts_reversed_ranges():
    assert parse_block_list("5-2,7", 8, feature="DAVE") == {2, 3, 4, 5, 7}
    assert parse_block_list(" 3 - 3 ", 8, feature="DAVE") == {3}


@pytest.mark.parametrize("text", ["-1", "a", "1-", "2-x", "1-2-3"])
def test_block_list_names_the_feature_and_the_bad_part(text):
    with pytest.raises(ValueError, match=rf"DAVE block selection .*'{text}'"):
        parse_block_list(f"0,{text}", 8, feature="DAVE")


def test_positive_rows_pick_the_cond_rows_of_a_cfg_batch():
    cache = {}
    branch = positive_rows([1, 0], 4, cache, "cpu")  # uncond rows 0-1, cond rows 2-3
    assert branch.rows == (2, 3) and branch.index.tolist() == [2, 3]
    assert positive_rows((1, 0), 4, cache, "cpu").index is branch.index  # one index tensor per rows and device

    batch = torch.arange(4.0).view(4, 1)
    assert branch.select(batch).flatten().tolist() == [2.0, 3.0]
    assert branch.merge(batch, torch.full((2, 1), -1.0)).flatten().tolist() == [0.0, 1.0, -1.0, -1.0]


@pytest.mark.parametrize("markers, batch, every_row", [([0], 2, False), ([0, 1, 0], 4, False), ([], 2, False), (None, 2, False), ([0, 1], 4, True), ([1], 2, True)])
def test_positive_rows_cover_the_batch_when_the_markers_do(markers, batch, every_row):
    # All positive, markers that do not describe the batch, or every row asked for
    branch = positive_rows(markers, batch, {}, "cpu", every_row=every_row)
    tensor = torch.ones(batch, 1)
    assert branch.rows is None and branch.select(tensor) is tensor and branch.merge(None, tensor) is tensor


def test_positive_rows_are_none_without_a_positive_row():
    assert positive_rows([1], 2, {}, "cpu") is None


def test_branch_options_describe_the_rows_and_the_swapped_in_context():
    base_mask, context_mask = torch.ones(4, 3, 1), -torch.ones(2, 5, 1)
    options = {"cond_or_uncond": [1, 0], NEGPIP_MASK_KEY: base_mask, "other": 1}

    rows = positive_rows(options["cond_or_uncond"], 4, {}, "cpu").options(options, "regional_context", context_mask)
    assert rows == {"cond_or_uncond": [0], NEGPIP_MASK_KEY: base_mask, NEGPIP_CONTEXT_MASK_KEY: context_mask, "anima_nag_skip": "regional_context", "other": 1}
    assert options["cond_or_uncond"] == [1, 0] and NEGPIP_CONTEXT_MASK_KEY not in options  # not modified

    every = positive_rows([0, 1], 4, {}, "cpu", every_row=True).options(options, "artist_context", None)
    assert every["cond_or_uncond"] == [1, 0] and every[NEGPIP_CONTEXT_MASK_KEY] is None
    assert positive_rows(None, 2, {}, "cpu").options(None, "artist_context", None) == {"anima_nag_skip": "artist_context", NEGPIP_CONTEXT_MASK_KEY: None}


def test_wrapper_flags_hold_only_along_the_whole_chain():
    from modules.anima_support import WRAPPER_FLAGS, inherit_wrapper_flags

    def make():
        return lambda function, args: None

    first = inherit_wrapper_flags(make(), None)
    assert all(getattr(first, flag) for flag in WRAPPER_FLAGS)

    spatial = inherit_wrapper_flags(make(), first, resolution_independent=False)
    assert spatial.__spectrum_cache_safe__ and not spatial.__resolution_independent__

    outer = inherit_wrapper_flags(make(), spatial)
    assert outer.__spectrum_cache_safe__ and not outer.__resolution_independent__
    unmarked = inherit_wrapper_flags(make(), make())  # a foreign wrapper declares nothing
    assert not any(getattr(unmarked, flag) for flag in WRAPPER_FLAGS)
