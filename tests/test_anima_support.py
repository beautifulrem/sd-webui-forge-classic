import sys
import types
from types import SimpleNamespace

import pytest
import torch

from modules.anima_support import (
    NEGPIP_CONTEXT_MASK_KEY,
    NEGPIP_MASK_KEY,
    anima_auxiliary_denoiser,
    generated_token_frames,
    is_anima_engine,
    negpip_mask_for,
    pad_to_sequence,
    parse_block_list,
    positive_rows,
    process_ref,
    prompt_schedule,
    prompt_step,
    register_negpip_prompts,
    schedule_index,
    schedule_steps,
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


def test_prompts_whose_lengths_do_not_tile_are_padded_to_the_longest():
    """512 and 600 tokens tiled to their LCM of 38400 (and 512/600/530 to
    ~2M tokens, ~12.5 GB): past ConditionCrossAttn's factor 4, rows are
    padded to the longest the way Anima and NegPiP pad prompts."""

    short, long = torch.randn(512, 4), torch.randn(600, 4)
    short_mask = torch.ones(512, 1)
    short_mask[3] = -1.0
    masks = [short_mask, torch.ones(600, 1)]

    context, mask = split_conditioning({"crossattn": [short, long], "c_negpip_mask": masks})

    assert context.shape == (2, 600, 4) and mask.shape == (2, 600, 1)
    assert torch.equal(context[0, :512], short) and torch.equal(context[1], long)
    assert not context[0, 512:].any()  # zero embeddings, Anima's padding
    assert torch.equal(mask[0, :512], short_mask) and bool((mask[0, 512:] == 1).all())  # neutral
    assert bool((mask[1] == 1).all())

    from modules.anima_support import encode_extra_prompts

    model = SimpleNamespace(get_learned_conditioning=lambda prompt: [torch.ones(1, 512, 4), torch.ones(1, 600, 4), torch.ones(1, 530, 4)])
    context, mask = encode_extra_prompts(SimpleNamespace(sd_model=model), ["a", "b", "c"])
    assert context.shape == (3, 600, 4) and mask is None
    assert [int(row.any(dim=-1).sum()) for row in context] == [512, 600, 530]


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


def test_generated_token_frames_come_from_the_call_input():
    assert generated_token_frames(torch.zeros(2, 16, 1, 8, 8)) == 1
    assert generated_token_frames(torch.zeros(2, 16, 3, 8, 8), patch_temporal=2) == 2
    assert generated_token_frames(torch.zeros(2, 16, 3, 8, 8), patch_temporal=None) == 3
    assert generated_token_frames(torch.zeros(2, 16, 8, 8)) is None


def test_pad_to_sequence_fills_the_reference_tokens():
    mask = torch.tensor([[[0.5], [1.0]]])
    padded = pad_to_sequence(mask, 5)
    assert torch.equal(padded[0, :, 0], torch.tensor([0.5, 1.0, 0.0, 0.0, 0.0]))
    assert pad_to_sequence(mask, 2) is mask
    frames = pad_to_sequence(torch.zeros(1, 1, 2, 2), 3, 1.0, dim=2)
    assert frames.shape == (1, 1, 3, 2) and torch.equal(frames[:, :, 2:], torch.ones(1, 1, 1, 2))
    with pytest.raises(RuntimeError, match="does not match Anima sequence 1"):
        pad_to_sequence(mask, 1)


def test_block_list_accepts_reversed_ranges():
    assert parse_block_list("5-2,7", 8, feature="DAVE") == {2, 3, 4, 5, 7}
    assert parse_block_list(" 3 - 3 ", 8, feature="DAVE") == {3}


@pytest.mark.parametrize("text", ["0-2，5", "０－２，５", "0–2, 5", "0—2,５", "0\u22122,5"])
def test_block_list_reads_ime_input_like_the_weight_fields(text):
    # Full-width forms, "，" and dash variants, as fold_weight_text reads them
    assert parse_block_list(text, 8, feature="Regional") == {0, 1, 2, 5}


@pytest.mark.parametrize("text", ["-1", "a", "1-", "2-x", "1-2-3", "0_5", "1_0-3", "2-1_0"])
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


def _host_reads(action):
    """The host reads (``float``/``item``/``tolist``) of tensors ``action`` makes."""

    from torch.overrides import TorchFunctionMode

    class Count(TorchFunctionMode):
        reads = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            if getattr(func, "__name__", "") in ("__float__", "item", "tolist"):
                Count.reads += 1
            return func(*args, **(kwargs or {}))

    with Count():
        result = action()
    return result, Count.reads


def test_sigma_value_reads_a_timestep_tensor_once_per_wrapper_chain():
    from modules.anima_support import sigma_value

    timestep = torch.tensor([0.75, 0.75], dtype=torch.float64)
    values, reads = _host_reads(lambda: [sigma_value(timestep) for _ in range(6)])
    assert values == [0.75] * 6 and reads == 1

    # The sampler's raw sigma and the batched timestep alternate: both stay cached.
    sigmas = torch.tensor([0.5])
    values, reads = _host_reads(lambda: [sigma_value(tensor) for tensor in (timestep, sigmas) * 3])
    assert values == [0.75, 0.5] * 3 and reads == 1

    # A new tensor with the same value is read again.
    _, reads = _host_reads(lambda: sigma_value(torch.tensor([0.75])))
    assert reads == 1


def test_sigma_value_rereads_a_timestep_changed_in_place():
    from modules.anima_support import sigma_value

    timestep = torch.tensor([0.75, 0.75])
    assert sigma_value(timestep) == 0.75
    timestep.fill_(0.25)
    assert sigma_value(timestep) == 0.25
    timestep[:1].add_(0.5)  # through a view: shares the version counter
    assert sigma_value(timestep) == 0.75
    assert sigma_value([0.5, 1.0]) == 0.5 and sigma_value(2) == 2.0
    with pytest.raises(IndexError):
        sigma_value(torch.tensor([]))


def test_sigma_value_caches_inference_tensors_by_identity():
    import gc

    from modules.anima_support import sigma_value

    with torch.inference_mode():
        timestep = torch.tensor([0.5, 0.5])
    _, reads = _host_reads(lambda: [sigma_value(timestep) for _ in range(3)])
    assert reads == 1
    timestep = None
    gc.collect()
    with torch.inference_mode():
        other = torch.tensor([0.25])  # may reuse the freed tensor's id
    assert sigma_value(other) == 0.25


class _WindowPatcher:
    """The slice of a Forge UNet patcher ``windowed_override_patch`` touches."""

    def __init__(self, options=None):
        self.model = SimpleNamespace(diffusion_model=None)
        self.model_options = dict(options or {})

    def clone(self):
        return _WindowPatcher(self.model_options)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def _window_patch(model, module, tag, log, *, outermost=False, admit=True):
    import threading

    from modules.anima_support import windowed_override_patch

    def make(original):
        return lambda x: [tag, *original(x)]

    return windowed_override_patch(
        model,
        gate=lambda args: args["input"] if admit else None,
        begin=lambda token: log.append(("begin", tag)),
        overrides=lambda token: [(module, make(module.forward))],
        end=lambda: log.append(("end", tag)),
        lock=threading.RLock(),
        force_actual=lambda sigma: sigma > 0.5,
        outermost=outermost,
    )


def _window_args():
    return {"input": torch.zeros(1), "timestep": torch.ones(1), "c": {}}


class _Leaf(torch.nn.Module):
    def forward(self, x):
        return ["leaf"]


def test_windowed_override_patch_installs_for_admitted_calls_only():
    from modules import spectrum_force

    module, log = _Leaf(), []
    base = _WindowPatcher()
    patched = _window_patch(base, module, "a", log)
    wrapper = patched.model_options["model_function_wrapper"]
    assert patched is not base and "model_function_wrapper" not in base.model_options
    assert wrapper(lambda x, t, **c: module(x), _window_args()) == ["a", "leaf"]
    assert log == [("begin", "a"), ("end", "a")] and "forward" not in module.__dict__
    assert wrapper.__spectrum_cache_safe__ and wrapper.__resolution_independent__
    force = patched.model_options["transformer_options"][spectrum_force.KEY]
    assert spectrum_force.is_forced(force, 1.0) and not spectrum_force.is_forced(force, 0.25)

    log.clear()
    passing = _window_patch(base, module, "a", log, admit=False).model_options["model_function_wrapper"]
    assert passing(lambda x, t, **c: module(x), _window_args()) == ["leaf"] and log == []


@pytest.mark.parametrize("outermost", [False, True])
def test_windowed_override_patch_restores_and_ends_when_the_call_raises(outermost):
    module, log = _Leaf(), []
    wrapper = _window_patch(_WindowPatcher(), module, "a", log, outermost=outermost).model_options["model_function_wrapper"]

    def model_function(x, t, **c):
        assert "forward" in module.__dict__
        raise RuntimeError("model call failed")

    with pytest.raises(RuntimeError, match="model call failed"):
        wrapper(model_function, _window_args())
    assert "forward" not in module.__dict__ and log == [("begin", "a"), ("end", "a")]


def test_windowed_override_patch_outermost_wraps_the_others_in_either_order():
    module = _Leaf()

    def run(*patches):
        model = _WindowPatcher()
        for tag, outermost in patches:
            model = _window_patch(model, module, tag, [], outermost=outermost)
        out = model.model_options["model_function_wrapper"](lambda x, t, **c: module(x), _window_args())
        assert "forward" not in module.__dict__
        return out

    # Without it the wrapper that runs first (applied last) installs first: innermost.
    assert run(("a", False), ("b", False)) == ["a", "b", "leaf"]
    assert run(("a", True), ("b", False)) == run(("b", False), ("a", True)) == ["a", "b", "leaf"]


@pytest.fixture
def samplers(monkeypatch):
    """``modules.sd_samplers`` with Euler and a second-order Heun (the real
    module needs the full WebUI install)."""

    import modules

    class Config:
        def __init__(self, second_order):
            self.second_order = second_order

        def total_steps(self, steps):
            return steps * 2 if self.second_order else steps

    configs = {"Euler": Config(False), "Heun": Config(True)}
    stub = types.ModuleType("modules.sd_samplers")
    stub.find_sampler_config = lambda name: configs["Euler"] if name is None else configs.get(name)
    monkeypatch.setitem(sys.modules, "modules.sd_samplers", stub)
    monkeypatch.setattr(modules, "sd_samplers", stub, raising=False)


def test_schedule_steps_are_the_ones_processing_builds_each_pass_conds_with(samplers):
    # Base pass: setup_conds' total steps (a second-order sampler doubles them)
    assert schedule_steps(SimpleNamespace(steps=10, firstpass_steps=20, sampler_name="Heun")) == (20, None)
    assert schedule_steps(SimpleNamespace(steps=10, sampler_name="Heun")) == (20, None)
    # Hires pass: calculate_hr_conds' base firstpass_steps and the hires sampler's total steps
    hires = SimpleNamespace(steps=10, firstpass_steps=10, sampler_name="Euler", is_hr_pass=True, hr_second_pass_steps=0, hr_sampler_name="Heun")
    assert schedule_steps(hires) == (10, 20)
    hires.hr_second_pass_steps, hires.hr_sampler_name = 4, None
    assert schedule_steps(hires) == (10, 4)


def test_prompt_schedule_resolves_prompt_editing_for_the_base_and_hires_pass():
    assert prompt_schedule(["a [cat:dog:5]"], 10) == [(5, ("a cat",)), (10, ("a dog",))]
    assert prompt_schedule(["[red|blue]"], 4) == [(1, ("red",)), (2, ("blue",)), (3, ("red",)), (4, ("blue",))]
    # In a hires pass whole steps count on from the base pass's, fractions span both passes
    assert prompt_schedule(["a [cat:dog:15]"], 10, 10) == [(5, ("a cat",)), (10, ("a dog",))]
    assert prompt_schedule(["a [cat:dog:5]"], 10, 10) == [(10, ("a dog",))]
    assert prompt_schedule(["a [cat:dog:0.5]"], 10, 10) == [(10, ("a dog",))]
    assert prompt_schedule(["a [cat:dog:1.5]"], 10, 10) == [(5, ("a cat",)), (10, ("a dog",))]
    # Without prompt editing the text is kept as written, emphasis and escapes included
    for plain in ("a (red:1.2) cat, [soft]", r"\[literal\] BREAK x", "a [unbalanced", ""):
        assert prompt_schedule([plain], 10) == [(10, (plain,))]


def test_a_batch_schedule_merges_the_switch_steps_of_its_prompts():
    assert prompt_schedule(["a [cat:dog:5]", "plain", "[x:y:3]"], 10) == [
        (3, ("a cat", "plain", "x")),
        (5, ("a cat", "plain", "y")),
        (10, ("a dog", "plain", "y")),
    ]
    assert prompt_schedule(["[a:a:5]"], 10) == [(10, ("a",))]  # no change of text: one entry


def test_schedule_index_picks_the_entry_reconstruct_cond_batch_picks():
    from modules import prompt_parser

    ends = (3, 5, 10)
    schedule = [[prompt_parser.ScheduledPromptConditioning(end, torch.tensor([float(index)])) for index, end in enumerate(ends)]]
    for step in range(13):  # past the last end Forge falls back to the first entry
        assert schedule_index(ends, step) == int(prompt_parser.reconstruct_cond_batch(schedule, step)[0])


def test_prompt_step_is_the_cfg_denoiser_step_of_the_current_sampler():
    import gc

    class Process:
        pass

    process = Process()
    process.sampler = SimpleNamespace(model_wrap_cfg=SimpleNamespace(step=7))
    reference = process_ref(process)
    assert prompt_step(process) == prompt_step(reference) == 7
    assert prompt_step(SimpleNamespace(sampler=None)) == 0
    del process
    gc.collect()
    assert prompt_step(reference) == 0  # the generation has ended
