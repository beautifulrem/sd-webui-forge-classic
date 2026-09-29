import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_artist_scheduled_mixer.py"


def _load():
    import modules

    try:
        from modules import sd_models  # noqa: F401
    except ImportError:
        # The model loader needs the full WebUI install; the dispatch does not use it
        modules.sd_models = sys.modules["modules.sd_models"] = types.ModuleType("modules.sd_models")
    spec = importlib.util.spec_from_file_location("anima_artist_mixer_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mixer = _load()


class _Attention:
    """Row-independent stand-in for a cross-attention forward that records calls."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, x, context=None, rope_emb=None, transformer_options={}):
        self.calls.append((x.shape[0], context, dict(transformer_options)))
        if self.fail and transformer_options.get("anima_nag_skip"):
            raise RuntimeError("artist forward failed")
        return attend(x, context)


def attend(x, context):
    return torch.tanh(x * context.mean(dim=(1, 2)).view(-1, 1, 1) + context.sum(dim=(1, 2)).view(-1, 1, 1) * 0.01)


def _state(combine=mixer.COMBINE_OUTPUT_AVG, fusion=mixer.FUSION_INTERPOLATE, apply_uncond=False, batched=False, weights=(0.6, 0.3)):
    torch.manual_seed(0)
    artists = [
        mixer.ArtistRuntime(f"artist{index}", weight, {0}, 0.0, 1.0, 0.5, mixer.CURVE_HOLD, torch.randn(1, 3, 4))
        for index, weight in enumerate(weights)
    ]
    # Output average batches its artist forwards until that fails once:
    # batched=False takes the serial path it falls back to.
    return mixer.MixerState("run", True, 0.8, combine, fusion, apply_uncond, artists, batched_disabled=not batched)


def _inputs(batch=2):
    torch.manual_seed(1)
    return torch.randn(batch, 6, 4), torch.randn(batch, 3, 4)


def _dispatch(state, forward, x, context, markers):
    return mixer._dispatch_cross_attn(forward, 0, state, x, context=context, transformer_options={"cond_or_uncond": markers})


@pytest.mark.parametrize("batched", [False, True])
def test_output_avg_blends_weighted_artists_into_positive_rows(batched):
    state = _state(batched=batched)
    x, context = _inputs()
    out = _dispatch(state, _Attention(), x, context, [0, 1])

    base = attend(x, context)
    weights = [0.6 / 0.9, 0.3 / 0.9]
    artist = sum(w * attend(x, a.cond.expand(2, -1, -1)) for w, a in zip(weights, state.artists))
    strength = 0.8 * 0.9
    assert torch.allclose(out[0], base[0] * (1.0 - strength) + artist[0] * strength, atol=1e-6)
    assert torch.equal(out[1], base[1])  # negative-prompt row
    assert state.active_calls == 1


def test_batched_and_serial_artist_forwards_agree():
    x, context = _inputs()
    serial = _dispatch(_state(batched=False), _Attention(), x, context, [0, 1])
    forward = _Attention()
    batched = _dispatch(_state(batched=True), forward, x, context, [0, 1])
    assert torch.allclose(serial, batched, atol=1e-6)
    # One forward for both artists on the positive row (repeated per artist), one for the base
    assert [calls[0] for calls in forward.calls] == [2, 2]
    assert forward.calls[0][2]["cond_or_uncond"] == [0, 0]


@pytest.mark.parametrize("combine", [mixer.COMBINE_OUTPUT_AVG, mixer.COMBINE_CONCAT])
def test_calls_without_a_row_to_mix_skip_the_artist_forwards(combine):
    state = _state(combine=combine)
    x, context = _inputs()
    forward = _Attention()
    out = _dispatch(state, forward, x, context, [1])

    assert len(forward.calls) == 1 and forward.calls[0][1] is context
    assert torch.equal(out, attend(x, context))


def test_apply_uncond_mixes_every_row():
    state = _state(apply_uncond=True)
    x, context = _inputs()
    out = _dispatch(state, _Attention(), x, context, [0, 1])
    base = attend(x, context)
    assert not torch.equal(out[1], base[1])


@pytest.mark.parametrize("fusion", [mixer.FUSION_INTERPOLATE, mixer.FUSION_CONCAT_WITH_BASE])
def test_concat_attends_to_weighted_artist_tokens(fusion):
    state = _state(combine=mixer.COMBINE_CONCAT, fusion=fusion)
    x, context = _inputs()
    forward = _Attention()
    out = _dispatch(state, forward, x, context, [0, 1])

    combined = torch.cat([a.cond * (a.weight / 0.9) for a in state.artists], dim=1)
    kv = torch.cat([context[:1], combined], dim=1) if fusion == mixer.FUSION_CONCAT_WITH_BASE else combined
    # The artist forward runs on the positive-prompt row only
    assert forward.calls[0][0] == 1 and torch.allclose(forward.calls[0][1], kv)
    assert forward.calls[0][2]["anima_nag_skip"] == "artist_context"
    assert forward.calls[0][2]["cond_or_uncond"] == [0]
    strength = 0.8 * 0.9
    base = attend(x, context)
    assert torch.allclose(out[0], base[0] * (1.0 - strength) + attend(x[:1], kv)[0] * strength, atol=1e-6)
    assert torch.equal(out[1], base[1])


def test_quality_delta_limits_the_artist_delta():
    state = _state(fusion=mixer.FUSION_QUALITY_DELTA)
    x, context = _inputs()
    out = _dispatch(state, _Attention(), x, context, [0, 1])

    base = attend(x, context)
    weights = [0.6 / 0.9, 0.3 / 0.9]
    artist = sum(w * attend(x, a.cond.expand(2, -1, -1)) for w, a in zip(weights, state.artists))
    expected = mixer._norm_limited_delta(base, artist, 0.8 * 0.9)
    assert torch.allclose(out[0], expected[0], atol=1e-6)
    assert torch.equal(out[1], base[1])


def test_zero_weights_and_inactive_blocks_fall_back_to_the_base_forward():
    x, context = _inputs()
    forward = _Attention()
    out = _dispatch(_state(weights=(0.0, 0.0)), forward, x, context, [0, 1])
    assert len(forward.calls) == 1 and torch.equal(out, attend(x, context))

    state = _state()
    forward = _Attention()
    out = mixer._dispatch_cross_attn(forward, 5, state, x, context=context, transformer_options={})
    assert len(forward.calls) == 1 and state.fallback_calls == 1


def test_a_failing_artist_forward_turns_the_mixer_off_for_the_run():
    state = _state()
    state.infotext = {}
    x, context = _inputs()
    out = _dispatch(state, _Attention(fail=True), x, context, [0, 1])

    assert torch.equal(out, attend(x, context))
    assert not state.enabled
    assert state.infotext["Anima Artist Mixer"] == "disabled: RuntimeError at block 0"


def test_artist_mixer_yields_to_regional_in_headless_runs():
    from types import SimpleNamespace

    from modules.anima_feature_conflicts import ARTIST_MIXER, REGIONAL, is_feature_active, set_feature_active

    p = SimpleNamespace(is_api=True, extra_generation_params={})
    set_feature_active(p, REGIONAL)
    args = [True, 1, 1, False, False, 0, 0, 1.0, "balance", mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE, False, False]
    mixer.Script().process_before_every_sampling(p, *args)

    message = "disabled: Anima Regional Conditioning has priority in headless processing"
    assert p.extra_generation_params["Anima Artist Mixer"] == message
    assert not is_feature_active(p, ARTIST_MIXER)


class _CrossAttn(torch.nn.Module):
    context_dim = 4

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        return attend(x, context)


class _Unet:
    """Anima-shaped patcher stand-in: one block whose cross-attn can be patched."""

    def __init__(self):
        self.model = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(blocks=[types.SimpleNamespace(cross_attn=_CrossAttn())]))
        self.model_options = {}

    def clone(self):
        return self


def _run_args(enable_hires_rows=False, disable_hires_mixing=False):
    width = len(mixer.TABLE_HEADERS)
    row = lambda artist: [True, artist, 1.0, "0-51", 0.0, 1.0, 0.5, mixer.CURVE_HOLD, mixer.PRESET_CUSTOM, False]
    base = row("(base artist:-1)") + [None] * (width * (mixer.MAX_ARTIST_ROWS - 1))
    hires = row("hires artist") + [None] * (width * (mixer.MAX_ARTIST_ROWS - 1))
    head = [True, 1, 1, enable_hires_rows, disable_hires_mixing, 3.0, 3.0, 0.7, mixer.OPT_BALANCE, mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE, False, False]
    return head + base + hires


def _run_p(**kwargs):
    from types import SimpleNamespace

    return SimpleNamespace(is_api=True, extra_generation_params={}, sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=_Unet())), **kwargs)


def test_superseded_and_finished_runs_release_their_device_tensors(monkeypatch):
    monkeypatch.setattr(mixer, "_build_artists", lambda *args: _state().artists)
    p = _run_p()
    x, context = _inputs()

    def run_and_use():
        mixer.Script().process_before_every_sampling(p, *_run_args())
        state = mixer._RUN_STATES[-1]
        _dispatch(state, _Attention(), x, context, [1, 0])  # fills the per-call caches
        assert state.call_contexts and state.row_masks and all(artist.on_device for artist in state.artists)
        return state

    try:
        first = run_and_use()
        second = run_and_use()  # e.g. the hires pass
        assert mixer._RUN_STATES == [second]
        assert not first.call_contexts and not first.row_masks and not any(artist.on_device for artist in first.artists)

        mixer.Script().postprocess(p, None, *_run_args())
        assert not second.call_contexts and not any(artist.on_device for artist in second.artists)
        assert second.dispatch_calls == 1  # the counters stay for the summary
    finally:
        mixer._cleanup_runtime()


def _sampling_pass(monkeypatch, **kwargs):
    """A pass set up by process_before_every_sampling: ``(p, unet, cross_attn)``."""
    monkeypatch.setattr(mixer, "_build_artists", lambda *args: _state().artists)
    p = _run_p(**kwargs)
    mixer.Script().process_before_every_sampling(p, *_run_args())
    unet = p.sd_model.forge_objects.unet
    return p, unet, unet.model.diffusion_model.blocks[0].cross_attn


def _model_call(unet, attention, apply_model=None):
    """Run one model call through the unet's wrapper; ``(patched during the call, attention output)``."""
    x, context = _inputs()
    seen = {}

    def default_apply_model(input, timestep, **c):
        seen["patched"] = "forward" in attention.__dict__
        seen["out"] = attention(x, context=context, transformer_options={"cond_or_uncond": [0, 1]})
        return input

    unet.model_options["model_function_wrapper"](apply_model or default_apply_model, {"input": x, "timestep": torch.zeros(1), "c": {}})
    return seen.get("patched"), seen.get("out")


def test_artist_overrides_exist_only_during_a_model_call(monkeypatch):
    p, unet, attention = _sampling_pass(monkeypatch)
    try:
        assert "forward" not in attention.__dict__  # nothing is installed for the whole pass
        patched, out = _model_call(unet, attention)
        assert patched and "forward" not in attention.__dict__

        x, context = _inputs()
        base = attend(x, context)
        assert not torch.equal(out[0], base[0]) and torch.equal(out[1], base[1])  # mixed into the positive row
        state = mixer._RUN_STATES[-1]
        assert state.wrapper_checks == 1 and state.active_calls == 1 and state.patched_blocks == [0]
    finally:
        mixer._cleanup_runtime()


def test_artist_overrides_are_restored_when_the_model_call_raises(monkeypatch):
    p, unet, attention = _sampling_pass(monkeypatch)

    def interrupted(input, timestep, **c):
        assert "forward" in attention.__dict__
        raise RuntimeError("interrupted")

    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            _model_call(unet, attention, interrupted)
        assert "forward" not in attention.__dict__
    finally:
        mixer._cleanup_runtime()


def test_nothing_is_mixed_after_sampling_not_even_in_an_adetailer_inner_run(monkeypatch):
    # ADetailer's inner runs (postprocess_image) skip this script by default,
    # or yield to it (_ad_inner); they must not inherit the artist overrides,
    # even through the outer pass's patcher.
    p, unet, attention = _sampling_pass(monkeypatch)
    state = mixer._RUN_STATES[-1]
    try:
        mixer.Script().postprocess_batch(p, images=None, batch_number=0)
        inner = _run_p(_ad_inner=True)
        mixer.Script().process_before_every_sampling(inner, *_run_args())
        assert "model_function_wrapper" not in inner.sd_model.forge_objects.unet.model_options

        patched, out = _model_call(unet, attention)
        x, context = _inputs()
        assert not patched and torch.equal(out, attend(x, context))
        assert state.wrapper_checks == 0 and state.dispatch_calls == 0
        assert mixer._RUN_STATES == [state]  # kept for postprocess's run summary
    finally:
        mixer._cleanup_runtime()


def test_an_artist_batch_the_attention_batch_does_not_tile_turns_the_mixer_off():
    # Two prompts' artist contexts cannot be spread over three rows; every
    # row used to get the first prompt's artist context.
    state = _state()
    for artist in state.artists:
        artist.cond = artist.cond.expand(2, -1, -1)
    state.infotext = {}
    x, context = _inputs(batch=3)
    out = _dispatch(state, _Attention(), x, context, [0])

    assert torch.equal(out, attend(x, context))
    assert not state.enabled and state.infotext["Anima Artist Mixer"] == "disabled: RuntimeError at block 0"


@pytest.mark.parametrize("total, expected", [(28, range(10, 19)), (40, range(15, 28)), (52, range(20, 37))])
def test_blocks_fields_are_written_in_the_2b_layout(total, expected):
    # Performance's "10-18" is the same depth band on every model size; a
    # block brings its inserted copies along
    assert mixer._model_blocks(mixer._default_blocks_for_optimization(mixer.OPT_PERFORMANCE), total) == set(expected)
    assert mixer._model_blocks("0-51", total) == set(range(total))
    assert mixer._model_blocks("-1", total) == {total - 1}
    assert mixer._model_blocks("10-18", 30) == set(range(10, 19))  # unknown layout: the model's own indices


def test_process_registers_the_artists_of_the_passes_that_run():
    from modules.anima_feature_conflicts import REGIONAL, set_feature_active
    from modules.anima_support import NEGPIP_EXTRA_PROMPTS

    def registered(p, args):
        mixer.Script().process(p, *args)
        return getattr(p, NEGPIP_EXTRA_PROMPTS)[mixer.ARTIST_MIXER]

    assert registered(_run_p(enable_hr=True), _run_args(enable_hires_rows=True)) == ["base artist", "hires artist"]
    assert registered(_run_p(enable_hr=True), _run_args(enable_hires_rows=True, disable_hires_mixing=True)) == ["base artist"]
    assert registered(_run_p(enable_hr=True), _run_args()) == ["base artist"]
    p = _run_p(enable_hr=True)
    set_feature_active(p, REGIONAL)  # like process_before_every_sampling, yield to Regional
    assert registered(p, _run_args(enable_hires_rows=True)) == []
    assert "Anima Artist Mixer" not in p.extra_generation_params  # reported once, at sampling

    # The hires pass samples with its own rows only when they are independent
    run = mixer._resolve_run_rows(_run_p(), mixer._MixerArgs.from_args(_run_args(enable_hires_rows=True)), True)
    assert run[3][0][1] == "hires artist"
    assert mixer._resolve_run_rows(_run_p(), mixer._MixerArgs.from_args(_run_args()), True)[3][0][1] == "(base artist:-1)"
    # A caller without the whole header runs nothing
    assert mixer._MixerArgs.from_args(_run_args()[:12]) is None
    assert mixer._resolve_run_rows(_run_p(), None, False) is None


def test_conditioning_cache_key_follows_the_text_encoder_patches(monkeypatch):
    # Qwen layer weights re-apply the TE LoRA patches; by sampling time the
    # LoRA hash is back to the plain LoRA list, so it cannot tell them apart.
    from types import SimpleNamespace

    import modules

    monkeypatch.setattr(modules, "sd_models", SimpleNamespace(model_data=SimpleNamespace(forge_hash="h")), raising=False)
    patcher = SimpleNamespace(patches_uuid="te-1")
    model = SimpleNamespace(current_lora_hash="['a.safetensors']", forge_objects=SimpleNamespace(clip=SimpleNamespace(patcher=patcher)))
    p = SimpleNamespace(sd_model=model)

    first = mixer._cache_key(p, ["artist"])
    assert mixer._cache_key(p, ["artist"]) == first
    patcher.patches_uuid = "te-2"
    assert mixer._cache_key(p, ["artist"]) != first


def _negpip_state(combine, fusion, batched=False):
    state = _state(combine=combine, fusion=fusion, batched=batched)
    state.artists[1].negpip_mask = torch.tensor([[[1.0], [-1.0], [1.0]]])
    return state


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("fusion", [mixer.FUSION_INTERPOLATE, mixer.FUSION_CONCAT_WITH_BASE])
def test_output_avg_hands_each_artist_context_its_own_negpip_mask(fusion, batched):
    from modules.anima_support import NEGPIP_CONTEXT_MASK_KEY, NEGPIP_MASK_KEY

    state = _negpip_state(mixer.COMBINE_OUTPUT_AVG, fusion, batched)
    x, context = _inputs(batch=4)
    base_mask = torch.ones(4, 3, 1)
    base_mask[:, 0] = -1.0
    forward = _Attention()
    mixer._dispatch_cross_attn(forward, 0, state, x, context=context, transformer_options={"cond_or_uncond": [0, 1], NEGPIP_MASK_KEY: base_mask})

    artist_calls, base_call = forward.calls[:-1], forward.calls[-1]
    assert NEGPIP_CONTEXT_MASK_KEY not in base_call[2] and base_call[2][NEGPIP_MASK_KEY] is base_mask
    ones, artist = torch.ones(2, 3, 1), state.artists[1].negpip_mask.expand(2, -1, -1)
    if fusion == mixer.FUSION_CONCAT_WITH_BASE:  # the base tokens (positive rows) keep the base mask
        ones, artist = torch.cat([base_mask[:2], ones], dim=1), torch.cat([base_mask[:2], artist], dim=1)
    masks = [call[2][NEGPIP_CONTEXT_MASK_KEY] for call in artist_calls]
    if batched:
        assert len(masks) == 1 and torch.equal(masks[0], torch.cat([ones, artist]))
    elif fusion == mixer.FUSION_CONCAT_WITH_BASE:
        assert torch.equal(masks[0], ones) and torch.equal(masks[1], artist)
    else:
        assert masks[0] is None and torch.equal(masks[1], artist)


@pytest.mark.parametrize("fusion", [mixer.FUSION_INTERPOLATE, mixer.FUSION_CONCAT_WITH_BASE])
def test_concat_hands_the_joined_context_its_own_negpip_mask(fusion):
    from modules.anima_support import NEGPIP_CONTEXT_MASK_KEY, NEGPIP_MASK_KEY

    state = _negpip_state(mixer.COMBINE_CONCAT, fusion)
    x, context = _inputs()
    forward = _Attention()
    options = {"cond_or_uncond": [0, 1]}
    mixer._dispatch_cross_attn(forward, 0, state, x, context=context, transformer_options=options)
    expected = torch.cat([torch.ones(1, 3, 1), state.artists[1].negpip_mask], dim=1)
    if fusion == mixer.FUSION_CONCAT_WITH_BASE:  # no base mask: ones for the base tokens
        expected = torch.cat([torch.ones(1, 3, 1), expected], dim=1)
    assert torch.equal(forward.calls[0][2][NEGPIP_CONTEXT_MASK_KEY], expected)

    # Without negative weights anywhere, the artist context carries None, not the base mask
    state = _state(combine=mixer.COMBINE_CONCAT, fusion=fusion)
    forward = _Attention()
    options[NEGPIP_MASK_KEY] = torch.ones(2, 3, 1)
    mixer._dispatch_cross_attn(forward, 0, state, x, context=context, transformer_options=options)
    if fusion == mixer.FUSION_CONCAT_WITH_BASE:
        assert torch.equal(forward.calls[0][2][NEGPIP_CONTEXT_MASK_KEY], torch.ones(1, 9, 1))
    else:
        assert forward.calls[0][2][NEGPIP_CONTEXT_MASK_KEY] is None


def test_base_fusion_contexts_are_built_once_per_model_call():
    from modules.anima_support import NEGPIP_MASK_KEY

    state = _negpip_state(mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_CONCAT_WITH_BASE)
    state.artists[0].blocks = state.artists[1].blocks = {0, 1}
    x, context = _inputs()
    options = {"cond_or_uncond": [0, 1], NEGPIP_MASK_KEY: -torch.ones(2, 3, 1)}
    forward = _Attention()
    for block in (0, 1):  # two blocks of one model call
        mixer._dispatch_cross_attn(forward, block, state, x, context=context, transformer_options=options)
    first, second = forward.calls[:2], forward.calls[3:5]
    for one, two in zip(first, second):
        assert one[1] is two[1] and one[2]["negpip_context_mask"] is two[2]["negpip_context_mask"]


@pytest.mark.parametrize("combine, fusion", [
    (mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE),  # batched: the artist contexts stacked
    (mixer.COMBINE_CONCAT, mixer.FUSION_INTERPOLATE),  # the weighted artist tokens joined
    (mixer.COMBINE_CONCAT, mixer.FUSION_CONCAT_WITH_BASE),  # ... after the base tokens
])  # fmt: skip
def test_joined_artist_contexts_are_built_once_per_model_call(combine, fusion):
    state = _state(combine, fusion, batched=True)
    state.artists[0].blocks = {0, 1, 2}
    state.artists[1].blocks = {0, 1}
    x, context = _inputs()
    forward = _Attention()

    def artist_kv(block, context):
        """The context the block's one artist forward attends to."""
        forward.calls.clear()
        mixer._dispatch_cross_attn(forward, block, state, x, context=context, transformer_options={"cond_or_uncond": [0, 1]})
        (kv,) = [call[1] for call in forward.calls if call[1] is not context]
        return kv

    first = artist_kv(0, context)
    assert artist_kv(1, context) is first  # another block of the same model call
    assert artist_kv(2, context) is not first  # other active artists: their own context
    again = artist_kv(0, context.clone())  # the next model call
    assert again is not first and torch.equal(again, first)


def test_artist_forwards_skip_the_negative_rows():
    state = _state()
    x, context = _inputs(batch=4)
    forward = _Attention()
    out = _dispatch(state, forward, x, context, [1, 0])  # positive rows are 2 and 3

    assert [call[0] for call in forward.calls] == [2, 2, 4]
    assert all(call[2]["cond_or_uncond"] == [0] for call in forward.calls[:2])
    base = attend(x, context)
    weights = [0.6 / 0.9, 0.3 / 0.9]
    artist = sum(w * attend(x, a.cond.expand(4, -1, -1)) for w, a in zip(weights, state.artists))
    strength = 0.8 * 0.9
    assert torch.equal(out[:2], base[:2])
    assert torch.allclose(out[2:], base[2:] * (1.0 - strength) + artist[2:] * strength, atol=1e-6)


def test_rows_mixing_into_no_block_take_no_slot_of_the_artist_cap(monkeypatch):
    from modules.anima_support import NEGPIP_EXTRA_PROMPTS

    cap = mixer._MAX_ARTISTS[mixer.OPT_PERFORMANCE]
    width, rows = len(mixer.TABLE_HEADERS), cap + 2

    def row(artist, blocks="0-51"):
        return [True, artist, 1.0, blocks, 0.0, 1.0, 0.5, mixer.CURVE_HOLD, mixer.PRESET_CUSTOM, False]

    # The test model has one block: "5" selects none of it, so that row is skipped
    base = row("ghost", "5") + [value for index in range(rows - 1) for value in row(f"artist{index}")]
    base += [None] * (width * (mixer.MAX_ARTIST_ROWS - rows))
    head = [True, rows, 1, False, False, 3.0, 3.0, 0.7, mixer.OPT_PERFORMANCE, mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE, False, False]
    args = head + base + [None] * (width * mixer.MAX_ARTIST_ROWS)
    expected = [f"artist{index}" for index in range(cap)]

    p = _run_p()
    mixer.Script().process(p, *args)
    assert getattr(p, NEGPIP_EXTRA_PROMPTS)[mixer.ARTIST_MIXER] == expected  # NegPiP sees the mixed artists only

    encoded = []
    monkeypatch.setattr(mixer, "effective_prompt_batch", lambda p: [[""]])
    monkeypatch.setattr(mixer, "_encode_text_batch", lambda p, texts, use_cache=True: encoded.extend(texts) or (torch.zeros(1, 3, 4), None))
    try:
        mixer.Script().process_before_every_sampling(p, *args)
        assert encoded == expected  # nothing past the cap is encoded
        assert [artist.name for artist in mixer._RUN_STATES[-1].artists] == expected
        assert p.extra_generation_params["Anima Artist Mixer"].startswith(f"{cap} artists")
    finally:
        mixer._cleanup_runtime()
