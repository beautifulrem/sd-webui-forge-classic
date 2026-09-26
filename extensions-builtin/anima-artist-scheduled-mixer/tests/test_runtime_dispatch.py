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
    return mixer.MixerState("run", True, 0.8, combine, fusion, apply_uncond, batched, artists)


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


def test_blocks_are_unpatched_after_the_batch_before_postprocess_image():
    # ADetailer's inner runs (postprocess_image) skip this script by default,
    # so they must not inherit the artist cross-attn overrides.
    from types import SimpleNamespace

    attention = torch.nn.Module()
    dm = SimpleNamespace(blocks=[SimpleNamespace(cross_attn=attention)])
    state = _state()
    mixer._install_cross_attn_patch(dm, state)
    mixer._ACTIVE_STATE = state
    mixer._RUN_STATES.append(state)
    try:
        assert "forward" in attention.__dict__
        mixer.Script().postprocess_batch(SimpleNamespace(), images=None, batch_number=0)

        assert "forward" not in attention.__dict__
        assert mixer._ACTIVE_STATE is None
        assert mixer._RUN_STATES == [state]  # kept for postprocess's run summary
    finally:
        mixer._cleanup_runtime()


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
