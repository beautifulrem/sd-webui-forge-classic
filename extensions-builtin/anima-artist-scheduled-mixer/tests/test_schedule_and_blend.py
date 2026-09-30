import pytest
import torch

from lib_artist_mixer.contexts import artist_mask_like, broadcast_batch, norm_limited_delta, to_context_like
from lib_artist_mixer.schedule import CURVE_BACK, CURVE_FRONT, CURVE_HOLD, CURVE_SMOOTH, CURVE_TRIANGLE, curve_factor, parse_blocks, parse_inline_weight


@pytest.mark.parametrize(
    "curve, progress, expected",
    [
        (CURVE_HOLD, 0.5, 1.0),
        (CURVE_FRONT, 0.2, 1.0),
        (CURVE_FRONT, 0.6, 0.25),
        (CURVE_BACK, 0.2, 0.25),
        (CURVE_TRIANGLE, 0.4, 1.0),
        (CURVE_TRIANGLE, 0.3, 0.5),
        (CURVE_SMOOTH, 0.4, 1.0),
        (CURVE_SMOOTH, 0.2, 0.0),
    ],
)
def test_curve_factor_shapes_the_window(curve, progress, expected):
    assert curve_factor(progress, 0.2, 0.6, 0.4, curve) == pytest.approx(expected)


def test_curve_factor_outside_or_without_progress():
    assert curve_factor(0.9, 0.2, 0.6, 0.4, CURVE_HOLD) == 0.0
    assert curve_factor(None, 0.2, 0.6, 0.4, CURVE_HOLD) == 1.0
    assert curve_factor(0.3, 0.6, 0.2, 0.4, CURVE_HOLD) == 1.0  # start/end swapped


def test_block_field_syntax():
    assert parse_blocks("", 4) == {0, 1, 2, 3}
    assert parse_blocks("0-1，-1", 28) == {0, 1, 27}
    assert parse_blocks("3-1, x, 9", 5) == {1, 2, 3}
    # IME input as in the weight fields: full-width forms and dash variants
    assert parse_blocks("０－２，５", 28) == parse_blocks("0–2, 5", 28) == parse_blocks("0—2,-23", 28) == {0, 1, 2, 5}
    # A "_" is refused like in the weight fields ("0_5" is not block 5), the
    # rest of the field still counts
    assert parse_blocks("0_5, 7", 28) == {7} and parse_blocks("1_0-3, 9", 28) == {9}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0 - 5", set(range(6))),
        ("0　－　5", set(range(6))),
        ("- 1, 0 - -27", {0, 1, 27}),
        ("1 0, 5", {5}),  # "1 0" is not block 10: skipped, the rest counts
        ("１　０, 5", {5}),
        ("1\t0, 5", {5}),
        ("1 0-3, 9", {9}),
    ],
)
def test_block_field_whitespace(text, expected):
    assert parse_blocks(text, 28) == expected


@pytest.mark.parametrize("text", ["1 0", "１　０", "1\t0"])
def test_a_block_field_with_a_spaced_number_selects_nothing(text, caplog):
    with caplog.at_level("WARNING"):
        assert parse_blocks(text, 28) == set()
    assert "selects none" in caplog.text


def test_a_block_field_selecting_nothing_is_no_block(caplog):
    assert parse_blocks(" ，", 4) == {0, 1, 2, 3}  # only separators: an empty field
    with caplog.at_level("WARNING"):
        assert parse_blocks("x", 28) == set()
        assert parse_blocks("30-40", 28) == set()  # past the 2B layout
    assert "selects none" in caplog.text


def test_inline_weight_syntax():
    assert parse_inline_weight("(artist:1.2)") == ("artist", 1.2)
    assert parse_inline_weight("artist：0.5") == ("artist", 0.5)
    assert parse_inline_weight("<lora:x:0.5>") == ("<lora:x:0.5>", 1.0)
    assert parse_inline_weight("") == ("", 1.0)
    # IME input as in the weight fields; the last colon separates a name that has colons
    assert parse_inline_weight("(artist:１．５)") == ("artist", 1.5)
    assert parse_inline_weight("[artist：−0.5]") == ("artist", -0.5)
    assert parse_inline_weight("(a:b:1.2)") == ("a:b", 1.2)
    # IME brackets, with either colon
    assert parse_inline_weight("（wlop：1.5）") == ("wlop", 1.5)
    assert parse_inline_weight("【wlop：1.5】") == ("wlop", 1.5)
    assert parse_inline_weight("［wlop:0.5］") == ("wlop", 0.5)
    # Not a number: the whole text is the artist, at weight 1
    assert parse_inline_weight("(artist:foo)") == ("(artist:foo)", 1.0)
    assert parse_inline_weight("artist:0_5") == ("artist:0_5", 1.0)
    # An empty name is empty with or without brackets (the row is then skipped)
    assert parse_inline_weight("(:1.5)") == parse_inline_weight("( :1.5)") == parse_inline_weight(":1.5") == ("", 1.5)


def test_a_block_field_parses_in_linear_time_on_long_whitespace_runs(caplog):
    # Blocks comes from pasted infotext, the API and saved settings, unbounded;
    # four times the text may take about four times as long, not sixteen
    import time

    def seconds(text):
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            parse_blocks(text, 28)
            best = min(best, time.perf_counter() - start)
        return best

    for shape in (lambda spaces: f"1{spaces}2", lambda spaces: f"0{spaces}-{spaces}5"):
        short, long = (seconds(shape(" " * n)) for n in (10000, 40000))
        assert long < 8 * short + 1e-3, (short, long)
    assert all(len(record.getMessage()) < 400 for record in caplog.records)  # the warning quotes a bounded excerpt


def test_norm_limited_delta_caps_each_row_separately():
    base = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    artist = torch.tensor([[1.0, 5.0], [10.0, 1.0]])

    out = norm_limited_delta(base, artist, 1.0, max_ratio=0.6)

    assert torch.allclose(out[0], torch.tensor([1.0, 0.6]))  # capped at 0.6 * |base|
    assert torch.allclose(out[1], torch.tensor([10.0, 1.0]))  # within the cap


def test_artist_context_is_aligned_to_the_base_context():
    context = torch.zeros(2, 3, 4)
    assert to_context_like(torch.ones(3, 4), context).shape == (2, 3, 4)
    with pytest.raises(RuntimeError):
        to_context_like(torch.ones(1, 3, 5), context)


def test_artist_prompt_batches_tile_to_the_attention_batch_or_fail():
    prompts = torch.arange(2.0).view(2, 1, 1).expand(2, 3, 4)
    assert broadcast_batch(prompts, 4)[:, 0, 0].tolist() == [0.0, 1.0, 0.0, 1.0]  # cond + uncond per prompt
    assert broadcast_batch(prompts[:1], 3).shape == (3, 3, 4)
    # A batch it does not divide cannot say which prompt a row belongs to:
    # it used to give every row the first prompt's context
    with pytest.raises(RuntimeError, match="Artist prompt batch 2 does not match attention batch 3"):
        to_context_like(prompts, torch.zeros(3, 3, 4))
    with pytest.raises(RuntimeError, match="does not match attention batch 3"):
        artist_mask_like(torch.ones(2, 3, 1), torch.zeros(3, 3, 4))
