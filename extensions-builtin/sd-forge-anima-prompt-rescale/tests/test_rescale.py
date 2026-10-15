"""The prompt-schedule rescaler's math (verified Forge neo semantics)."""

import pytest

from lib_anima_rescale.rescale import executed_steps, find_schedule_whens, remap_when, rescale_prompt_schedule


@pytest.mark.parametrize(
    ("steps", "denoise", "expected"),
    [(35, 0.5, 17), (35, 1.0, 34), (35, 0.1, 3), (35, 0.001, 1)],
    ids=["half", "0.999 cap", "tenth", "at least one"],
)
def test_executed_steps_match_forge_setup_img2img_steps(steps, denoise, expected):
    assert executed_steps(steps, denoise) == expected


@pytest.mark.parametrize(
    ("fraction", "steps", "denoise", "expected"),
    [(15 / 35, 35, 0.5, 7), (15 / 35, 35, 1.0, 15), (15 / 35, 35, 0.1, 1), (0.9, 35, 0.5, 15)],
)
def test_remap_when(fraction, steps, denoise, expected):
    assert remap_when(fraction, steps, denoise) == expected


@pytest.mark.parametrize(
    ("prompt", "source", "target", "denoise", "expected"),
    [
        ("[@guweiz:@wlop:15]", 35, 35, 0.5, "[@guweiz:@wlop:7]"),
        ("[a:b:15]", 35, 35, 1.0, "[a:b:15]"),
        ("[a:b:0.4286]", 35, 35, 0.5, "[a:b:7]"),
        ("[a:b:15]", 35, 50, 1.0, "[a:b:21]"),
        ("[blurry:sharp:10]", 35, 35, 0.5, "[blurry:sharp:5]"),
    ],
)
def test_absolute_rescale(prompt, source, target, denoise, expected):
    assert rescale_prompt_schedule(prompt, source, target, denoise) == expected


def test_nested_schedules_are_rescaled_at_every_depth():
    def grounded(when, source, target, denoise):
        fraction = when if when < 1 else when / source
        executed = max(1, int(min(denoise, 0.999) * target))
        return max(1, min(executed, round(fraction * executed)))

    inner, outer = grounded(0.3, 35, 35, 0.5), grounded(20, 35, 35, 0.5)
    assert rescale_prompt_schedule("[a:[b:c:0.3]:20]", 35, 35, 0.5) == f"[a:[b:c:{inner}]:{outer}]"


def test_fraction_output():
    assert rescale_prompt_schedule("[a:b:15]", 35, 35, 0.5, as_fraction=True) == "[a:b:0.2]"


def test_only_schedules_are_found():
    assert find_schedule_whens("[a|b|c]") == []
    assert find_schedule_whens("(cat:1.3)") == []
    assert find_schedule_whens("[(cat:1.3):dog:10]") == [(15, 17, 10.0)]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("a plain prompt", "a plain prompt"),
        ("[a|b|c]", "[a|b|c]"),
        ("(masterpiece:1.4)", "(masterpiece:1.4)"),
        ("(best:1.2), [a:b:15], [x|y]", "(best:1.2), [a:b:7], [x|y]"),
        ("[broken:::]", "[broken:::]"),
    ],
    ids=["plain", "alternation", "weights", "mixed", "malformed"],
)
def test_everything_else_is_left_untouched(prompt, expected):
    assert rescale_prompt_schedule(prompt, 35, 35, 0.5) == expected
