import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.text_processing.parsing import parse_prompt_attention  # noqa: E402
from lib_negpip.utils import has_negative, negative_weights  # noqa: E402


@pytest.mark.parametrize(
    ("prompt", "found"),
    [
        ("a cat, (dog:-1)", ["(dog:-1)"]),
        ("( hat :- 1.5 ), (cap: -.5)", ["(cap: -.5)"]),
        (r"(a\(b\):-0.5)", [r"(a\(b\):-0.5)"]),
        ("(  :-1)", ["(  :-1)"]),
        ("((x:-1)", ["(x:-1)"]),
        ("(\tx\n:\t-2\n)", ["(\tx\n:\t-2\n)"]),
        ("(\tx\n:\t-\n2\n)", []),
        ("(:-1), (x:-1.), (x:1), (x:-a), (a(b:-1)", ["(x:-1.)", "(b:-1)"]),
        ("(x:-.), (x:-1..5), (x:-1.5.), (x:- 1 . 5)", []),
        ("(x:-007), (y:-0.25), (z:-0), (w:-0.0)", ["(x:-007)", "(y:-0.25)"]),
        # The attention parser reads \( as a literal bracket, \\( as a real one.
        (r"\(x:-1)", []),
        (r"\\(x:-1)", ["(x:-1)"]),
        (r"(a \(b\):-1)", [r"(a \(b\):-1)"]),
    ],
)
def test_negative_weights_are_found(prompt, found):
    assert [match for match, _, _ in negative_weights(prompt)] == found
    assert has_negative(prompt) is bool(found)


def _parser_negatives(prompt: str) -> list[tuple[str, float]]:
    """The (text, weight) pieces Forge's attention parser weights below zero."""
    try:
        parsed = parse_prompt_attention(prompt)
    except ValueError:  # e.g. "(x:-1..5)": float() of the parser's [.\d]+ fails
        return []
    return [(text, weight) for text, weight in parsed if weight < 0 and (text, weight) != ("BREAK", -1)]


# Each form alone: the parser weights one piece below zero exactly when NegPiP
# takes it, with the same text and weight.
@pytest.mark.parametrize(
    "prompt",
    [
        "(foo:-1)", "(foo:-0.5)", "(foo:-.5)", "(foo:-1.)", "(foo:-007)", "(foo:-1.50)", "(foo:-\u0661)",
        "(foo :-1)", "(foo: -1)", "(foo : -1 )", "(foo:\t-2\n)", "(foo  :  -3  )",
        "(foo:- 1)", "(foo:-\t1)", "(foo:- .5)", "(foo:-0)", "(foo:-0.0)", "(foo:-.0)", "(foo:-0.)",
        "(foo:+-1)", "(foo:--1)", "(foo:-+1)", "(foo:1)", "(foo:+1)", "(foo:-1e2)", "(foo:-1.5.)",
        "(foo:-1..5)", "(foo:-.)", "(foo:-)", "(foo:-1", "foo:-1)", r"\(foo:-1)", r"\\(foo:-1)",
    ],
)
def test_negative_weights_agree_with_forges_attention_parser(prompt):
    expected = _parser_negatives(prompt)
    assert len(expected) <= 1
    assert [(text, weight) for _, text, weight in negative_weights(prompt)] == expected
    assert has_negative(prompt) is bool(expected)


@pytest.mark.parametrize(
    "prompt",
    [
        "(" + " " * 2000,
        "(" + " " * 2000 + ":",
        "(" + " " * 2000 + ":-",
        "( " * 2000 + ":-",
        "(" + r"\(" * 2000,
        "(a:-" + "1" * 200_000 + "x",
        "(a:-" + "1" * 100_000 + "." + "1" * 100_000 + "x",
        r"\(" * 200_000 + ":-1",
        "(" + r"\(" * 200_000 + ":-",
    ],
    ids=["spaces", "colon", "minus", "repeated", "escaped", "digits", "decimal", "escaped-run", "escaped-open"],
)
def test_unclosed_weights_do_not_backtrack_for_seconds(prompt):
    # The pattern runs on every prompt of every generation; these took
    # seconds when whitespace or a digit run could be split between parts of
    # it, or when every \( of a run started a scan of its own.
    start = time.perf_counter()
    assert not has_negative(prompt)
    assert time.perf_counter() - start < 0.5


def _search_seconds(prompt: str) -> float:
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        has_negative(prompt)
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize(
    "make",
    [lambda n: "(a:-" + "1" * n + "x", lambda n: r"\(" * n + ":-1", lambda n: "(" + " " * n + ":-"],
    ids=["digits", "escaped-run", "spaces"],
)
def test_search_time_grows_linearly(make):
    # Four times the input may cost about four times as long; a quadratic
    # pattern costs sixteen times (seconds already at 20k characters).
    small, large = _search_seconds(make(5_000)), _search_seconds(make(20_000))
    assert large < max(8 * small, 0.02)
