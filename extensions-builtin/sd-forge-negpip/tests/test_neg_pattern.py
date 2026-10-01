import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.text_processing.parsing import parse_prompt_attention, re_attention  # noqa: E402
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
    # BREAK markers carry weight -1, scaled by any bracket around them; real
    # text is never exactly "BREAK" (the parser splits it away)
    return [(text, weight) for text, weight in parsed if weight < 0 and text and text != "BREAK"]


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
        # Colons and escapes in the text, as Danbooru tags and escaped brackets write them
        "(:d:-1)", "(re:zero:-1)", "(:o:-0.8)", "(a:b:c:-.5)", r"(a\[b:-1)", r"(a\]b:-1)", r"(foo\\:-1)",
        r"(\(x\):-1)", r"\\\(x:-1)", r"(x\:-1)",
        # Groups the parser reads as no text weight nothing; whitespace is text
        r"(\:-1)", "([:-1)", "( BREAK [:-1.2)", "(:-1)",
        "(  :-1)",
    ],
)
def test_negative_weights_agree_with_forges_attention_parser(prompt):
    expected = _parser_negatives(prompt)
    assert len(expected) <= 1
    # NegPiP keeps the text as written (it is encoded again); the parser unescapes it
    found = [("".join(part for part, _ in parse_prompt_attention(text)), weight) for _, text, weight in negative_weights(prompt)]
    assert found == expected
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


@pytest.mark.parametrize("prompt", ["(cat, (dog):-1)", "((a):-1)", "((foo:1.2):-1)", "a, ((b) c:-0.5), d"])
def test_a_negative_weight_around_a_nested_group_counts_but_stays_in_the_prompt(prompt):
    # On Anima the hooks must go in whatever the other prompts hold (the mask
    # comes from the parser); the SD path only moves innermost groups out
    assert _parser_negatives(prompt)
    assert has_negative(prompt) and negative_weights(prompt) == []


def _round_depth(prompt: str) -> int:
    """The deepest nesting of round brackets as the parser reads them."""
    depth = deepest = 0
    for token in re_attention.finditer(prompt):
        if token[0] == "(":
            depth += 1
            deepest = max(deepest, depth)
        elif depth and (token[0] == ")" or token[1] is not None):
            depth -= 1
    return deepest


def test_has_negative_agrees_with_the_parser_on_random_prompts():
    # Seeded combinations of every token kind the parser knows. NegPiP never
    # misses a negative weight the parser applies. It may over-report only
    # where round groups nest, as nested weights can cancel ("((a:-1):-1)") or
    # zero ("((a:-0.5):-0)") a sign: harmless, the Anima mask comes from the
    # parser's own multipliers
    import random

    pieces = ["(", ")", "[", "]", "a", "dog", ":", ":d", " ", ",", "BREAK", " BREAK ", "\\(", "\\)", "\\", "\\\\", ":-1)", ":-0.5)", ":1.2)", ":-0)", ":- 1)"]
    rng = random.Random(43)
    checked = 0
    for _ in range(20000):
        prompt = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 12)))
        try:
            parse_prompt_attention(prompt)
        except ValueError:
            continue
        expected = bool(_parser_negatives(prompt))
        found = has_negative(prompt)
        if found and not expected:
            assert _round_depth(prompt) >= 2, prompt
        else:
            assert found is expected, prompt
        checked += 1
    assert checked > 15000


def test_deep_nesting_stays_linear_in_memory_and_time():
    # Nested spans overlap: slicing each would take memory quadratic in the
    # depth (gigabytes for a 200 KB prompt); only innermost groups are sliced
    import tracemalloc

    def measure(depth):
        prompt = "(" * depth + "a" + ":-1)" * depth
        tracemalloc.start()
        start = time.perf_counter()
        assert has_negative(prompt) and len(negative_weights(prompt)) == 1
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return seconds, peak, len(prompt)

    small_seconds, _, _ = measure(5_000)
    large_seconds, large_peak, length = measure(20_000)
    assert large_peak < 64 * length, large_peak  # bytes per prompt character, not per character squared
    assert large_seconds < max(8 * small_seconds, 0.05), (small_seconds, large_seconds)
