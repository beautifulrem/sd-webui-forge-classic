import re
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_negpip.utils import NEG_PATTERN, has_negative  # noqa: E402


@pytest.mark.parametrize(
    ("prompt", "found"),
    [
        ("a cat, (dog:-1)", ["(dog:-1)"]),
        ("( hat :- 1.5 ), (cap: -.5)", ["( hat :- 1.5 )", "(cap: -.5)"]),
        (r"(a\(b\):-0.5)", [r"(a\(b\):-0.5)"]),
        ("(  :-1)", ["(  :-1)"]),
        ("((x:-1)", ["(x:-1)"]),
        ("(\tx\n:\t-\n2\n)", ["(\tx\n:\t-\n2\n)"]),
        ("(:-1), (x:-1.), (x:1), (x:-a), (a(b:-1)", ["(b:-1)"]),
        ("(x:-.), (x:-1..5), (x:-1.5.), (x:- 1 . 5)", []),
        ("(x:-007), (y:-0.25)", ["(x:-007)", "(y:-0.25)"]),
        # The attention parser reads \( as a literal bracket, \\( as a real one.
        (r"\(x:-1)", []),
        (r"\\(x:-1)", ["(x:-1)"]),
        (r"(a \(b\):-1)", [r"(a \(b\):-1)"]),
    ],
)
def test_negative_weights_are_found(prompt, found):
    assert re.findall(NEG_PATTERN, prompt) == found
    assert has_negative(prompt) is bool(found)


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
