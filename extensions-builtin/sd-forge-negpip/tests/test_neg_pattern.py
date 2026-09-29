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
    ],
)
def test_negative_weights_are_found(prompt, found):
    assert re.findall(NEG_PATTERN, prompt) == found
    assert has_negative(prompt) is bool(found)


@pytest.mark.parametrize(
    "prompt",
    ["(" + " " * 2000, "(" + " " * 2000 + ":", "(" + " " * 2000 + ":-", "( " * 2000 + ":-", "(" + r"\(" * 2000],
    ids=["spaces", "colon", "minus", "repeated", "escaped"],
)
def test_unclosed_weights_do_not_backtrack_for_seconds(prompt):
    # The pattern runs on every prompt of every generation; this took
    # seconds when whitespace could be split between three parts of it.
    start = time.perf_counter()
    assert not has_negative(prompt)
    assert time.perf_counter() - start < 0.5
