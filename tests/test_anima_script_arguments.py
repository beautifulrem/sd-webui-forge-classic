"""The panels' one script-argument policy: extras refused, missing values
filled from the panel's defaults when it gives them, otherwise refused;
non-finite float values refused by name."""

import pytest

from modules.anima_support import script_arguments

NAMES = ("enable", "strength", "mode")


def test_values_are_named_in_order():
    args = script_arguments("Panel", NAMES, [True, 0.5, "fast"])
    assert vars(args) == {"enable": True, "strength": 0.5, "mode": "fast"}


def test_extra_values_are_refused_even_with_defaults():
    for defaults in (None, (False, 1.0, "slow"), {"mode": "slow"}):
        with pytest.raises(TypeError, match=r"^Panel takes 3 script arguments, got 4$"):
            script_arguments("Panel", NAMES, [True, 0.5, "fast", "extra"], defaults=defaults)


def test_missing_values_without_defaults_are_refused_by_name():
    with pytest.raises(TypeError, match=r"^Panel takes 3 script arguments, got 1; missing: strength, mode$"):
        script_arguments("Panel", NAMES, [True])


def test_missing_values_are_filled_from_a_sequence_of_defaults():
    args = script_arguments("Panel", NAMES, [True], defaults=(False, 1.0, "slow"))
    assert vars(args) == {"enable": True, "strength": 1.0, "mode": "slow"}
    # Given values are kept
    assert script_arguments("Panel", NAMES, [False, 0.5, "fast"], defaults=(True, 1.0, "slow")).strength == 0.5


def test_missing_values_are_filled_from_a_mapping_of_defaults():
    args = script_arguments("Panel", NAMES, [True, 0.5], defaults={"mode": "slow"})
    assert args.mode == "slow"
    # A missing name the mapping has no default for is refused
    with pytest.raises(TypeError, match=r"^Panel takes 3 script arguments, got 1; missing: strength$"):
        script_arguments("Panel", NAMES, [True], defaults={"mode": "slow"})


def test_a_sequence_of_defaults_must_cover_every_name():
    with pytest.raises(ValueError, match="Panel has 2 script argument defaults for 3 names"):
        script_arguments("Panel", NAMES, [True], defaults=(1.0, "slow"))


def test_a_long_missing_tail_names_its_first_arguments_only():
    names = [f"slot{index}" for index in range(12)]
    with pytest.raises(TypeError, match=r"^Panel takes 12 script arguments, got 2; missing: slot2, slot3, slot4, slot5, slot6 and 5 more$"):
        script_arguments("Panel", names, [1, 2])


NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("value", NON_FINITE, ids=repr)
def test_a_non_finite_float_is_refused_by_name(value):
    # The API reads JSON NaN / Infinity as floats; min(max(nan, 0), 1) is nan
    with pytest.raises(ValueError, match=rf"^Panel refuses non-finite script arguments: strength={value!r}$"):
        script_arguments("Panel", NAMES, [True, value, "fast"])


def test_every_non_finite_argument_is_named_filled_defaults_included():
    import numpy as np

    with pytest.raises(ValueError, match=r"^Panel refuses non-finite script arguments: enable=nan, mode=np\.float32\(inf\)$"):
        script_arguments("Panel", NAMES, [float("nan"), 0.5], defaults={"mode": np.float32("inf")})


def test_finite_numbers_and_text_that_reads_as_nan_are_kept():
    # A text field may hold "nan" (a prompt, a block list); only float values are checked
    args = script_arguments("Panel", NAMES, [True, 1e308, "nan"])
    assert (args.enable, args.strength, args.mode) == (True, 1e308, "nan")
    assert script_arguments("Panel", NAMES, [False, 3, "inf"]).strength == 3


def test_require_finite_checks_the_values_it_is_given():
    from modules.anima_support import require_finite

    require_finite("Panel", strength=0.5, blocks="nan", enable=True, missing=None)
    with pytest.raises(ValueError, match=r"^Panel refuses non-finite script arguments: tau=nan, gain=-inf$"):
        require_finite("Panel", strength=0.5, tau=float("nan"), gain=float("-inf"))
