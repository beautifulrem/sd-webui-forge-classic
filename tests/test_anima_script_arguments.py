"""The panels' one script-argument policy: extras refused, missing values
filled from the panel's defaults when it gives them, otherwise refused."""

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
