from types import SimpleNamespace

from modules import spectrum_force


def test_reasons_accumulate_instead_of_overwriting():
    patcher = SimpleNamespace(model_options={})
    spectrum_force.request(patcher, "anima_nag")
    spectrum_force.request(patcher, spectrum_force.sigma_window(0.2, 0.6))

    value = patcher.model_options["transformer_options"][spectrum_force.KEY]
    assert len(value) == 2
    assert spectrum_force.is_forced(value, 0.9)  # the whole-pass reason wins


def test_callable_reasons_only_force_their_window():
    value = spectrum_force.combine(None, spectrum_force.sigma_window(0.6, 0.2))
    assert spectrum_force.is_forced(value, 0.4)
    assert not spectrum_force.is_forced(value, 0.8)
    assert not spectrum_force.is_forced(None, 0.4)
    assert spectrum_force.is_forced("legacy string", 0.4)
