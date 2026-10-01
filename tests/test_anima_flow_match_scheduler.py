import sys
from types import SimpleNamespace

import pytest
import torch

import k_diffusion.sampling  # noqa: F401  (sd_schedulers reads k_diffusion.sampling at import)

pytest.importorskip("diffusers")
TOGGLES = ("use_dynamic_shifting", "invert_sigmas", "use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "stochastic_sampling")


def _import_fresh(monkeypatch):
    """modules.sd_schedulers imported again (it reads the options at import);
    the module and the package attribute the import sets are both put back
    after the test, so later imports never see this copy."""
    import modules

    # setitem records the entry (or its absence) for monkeypatch to restore;
    # delitem records nothing when the module was never imported
    monkeypatch.setitem(sys.modules, "modules.sd_schedulers", None)
    del sys.modules["modules.sd_schedulers"]
    monkeypatch.setattr(modules, "sd_schedulers", None, raising=False)
    import modules.sd_schedulers

    return modules.sd_schedulers


@pytest.fixture
def sd_schedulers(monkeypatch):
    """The module imported against stand-in options."""
    from modules import shared

    monkeypatch.setattr(shared, "opts", SimpleNamespace(hide_schedulers=[], **dict.fromkeys(TOGGLES, False)))
    return _import_fresh(monkeypatch)


def _inner_model(*, use_shift, shift=3.0):
    predictor = SimpleNamespace(shift=shift)
    unet = SimpleNamespace(model=SimpleNamespace(predictor=predictor))
    return SimpleNamespace(inner_model=SimpleNamespace(use_shift=use_shift, forge_objects=SimpleNamespace(unet=unet)))


def test_non_flow_models_fall_back_to_the_simple_schedule(sd_schedulers, monkeypatch):
    calls = []
    monkeypatch.setattr(sd_schedulers, "simple_scheduler", lambda *args: calls.append(args) or torch.ones(3))
    model = _inner_model(use_shift=False)

    result = sd_schedulers.anima_flow_match_scheduler(4, 0.03, 14.6, model, "cpu")

    assert torch.equal(result, torch.ones(3))
    assert calls == [(4, 0.03, 14.6, model, "cpu")]


def test_anima_grid_ignores_the_global_schedule_toggles(sd_schedulers, monkeypatch):
    model = _inner_model(use_shift=True)
    plain = sd_schedulers.flow_match_euler_discrete_scheduler(8, 0.0, 1.0, model, "cpu")
    monkeypatch.setattr(sd_schedulers.shared.opts, "use_karras_sigmas", True)

    anima = sd_schedulers.anima_flow_match_scheduler(8, 0.0, 1.0, model, "cpu")

    assert torch.equal(anima, plain)
    assert not torch.equal(sd_schedulers.flow_match_euler_discrete_scheduler(8, 0.0, 1.0, model, "cpu"), plain)
    # Diffusers' grid: t runs linearly from 1 to the shifted smallest training
    # sigma, then shift 3 maps it to 3t / (1 + 2t); a final 0 is appended.
    shift = lambda t: 3 * t / (1 + 2 * t)  # noqa: E731
    t = torch.linspace(1.0, shift(1.0 / 1000), 8)
    assert torch.allclose(anima[:-1], shift(t), atol=1e-5)
    assert anima[-1] == 0


def test_beta57_is_registered_as_the_fixed_beta_preset(sd_schedulers, monkeypatch):
    calls = []
    monkeypatch.setattr(sd_schedulers, "beta_scheduler", lambda *args, **kwargs: calls.append(kwargs) or "sigmas")
    scheduler = sd_schedulers.schedulers_map["beta57"]

    assert scheduler.function(8, 0.03, 14.6, None, "cpu") == "sigmas"
    assert calls == [{"alpha": 0.5, "beta": 0.7}]


def test_every_scheduler_alias_resolves_like_its_label(sd_schedulers):
    aliased = [scheduler for scheduler in sd_schedulers.schedulers if scheduler.aliases]
    assert {"Beta 57", "RES4LYF Beta57", "Anima Diffusers FlowMatch"} <= {alias for scheduler in aliased for alias in scheduler.aliases}
    for scheduler in aliased:
        for alias in scheduler.aliases:
            assert sd_schedulers.schedulers_map[alias] is scheduler is sd_schedulers.schedulers_map[scheduler.label]


def test_hidden_schedulers_leave_the_dropdown_map_but_not_the_full_one(monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared, "opts", SimpleNamespace(hide_schedulers=["Anima FlowMatch"], **dict.fromkeys(TOGGLES, False)))
    sd_schedulers = _import_fresh(monkeypatch)

    assert "Anima FlowMatch" not in sd_schedulers.schedulers_map
    for key in ("anima_flow_match", "Anima FlowMatch", "Anima Diffusers FlowMatch"):
        assert sd_schedulers.all_schedulers_map[key].name == "anima_flow_match"
