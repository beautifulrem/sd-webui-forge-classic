"""``lock_scheduler_override``: Anima Flow samplers keep their own grid even
when a script (e.g. XYZ/Hires) installs ``p.sampler_noise_scheduler_override``.

sd_samplers_kdiffusion cannot be imported outside the WebUI (circular
imports), so ``KDiffusionSampler.get_sigmas`` is loaded from its source."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[1] / "modules" / "sd_samplers_kdiffusion.py"
_class = next(node for node in ast.parse(_SOURCE.read_text(encoding="utf-8")).body if getattr(node, "name", None) == "KDiffusionSampler")
_method = next(node for node in _class.body if getattr(node, "name", None) == "get_sigmas")
_OPTS = SimpleNamespace(always_discard_next_to_last_sigma=False, sigma_min=0, sigma_max=0, rho=0)
_GRID = torch.tensor([1.0, 0.5, 0.0])
_KARRAS = torch.tensor([14.6, 1.0, 0.0])
_FLOW = SimpleNamespace(label="Anima FlowMatch", function=lambda n, **_: _GRID, default_rho=-1, need_inner_model=False)
_ns = {
    "opts": _OPTS,
    "torch": torch,
    "devices": SimpleNamespace(cpu="cpu"),
    "logger": logging.getLogger("modules.sd_samplers_kdiffusion"),
    "sd_schedulers": SimpleNamespace(),
}
_KARRAS_SCHEDULER = SimpleNamespace(label="Karras", function=lambda n, **_: _KARRAS, default_rho=7.0, need_inner_model=False)
# The real maps: every scheduler (with aliases), and the ones the user did not hide
_ALL = {"anima_flow_match": _FLOW, "Anima FlowMatch": _FLOW, "Anima Diffusers FlowMatch": _FLOW, "Karras": _KARRAS_SCHEDULER}
_ns["sd_schedulers"].all_schedulers_map = _ns["sd_schedulers"].schedulers_map = _ALL
exec(compile(ast.Module([_method], []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
get_sigmas = _ns["get_sigmas"]


def _sampler(lock):
    options = {"scheduler": "anima_flow_match", "lock_scheduler_override": lock}
    return SimpleNamespace(config=SimpleNamespace(options=options), model_wrap=SimpleNamespace(sigmas=torch.tensor([0.03, 14.6])))


def _process():
    return SimpleNamespace(
        is_hr_pass=False,
        scheduler="Automatic",
        extra_generation_params={},
        sampler_noise_scheduler_override=lambda steps: torch.full((steps + 1,), 9.0),
    )


@pytest.mark.parametrize("lock", [True, False])
def test_locked_samplers_ignore_script_schedule_overrides(lock, caplog):
    p = _process()

    with caplog.at_level(logging.WARNING, logger="modules.sd_samplers_kdiffusion"):
        sigmas = get_sigmas(_sampler(lock), p, 2)

    warned = [record for record in caplog.records if "schedule override" in record.getMessage()]
    if lock:
        assert torch.equal(sigmas, _GRID)
        assert p.extra_generation_params["Anima Flow scheduler lock"] is True
        assert len(warned) == 1  # not only an infotext flag
    else:
        assert torch.equal(sigmas, torch.full((3,), 9.0))
        assert "Anima Flow scheduler lock" not in p.extra_generation_params
        assert warned == []


@pytest.mark.parametrize("lock", [True, False])
@pytest.mark.parametrize("hires", [False, True])
def test_locked_samplers_replace_an_api_scheduler_choice(lock, hires, caplog):
    """API/XYZ runs bypass the UI's greyed-out Schedule type; the lock keeps the
    Anima FlowMatch grid the sampler's tail thresholds were tuned for."""
    p = SimpleNamespace(is_hr_pass=hires, scheduler="Karras", hr_scheduler="Karras", extra_generation_params={}, sampler_noise_scheduler_override=None)

    with caplog.at_level(logging.WARNING, logger="modules.sd_samplers_kdiffusion"):
        sigmas = get_sigmas(_sampler(lock), p, 2)

    key = "Anima Flow replaced hires scheduler" if hires else "Anima Flow replaced scheduler"
    warned = [record for record in caplog.records if "replaced by the sampler" in record.getMessage()]
    if lock:
        assert torch.equal(sigmas, _GRID)
        assert p.extra_generation_params[key] == "Karras"
        assert len(warned) == 1
    else:
        assert torch.equal(sigmas, _KARRAS)
        assert key not in p.extra_generation_params
        assert warned == []


@pytest.mark.parametrize("choice", ["Anima FlowMatch", "anima_flow_match", "Anima Diffusers FlowMatch", "Automatic", None])
def test_locked_samplers_do_not_flag_their_own_scheduler(choice, caplog):
    p = SimpleNamespace(is_hr_pass=False, scheduler=choice, extra_generation_params={}, sampler_noise_scheduler_override=None)

    with caplog.at_level(logging.WARNING, logger="modules.sd_samplers_kdiffusion"):
        sigmas = get_sigmas(_sampler(True), p, 2)

    assert torch.equal(sigmas, _GRID)
    assert "Anima Flow replaced scheduler" not in p.extra_generation_params
    assert p.extra_generation_params["Schedule type"] == "Anima FlowMatch"
    assert caplog.records == []


def test_a_locked_sampler_keeps_its_scheduler_when_the_user_hides_it(monkeypatch, caplog):
    # "Hide schedulers" drops Anima FlowMatch from the dropdowns' map only
    monkeypatch.setattr(_ns["sd_schedulers"], "schedulers_map", {"Karras": _KARRAS_SCHEDULER})
    p = SimpleNamespace(is_hr_pass=False, scheduler="Automatic", extra_generation_params={}, sampler_noise_scheduler_override=None)

    with caplog.at_level(logging.WARNING, logger="modules.sd_samplers_kdiffusion"):
        sigmas = get_sigmas(_sampler(True), p, 2)

    assert torch.equal(sigmas, _GRID)
    assert p.extra_generation_params["Schedule type"] == "Anima FlowMatch"
    assert caplog.records == []
