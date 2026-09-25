"""``lock_scheduler_override``: Anima Flow samplers keep their own grid even
when a script (e.g. XYZ/Hires) installs ``p.sampler_noise_scheduler_override``.

sd_samplers_kdiffusion cannot be imported outside the WebUI (circular
imports), so ``KDiffusionSampler.get_sigmas`` is loaded from its source."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[1] / "modules" / "sd_samplers_kdiffusion.py"
_class = next(node for node in ast.parse(_SOURCE.read_text(encoding="utf-8")).body if getattr(node, "name", None) == "KDiffusionSampler")
_method = next(node for node in _class.body if getattr(node, "name", None) == "get_sigmas")
_OPTS = SimpleNamespace(always_discard_next_to_last_sigma=False, sigma_min=0, sigma_max=0, rho=0)
_GRID = torch.tensor([1.0, 0.5, 0.0])
_ns = {
    "opts": _OPTS,
    "torch": torch,
    "devices": SimpleNamespace(cpu="cpu"),
    "sd_schedulers": SimpleNamespace(
        schedulers_map={"anima_flow_match": SimpleNamespace(label="Anima FlowMatch", function=lambda n, **_: _GRID, default_rho=-1, need_inner_model=False)}
    ),
}
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
def test_locked_samplers_ignore_script_schedule_overrides(lock):
    p = _process()

    sigmas = get_sigmas(_sampler(lock), p, 2)

    if lock:
        assert torch.equal(sigmas, _GRID)
        assert p.extra_generation_params["Anima Flow scheduler lock"] is True
    else:
        assert torch.equal(sigmas, torch.full((3,), 9.0))
        assert "Anima Flow scheduler lock" not in p.extra_generation_params
