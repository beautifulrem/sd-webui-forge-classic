"""Anima Flow UniPC2 / PC3 against upstream KeithZ117/Comfyui-anima-sampler
(MIT) at effba3c5dc232938aef0a8d19f2c67d4598b46cc: the reference outputs were
produced by upstream's own sampling loop (tests/fixtures/make_anima_flow_upstream_reference.py)
on the closed-form Gaussian denoiser, in float64."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

import modules.sd_samplers_anima as anima_samplers

_FIXTURES = Path(__file__).parent / "fixtures"
_spec = importlib.util.spec_from_file_location("make_anima_flow_upstream_reference", _FIXTURES / "make_anima_flow_upstream_reference.py")
_generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generator)
MU, STD, SCHEDULES, VARIANTS = _generator.MU, _generator.STD, _generator.SCHEDULES, _generator.VARIANTS
_REFERENCE = torch.load(_FIXTURES / "anima_flow_upstream_reference.pt")
_SAMPLERS = {"unipc2": anima_samplers.sample_anima_flow_unipc2, "pc3": anima_samplers.sample_anima_flow_pc3}


def _case_params():
    for key in _REFERENCE["cases"]:
        schedule, variant = key.split("/")
        yield pytest.param(schedule, variant, id=key)


class _GaussianDenoiser:
    """Closed-form RF denoiser for i.i.d. N(MU, STD^2) data, recording its sigmas."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, sigma, **_extra_args):
        self.calls.append(float(sigma.flatten()[0]))
        t = sigma.view(-1, *([1] * (x.ndim - 1)))
        return MU + (1 - t) * STD**2 / ((1 - t) ** 2 * STD**2 + t**2) * (x - (1 - t) * MU)


@pytest.mark.parametrize("schedule, variant", list(_case_params()))
def test_flow_samplers_match_upstream(monkeypatch, schedule, variant):
    monkeypatch.setattr(anima_samplers, "require_anima_flow_denoiser", lambda _model, _feature: None)
    monkeypatch.setattr(anima_samplers, "anima_auxiliary_denoiser", lambda _model: nullcontext())
    reference = _REFERENCE["cases"][f"{schedule}/{variant}"]
    _, _, sampler, kwargs = VARIANTS[variant]
    model = _GaussianDenoiser()

    x = _SAMPLERS[sampler](model, _REFERENCE["noise"].clone(), SCHEDULES[schedule], disable=True, **kwargs)

    assert torch.allclose(torch.tensor(model.calls, dtype=torch.float64), reference["calls"], rtol=0, atol=1e-12)
    torch.testing.assert_close(x, reference["x"], rtol=1e-9, atol=1e-9)


def test_reference_covers_every_variant_and_thresholding_is_exercised():
    assert set(_REFERENCE["cases"]) == {f"{schedule}/{variant}" for schedule in SCHEDULES for variant in VARIANTS}
    cases = _REFERENCE["cases"]
    # the Gaussian data (MU + STD > 1) gets clipped: thresholding is not a no-op here
    assert not torch.allclose(cases["shift3_16/unipc2_thresholding"]["x"], cases["shift3_16/unipc2_bh2"]["x"])
    assert not torch.allclose(cases["shift3_16/unipc2_thresholding_quantile"]["x"], cases["shift3_16/unipc2_thresholding"]["x"])


def test_reference_covers_truncated_and_shortest_schedules():
    cases = _REFERENCE["cases"]
    for variant in VARIANTS:
        # an img2img / Hires pass starts mid-schedule: the first model call is below sigma 1
        assert cases[f"shift3_16_from6/{variant}"]["calls"][0] == SCHEDULES["shift3_16_from6"][0] < 1
        # 1 and 2 steps: one model call per step, the predictor/corrector never has more history
        assert len(cases[f"shift3_1/{variant}"]["calls"]) == 1
        assert len(cases[f"shift3_2/{variant}"]["calls"]) == 2
