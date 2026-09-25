from types import SimpleNamespace

import torch

import modules.sd_samplers_anima as anima_samplers

from modules.sd_samplers_anima import (
    _PC3State,
    _bh_rhos,
    _pc3_correct,
    _rf_lambda,
    _threshold_sample,
)


def test_unipc_bh_selector_changes_integration_coefficient():
    like = torch.zeros(1)
    hh = torch.tensor(-0.5)

    _, bh1, _ = _bh_rhos(
        hh,
        order=1,
        like=like,
        predictor=True,
        solver_type="bh1",
    )
    _, bh2, _ = _bh_rhos(
        hh,
        order=1,
        like=like,
        predictor=True,
        solver_type="bh2",
    )

    assert torch.allclose(bh1, hh)
    assert torch.allclose(bh2, torch.expm1(hh))


def test_dynamic_thresholding_is_independent_per_batch_row():
    sample = torch.tensor([[[-4.0, 2.0]], [[-40.0, 20.0]]])

    result = _threshold_sample(sample, 1.0, 10.0)

    assert result.shape == sample.shape
    assert torch.allclose(result[0], torch.tensor([[-1.0, 0.5]]))
    assert torch.allclose(result[1], torch.tensor([[-1.0, 1.0]]))


def test_pc3_zero_gamma_returns_predictor_exactly():
    x = torch.tensor([1.0, 0.5])
    denoised = torch.tensor([0.8, 0.2])
    denoised_pred = torch.tensor([0.7, 0.1])
    x_pred = torch.tensor([0.75, 0.15])
    state = _PC3State(
        previous_denoised=torch.tensor([0.9, 0.3]),
        previous_lambda=_rf_lambda(torch.tensor(0.8)),
        previous_previous_denoised=torch.tensor([1.0, 0.4]),
        previous_previous_lambda=_rf_lambda(torch.tensor(0.9)),
    )

    result = _pc3_correct(
        x,
        denoised,
        denoised_pred,
        torch.tensor(0.7),
        torch.tensor(0.5),
        state,
        x_pred,
        3,
        max_gamma=0.0,
        tolerance=0.005,
    )

    assert torch.equal(result, x_pred)


def test_pc3_marks_endpoint_correction_as_auxiliary(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.p = SimpleNamespace()
            self.step = 0
            self.total_steps = 0
            self.auxiliary_calls = []

        def __call__(self, x, sigma, **_extra_args):
            self.auxiliary_calls.append(
                bool(getattr(self.p, "_anima_auxiliary_denoiser", False))
            )
            return x * 0.9

    monkeypatch.setattr(anima_samplers, "_require_discrete_flow", lambda _model: None)
    model = FakeModel()
    x = torch.ones(1, 1, 2, 2)
    sigmas = torch.tensor([0.95, 0.9, 0.85, 0.8, 0.75, 0.0])

    anima_samplers.sample_anima_flow_pc3(model, x, sigmas, disable=True)

    assert any(model.auxiliary_calls)
    assert not hasattr(model.p, "_anima_auxiliary_denoiser")


def test_unipc_applies_dynamic_thresholding_on_terminal_step(monkeypatch):
    class FakeModel:
        def __init__(self, output):
            self.output = output

        def __call__(self, _x, _sigma, **_extra_args):
            return self.output

    monkeypatch.setattr(anima_samplers, "_require_discrete_flow", lambda _model: None)
    denoised = torch.tensor([[[[-4.0, 2.0]]], [[[-40.0, 20.0]]]])
    model = FakeModel(denoised)

    result = anima_samplers.sample_anima_flow_unipc2(
        model,
        torch.zeros_like(denoised),
        torch.tensor([0.8, 0.0]),
        disable=True,
        flow_unipc_thresholding=True,
        flow_unipc_dynamic_thresholding_ratio=1.0,
        flow_unipc_sample_max_value=10.0,
    )

    assert torch.equal(result, _threshold_sample(denoised, 1.0, 10.0))
