"""Forge-native ports of KeithZ117's Anima Flow corrective solvers.

The sampler ABI is k-diffusion's, but the math follows the upstream
``flow_unipc2_diffusers_x0`` and ``flow_pc3_diffusers_damped`` paths at commit
effba3c5dc232938aef0a8d19f2c67d4598b46cc:
https://github.com/KeithZ117/Comfyui-anima-sampler

Portions copyright (c) 2026 Comfyui-anima-sampler contributors, used under
the MIT License. Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, subject to inclusion of this
notice. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from dataclasses import dataclass

import torch
from tqdm.auto import trange

from modules.anima_support import require_anima_flow_denoiser


_EPS = 1e-6
_TAIL_SIGMA = 0.10
_TAIL_GAP = 0.65
_TAIL_GAP_SIGMA = 0.25


def _require_discrete_flow(model):
    """Reject non-Anima and VP/VE models before treating sigma as RF time."""

    require_anima_flow_denoiser(model, "Anima Flow samplers")


def _rf_lambda(t):
    t = torch.clamp(torch.as_tensor(t), min=_EPS, max=1.0 - _EPS)
    return torch.log1p(-t) - torch.log(t)


def _scalar(value):
    return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)


def _rms(value):
    value = value.float()
    return torch.sqrt(torch.mean(value * value))


def _euler_step(x, denoised, t, t_next):
    if _scalar(t_next) <= 0.0 or _scalar(t) <= 0.0:
        return denoised
    return x + (t_next - t) * (x - denoised) / t


def _set_denoiser_step(model, step, total_steps):
    """Keep Forge prompt schedules tied to RF intervals, not PC3 call count."""

    if hasattr(model, "step"):
        model.step = int(step)
    if hasattr(model, "total_steps"):
        model.total_steps = int(total_steps)


@torch.no_grad()
def sample_anima_flow_euler(
    model, x, sigmas, extra_args=None, callback=None, disable=None
):
    """Official Anima FlowMatch Euler path without stochastic churn."""

    _require_discrete_flow(model)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    for i in trange(total_steps, disable=disable):
        t, t_next = sigmas[i], sigmas[i + 1]
        _set_denoiser_step(model, i, total_steps)
        denoised = model(x, t * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": t, "sigma_hat": t, "denoised": denoised})
        x = _euler_step(x, denoised, t, t_next)
    return x


@dataclass
class _UniPCState:
    model_outputs: tuple = (None, None)
    times: tuple = (None, None)
    lambdas: tuple = (None, None)
    last_sample: object = None
    lower_order_nums: int = 0
    this_order: int = 1


def _tail_policy(t, t_next):
    t_value = _scalar(t)
    t_next_value = _scalar(t_next)
    lambda_gap = _scalar(_rf_lambda(t_next) - _rf_lambda(t))
    terminal = t_next_value <= 0.0
    tail_interval = (
        terminal
        or t_next_value <= _TAIL_SIGMA
        or (t_value <= _TAIL_GAP_SIGMA and lambda_gap >= _TAIL_GAP)
    )
    tail_corrector = terminal or t_value <= _TAIL_SIGMA
    return tail_interval, tail_corrector


def _bh_rhos(hh, *, order, like, predictor, solver_type, rks=()):
    hh = hh.to(device=like.device, dtype=like.dtype)
    h_phi_1 = torch.expm1(hh)
    if solver_type == "bh1":
        b_h = hh
    elif solver_type == "bh2":
        b_h = h_phi_1
    else:
        raise ValueError("Anima Flow UniPC solver type must be 'bh1' or 'bh2'")

    if order <= 1:
        return h_phi_1, b_h, [like.new_tensor(0.5)]
    # The Diffusers-grid variant deliberately caps body order at two. Its
    # predictor coefficient has the closed-form UniP value 1/2.
    if predictor:
        return h_phi_1, b_h, [like.new_tensor(0.5)]

    rks = torch.stack([value.to(like) for value in rks] + [like.new_tensor(1.0)])
    h_phi_2 = h_phi_1 / hh - 1.0
    h_phi_3 = h_phi_2 / hh - 0.5
    matrix = torch.stack([torch.ones_like(rks), rks])
    vector = torch.stack([h_phi_2 / b_h, 2.0 * h_phi_3 / b_h]).to(like)
    rhos = torch.linalg.solve(matrix, vector)
    return h_phi_1, b_h, list(rhos)


def _unipc_predict(
    x, model_outputs, times, lambdas, t_next, order, *, solver_type
):
    t = torch.clamp(times[-1].to(x), min=_EPS, max=1.0 - _EPS)
    t_next = torch.clamp(torch.as_tensor(t_next).to(x), min=_EPS, max=1.0 - _EPS)
    current_lambda = lambdas[-1]
    h = _rf_lambda(t_next).to(x) - current_lambda
    if _scalar(h) <= _EPS:
        return _euler_step(x, model_outputs[-1], t, t_next)

    denoised = model_outputs[-1]
    hh = -h
    d1s = []
    if order >= 2:
        rk = (lambdas[-2] - current_lambda) / h
        if abs(_scalar(rk)) <= _EPS:
            order = 1
        else:
            d1s.append((model_outputs[-2] - denoised) / rk)

    h_phi_1, b_h, rhos = _bh_rhos(
        hh,
        order=order,
        like=x,
        predictor=True,
        solver_type=solver_type,
    )
    base = (t_next / t) * x - (1.0 - t_next) * h_phi_1 * denoised
    if order < 2:
        return base
    return base - (1.0 - t_next) * b_h * rhos[0] * d1s[0]


def _unipc_correct(
    state,
    x,
    current_denoised,
    current_t,
    current_lambda,
    order,
    *,
    solver_type,
):
    previous_denoised = state.model_outputs[-1]
    previous_t = torch.clamp(state.times[-1].to(x), min=_EPS, max=1.0 - _EPS)
    current_t = torch.clamp(current_t.to(x), min=_EPS, max=1.0 - _EPS)
    h = current_lambda - state.lambdas[-1]
    if _scalar(h) <= _EPS:
        return x

    hh = -h
    d1s = []
    if order >= 2:
        rk = (state.lambdas[-2] - state.lambdas[-1]) / h
        if abs(_scalar(rk)) <= _EPS:
            order = 1
        else:
            d1s.append((state.model_outputs[-2] - previous_denoised) / rk)

    h_phi_1, b_h, rhos = _bh_rhos(
        hh,
        order=order,
        like=state.last_sample,
        predictor=False,
        solver_type=solver_type,
        rks=(rk,) if order >= 2 and d1s else (),
    )
    base = (current_t / previous_t) * state.last_sample - (
        1.0 - current_t
    ) * h_phi_1 * previous_denoised
    residual = rhos[-1] * (current_denoised - previous_denoised)
    if order >= 2 and d1s:
        residual = residual + rhos[0] * d1s[0]
    return base - (1.0 - current_t) * b_h * residual


def _threshold_sample(sample, ratio, maximum):
    """Diffusers/UniPC dynamic thresholding applied per latent sample."""

    dtype = sample.dtype
    value = sample.float() if dtype not in (torch.float32, torch.float64) else sample
    batch = int(value.shape[0])
    flattened = value.reshape(batch, -1)
    thresholds = torch.quantile(flattened.abs(), float(ratio), dim=1)
    thresholds = torch.clamp(thresholds, min=1.0, max=float(maximum)).reshape(
        batch, 1
    )
    return (torch.clamp(flattened, -thresholds, thresholds) / thresholds).reshape(
        value.shape
    ).to(dtype)


@torch.no_grad()
def sample_anima_flow_unipc2(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    flow_unipc_solver_type="bh2",
    flow_unipc_disable_corrector_first=0,
    flow_unipc_thresholding=False,
    flow_unipc_dynamic_thresholding_ratio=0.995,
    flow_unipc_sample_max_value=1.0,
):
    """Anima RF x0 UniPC2 with conservative Diffusers-grid tail handling."""

    _require_discrete_flow(model)
    extra_args = {} if extra_args is None else extra_args
    state = _UniPCState()
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    solver_type = str(flow_unipc_solver_type)
    if solver_type not in {"bh1", "bh2"}:
        raise ValueError("Anima Flow UniPC solver type must be 'bh1' or 'bh2'")
    disable_corrector_first = max(0, int(flow_unipc_disable_corrector_first))
    threshold_ratio = float(flow_unipc_dynamic_thresholding_ratio)
    threshold_maximum = float(flow_unipc_sample_max_value)
    if not 0.0 < threshold_ratio <= 1.0:
        raise ValueError("Anima Flow UniPC threshold ratio must be in (0, 1]")
    if threshold_maximum < 1.0:
        raise ValueError("Anima Flow UniPC threshold maximum must be at least 1")

    for i in trange(total_steps, disable=disable):
        t, t_next = sigmas[i], sigmas[i + 1]
        _set_denoiser_step(model, i, total_steps)
        denoised = model(x, t * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": t, "sigma_hat": t, "denoised": denoised})
        model_output = (
            _threshold_sample(denoised, threshold_ratio, threshold_maximum)
            if bool(flow_unipc_thresholding)
            else denoised
        )

        tail_interval, tail_corrector = _tail_policy(t, t_next)
        current_t = torch.clamp(t.to(x), min=_EPS, max=1.0 - _EPS)
        current_lambda = _rf_lambda(t).to(x)
        available = 1 + int(state.model_outputs[-1] is not None)
        available += int(state.model_outputs[-2] is not None)

        x_corrected = x
        use_corrector = (
            i > 0
            and not tail_corrector
            and i - 1 >= disable_corrector_first
            and state.last_sample is not None
            and state.model_outputs[-1] is not None
        )
        if use_corrector:
            corrector_order = min(max(1, state.this_order), available)
            x_corrected = _unipc_correct(
                state,
                x,
                model_output,
                current_t,
                current_lambda,
                corrector_order,
                solver_type=solver_type,
            )

        model_outputs = (state.model_outputs[-1], model_output)
        times = (state.times[-1], current_t)
        lambdas = (state.lambdas[-1], current_lambda)

        if _scalar(t_next) <= 0.0:
            x_next = denoised
            predictor_order = 1
        else:
            requested_order = 1 if tail_interval else 2
            predictor_order = min(requested_order, state.lower_order_nums + 1)
            if model_outputs[-2] is None:
                predictor_order = 1
            x_next = _unipc_predict(
                x_corrected,
                model_outputs,
                times,
                lambdas,
                t_next,
                predictor_order,
                solver_type=solver_type,
            )

        state = _UniPCState(
            model_outputs=model_outputs,
            times=times,
            lambdas=lambdas,
            last_sample=x_corrected,
            lower_order_nums=min(2, state.lower_order_nums + 1),
            this_order=predictor_order,
        )
        x = x_next

    return x


@dataclass
class _PC3State:
    previous_denoised: object = None
    previous_lambda: object = None
    previous_previous_denoised: object = None
    previous_previous_lambda: object = None


def _pc3_next_state(state, denoised, t):
    return _PC3State(
        previous_denoised=denoised,
        previous_lambda=_rf_lambda(t).to(denoised),
        previous_previous_denoised=state.previous_denoised,
        previous_previous_lambda=state.previous_lambda,
    )


def _pc3_predict(x, denoised, t, t_next, state, max_order):
    if _scalar(t_next) <= 0.0:
        return denoised, 1

    t = torch.clamp(t.to(x), min=_EPS)
    t_next = torch.clamp(t_next.to(x), min=0.0)
    current_lambda = _rf_lambda(t).to(x)
    lambda_next = _rf_lambda(t_next).to(x)
    h = lambda_next - current_lambda
    ratio = t_next / t
    if _scalar(h) <= _EPS:
        return ratio * x + (1.0 - ratio) * denoised, 1

    c = t_next * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    x_p1 = ratio * x + c * k0 * denoised
    if max_order <= 1 or state.previous_denoised is None:
        return x_p1, 1

    h_previous = current_lambda - state.previous_lambda
    if _scalar(h_previous) <= _EPS:
        return x_p1, 1
    x_p2 = ratio * x + c * (
        (k0 + k1 / h_previous) * denoised - (k1 / h_previous) * state.previous_denoised
    )
    if max_order <= 2 or state.previous_previous_denoised is None:
        return x_p2, 2

    h_previous_previous = state.previous_lambda - state.previous_previous_lambda
    if _scalar(h_previous_previous) <= _EPS:
        return x_p2, 2
    r1 = _scalar(h / h_previous)
    r2 = _scalar(h_previous / h_previous_previous)
    k2 = torch.exp(h) * h * h - 2.0 * k1
    a = h_previous
    b = h_previous + h_previous_previous
    w0 = k0 + ((a + b) / (a * b)) * k1 + k2 / (a * b)
    w1 = -(k2 + b * k1) / (a * (b - a))
    w2 = (k2 + a * k1) / (b * (b - a))
    coeff_l1 = _scalar(
        (torch.abs(w0) + torch.abs(w1) + torch.abs(w2)) / (torch.abs(k0) + _EPS)
    )
    if (
        not (0.5 <= r1 <= 2.0 and 0.5 <= r2 <= 2.0)
        or _scalar(h) > 0.55
        or coeff_l1 > 5.0
    ):
        return x_p2, 2
    return ratio * x + c * (
        w0 * denoised
        + w1 * state.previous_denoised
        + w2 * state.previous_previous_denoised
    ), 3


def _pc3_correct(
    x,
    denoised,
    denoised_pred,
    t,
    t_next,
    state,
    x_pred,
    predictor_order,
    *,
    max_gamma,
    tolerance,
):
    current_lambda = _rf_lambda(t).to(x)
    lambda_next = _rf_lambda(t_next).to(x)
    h = lambda_next - current_lambda
    h_previous = current_lambda - state.previous_lambda
    if _scalar(h) <= _EPS or _scalar(h_previous) <= _EPS:
        return x_pred

    t_current = torch.clamp(t.to(x), min=_EPS)
    t_next = torch.clamp(t_next.to(x), min=0.0)
    ratio = t_next / t_current
    c = t_next * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    k2 = torch.exp(h) * h * h - 2.0 * k1
    previous_node = state.previous_lambda - current_lambda
    w_prev = (k2 - h * k1) / (previous_node * (previous_node - h))
    w_current = (k2 - (previous_node + h) * k1 + previous_node * h * k0) / (
        previous_node * h
    )
    w_endpoint = (k2 - previous_node * k1) / ((h - previous_node) * h)
    x_corrected = ratio * x + c * (
        w_prev * state.previous_denoised
        + w_current * denoised
        + w_endpoint * denoised_pred
    )

    error = _rms(x_corrected - x_pred) / (_rms(x_pred) + _EPS)
    gamma_error = torch.sqrt(
        torch.clamp(
            x.new_tensor(float(tolerance)) / (error + _EPS), min=0.0, max=1.0
        )
    )
    gamma_lambda = torch.sigmoid((current_lambda + 2.5) / 0.5) * torch.sigmoid(
        (4.5 - lambda_next) / 0.8
    )
    gamma = float(max_gamma) * gamma_lambda * gamma_error

    correction_rms = _rms(x_corrected - x_pred)
    if predictor_order >= 3 and _scalar(correction_rms) > _EPS:
        predictor_step_rms = _rms(x_pred - x)
        anchor_rms = torch.maximum(_rms(x_pred), _rms(x))
        cap_ratio = 0.65 if _scalar(lambda_next) >= 3.5 else 0.90
        correction_cap = torch.maximum(
            predictor_step_rms * cap_ratio, anchor_rms * 0.015
        )
        gamma = gamma * torch.clamp(
            correction_cap / (torch.abs(gamma) * correction_rms + _EPS), 0.0, 1.0
        )
    return x_pred + gamma * (x_corrected - x_pred)


@torch.no_grad()
def sample_anima_flow_pc3(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    flow_pc3_gamma=1.0,
    flow_pc3_tolerance=0.005,
):
    """Anima damped PC3 with upstream's conservative Diffusers-tail policy."""

    _require_discrete_flow(model)
    extra_args = {} if extra_args is None else extra_args
    state = _PC3State()
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    max_gamma = float(flow_pc3_gamma)
    tolerance = float(flow_pc3_tolerance)
    if not 0.0 <= max_gamma <= 1.0:
        raise ValueError("Anima Flow PC3 gamma must be in [0, 1]")
    if not 0.0 < tolerance <= 1.0:
        raise ValueError("Anima Flow PC3 tolerance must be in (0, 1]")

    for i in trange(total_steps, disable=disable):
        t, t_next = sigmas[i], sigmas[i + 1]
        _set_denoiser_step(model, i, total_steps)
        denoised = model(x, t * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": t, "sigma_hat": t, "denoised": denoised})
        if _scalar(t_next) <= 0.0:
            x = denoised
            continue

        tail_interval, _ = _tail_policy(t, t_next)
        max_order = 2 if total_steps - i <= 2 else 3
        if tail_interval:
            max_order = 1
        x_pred, predictor_order = _pc3_predict(x, denoised, t, t_next, state, max_order)

        can_correct = (
            not tail_interval
            and total_steps - i > 2
            and predictor_order >= 3
            and state.previous_previous_denoised is not None
            and max_gamma > 0.0
        )
        if can_correct:
            _set_denoiser_step(model, min(i + 1, total_steps - 1), total_steps)
            denoised_pred = model(x_pred, t_next * s_in, **extra_args)
            x_next = _pc3_correct(
                x,
                denoised,
                denoised_pred,
                t,
                t_next,
                state,
                x_pred,
                predictor_order,
                max_gamma=max_gamma,
                tolerance=tolerance,
            )
        else:
            x_next = x_pred

        state = _pc3_next_state(state, denoised, t)
        x = x_next

    return x
