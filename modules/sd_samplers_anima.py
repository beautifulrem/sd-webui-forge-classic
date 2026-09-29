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

from modules.anima_support import anima_auxiliary_denoiser, require_anima_flow_denoiser


_EPS = 1e-6
_TAIL_SIGMA = 0.10
_TAIL_GAP = 0.65
_TAIL_GAP_SIGMA = 0.25


def _rf_lambda(t):
    t = torch.clamp(torch.as_tensor(t), min=_EPS, max=1.0 - _EPS)
    return torch.log1p(-t) - torch.log(t)


def _rms(value):
    """Per-sample RMS, shaped [B, 1, ...] to broadcast against the batch: a
    batch-wide mean would couple each image's PC3 correction to its batch-mates.
    Tensors without a batch dim (ndim < 2) reduce to one scalar."""
    value = value.float()
    dims = tuple(range(1, value.ndim))
    if not dims:
        return torch.sqrt(torch.mean(value * value))
    return torch.sqrt(torch.mean(value * value, dim=dims, keepdim=True))


def euler_step(x, denoised, t, t_next):
    """One rectified-flow Euler step from ``t`` to ``t_next``."""
    return x + (t_next - t) * (x - denoised) / t


def _set_denoiser_step(model, step, total_steps):
    """Keep Forge prompt schedules tied to RF intervals, not PC3 call count."""

    if hasattr(model, "step"):
        model.step = int(step)
    if hasattr(model, "total_steps"):
        model.total_steps = int(total_steps)


def denoise_step(model, x, sigmas, i, extra_args, callback):
    """The model call and callback every Flow sampler step starts with."""

    t = sigmas[i]
    _set_denoiser_step(model, i, len(sigmas) - 1)
    denoised = model(x, t * x.new_ones([x.shape[0]]), **extra_args)
    if callback is not None:
        callback({"x": x, "i": i, "sigma": t, "sigma_hat": t, "denoised": denoised})
    return denoised


class _Schedule:
    """The sigma schedule, read to the CPU once.

    Every branch the solvers take (tail policy, degenerate steps, solver
    order) depends only on the schedule, so it is decided from these values
    instead of syncing a device scalar each step. ``lambdas`` is the RF
    log-SNR of each sigma, computed the way the solvers compute it."""

    def __init__(self, sigmas):
        sigmas = sigmas.detach().cpu()
        self.sigmas = sigmas.tolist()
        self.lambdas = _rf_lambda(sigmas)

    def gap(self, i: int, j: int) -> float:
        """``lambda[j] - lambda[i]``."""
        return float(self.lambdas[j] - self.lambdas[i])

    def ratio(self, i: int, j: int, k: int, m: int) -> float:
        """``(lambda[j] - lambda[i]) / (lambda[m] - lambda[k])``."""
        lambdas = self.lambdas
        return float((lambdas[j] - lambdas[i]) / (lambdas[m] - lambdas[k]))

    def tail_policy(self, i: int):
        return _tail_policy(self.sigmas[i], self.sigmas[i + 1], self.gap(i, i + 1))


@torch.no_grad()
def sample_anima_flow_euler(
    model, x, sigmas, extra_args=None, callback=None, disable=None
):
    """Official Anima FlowMatch Euler path without stochastic churn."""

    require_anima_flow_denoiser(model, "Anima Flow samplers")
    extra_args = {} if extra_args is None else extra_args
    values = _Schedule(sigmas).sigmas
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = denoise_step(model, x, sigmas, i, extra_args, callback)
        if values[i + 1] <= 0.0 or values[i] <= 0.0:
            x = denoised
        else:
            x = euler_step(x, denoised, sigmas[i], sigmas[i + 1])
    return x


@dataclass
class _UniPCState:
    model_outputs: tuple = (None, None)
    times: tuple = (None, None)
    lambdas: tuple = (None, None)
    last_sample: object = None
    lower_order_nums: int = 0
    this_order: int = 1


def _tail_policy(t_value, t_next_value, lambda_gap):
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
    b_h = hh if solver_type == "bh1" else h_phi_1

    # The Diffusers-grid variant deliberately caps body order at two. Its
    # predictor coefficient has the closed-form UniP value 1/2.
    if order <= 1 or predictor:
        return h_phi_1, b_h, [0.5]

    (rk,) = rks
    rk = rk.to(like)
    h_phi_2 = h_phi_1 / hh - 1.0
    h_phi_3 = h_phi_2 / hh - 0.5
    v0 = (h_phi_2 / b_h).to(like)
    v1 = (2.0 * h_phi_3 / b_h).to(like)
    # Solves [[1, 1], [rk, 1]] @ rhos = [v0, v1] without torch.linalg.solve,
    # which syncs with the host on CUDA every step. Eliminates in LAPACK's
    # order (partial pivoting, reciprocal scaling) so the rounding matches it.
    swap = torch.abs(rk) > 1.0
    inv_rk = 1.0 / rk
    first, second = torch.where(swap, v1, v0), torch.where(swap, v0, v1)
    multiplier = torch.where(swap, inv_rk, rk)
    rho1 = (second - multiplier * first) * (1.0 / (1.0 - multiplier))
    rho0 = (first - rho1) * torch.where(swap, inv_rk, torch.ones_like(rk))
    return h_phi_1, b_h, [rho0, rho1]


def _unipc_predict(
    x, model_outputs, times, lambdas, t_next, order, *, solver_type, schedule, step
):
    t = torch.clamp(times[-1].to(x), min=_EPS, max=1.0 - _EPS)
    t_next = torch.clamp(torch.as_tensor(t_next).to(x), min=_EPS, max=1.0 - _EPS)
    current_lambda = lambdas[-1]
    h = _rf_lambda(t_next).to(x) - current_lambda
    if schedule.gap(step, step + 1) <= _EPS:
        return euler_step(x, model_outputs[-1], t, t_next)

    denoised = model_outputs[-1]
    hh = -h
    d1s = []
    if order >= 2:
        rk = (lambdas[-2] - current_lambda) / h
        if abs(schedule.ratio(step, step - 1, step, step + 1)) <= _EPS:
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
    schedule,
    step,
):
    previous_denoised = state.model_outputs[-1]
    previous_t = torch.clamp(state.times[-1].to(x), min=_EPS, max=1.0 - _EPS)
    current_t = torch.clamp(current_t.to(x), min=_EPS, max=1.0 - _EPS)
    h = current_lambda - state.lambdas[-1]
    if schedule.gap(step - 1, step) <= _EPS:
        return x

    hh = -h
    d1s = []
    if order >= 2:
        rk = (state.lambdas[-2] - state.lambdas[-1]) / h
        if abs(schedule.ratio(step - 1, step - 2, step - 1, step)) <= _EPS:
            order = 1
        else:
            d1s.append((state.model_outputs[-2] - previous_denoised) / rk)

    h_phi_1, b_h, rhos = _bh_rhos(
        hh,
        order=order,
        like=state.last_sample,
        predictor=False,
        solver_type=solver_type,
        rks=(rk,) if order >= 2 else (),
    )
    base = (current_t / previous_t) * state.last_sample - (
        1.0 - current_t
    ) * h_phi_1 * previous_denoised
    residual = rhos[-1] * (current_denoised - previous_denoised)
    if order >= 2:
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

    require_anima_flow_denoiser(model, "Anima Flow samplers")
    extra_args = {} if extra_args is None else extra_args
    state = _UniPCState()
    total_steps = len(sigmas) - 1
    schedule = _Schedule(sigmas)
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
        denoised = denoise_step(model, x, sigmas, i, extra_args, callback)
        model_output = (
            _threshold_sample(denoised, threshold_ratio, threshold_maximum)
            if bool(flow_unipc_thresholding)
            else denoised
        )

        tail_interval, tail_corrector = schedule.tail_policy(i)
        current_t = torch.clamp(t.to(x), min=_EPS, max=1.0 - _EPS)
        current_lambda = _rf_lambda(t).to(x)
        available = 1 + sum(output is not None for output in state.model_outputs)

        x_corrected = x
        use_corrector = (
            i > 0
            and not tail_corrector
            and i - 1 >= disable_corrector_first
            and state.last_sample is not None
            and state.model_outputs[-1] is not None
        )
        if use_corrector:
            # Upstream truncates the history to the solver order on tail
            # intervals, so a corrector that still runs there is first order.
            corrector_order = 1 if tail_interval else min(max(1, state.this_order), available)
            x_corrected = _unipc_correct(
                state,
                x,
                model_output,
                current_t,
                current_lambda,
                corrector_order,
                solver_type=solver_type,
                schedule=schedule,
                step=i,
            )

        model_outputs = (state.model_outputs[-1], model_output)
        times = (state.times[-1], current_t)
        lambdas = (state.lambdas[-1], current_lambda)

        if schedule.sigmas[i + 1] <= 0.0:
            x_next = model_output
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
                schedule=schedule,
                step=i,
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


def _pc3_third_order_weights(h, k0, k1, h_previous, h_previous_previous):
    k2 = torch.exp(h) * h * h - 2.0 * k1
    a = h_previous
    b = h_previous + h_previous_previous
    w0 = k0 + ((a + b) / (a * b)) * k1 + k2 / (a * b)
    w1 = -(k2 + b * k1) / (a * (b - a))
    w2 = (k2 + a * k1) / (b * (b - a))
    return w0, w1, w2


def _pc3_third_order_is_stable(schedule, step):
    """Upstream's guard for the third-order predictor: similar step ratios, a
    small step, and weights that do not blow up. All of it is schedule-only."""

    lambdas = schedule.lambdas
    h = lambdas[step + 1] - lambdas[step]
    h_previous = lambdas[step] - lambdas[step - 1]
    h_previous_previous = lambdas[step - 1] - lambdas[step - 2]
    r1 = float(h / h_previous)
    r2 = float(h_previous / h_previous_previous)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    w0, w1, w2 = _pc3_third_order_weights(h, k0, k1, h_previous, h_previous_previous)
    coeff_l1 = float((torch.abs(w0) + torch.abs(w1) + torch.abs(w2)) / (torch.abs(k0) + _EPS))
    return 0.5 <= r1 <= 2.0 and 0.5 <= r2 <= 2.0 and float(h) <= 0.55 and coeff_l1 <= 5.0


def _pc3_predict(x, denoised, t, t_next, state, max_order, *, schedule, step):
    t = torch.clamp(t.to(x), min=_EPS)
    t_next = torch.clamp(t_next.to(x), min=0.0)
    current_lambda = _rf_lambda(t).to(x)
    lambda_next = _rf_lambda(t_next).to(x)
    h = lambda_next - current_lambda
    ratio = t_next / t
    if schedule.gap(step, step + 1) <= _EPS:
        return ratio * x + (1.0 - ratio) * denoised, 1

    c = t_next * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    x_p1 = ratio * x + c * k0 * denoised
    if max_order <= 1 or state.previous_denoised is None:
        return x_p1, 1

    h_previous = current_lambda - state.previous_lambda
    if schedule.gap(step - 1, step) <= _EPS:
        return x_p1, 1
    x_p2 = ratio * x + c * (
        (k0 + k1 / h_previous) * denoised - (k1 / h_previous) * state.previous_denoised
    )
    if max_order <= 2 or state.previous_previous_denoised is None:
        return x_p2, 2

    if schedule.gap(step - 2, step - 1) <= _EPS or not _pc3_third_order_is_stable(schedule, step):
        return x_p2, 2
    h_previous_previous = state.previous_lambda - state.previous_previous_lambda
    w0, w1, w2 = _pc3_third_order_weights(h, k0, k1, h_previous, h_previous_previous)
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
    *,
    max_gamma,
    tolerance,
    schedule,
    step,
):
    """Damped correction of a third-order prediction; _pc3_predict only
    returns order 3 when neither neighbouring step gap is empty."""

    current_lambda = _rf_lambda(t).to(x)
    lambda_next = _rf_lambda(t_next).to(x)
    h = lambda_next - current_lambda
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

    # The RMS are per-image [B, 1, ...] float32 tensors. As 0-d tensors they
    # used to promote to the schedule's float64 in the scalar math below; the
    # casts keep that precision (and upstream parity) now that they have dims.
    scalar_dtype = torch.promote_types(torch.float32, x.dtype)
    error = (_rms(x_corrected - x_pred) / (_rms(x_pred) + _EPS)).to(scalar_dtype)
    # A no-op for the driver's device tensor; a float would be a host upload.
    tolerance = torch.as_tensor(tolerance, dtype=x.dtype, device=x.device)
    gamma_error = torch.sqrt(
        torch.clamp(tolerance / (error + _EPS), min=0.0, max=1.0)
    )
    gamma_lambda = torch.sigmoid((current_lambda + 2.5) / 0.5) * torch.sigmoid(
        (4.5 - lambda_next) / 0.8
    )
    gamma = float(max_gamma) * gamma_lambda * gamma_error

    # Cap the correction relative to the predictor step; skipped (via where,
    # without a host sync) when the correction is negligible.
    correction_rms = _rms(x_corrected - x_pred).to(scalar_dtype)
    predictor_step_rms = _rms(x_pred - x)
    anchor_rms = torch.maximum(_rms(x_pred), _rms(x))
    cap_ratio = 0.65 if float(schedule.lambdas[step + 1]) >= 3.5 else 0.90
    correction_cap = torch.maximum(predictor_step_rms * cap_ratio, anchor_rms * 0.015).to(scalar_dtype)
    capped = gamma * torch.clamp(
        correction_cap / (torch.abs(gamma) * correction_rms + _EPS), 0.0, 1.0
    )
    gamma = torch.where(correction_rms > _EPS, capped, gamma)
    # A [B, 1, ...] gamma would promote a half-precision latent to float32.
    return x_pred + gamma.to(x_pred.dtype) * (x_corrected - x_pred)


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

    require_anima_flow_denoiser(model, "Anima Flow samplers")
    extra_args = {} if extra_args is None else extra_args
    state = _PC3State()
    total_steps = len(sigmas) - 1
    schedule = _Schedule(sigmas)
    max_gamma = float(flow_pc3_gamma)
    tolerance = float(flow_pc3_tolerance)
    if not 0.0 <= max_gamma <= 1.0:
        raise ValueError("Anima Flow PC3 gamma must be in [0, 1]")
    if not 0.0 < tolerance <= 1.0:
        raise ValueError("Anima Flow PC3 tolerance must be in (0, 1]")
    # Filled on the device once (no host copy), not uploaded every step.
    tolerance = torch.full((), tolerance, dtype=x.dtype, device=x.device)

    for i in trange(total_steps, disable=disable):
        t, t_next = sigmas[i], sigmas[i + 1]
        denoised = denoise_step(model, x, sigmas, i, extra_args, callback)
        if schedule.sigmas[i + 1] <= 0.0:
            x = denoised
            continue

        tail_interval, _ = schedule.tail_policy(i)
        max_order = 2 if total_steps - i <= 2 else 3
        if tail_interval:
            max_order = 1
        x_pred, predictor_order = _pc3_predict(
            x, denoised, t, t_next, state, max_order, schedule=schedule, step=i
        )

        # Order 3 implies a non-tail step with two previous outputs
        if predictor_order >= 3 and max_gamma > 0.0:
            _set_denoiser_step(model, min(i + 1, total_steps - 1), total_steps)
            with anima_auxiliary_denoiser(model):
                denoised_pred = model(x_pred, t_next * x.new_ones([x.shape[0]]), **extra_args)
            x_next = _pc3_correct(
                x,
                denoised,
                denoised_pred,
                t,
                t_next,
                state,
                x_pred,
                max_gamma=max_gamma,
                tolerance=tolerance,
                schedule=schedule,
                step=i,
            )
        else:
            x_next = x_pred

        state = _pc3_next_state(state, denoised, t)
        x = x_next

    return x
