"""Anima DiT feature forecasting for Forge Neo's built-in Spectrum panel.

The cache lives immediately before Anima's ``final_layer``.  Cached steps still
run the timestep embedding, final projection, unpatchify, and the model's native
flow-prediction conversion.  This is intentionally narrower than caching the
whole denoised output: the latter silently reuses stale timestep semantics.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Sequence

import torch

from lib_spectrum.sea import SeaCalibration, l1rel, sea_filter

logger = logging.getLogger("Spectrum")

COMPAT_POLICIES = ("Conservative", "Strict", "Legacy / fastest")


class ChebyshevForecaster:
    """Sliding-window Chebyshev ridge regression with Taylor blending."""

    def __init__(self, degree: int, history_size: int, lam: float, steps: int, weight: float):
        if history_size < degree + 2:
            raise ValueError("Spectrum history size must be at least polynomial degree + 2")
        self.degree = int(degree)
        self.history_size = int(history_size)
        self.lam = float(lam)
        self.steps = max(1, int(steps))
        self.weight = float(weight)
        self.features: list[torch.Tensor] = []
        self.times: list[float] = []
        self.shape = None

    def _tau(self, value):
        return 2.0 * value / self.steps - 1.0

    def _design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        columns = [torch.ones_like(taus)]
        if self.degree > 0:
            columns.append(taus)
        for _ in range(2, self.degree + 1):
            columns.append(2.0 * taus * columns[-1] - columns[-2])
        return torch.cat(columns[: self.degree + 1], dim=1)

    @property
    def ready(self) -> bool:
        return len(self.features) >= max(2, self.degree + 2)

    def update(self, step: int, feature: torch.Tensor) -> None:
        if self.shape is not None and tuple(feature.shape) != self.shape:
            self.clear()
        self.shape = tuple(feature.shape)
        self.features.append(feature.detach().clone())
        self.times.append(float(step))
        if len(self.features) > self.history_size:
            del self.features[0]
            del self.times[0]

    @torch.no_grad()
    def predict(self, step: int) -> torch.Tensor:
        if not self.features:
            raise RuntimeError("Spectrum forecast requested before warmup")
        device = self.features[-1].device
        original_dtype = self.features[-1].dtype
        observations = torch.stack([item.reshape(-1) for item in self.features]).float()
        times = torch.tensor(self.times, device=device, dtype=torch.float32)
        design = self._design(self._tau(times))
        regularizer = self.lam * torch.eye(design.shape[1], device=device, dtype=torch.float32)
        normal = design.T @ design + regularizer
        rhs = design.T @ observations
        try:
            factor = torch.linalg.cholesky(normal)
            coefficients = torch.cholesky_solve(rhs, factor)
        except (RuntimeError, torch.linalg.LinAlgError):
            coefficients = torch.linalg.pinv(normal) @ rhs
        query = torch.tensor([self._tau(float(step))], device=device, dtype=torch.float32)
        chebyshev = (self._design(query) @ coefficients).squeeze(0)

        if len(self.features) >= 2 and self.weight < 1.0:
            latest = self.features[-1].reshape(-1).float()
            previous = self.features[-2].reshape(-1).float()
            dt = max(self.times[-1] - self.times[-2], 1e-8)
            scale = (float(step) - self.times[-1]) / dt
            taylor = latest + scale * (latest - previous)
            prediction = (1.0 - self.weight) * taylor + self.weight * chebyshev
        else:
            prediction = chebyshev
        return prediction.to(original_dtype).reshape(self.shape)

    def clear(self) -> None:
        self.features.clear()
        self.times.clear()
        self.shape = None


def _capture_feature(module, args):
    state = getattr(module, "_forge_spectrum_state", None)
    if state is not None and args:
        state.captured_feature = args[0].detach().clone()


def _ensure_capture_hook(dit) -> None:
    final_layer = dit.final_layer
    if not getattr(final_layer, "_forge_spectrum_hook_installed", False):
        final_layer.register_forward_pre_hook(_capture_feature)
        final_layer._forge_spectrum_hook_installed = True


def _fast_forward(dit, predictor, sigma, input_x, feature):
    model_dtype = next(
        (parameter.dtype for parameter in dit.final_layer.parameters() if parameter.dtype.is_floating_point),
        feature.dtype,
    )
    feature = feature.to(model_dtype)
    internal_timestep = predictor.timestep(sigma).float()
    if internal_timestep.ndim == 1:
        internal_timestep = internal_timestep.unsqueeze(1)
    sinusoidal = dit.t_embedder[0](internal_timestep)
    time_embedding, adaln = dit.t_embedder[1](sinusoidal.to(feature.dtype))
    time_embedding = dit.t_embedding_norm(time_embedding)
    model_output = dit.final_layer(feature, time_embedding, adaln_lora_B_T_3D=adaln)
    model_output = dit.unpatchify(model_output)
    spatial = input_x.shape[-3:]
    model_output = model_output[:, :, : spatial[-3], : spatial[-2], : spatial[-1]]
    return predictor.calculate_denoised(sigma, model_output.float(), input_x)


def _branch_info(args, batch: int):
    raw = args.get("cond_or_uncond", [])
    try:
        branches = [int(item) for item in raw]
    except (TypeError, ValueError):
        return [], False
    valid = bool(branches) and batch % len(branches) == 0
    return branches, valid


def _safe_simple_branches(branches: Sequence[int]) -> bool:
    return len(branches) in (1, 2) and len(set(branches)) == len(branches) and all(item in (0, 1) for item in branches)


def _conditioning_signature(conditioning):
    """Small content signature used to invalidate forecasts on prompt changes."""
    if not isinstance(conditioning, dict):
        return None
    context = conditioning.get("c_crossattn")
    if not torch.is_tensor(context):
        return None
    flattened = context.detach().reshape(-1)
    if flattened.numel() == 0:
        sample = ()
    else:
        stride = max(1, flattened.numel() // 16)
        values = flattened[::stride][:16].float().cpu().tolist()
        sample = tuple(round(float(value), 6) for value in values)
    return tuple(context.shape), str(context.dtype), sample


class SpectrumState:
    def __init__(
        self,
        *,
        steps,
        weight,
        degree,
        lam,
        window_size,
        flex_window,
        warmup_steps,
        tail_actual_steps,
        history_size,
        schedule,
        refresh_ratio,
        sea_beta,
        sea_calibration,
        compat_policy,
        verbose,
    ):
        self.steps = int(steps)
        self.weight = float(weight)
        self.degree = int(degree)
        self.lam = float(lam)
        self.window_size = float(window_size)
        self.flex_window = float(flex_window)
        self.warmup_steps = int(warmup_steps)
        self.tail_actual_steps = int(tail_actual_steps)
        self.history_size = int(history_size)
        self.schedule = str(schedule)
        self.refresh_ratio = float(refresh_ratio)
        self.sea_beta = float(sea_beta)
        self.sea_calibration = sea_calibration
        self.compat_policy = str(compat_policy)
        self.verbose = bool(verbose)
        self.forecasters: dict[Hashable, ChebyshevForecaster] = {}
        self.step = -1
        self.last_sigma = None
        self.current_window = self.window_size
        self.consecutive_cached = 0
        self.mode = "actual"
        self.captured_feature = None
        self.input_shape = None
        self.conditioning_signature = None
        self.sea_previous = None
        self.sea_accumulated = 0.0
        self.sea_distances: list[float] = []
        self.sea_delta = sea_calibration.load() if sea_calibration is not None else None
        self.sea_saved = self.sea_delta is not None
        self.warned_wrapper = False

    def reset(self):
        self.forecasters.clear()
        self.step = -1
        self.last_sigma = None
        self.current_window = self.window_size
        self.consecutive_cached = 0
        self.mode = "actual"
        self.captured_feature = None
        self.input_shape = None
        self.conditioning_signature = None
        self.sea_previous = None
        self.sea_accumulated = 0.0
        self.sea_distances.clear()

    def advance(self, sigma: float, latent: torch.Tensor) -> bool:
        if self.last_sigma is not None and sigma > self.last_sigma + 1e-7:
            self.reset()
        new_step = self.last_sigma is None or abs(sigma - self.last_sigma) > 1e-7
        if not new_step:
            return False
        self.step += 1
        self.last_sigma = sigma
        if self.schedule == "SEA (auto-calibrated)":
            current = sea_filter(latent[0:1], sigma, self.sea_beta)
            if self.sea_previous is not None:
                distance = l1rel(current, self.sea_previous)
                self.sea_accumulated += distance
                if self.warmup_steps <= self.step < self.stop_at and self.sea_delta is None:
                    self.sea_distances.append(distance)
            self.sea_previous = current
        self._finish_sea_calibration_if_ready()
        return True

    @property
    def stop_at(self):
        return max(self.warmup_steps, self.steps - self.tail_actual_steps)

    def _finish_sea_calibration_if_ready(self):
        if self.schedule != "SEA (auto-calibrated)" or self.sea_saved or self.step < self.stop_at:
            return
        if self.sea_calibration is not None and self.sea_distances:
            self.sea_saved = True
            try:
                self.sea_delta = self.sea_calibration.solve_and_save(self.sea_distances)
            except Exception as error:
                logger.warning("Spectrum SEA calibration could not be saved: %s", error)
            else:
                logger.info("Spectrum SEA calibration saved (delta=%.7g); it will be used on the next matching generation.", self.sea_delta)

    def ready(self, branches):
        return all(key in self.forecasters and self.forecasters[key].ready for key in branches)

    def decide(self, branches):
        if self.step < self.warmup_steps or self.step >= self.stop_at or not self.ready(branches):
            return "actual"
        if self.schedule == "SEA (auto-calibrated)" and self.sea_delta is not None:
            return "cached" if self.sea_accumulated < self.sea_delta else "actual"
        divisor = max(1, math.floor(self.current_window))
        return "actual" if (self.consecutive_cached + 1) % divisor == 0 else "cached"

    def actual_completed(self):
        if self.step >= self.warmup_steps:
            self.current_window = round(self.current_window + self.flex_window, 3)
        self.consecutive_cached = 0
        if self.schedule == "SEA (auto-calibrated)" and self.sea_delta is not None:
            self.sea_accumulated = 0.0

    def cached_completed(self):
        self.consecutive_cached += 1


class SpectrumNode:
    @staticmethod
    def patch(
        model,
        steps: int,
        weight: float,
        degree: int,
        lam: float,
        window_size: int,
        flex_window: float,
        warmup_steps: int,
        stop_caching_step: float,
        tail_actual_steps: int,
        history_size: int,
        schedule: str,
        refresh_ratio: float,
        sea_beta: float,
        compat_policy: str,
        verbose: bool,
        sea_cache_dir: str,
        sea_cache_context: dict,
    ):
        history_size = max(int(history_size), int(degree) + 2)
        if compat_policy not in COMPAT_POLICIES:
            logger.warning("Unknown Spectrum compatibility policy %r; using Conservative.", compat_policy)
            compat_policy = "Conservative"

        new_model = model.clone()
        kmodel = new_model.model
        dit = getattr(kmodel, "diffusion_model", None)
        predictor = getattr(kmodel, "predictor", None)
        required = ("final_layer", "t_embedder", "t_embedding_norm", "unpatchify")
        if dit is None or predictor is None or not all(hasattr(dit, name) for name in required):
            raise RuntimeError("Spectrum vNext requires the Anima DiT architecture")

        old_wrapper = new_model.model_options.get("model_function_wrapper")
        # process_before_every_sampling runs again for hires/img2img passes. Do
        # not nest a second Spectrum closure around the first pass's state.
        if getattr(old_wrapper, "__spectrum_feature_cache__", False):
            old_wrapper = getattr(old_wrapper, "__spectrum_previous_wrapper__", None)
        tail_from_fraction = max(0, int(steps) - int(math.ceil(float(steps) * float(stop_caching_step))))
        tail_actual_steps = max(int(tail_actual_steps), tail_from_fraction)
        calibration = None
        if schedule == "SEA (auto-calibrated)":
            calibration = SeaCalibration(
                cache_dir=sea_cache_dir,
                context=sea_cache_context,
                steps=int(steps),
                warmup_steps=int(warmup_steps),
                tail_actual_steps=tail_actual_steps,
                window_size=float(window_size),
                flex_window=float(flex_window),
                refresh_ratio=float(refresh_ratio),
            )
        state = SpectrumState(
            steps=steps,
            weight=weight,
            degree=degree,
            lam=lam,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            history_size=history_size,
            schedule=schedule,
            refresh_ratio=refresh_ratio,
            sea_beta=sea_beta,
            sea_calibration=calibration,
            compat_policy=compat_policy,
            verbose=verbose,
        )
        _ensure_capture_hook(dit)

        def actual_forward(model_function, args):
            if old_wrapper is not None:
                return old_wrapper(model_function, args)
            return model_function(args["input"], args["timestep"], **args["c"])

        def wrapper(model_function, args):
            input_x = args["input"]
            sigma = args["timestep"]
            sigma_value = float(sigma.flatten()[0].item())
            branches, valid_branches = _branch_info(args, input_x.shape[0])
            new_step = state.advance(sigma_value, input_x)

            shape_changed = state.input_shape is not None and tuple(input_x.shape[1:]) != state.input_shape
            conditioning_signature = _conditioning_signature(args.get("c"))
            conditioning_changed = state.conditioning_signature is not None and conditioning_signature != state.conditioning_signature
            if shape_changed or conditioning_changed:
                state.forecasters.clear()
            state.input_shape = tuple(input_x.shape[1:])
            state.conditioning_signature = conditioning_signature

            compatible = valid_branches
            if compat_policy != "Legacy / fastest":
                compatible = compatible and _safe_simple_branches(branches)
                if old_wrapper is not None and not getattr(old_wrapper, "__spectrum_cache_safe__", False):
                    compatible = False
                    if not state.warned_wrapper:
                        logger.warning("Spectrum conservative mode is using actual forwards because an earlier model wrapper is not cache-safe.")
                        state.warned_wrapper = True
            if compat_policy == "Strict":
                compatible = False  # Forge currently exposes no stable per-conditioning UUIDs.

            if new_step:
                state.mode = state.decide(branches) if compatible else "actual"
                if state.verbose:
                    logger.info("Spectrum step %d/%d sigma=%.6g mode=%s", state.step + 1, state.steps, sigma_value, state.mode)
            elif compat_policy != "Legacy / fastest":
                state.mode = "actual"

            if state.mode == "cached" and compatible and state.ready(branches):
                predictions = [state.forecasters[key].predict(state.step) for key in branches]
                feature = torch.cat(predictions, dim=0)
                result = _fast_forward(dit, predictor, sigma, input_x, feature)
                state.cached_completed()
                return result

            dit.final_layer._forge_spectrum_state = state
            state.captured_feature = None
            result = actual_forward(model_function, args)
            feature = state.captured_feature
            dit.final_layer._forge_spectrum_state = None
            if new_step and compatible and feature is not None and feature.shape[0] % len(branches) == 0:
                chunks = feature.chunk(len(branches), dim=0)
                for key, chunk in zip(branches, chunks):
                    forecaster = state.forecasters.get(key)
                    if forecaster is None:
                        forecaster = ChebyshevForecaster(degree, history_size, lam, steps, weight)
                        state.forecasters[key] = forecaster
                    forecaster.update(state.step, chunk)
            if new_step:
                state.actual_completed()
            return result

        wrapper.__spectrum_feature_cache__ = True
        wrapper.__spectrum_previous_wrapper__ = old_wrapper
        new_model.set_model_unet_function_wrapper(wrapper)
        return new_model
