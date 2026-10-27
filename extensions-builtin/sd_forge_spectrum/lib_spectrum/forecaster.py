"""Spectrum feature forecasting for Forge Neo's built-in Spectrum panel.

On Anima the cache lives immediately before the DiT ``final_layer``.  Cached
steps still run the timestep embedding, final projection, unpatchify, and the
model's native flow-prediction conversion.  This is intentionally narrower than
caching the whole denoised output: the latter silently reuses stale timestep
semantics.  Other architectures fall back to forecasting the model output, as
upstream Forge Neo's Spectrum does.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Hashable, Sequence

import torch

from lib_spectrum.sea import SeaCalibration, flow_time, l1rel, sea_filter
from modules import spectrum_force
from modules.anima_support import call_model_wrapper, is_anima_auxiliary_denoiser, process_ref, sigma_value

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
    def predict(self, step: int, out: torch.Tensor | None = None) -> torch.Tensor:
        """The forecast feature at ``step`` in the stored features' dtype,
        written into ``out`` (e.g. a branch's rows of a batched feature) when
        given."""

        if not self.features:
            raise RuntimeError("Spectrum forecast requested before warmup")
        if out is not None and tuple(out.shape) != self.shape:
            raise ValueError(f"Spectrum forecast of shape {self.shape} cannot be written to {tuple(out.shape)}")
        original_dtype = self.features[-1].dtype
        # The ridge system is only (history x degree): solve it on the CPU
        # instead of a dozen tiny GPU kernels plus a readback per branch.
        # In float64: at Regularization 0 the normal matrix's condition
        # number reaches ~1e9 at degree 8, beyond float32's precision.
        times = torch.tensor(self.times, dtype=torch.float64)
        design = self._design(self._tau(times))
        regularizer = self.lam * torch.eye(design.shape[1], dtype=torch.float64)
        normal = design.T @ design + regularizer
        query = torch.tensor([self._tau(float(step))], dtype=torch.float64)
        query_design = self._design(query)
        try:
            factor = torch.linalg.cholesky(normal)
            history_projection = torch.cholesky_solve(design.T, factor)
        except (RuntimeError, torch.linalg.LinAlgError):
            history_projection = torch.linalg.pinv(normal) @ design.T
        weights = (query_design @ history_projection).squeeze(0).tolist()
        # The forecast is a weighted sum of the stored features: the
        # Chebyshev fit and the Taylor extrapolation
        # ``latest + scale * (latest - previous)`` are both linear in them.
        # Blending in this tiny weight space leaves one accumulator as the
        # only feature-sized temporary.
        if len(self.features) >= 2 and self.weight < 1.0:
            dt = max(self.times[-1] - self.times[-2], 1e-8)
            scale = (float(step) - self.times[-1]) / dt
            taylor_share = 1.0 - self.weight
            weights = [self.weight * weight for weight in weights]
            weights[-1] += taylor_share * (1.0 + scale)
            weights[-2] -= taylor_share * scale
        if out is None:
            out = torch.empty(self.shape, dtype=original_dtype, device=self.features[-1].device)
        # Accumulate in float32: straight into ``out`` when that is float32,
        # else in one scratch cast into ``out`` at the end.
        prediction = out if out.dtype == torch.float32 else torch.empty(self.shape, dtype=torch.float32, device=out.device)
        prediction.zero_()
        for weight, feature in zip(weights, self.features):
            prediction.add_(feature.reshape(self.shape), alpha=float(weight))
        if prediction is not out:
            out.copy_(prediction)
        return out

    def clear(self) -> None:
        self.features.clear()
        self.times.clear()
        self.shape = None


def predict_branches(forecasters: Sequence[ChebyshevForecaster], step: int) -> torch.Tensor:
    """The forecasts of ``forecasters`` (one per branch, in batch order)
    concatenated along the batch, each written into its rows of one output."""

    latest = forecasters[0].features[-1]
    rows = [forecaster.shape[0] for forecaster in forecasters]
    output = torch.empty((sum(rows), *forecasters[0].shape[1:]), dtype=latest.dtype, device=latest.device)
    start = 0
    for forecaster, count in zip(forecasters, rows):
        forecaster.predict(step, out=output[start : start + count])
        start += count
    return output


def _is_flow_predictor(predictor) -> bool:
    from backend.modules.k_prediction import PredictionDiscreteFlow, PredictionFlux

    return isinstance(predictor, (PredictionDiscreteFlow, PredictionFlux))


def _capture_feature(module, args):
    local = getattr(module, "_forge_spectrum_local", None)
    state = getattr(local, "state", None)
    if state is not None and args:
        # No copy here: the forecasters clone the per-branch slices they keep.
        state.captured_feature = args[0].detach()


def _ensure_capture_hook(dit) -> None:
    final_layer = dit.final_layer
    if not hasattr(final_layer, "_forge_spectrum_local"):
        final_layer._forge_spectrum_local = threading.local()
    if not getattr(final_layer, "_forge_spectrum_hook_installed", False):
        final_layer.register_forward_pre_hook(_capture_feature)
        final_layer._forge_spectrum_hook_installed = True


def _fast_forward(dit, predictor, sigma, input_x, feature):
    # The feature keeps the compute dtype final_layer saw in the real forward
    # (Anima casts to the context dtype); the parameters' storage dtype (bf16,
    # FP8, ...) is not what the manual-cast layers compute in.
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


def _branch_keys(branches: Sequence[int]) -> list:
    """Per-chunk keys of a call's forecasters and signatures: the branch
    value, and ``(branch, n)`` for its ``n``-th repeat in the call (AND
    sub-prompts share ``cond_or_uncond`` 0), so repeated chunks are never
    merged into one history."""
    seen: dict[int, int] = {}
    keys = []
    for branch in branches:
        count = seen.get(branch, 0)
        seen[branch] = count + 1
        keys.append(branch if count == 0 else (branch, count))
    return keys


def _safe_simple_branches(branches: Sequence[int]) -> bool:
    return len(branches) in (1, 2) and len(set(branches)) == len(branches) and all(item in (0, 1) for item in branches)


def _call_samples(timestep, conditioning, branches: Sequence[int]):
    """A model call's sigma and its per-branch conditioning signatures (None
    when the conditioning cannot be split), read with one host transfer.

    A signature is a small content sample used to invalidate forecasts on
    prompt changes: ``(branch shape, dtype, up to 16 strided values)``,
    computed every call.  The sigma and every branch's sample are gathered on
    the device and copied together; each branch is a contiguous run of the
    flattened context.  Without a splittable context (or with the timestep on
    another device) the sigma is read on its own through ``sigma_value``.
    """
    context = conditioning.get("c_crossattn") if isinstance(conditioning, dict) and branches else None
    if not torch.is_tensor(context) or context.shape[0] % len(branches):
        return sigma_value(timestep), None
    count = len(branches)
    shape = (context.shape[0] // count, *context.shape[1:])
    per_branch = context.numel() // count
    fused = torch.is_tensor(timestep) and timestep.numel() > 0 and timestep.device == context.device
    # float64 sigmas stay exact; context samples upcast losslessly.
    work_dtype = torch.promote_types(timestep.dtype, torch.float32) if fused else torch.float32
    parts = [timestep.detach().reshape(-1)[:1].to(work_dtype)] if fused else []
    size = 0
    if per_branch:
        stride = max(1, per_branch // 16)
        offsets = torch.arange(0, per_branch, stride, device=context.device)[:16]
        index = (torch.arange(count, device=context.device)[:, None] * per_branch + offsets).reshape(-1)
        parts.append(context.detach().reshape(-1).index_select(0, index).to(work_dtype))
        size = offsets.numel()
    values = torch.cat(parts).cpu().tolist() if parts else []
    sigma = float(values.pop(0)) if fused else sigma_value(timestep)
    samples = [tuple(round(float(value), 6) for value in values[i * size : (i + 1) * size]) for i in range(count)]
    dtype = str(context.dtype)
    return sigma, {key: (shape, dtype, sample) for key, sample in zip(branches, samples)}


def _replaced_blocks(transformer_options) -> tuple:
    """The blocks a call's ``patches_replace`` swaps out (``("dit",
    "('single_block', 22)")``, ...), as a sorted tuple; empty without any.

    A replaced block changes what the model returns for that call, and a
    script can replace blocks for one branch over part of the pass (Skip
    Layer Guidance: the uncond call of a Moss-Image step skips layers between
    two percents). Part of the branch's signature, so its history is rebuilt
    instead of forecast across the switch."""

    replaced = transformer_options.get("patches_replace") if isinstance(transformer_options, dict) else None
    if not isinstance(replaced, dict):
        return ()
    return tuple(sorted((str(name), repr(block)) for name, blocks in replaced.items() if isinstance(blocks, dict) for block in blocks))


class SpectrumState:
    def __init__(
        self,
        *,
        steps,
        window_size,
        flex_window,
        warmup_steps,
        tail_actual_steps,
        schedule,
        sea_beta,
        sea_calibration,
        verbose,
        flow_sigmas=True,
    ):
        self.steps = int(steps)
        self.window_size = float(window_size)
        self.flex_window = float(flex_window)
        self.warmup_steps = int(warmup_steps)
        self.tail_actual_steps = int(tail_actual_steps)
        self.schedule = str(schedule)
        self.sea_beta = float(sea_beta)
        self.sea_calibration = sea_calibration
        self.flow_sigmas = bool(flow_sigmas)
        self.verbose = bool(verbose)
        self.forecasters: dict[Hashable, ChebyshevForecaster] = {}
        self.sea_distances: list[float] = []
        self.sea_delta = sea_calibration.load() if sea_calibration is not None else None
        self.sea_saved = self.sea_delta is not None
        self.warned_wrapper = False
        self.reset()

    def reset(self):
        self.forecasters.clear()
        self.step = -1
        # Branches the current step's model calls have run so far
        self.step_branches: set = set()
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
        self.was_forced = False

    def advance(self, sigma: float, latent: torch.Tensor, copies: int = 1) -> bool:
        """``copies``: how many branch chunks (cond, uncond, AND sub-prompts)
        the call batches ``latent`` into. Each chunk is the same x, so SEA
        filters and compares only the first one."""

        if self.last_sigma is not None and sigma > self.last_sigma + 1e-7:
            self.reset()
        new_step = self.last_sigma is None or abs(sigma - self.last_sigma) > 1e-7
        if not new_step:
            return False
        self.step += 1
        self.last_sigma = sigma
        if self.schedule == "SEA (auto-calibrated)":
            if copies > 1 and latent.shape[0] % copies == 0:
                latent = latent[: latent.shape[0] // copies]
            current = sea_filter(latent, flow_time(sigma, self.flow_sigmas), self.sea_beta)
            previous = self.sea_previous
            if previous is not None and current.shape[1:] == previous.shape[1:]:
                # Without branch info (copies 1) a CFG range dropping the
                # uncond copies still changes the batch (2 -> 1). Every branch
                # chunk is the same x, so row i is image i % batch in both:
                # compare the rows the two steps share.
                rows = min(current.shape[0], previous.shape[0])
                distance = l1rel(current[:rows], previous[:rows])
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

    @property
    def forecasting_ended(self) -> bool:
        """Every step from here on runs actual (see :meth:`decide`)."""
        return self.step >= self.stop_at

    @property
    def feeds_forecasts(self) -> bool:
        """Whether this step's feature can still serve a forecast: the last
        step that may be cached is ``stop_at - 1``, and it can only use
        features from the steps before it."""
        return self.step < self.stop_at - 1

    def ready(self, branches):
        return all(key in self.forecasters and self.forecasters[key].ready for key in branches)

    def decide(self, branches):
        if self.step < self.warmup_steps or self.forecasting_ended or not self.ready(branches):
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
        process=None,
    ):
        history_size = max(int(history_size), int(degree) + 2)
        if compat_policy not in COMPAT_POLICIES:
            logger.warning("Unknown Spectrum compatibility policy %r; using Conservative.", compat_policy)
            compat_policy = "Conservative"
        if compat_policy == "Strict":
            # Forge exposes no stable per-conditioning UUIDs, so Strict never
            # caches: skip the wrapper and its per-call bookkeeping entirely.
            logger.info("Spectrum Strict policy runs every step actual; Spectrum is not installed for this pass.")
            return model

        new_model = model.clone()
        kmodel = new_model.model
        dit = getattr(kmodel, "diffusion_model", None)
        predictor = getattr(kmodel, "predictor", None)
        required = ("final_layer", "t_embedder", "t_embedding_norm", "unpatchify")
        feature_cache = dit is not None and predictor is not None and all(hasattr(dit, name) for name in required)

        # Every pass starts from forge_objects_after_applying_lora, so any
        # existing wrapper was installed in this pass (scripts run in load
        # order, e.g. Guidance/Regional before Spectrum) and must be kept.
        old_wrapper = new_model.model_options.get("model_function_wrapper")
        # Upstream Forge Neo runs actual from step int(steps * stop) on (25 x
        # 0.9: 3 tail steps); the epsilon only absorbs float noise (90 x 0.7).
        tail_from_fraction = max(0, int(steps) - int(float(steps) * float(stop_caching_step) + 1e-9))
        tail_actual_steps = max(int(tail_actual_steps), tail_from_fraction)
        calibration = None
        if schedule == "SEA (auto-calibrated)":
            calibration = SeaCalibration(
                cache_dir=sea_cache_dir,
                # Latent statistics differ per architecture, so calibrations
                # must not be shared between e.g. SDXL and Anima.
                context={
                    **dict(sea_cache_context or {}),
                    "model": type(dit if dit is not None else kmodel).__name__,
                    "cache": "feature" if feature_cache else "output",
                },
                steps=int(steps),
                warmup_steps=int(warmup_steps),
                tail_actual_steps=tail_actual_steps,
                window_size=float(window_size),
                flex_window=float(flex_window),
                refresh_ratio=float(refresh_ratio),
                sea_beta=float(sea_beta),
            )
        state = SpectrumState(
            steps=steps,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            schedule=schedule,
            sea_beta=sea_beta,
            sea_calibration=calibration,
            verbose=verbose,
            flow_sigmas=_is_flow_predictor(predictor),
        )
        if feature_cache:
            _ensure_capture_hook(dit)
        if process is not None:
            # Lets the script free the (GPU) feature history after generation.
            # Earlier passes are finished, so free their history right away
            # instead of keeping it alive (e.g. during the larger hires pass).
            for previous_state in getattr(process, "_spectrum_states", None) or ():
                previous_state.reset()
            setattr(process, "_spectrum_states", [state])
        # The wrapper lives on the patcher past this generation: no strong p
        process = process_ref(process)

        def actual_forward(model_function, args):
            return call_model_wrapper(old_wrapper, model_function, args)

        def wrapper(model_function, args):
            if is_anima_auxiliary_denoiser(process):
                return actual_forward(model_function, args)

            input_x = args["input"]
            sigma = args["timestep"]
            branches, valid_branches = _branch_info(args, input_x.shape[0])
            keys = _branch_keys(branches)
            current_sigma, signatures = _call_samples(sigma, args.get("c") if valid_branches else None, keys)
            transformer_options = args.get("c", {}).get("transformer_options", {})
            if signatures is not None and (replaced := _replaced_blocks(transformer_options)):
                signatures = {key: (*signature, replaced) for key, signature in signatures.items()}
            new_step = state.advance(current_sigma, input_x, len(branches) if valid_branches else 1)
            # When one CFG batch does not fit, Forge runs uncond and cond as
            # separate calls at the same sigma (uncond first): each branch's
            # first call at a sigma is its step, not a repeated-sigma call.
            branch_step = not new_step and not state.step_branches.intersection(keys)

            shape_changed = state.input_shape is not None and tuple(input_x.shape[1:]) != state.input_shape
            previous_signatures = state.conditioning_signature
            if shape_changed or signatures is None or previous_signatures is None:
                if shape_changed or signatures != previous_signatures:
                    state.forecasters.clear()
                state.conditioning_signature = signatures
            else:
                # Per branch: the unconditional one may drop out of a step
                # (CFG 1, a limited CFG range) without invalidating the
                # conditional forecaster's history.
                for key, signature in signatures.items():
                    if key in previous_signatures and previous_signatures[key] != signature:
                        state.forecasters.pop(key, None)
                state.conditioning_signature = {**previous_signatures, **signatures}
            if new_step:
                # A branch missing from a step would forecast across the gap.
                # Judged on the whole previous step: a split step's later
                # calls bring the other branch.
                for key in [key for key in state.forecasters if key not in state.step_branches]:
                    del state.forecasters[key]
                state.step_branches = set(keys)
                if state.forecasting_ended:
                    # Nothing is forecast again this pass: free the
                    # (full-resolution) history before the tail, VAE decode
                    # and Hires rather than when the next pass starts.
                    state.forecasters.clear()
            else:
                state.step_branches.update(keys)
            state.input_shape = tuple(input_x.shape[1:])

            compatible = valid_branches
            forced = spectrum_force.is_forced(transformer_options.get(spectrum_force.KEY), current_sigma)
            if new_step:
                if state.was_forced and not forced:
                    # Forecasters were not updated inside the forced window;
                    # rebuild history instead of extrapolating across it.
                    state.forecasters.clear()
                    state.current_window = state.window_size
                    state.consecutive_cached = 0
                state.was_forced = forced
            if forced:
                compatible = False
            if compat_policy != "Legacy / fastest":
                compatible = compatible and _safe_simple_branches(branches)
                if old_wrapper is not None and not getattr(old_wrapper, "__spectrum_cache_safe__", False):
                    compatible = False
                    if not state.warned_wrapper:
                        logger.warning("Spectrum conservative mode is using actual forwards because an earlier model wrapper is not cache-safe.")
                        state.warned_wrapper = True

            if new_step:
                state.mode = state.decide(keys) if compatible else "actual"
                if state.verbose:
                    logger.info("Spectrum step %d/%d sigma=%.6g mode=%s", state.step + 1, state.steps, current_sigma, state.mode)
            elif compat_policy != "Legacy / fastest" and not branch_step:
                state.mode = "actual"

            # A split step's later branch follows the step's decision, as the
            # same branch would in one batch; it is not another cached step.
            if state.mode == "cached" and compatible and state.ready(keys):
                feature = predict_branches([state.forecasters[key] for key in keys], state.step)
                if feature_cache:
                    result = _fast_forward(dit, predictor, sigma, input_x, feature)
                else:
                    result = feature.to(input_x.dtype)
                if not branch_step:
                    state.cached_completed()
                return result

            # Only a new, cacheable step (of these branches) a later forecast
            # can use feeds the forecasters: skip capturing (a full-resolution
            # feature) on forced / repeated / other calls and from the last
            # cacheable step on.
            feeds = (new_step or branch_step) and compatible and state.feeds_forecasts
            if feature_cache and feeds:
                capture_local = dit.final_layer._forge_spectrum_local
                capture_local.state = state
                state.captured_feature = None
                feature = None
                try:
                    result = actual_forward(model_function, args)
                    feature = state.captured_feature
                finally:
                    capture_local.state = None
                    state.captured_feature = None
            elif feature_cache:
                result = actual_forward(model_function, args)
                feature = None
            else:
                result = actual_forward(model_function, args)
                feature = result
            if feeds and feature is not None and feature.shape[0] % len(keys) == 0:
                chunks = feature.chunk(len(keys), dim=0)
                for key, chunk in zip(keys, chunks):
                    forecaster = state.forecasters.get(key)
                    if forecaster is None:
                        forecaster = ChebyshevForecaster(degree, history_size, lam, steps, weight)
                        state.forecasters[key] = forecaster
                    forecaster.update(state.step, chunk)
            if new_step:
                state.actual_completed()
            return result

        new_model.set_model_unet_function_wrapper(wrapper)
        return new_model
