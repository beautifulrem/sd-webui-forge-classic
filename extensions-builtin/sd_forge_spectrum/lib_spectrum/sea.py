"""SEA cache-decision metric and persistent per-configuration calibration."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from collections.abc import Sequence

import torch

from modules.anima_user_data import atomic_replace

_EPS = 1e-8


def radial_frequency(height: int, width: int, device) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(gy.square() + gx.square())


def flow_time(sigma: float, flow: bool) -> float:
    """The noise fraction ``t`` of ``x = (1 - t) x0 + t noise`` that the SEA
    gain is written for. A variance-exploding sigma (``x = x0 + sigma noise``,
    eps/v models) has the same SNR at ``t = sigma / (1 + sigma)``."""

    sigma = float(sigma)
    return sigma if flow else sigma / (1.0 + sigma)


@functools.lru_cache(maxsize=8)
def _power_spectrum(height: int, width: int, beta: float, device: torch.device) -> torch.Tensor:
    """The ``|f|^-beta`` prior of :func:`sea_gain`. Only sigma changes between
    steps, so it is built once per latent size, beta and device (a few
    ``H x W`` float32 tensors); callers must not modify it in place."""

    frequency = radial_frequency(height, width, device)
    return frequency.clamp_min(1.0 / max(height, width)).pow(-beta)


def sea_gain(height: int, width: int, sigma: float, beta: float, device) -> torch.Tensor:
    """Wiener-style gain for a flow latent at noise fraction ``sigma``."""

    signal = 1.0 - float(sigma)
    noise = float(sigma)
    spectrum = _power_spectrum(int(height), int(width), float(beta), torch.device(device))
    gain = signal * spectrum / (signal * signal * spectrum + noise * noise + _EPS)
    return gain / gain.mean().clamp_min(_EPS)


def sea_filter(value: torch.Tensor, sigma: float, beta: float = 2.0) -> torch.Tensor:
    height, width = value.shape[-2:]
    gain = sea_gain(height, width, sigma, beta, value.device)
    transformed = torch.fft.fft2(value.float(), dim=(-2, -1))
    return torch.fft.ifft2(transformed * gain, dim=(-2, -1)).real.to(value.dtype)


def l1rel(current: torch.Tensor, previous: torch.Tensor) -> float:
    if current.shape != previous.shape:
        raise ValueError("SEA relative-distance tensors must have the same shape")
    if current.ndim == 0:
        return float((current - previous).abs() / (previous.abs() + _EPS))

    difference = (current - previous).abs().reshape(current.shape[0], -1).sum(dim=1)
    baseline = previous.abs().reshape(previous.shape[0], -1).sum(dim=1)
    return float((difference / (baseline + _EPS)).max())


def count_refreshes(distances: Sequence[float], delta: float) -> int:
    accumulated = 0.0
    refreshes = 0
    for distance in distances:
        accumulated += float(distance)
        if accumulated >= delta:
            refreshes += 1
            accumulated = 0.0
    return refreshes


def solve_delta_for_refresh_ratio(distances: Sequence[float], refresh_ratio: float, iterations: int = 60) -> float:
    values = [float(value) for value in distances]
    if not values:
        return _EPS
    target = max(1, round(float(refresh_ratio) * len(values)))
    low, high = 0.0, max(sum(values), _EPS)
    for _ in range(iterations):
        midpoint = (low + high) * 0.5
        if count_refreshes(values, midpoint) > target:
            low = midpoint
        else:
            high = midpoint
    return max(high, _EPS)


def window_refresh_fraction(steps, warmup_steps, tail_actual_steps, window_size, flex_window):
    stop_at = max(int(warmup_steps), int(steps) - int(tail_actual_steps))
    current_window = float(window_size)
    consecutive_cached = 0
    actual = 0
    eligible = 0
    for step in range(int(steps)):
        if step < int(warmup_steps) or step >= stop_at:
            do_actual = True
        else:
            divisor = max(1, math.floor(current_window))
            do_actual = (consecutive_cached + 1) % divisor == 0
            eligible += 1
            actual += int(do_actual)
        if do_actual:
            if step >= int(warmup_steps):
                current_window = round(current_window + float(flex_window), 3)
            consecutive_cached = 0
        else:
            consecutive_cached += 1
    return actual / max(1, eligible)


class SeaCalibration:
    """Loads and atomically saves a delta keyed by generation configuration."""

    def __init__(self, *, cache_dir, context, steps, warmup_steps, tail_actual_steps, window_size, flex_window, refresh_ratio, sea_beta):
        self.cache_dir = cache_dir
        self.refresh_ratio = float(refresh_ratio)
        payload = {
            "schema": 1,
            "context": context,
            "steps": int(steps),
            "warmup": int(warmup_steps),
            "tail": int(tail_actual_steps),
            "window": float(window_size),
            "growth": float(flex_window),
            "refresh_ratio": self.refresh_ratio,
            # The distances the delta is solved from are measured through the beta filter.
            "beta": float(sea_beta),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.key = hashlib.sha256(encoded).hexdigest()[:24]
        self.path = os.path.join(cache_dir, f"{self.key}.json")
        self.auto_ratio = window_refresh_fraction(steps, warmup_steps, tail_actual_steps, window_size, flex_window)

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
            value = float(data["delta"])
            return value if math.isfinite(value) and value > 0 else None
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def solve_and_save(self, distances):
        ratio = self.auto_ratio if self.refresh_ratio <= 0 else min(1.0, self.refresh_ratio)
        delta = solve_delta_for_refresh_ratio(distances, ratio)
        payload = json.dumps({"schema": 1, "delta": delta, "refresh_ratio": ratio}).encode("utf-8")
        atomic_replace(self.path, lambda handle: handle.write(payload))
        return delta
