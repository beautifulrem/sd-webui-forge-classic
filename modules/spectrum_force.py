"""Shared ``forge_spectrum_force_actual`` transformer option.

Features that change the DiT output in ways Spectrum cannot forecast ask it
to run actual forwards. Several can be active in one pass, so the option
holds a tuple of reasons: a truthy string forces every step, a callable of
the current sigma forces only the steps it returns True for.
"""

from __future__ import annotations

KEY = "forge_spectrum_force_actual"


def combine(existing, reason):
    """Return ``existing`` with ``reason`` added, without dropping others."""

    if existing is None or existing is False or existing == "":
        return (reason,)
    reasons = tuple(existing) if isinstance(existing, (tuple, list)) else (existing,)
    return (*reasons, reason)


def request(patcher, reason) -> None:
    """Add ``reason`` to ``patcher``'s transformer options (copy-on-write)."""

    options = dict(patcher.model_options.get("transformer_options", {}))
    options[KEY] = combine(options.get(KEY), reason)
    patcher.model_options["transformer_options"] = options


def is_forced(value, sigma: float) -> bool:
    """Whether any reason in ``value`` forces an actual forward at ``sigma``."""

    reasons = value if isinstance(value, (tuple, list)) else (value,)
    return any(reason(sigma) if callable(reason) else bool(reason) for reason in reasons)


def in_sigma_window(sigma: float, start: float, end: float) -> bool:
    """Inclusive sigma-window test shared by features and their requests."""

    low, high = sorted((float(start), float(end)))
    eps = max(1e-8, high * 1e-7)
    return low - eps <= float(sigma) <= high + eps


def sigma_window(start: float, end: float):
    """Callable reason that forces steps whose sigma lies in [start, end]."""

    return lambda sigma: in_sigma_window(sigma, start, end)
