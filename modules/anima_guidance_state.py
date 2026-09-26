"""Lifecycle helpers shared by Anima samplers and guidance extensions."""

from __future__ import annotations

_STATEFUL_GUIDANCE = "_anima_stateful_guidance"


def clear_stateful_guidance(process) -> None:
    """Start a sampling pass with no registered guidance state."""

    setattr(process, _STATEFUL_GUIDANCE, [])


def register_stateful_guidance(process, state):
    """Register an object with ``reset()`` whose history spans sampling steps
    (DCW, SMC-CFG, APG, CFG-Zero*, Momentum), so probe passes can discard it.
    Returns ``state``."""

    registered = getattr(process, _STATEFUL_GUIDANCE, None)
    if registered is None:
        registered = []
        setattr(process, _STATEFUL_GUIDANCE, registered)
    registered.append(state)
    return state


def reset_stateful_guidance(model) -> None:
    """Discard auxiliary/probe-pass history before a real sampling pass."""

    for state in getattr(getattr(model, "p", None), _STATEFUL_GUIDANCE, ()):
        state.reset()
