"""Temporary instance-level ``forward`` overrides that restore cleanly."""

from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager


_NO_INSTANCE_FORWARD = object()
# module -> number of overrides installed on it and not yet restored. Lets
# e.g. torch.compile wrappers notice per-step patches inside a compiled graph.
_ACTIVE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_ACTIVE_LOCK = threading.Lock()


def overridden_modules() -> list:
    """Modules that currently carry an override from this module."""
    with _ACTIVE_LOCK:
        return list(_ACTIVE.keys())


def install_forward_override(module, forward, name: str = "forward"):
    """Install an instance-level ``forward`` (or another method ``name``) and
    return a restore token.

    The token records whether the instance already had its own attribute.
    Restoring with :func:`restore_forward_override` deletes the override
    instead of pinning the old bound method on the instance, so later
    class-level ``forward`` patches (e.g. NegPiP) keep working.
    """

    previous = module.__dict__.get(name, _NO_INSTANCE_FORWARD)
    setattr(module, name, forward)
    try:
        with _ACTIVE_LOCK:
            _ACTIVE[module] = _ACTIVE.get(module, 0) + 1
    except TypeError:
        pass  # not weak-referenceable: not tracked
    return previous


def restore_forward_override(module, forward, previous, name: str = "forward") -> bool:
    """Undo :func:`install_forward_override` while ``forward`` is still active."""

    if module.__dict__.get(name) is not forward:
        return False
    if previous is _NO_INSTANCE_FORWARD:
        del module.__dict__[name]
    else:
        module.__dict__[name] = previous
    try:
        with _ACTIVE_LOCK:
            count = _ACTIVE.get(module, 0) - 1
            if count > 0:
                _ACTIVE[module] = count
            else:
                _ACTIVE.pop(module, None)
    except TypeError:
        pass
    return True


def cached_overrides(module_at, make_forward):
    """``index -> (module, forward)`` pairs for :func:`forward_overrides`.

    ``forward`` is ``make_forward(original, index)`` around the forward that
    ``module_at(index)`` has at the time. It is built once per index and
    rebuilt only when that forward changes (e.g. a later class-level patch),
    so each model call reuses the same wrappers.
    """

    wrappers = {}  # index -> (forward it wraps, wrapper)

    def override(index):
        module = module_at(index)
        original = module.forward
        cached = wrappers.get(index)
        if cached is None or cached[0] != original:
            cached = wrappers[index] = (original, make_forward(original, index))
        return module, cached[1]

    return override


@contextmanager
def forward_overrides(pairs, name: str = "forward"):
    """Install ``(module, forward)`` overrides for the ``with`` block; they are
    restored in reverse order on exit, also when the block raises."""

    installed = []
    try:
        for module, forward in pairs:
            installed.append((module, forward, install_forward_override(module, forward, name)))
        yield
    finally:
        for module, forward, previous in reversed(installed):
            restore_forward_override(module, forward, previous, name)
