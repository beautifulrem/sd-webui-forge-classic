"""Temporary instance-level ``forward`` overrides that restore cleanly."""

from __future__ import annotations


_NO_INSTANCE_FORWARD = object()


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
    return previous


def restore_forward_override(module, forward, previous, name: str = "forward") -> bool:
    """Undo :func:`install_forward_override` while ``forward`` is still active."""

    if module.__dict__.get(name) is not forward:
        return False
    if previous is _NO_INSTANCE_FORWARD:
        del module.__dict__[name]
    else:
        module.__dict__[name] = previous
    return True
