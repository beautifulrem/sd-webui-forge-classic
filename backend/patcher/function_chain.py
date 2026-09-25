"""Dependency-free helpers for composable runtime function wrappers."""


def function_chain_contains(function, marker: str) -> bool:
    """Return whether a wrapper chain carries marker, tolerating cycles."""

    visited = set()
    current = function
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if getattr(current, marker, False):
            return True
        current = getattr(current, "__wrapped__", None)
    return False
