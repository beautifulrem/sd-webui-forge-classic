from backend.patcher.function_chain import function_chain_contains


def _wrap(previous, marker=None):
    def wrapper(*args, **kwargs):
        return previous(*args, **kwargs)

    wrapper.__wrapped__ = previous
    if marker is not None:
        setattr(wrapper, marker, True)
    return wrapper


def test_finds_marker_below_unrelated_wrappers():
    def base():
        return None

    marked = _wrap(base, "_target")
    outer = _wrap(_wrap(marked, "_different"))
    assert function_chain_contains(outer, "_target")
    assert not function_chain_contains(outer, "_missing")


def test_cycle_is_safe():
    def first():
        return None

    def second():
        return None

    first.__wrapped__ = second
    second.__wrapped__ = first
    assert not function_chain_contains(first, "_missing")
