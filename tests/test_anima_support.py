from types import SimpleNamespace

from modules.anima_support import is_anima_engine


def test_anima_capability_marker_accepts_wrapped_engines():
    assert is_anima_engine(SimpleNamespace(is_anima_engine=True))


def test_models_without_anima_capability_are_rejected():
    assert not is_anima_engine(SimpleNamespace())
    assert not is_anima_engine(None)
