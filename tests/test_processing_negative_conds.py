"""CFG 1 skips encoding the negative prompt unless a script keeps it (Anima NAG)."""

import types

import pytest

from remi.testing import webui_state


@pytest.fixture(scope="module", autouse=True)
def webui():
    with webui_state() as state:
        yield state


@pytest.mark.parametrize(
    ("cfg", "keep", "skips"),
    [(1, False, True), (1.0, None, True), (1, True, False), (4.0, False, False), (4.0, True, False)],
)
def test_skips_negative_conds(cfg, keep, skips):
    from modules.processing import StableDiffusionProcessing

    p = types.SimpleNamespace() if keep is None else types.SimpleNamespace(keep_negative_conds=keep)
    assert StableDiffusionProcessing.skips_negative_conds(p, cfg) is skips
