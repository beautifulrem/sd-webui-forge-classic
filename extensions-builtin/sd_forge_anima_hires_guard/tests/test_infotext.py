"""A guarded run's infotext replays into the panel it was run from, through
the paste (onto the panel's values) and the API's infotext replay (onto the
defaults); both read it through the panel's params() layout."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import remi_script_harness as harness

ROOT = Path(__file__).resolve().parents[1]
hires_guard = harness.load_script(ROOT / "scripts" / "anima_hires_guard.py", "anima_hires_guard_infotext_under_test")

# Every control away from its default: report only, custom limits, alignment off
CHANGED = [True, "Report only", "Custom", False, 1.0, 3.0, 2.5]


def _record(args):
    """The infotext a guarded Hires run with the script args ``args`` records."""
    sd_model = SimpleNamespace(is_anima_engine=True)
    p = SimpleNamespace(sd_model=sd_model, width=1536, height=1024, enable_hr=True, hr_scale=2.0, hr_resize_x=0, hr_resize_y=0, extra_generation_params={})
    hires_guard.AnimaHiresGuardScript().after_model_load(p, *args)
    return p.extra_generation_params


@pytest.mark.parametrize("enabled", [True, False], ids=["every-control-changed", "off"])
def test_the_infotext_replays_into_the_same_script_args(enabled):
    layout = harness.script_layout(hires_guard.AnimaHiresGuardScript())
    defaults = layout.values_to_args({})
    args = [enabled, *CHANGED[1:]]
    assert all(value != default for value, default in zip(args[1:], defaults[1:]))
    params = _record(args)
    # Off records nothing: the paste and the API leave the panel at its defaults
    expected = args if enabled else defaults
    replayed = harness.replayed_args(layout, params)
    assert replayed == expected
    assert _record(replayed) == params
    if enabled:
        # The recorded "False" of the alignment switch replays as False
        assert params["Anima Hires align"] is False and "align 1," in params["Anima Hires limits"]
        assert params["Anima Hires adjustments"].startswith(hires_guard.REPORT_ONLY_PREFIX)
        # Onto a panel that was set otherwise, the recorded values win
        assert harness.replayed_args(layout, params, start=dict(zip(layout.names, defaults))) == args
    else:
        assert params == {}
