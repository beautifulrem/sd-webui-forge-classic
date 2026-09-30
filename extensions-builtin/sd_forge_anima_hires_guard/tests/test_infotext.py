"""A guarded run's infotext replays into the panel it was run from, through
the WebUI's paste and through the API's infotext replay."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the other panels' suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

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
    script = hires_guard.AnimaHiresGuardScript()
    controls = harness.build_ui(script)
    args = [enabled, *CHANGED[1:]]
    assert all(value != control.value for value, control in zip(args[1:], controls[1:]))
    # Off records nothing: the paste and the API leave the panel at its defaults
    expected = args if enabled else [control.value for control in controls]
    params = harness.assert_infotext_round_trip(script, controls, args, _record, expected)
    if enabled:
        # The recorded "False" of the alignment switch replays as False
        assert params["Anima Hires align"] is False and "align 1," in params["Anima Hires limits"]
        assert params["Anima Hires adjustments"].startswith(hires_guard.REPORT_ONLY_PREFIX)
    else:
        assert params == {}
