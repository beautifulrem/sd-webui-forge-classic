"""The stage-timeline widget redraws the schedule in TypeScript; its vitest
suites compare it with Python results recorded in JSON fixtures. These tests
rebuild the fixtures from the Python they record, so a change on either side
fails one of the suites. ``REMI_REGEN_FIXTURES=1`` rewrites them."""

import json
import os
from pathlib import Path

import pytest
from lib_stage_scheduler import options, runtime, templates, ui

from modules.anima_lora_support import stage_window, window_curve_factor

WIDGETS = Path(__file__).resolve().parents[3] / "webui" / "src" / "widgets"
SCHEDULE = WIDGETS / "_anima" / "schedule.fixture.json"
MODEL = WIDGETS / "stage-timeline" / "model.fixture.json"


def _check(path, rebuilt):
    if os.environ.get("REMI_REGEN_FIXTURES") == "1":
        path.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=0) + "\n", encoding="utf8")
    else:
        assert rebuilt == json.loads(path.read_text(encoding="utf8"))


def _schedule(fixture):
    """``stage_window`` / ``window_curve_factor`` at the fixture's inputs."""
    points = fixture["points"]
    return {
        **fixture,
        "windows": [{**case, "window": list(stage_window(case["stage"], case["shift"]))} for case in fixture["windows"]],
        "curves": [
            {**case, "values": [window_curve_factor(x, case["start"], case["end"], case["peak"], case["shape"], case["floor"]) for x in points]}
            for case in fixture["curves"]
        ],
    }


def _install(fixture):
    """The fixture's saved (not built-in) templates in the template store."""

    def add(data):
        for name, template in fixture["templates"].items():
            if not template["builtin"]:
                data[name] = {"auto": template["auto"], "stage": template["stage"], "parts": template["parts"], **({"curve": template["curve"]} if template["curve"] else {})}
        return None, ""

    templates._TEMPLATES.edit(add)


def _model(fixture):
    """The widget's template data and ``_stage_curve_factor`` of each case,
    run as ``start_run`` builds a pass."""
    payload = ui.templates_payload()
    cases = []
    for case in fixture["cases"]:
        timing = case["timing"]
        config = runtime._build_pass_config(
            options.MODE_PROMPT_ALL, "", "", case["template"], "", timing["auto"], None, timing["start"], timing["end"], timing["peak"], timing["strength"], timing["curve"], case["shift"]
        )
        cases.append({**case, "factors": [runtime._stage_curve_factor(config, x) for x in fixture["points"]]})
    return {**fixture, "templates": {name: payload[name] for name in fixture["templates"]}, "cases": cases}


@pytest.mark.parametrize("path, rebuild", [(SCHEDULE, _schedule), (MODEL, _model)], ids=["schedule", "model"])
def test_the_widget_fixtures_match_the_python_schedule(path, rebuild):
    fixture = json.loads(path.read_text(encoding="utf8"))
    if path == MODEL:
        _install(fixture)
    _check(path, rebuild(fixture))
