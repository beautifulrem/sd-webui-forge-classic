"""Inputs of the schedule parity fixture shared with the ``sigma-curve`` widget
(webui/src/widgets/sigma-curve/schedule.fixture.json). Regenerate with
``python extensions-builtin/sd-webui-detail-daemon/tests/schedule_cases.py``."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "webui" / "src" / "widgets" / "sigma-curve" / "schedule.fixture.json"

DEFAULTS = {"start": 0.2, "end": 0.8, "bias": 0.5, "amount": 0.1, "exponent": 1.0, "start_offset": 0.0, "end_offset": 0.0, "fade": 0.0, "smooth": True}

CASES = [
    (50, {}),
    (50, {"smooth": False}),
    (50, {"amount": -0.35, "exponent": 2.5}),
    (50, {"exponent": 0.0}),
    (50, {"start_offset": 0.2, "end_offset": -0.4, "amount": 0.7}),
    (50, {"fade": 0.65, "bias": 0.1}),
    (50, {"start": 0.9, "end": 0.3}),
    (50, {"start": 0.5, "end": 0.5}),
    (50, {"bias": 0.0}),
    (50, {"bias": 1.0, "smooth": False, "exponent": 0.3}),
    (50, {"start": 0.0, "end": 1.0, "amount": 1.0}),
    (20, {"start": 0.25, "end": 0.75}),
    (7, {"start": 0.25, "end": 0.75, "bias": 0.5}),
    (2, {}),
    (1, {"amount": 0.4}),
    (33, {"amount": 3.5, "start_offset": -1.5, "end_offset": 2.0, "fade": 0.2, "exponent": 10.0}),
]


def cases(make_schedule) -> list[dict]:
    """``[{steps, params, values}]`` from the extension's ``make_schedule``."""

    out = []
    for steps, overrides in CASES:
        params = {**DEFAULTS, **overrides}
        out.append({"steps": steps, "params": params, "values": [float(v) for v in make_schedule(steps, **params)]})
    return out


def dump(entries: list[dict]) -> str:
    """The fixture text: one case per line."""

    return "[\n" + ",\n".join(json.dumps(entry) for entry in entries) + "\n]\n"


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from detail_daemon_testlib import load_detail_daemon

    module = load_detail_daemon()
    FIXTURE.write_text(dump(cases(module.Script().make_schedule)), encoding="utf-8")
    print("wrote", FIXTURE)
