"""The block-weight-grid widget reads weight texts in TypeScript; its vitest
suite compares it with Python results recorded in weights.fixture.json. This
test rebuilds the fixture from the Python it records, so a change on either
side fails one of the suites. ``REMI_REGEN_FIXTURES=1`` rewrites it."""

import json
import os
from pathlib import Path

from modules.anima_lora_support import parse_layer_weights, parse_module_weights

FIXTURE = Path(__file__).resolve().parents[3] / "webui" / "src" / "widgets" / "block-weight-grid" / "weights.fixture.json"


def test_the_widget_fixture_matches_the_python_parsers():
    fixture = json.loads(FIXTURE.read_text(encoding="utf8"))
    layers = []
    for case in fixture["layers"]:
        parsed = parse_layer_weights(case["text"])
        layers.append({"text": case["text"], "fallback": parsed.default, "values": {str(index): value for index, value in parsed.values.items()}})
    modules = [{"text": case["text"], "values": parse_module_weights(case["text"])} for case in fixture["modules"]]
    rebuilt = {**fixture, "layers": layers, "modules": modules}
    if os.environ.get("REMI_REGEN_FIXTURES") == "1":
        FIXTURE.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=0) + "\n", encoding="utf8")
    else:
        assert rebuilt == fixture
