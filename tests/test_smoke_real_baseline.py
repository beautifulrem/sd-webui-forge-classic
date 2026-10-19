"""tools/smoke_real.py reads its parity baseline from the committed fixture by
default (a /tmp copy is purged by macOS), or from a recorded request/response pair."""

import importlib.util
import json
import sys
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "smoke_real.py"
_spec = importlib.util.spec_from_file_location("smoke_real", _TOOL)
smoke = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("smoke_real", smoke)  # its dataclasses look their module up
_spec.loader.exec_module(smoke)


def test_the_committed_fixture_is_the_default_baseline():
    params, infotext, image = smoke.load_baseline(smoke.FIXTURE)
    assert image.is_file() and all(path.is_file() for path in smoke.baseline_files(smoke.FIXTURE))
    # the values the pre-rewrite /sdapi/v1/txt2img request used, as its infotext records them
    assert (params["seed"], params["steps"], params["cfg_scale"], params["distilled_cfg_scale"]) == (12345, 8, 1.0, 3.5)
    assert (params["width"], params["height"], params["sampler_name"], params["scheduler"]) == (768, 768, "Euler", "Simple")
    assert infotext.startswith(params["prompt"]) and "Seed: 12345" in infotext and "Shift: 3.5" in infotext


def test_a_recorded_request_and_response_pair_still_works(tmp_path):
    request = {"prompt": "cat", "seed": 1}
    (tmp_path / "baseline_txt2img.json").write_text(json.dumps({"parameters": request, "info": json.dumps({"infotexts": ["cat\nSteps: 1"]})}))
    (tmp_path / "baseline_1.png").write_bytes(b"")
    params, infotext, image = smoke.load_baseline(tmp_path)
    assert params == request and infotext == "cat\nSteps: 1" and image == tmp_path / "baseline_1.png"


def test_the_report_is_committable_evidence_of_a_gradio_free_run(capsys):
    report = smoke.Report()
    report.check("server interpreter has no Gradio", True, "/venv/bin/python")
    report.check("txt2img parity", False, "differs")
    document = report.document("/venv/bin/python", [], features=False)
    json.dumps(document)
    assert document["server_python"] == "/venv/bin/python" and document["server_python_gradio_modules"] == []
    assert document["passed"] is False and document["model"] == smoke.MODEL and len(document["revision"]) in (0, 40)
    assert document["checks"] == [
        {"name": "server interpreter has no Gradio", "ok": True, "detail": "/venv/bin/python"},
        {"name": "txt2img parity", "ok": False, "detail": "differs"},
    ]
