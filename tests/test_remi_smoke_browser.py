"""tools/smoke_browser.py: the real-server browser smoke turns Playwright's JSON report
into one result row per ``@real`` test, and says why when the tests cannot run."""

import importlib.util
import sys
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "smoke_browser.py"
_spec = importlib.util.spec_from_file_location("smoke_browser", _TOOL)
smoke = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("smoke_browser", smoke)
_spec.loader.exec_module(smoke)


def _test(status: str, result: str, duration: int, error: str | None = None) -> dict:
    run = {"status": result, "duration": duration}
    if error is not None:
        run["error"] = {"message": error}
    return {"status": status, "results": [run]}


def test_one_row_per_test_with_its_outcome_and_first_error_line():
    report = {
        "suites": [
            {
                "file": "browser-apis.spec.ts",
                "specs": [],
                "suites": [
                    {
                        "file": "browser-apis.spec.ts",
                        "specs": [
                            {"title": "keeps the screen awake @real", "tests": [_test("expected", "passed", 2500)]},
                            {"title": "notifies @real", "tests": [_test("unexpected", "failed", 4000, "\x1b[2mexpect(\x1b[22mreceived).toHaveLength(1)\nmore lines")]},
                        ],
                    }
                ],
            }
        ]
    }
    rows = smoke.results(report)
    assert rows[0] == ("keeps the screen awake @real", True, "browser-apis.spec.ts, 2.5 s")
    title, passed, detail = rows[1]
    assert (title, passed) == ("notifies @real", False)
    assert detail == "browser-apis.spec.ts, 4.0 s, failed: expect(received).toHaveLength(1)"


def test_says_why_the_tests_cannot_run(tmp_path, monkeypatch):
    rows = []

    def check(name, ok, detail=""):
        rows.append((name, ok, detail))
        return ok

    monkeypatch.setattr(smoke.shutil, "which", lambda _name: None)
    smoke.run_browser("http://127.0.0.1:1", tmp_path, check)
    assert rows == [("browser smoke", False, "pnpm is not installed (Node and pnpm run the Playwright tests)")]

    rows.clear()
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "/usr/bin/pnpm")
    monkeypatch.setattr(smoke, "WEBUI", tmp_path)
    smoke.run_browser("http://127.0.0.1:1", tmp_path, check)
    assert rows == [("browser smoke", False, "no built SPA: run `pnpm build` in webui/ first")]
