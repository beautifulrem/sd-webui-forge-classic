"""Real-server browser smoke: the SPA's ``@real`` Playwright tests against a running WebUI.

    python tools/smoke_browser.py --base-url http://127.0.0.1:7860   # a WebUI that is running
    python tools/smoke_real.py --browser                             # one it starts (with the API checks)

The tests (``webui/tests/e2e/*.spec.ts`` tagged ``@real``, e.g. ``browser-apis.spec.ts``)
open the built SPA the server serves (``webui/dist``: run ``pnpm build`` in ``webui/``
first) in Playwright's Chromium and generate with the server's model. The browser APIs a
test cannot see on a real screen are replaced by recording probes: the screen wake lock
while generating, the notification permission prompt, the notification and the
``notification.mp3`` sound when images are ready in the background, gamepad navigation in
the viewer; also the live preview in the viewer and the extensions' "Check for updates"
feedback. They change no server setting (client options are overridden in the page only),
but each generates a few small images into the server's outputs.

Prints one row per test; exit 1 when one failed, 2 when the tests could not run (no
``pnpm``, no built SPA, the server not answering).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "webui"

Check = Callable[[str, bool, str], bool]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _specs(suite: dict) -> list[tuple[str, dict]]:
    """Every spec of a Playwright JSON report suite (nested suites included), with its file."""

    found = [(suite.get("file") or "", spec) for spec in suite.get("specs", [])]
    for child in suite.get("suites", []):
        found += _specs(child)
    return found


def results(report: dict) -> list[tuple[str, bool, str]]:
    """``(title, passed, detail)`` per test of a Playwright JSON report."""

    rows: list[tuple[str, bool, str]] = []
    for suite in report.get("suites", []):
        for file, spec in _specs(suite):
            for test in spec.get("tests", []):
                runs = test.get("results", [])
                last = runs[-1] if runs else {}
                status = last.get("status", "skipped")
                passed = test.get("status") in ("expected", "flaky") and status in ("passed", "skipped")
                error = ANSI.sub("", (last.get("error") or {}).get("message", "")) if not passed else ""
                seconds = sum(run.get("duration", 0) for run in runs) / 1000
                detail = f"{Path(file).name}, {seconds:.1f} s" + (f", {status}" if status != "passed" else "") + (f": {error.splitlines()[0][:300]}" if error else "")
                rows.append((spec.get("title", "?"), passed, detail))
    return rows


def run_browser(base_url: str, out: Path, check: Check, *, grep: str | None = None, env: dict[str, str] | None = None) -> None:
    """Runs the ``@real`` tests (or those matching ``grep``, see ``REMI_E2E_GREP`` in
    ``webui/playwright.config.ts``) against ``base_url``; one ``check`` row per test.
    ``env`` is added to the tests' environment."""

    pnpm = shutil.which("pnpm")
    if pnpm is None:
        check("browser smoke", False, "pnpm is not installed (Node and pnpm run the Playwright tests)")
        return
    if not (WEBUI / "dist" / "index.html").is_file():
        check("browser smoke", False, "no built SPA: run `pnpm build` in webui/ first")
        return
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **(env or {}), "REMI_E2E_BASE_URL": base_url, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", "FORCE_COLOR": "0"}
    if grep:
        env["REMI_E2E_GREP"] = grep
    env.pop("CI", None)  # no retries: a real-server failure is reported as it happened
    command = [pnpm, "exec", "playwright", "test", "--reporter=json", "--output", str(out / "browser-results")]
    print(f"browser: {' '.join(command[1:])} (REMI_E2E_BASE_URL={base_url})", flush=True)
    done = subprocess.run(command, cwd=WEBUI, env=env, capture_output=True, text=True, timeout=3600, check=False)
    (out / "browser-report.json").write_text(done.stdout, encoding="utf-8")
    try:
        report = json.loads(done.stdout)
    except json.JSONDecodeError:
        check("browser smoke", False, f"Playwright exited with {done.returncode}: {(done.stderr or done.stdout)[-1500:]}")
        return
    rows = results(report)
    if not rows:
        check("browser smoke", False, f"no {grep or '@real'} test ran: " + "; ".join(str(error.get("message", ""))[:300] for error in report.get("errors", [])))
        return
    for title, passed, detail in rows:
        check(f"browser: {re.sub(r' @[a-z-]+$', '', title)}", passed, detail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:7860", help="the running WebUI (default %(default)s)")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "remi-browser-smoke", help="Playwright's report and failure traces")
    options = parser.parse_args()

    try:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"{options.base_url.rstrip('/')}/api/v2/app", timeout=10).close()
    except OSError as error:
        print(f"the server does not answer at {options.base_url}: {error}", file=sys.stderr)
        return 2

    rows: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        rows.append((name, ok, detail))
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{f': {detail}' if detail else ''}", flush=True)
        return ok

    run_browser(options.base_url.rstrip("/"), options.out, check)
    if any(name == "browser smoke" for name, _, _ in rows):
        return 2
    return 1 if any(not ok for _, ok, _ in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
