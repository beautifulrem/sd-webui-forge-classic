"""Server startup time of this tree against the pre-rewrite tree (docs/research/ui-performance.md).

    python tools/perf/startup.py --old "$OLD_TREE" --old-python "$OLD_PY" --new-python "$NEW_PY" \
        --runs 5 --out "$PERF_OUT/startup.json"

``$OLD_TREE`` is a worktree of the pre-rewrite code and ``$OLD_PY`` a Python with Gradio 4.40.0;
docs/research/ui-performance.md ("Recreating the environments") builds both.

For each tree and mode (``ui``: the web interface, ``api``: ``--nowebui``) it starts ``launch.py``
``runs`` times, old and new alternating (after one unrecorded warm-up start each that fills the
OS file cache) with a fresh
``--data-dir`` holding a copy of this repo's ``config.json``, the models from ``--model-ref``, offline
(``--ad-no-huggingface``, ``HF_HUB_OFFLINE=1``, no proxy: see :data:`OFFLINE_ARGS`), and
measures the wall time from spawning the process until the server answers (``GET /`` for the UI,
``GET /sdapi/v1/sd-models`` for the API) and the process's own ``Startup time`` line, then
stops it (Ctrl+C). The RSS of the process tree once ready is recorded too. Refuses to run while
another ``launch.py`` runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_STARTUP = re.compile(r"Startup time: ([\d.]+)s")


def answers(url: str, timeout: float = 120) -> bool:
    """Whether ``url`` answers 200. Waits for a slow answer instead of giving up and asking again:
    a probe that times out leaves the server rendering a page nobody reads, and on a busy machine
    the old tree's Gradio page (seconds per request) then never catches up with the probes."""

    try:
        with _opener.open(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def tree_rss(pid: int) -> int:
    import psutil

    try:
        proc = psutil.Process(pid)
        return sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)])
    except psutil.Error:
        return 0


OFFLINE_ARGS = ["--ad-no-huggingface"]
"""Both trees start offline alike: the old tree's ADetailer downloads its missing stock detectors on
every start (``get_models``; with outbound connections blocked the start hangs on them for minutes),
the new one only when a unit runs, and both accept this flag (ADetailer's ``preload.py``)."""


def server_env(environ: dict[str, str]) -> dict[str, str]:
    """The servers' environment: offline (no proxy - Gradio's httpx client refuses a SOCKS proxy
    without socksio, so the old tree would not start at all -, ``HF_HUB_OFFLINE`` so no Hugging
    Face lookup can wait on the network), no SPA build."""

    env = {key: value for key, value in environ.items() if not key.lower().endswith("_proxy")}
    env.update({"REMI_SKIP_WEBUI_BUILD": "1", "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", "HF_HUB_OFFLINE": "1"})
    return env


def launch_args(python: str, mode: str, port: int, data: Path) -> list[str]:
    """``launch.py``'s command line for one start (the same for both trees)."""

    args = [python, "launch.py", "--skip-prepare-environment", "--skip-install", "--port", str(port), "--data-dir", str(data), "--model-ref", str(ROOT / "models"), *OFFLINE_ARGS]
    return args + (["--nowebui"] if mode == "api" else [])


def start_once(tree: Path, python: str, mode: str, port: int, timeout: float) -> dict:
    data = Path(tempfile.mkdtemp(prefix="remi-startup-"))
    if (ROOT / "config.json").is_file():
        shutil.copy2(ROOT / "config.json", data / "config.json")
    args = launch_args(python, mode, port, data)
    env = server_env(os.environ)
    url = f"http://127.0.0.1:{port}" + ("/sdapi/v1/sd-models" if mode == "api" else "/")
    log_path = data / "server.log"
    with open(log_path, "wb") as log:
        started = time.perf_counter()
        process = subprocess.Popen(args, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ready = None
        while time.perf_counter() - started < timeout and process.poll() is None:
            if answers(url):
                ready = time.perf_counter() - started
                break
            time.sleep(0.05)
        time.sleep(0.5)  # let the startup line be written
        rss = tree_rss(process.pid)
        for sig, wait in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, sig)
                process.wait(wait)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
    text = log_path.read_text(errors="replace")
    match = _STARTUP.search(text)
    reported = float(match.group(1)) if match else None
    detail = text[match.start() : text.find("\n", match.start())] if match else text[-1500:]
    shutil.rmtree(data, ignore_errors=True)
    return {"ready_s": round(ready, 2) if ready else None, "reported_s": reported, "rss_mb": round(rss / 1048576), "detail": detail.strip()}


def summarise(runs: list[dict]) -> dict:
    """Medians (and range) of one tree and mode's starts."""

    ready = [run["ready_s"] for run in runs if run["ready_s"]]
    reported = [run["reported_s"] for run in runs if run["reported_s"]]
    return {
        "runs": runs,
        "ready_median_s": round(statistics.median(ready), 2) if ready else None,
        "ready_min_s": min(ready) if ready else None,
        "ready_max_s": max(ready) if ready else None,
        "reported_median_s": round(statistics.median(reported), 2) if reported else None,
        "rss_median_mb": round(statistics.median(run["rss_mb"] for run in runs)) if runs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old", type=Path, required=True, help="worktree of the pre-rewrite code (git ref 'remi')")
    parser.add_argument("--old-python", required=True)
    parser.add_argument("--new-python", default=sys.executable)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--modes", default="api,ui")
    parser.add_argument("--out", type=Path)
    options = parser.parse_args()

    import psutil

    busy = [p.pid for p in psutil.process_iter(["cmdline"]) if any(str(part).endswith("launch.py") for part in (p.info.get("cmdline") or []))]
    if busy:
        print(f"another launch.py is running: {busy}", file=sys.stderr)
        return 2

    results: dict[str, dict] = {}
    trees = {"old": (options.old, options.old_python), "new": (ROOT, options.new_python)}
    for mode in options.modes.split(","):
        for name, (tree, python) in trees.items():
            print(f"{name}-{mode}: warm-up", flush=True)
            start_once(tree, python, mode, options.port, options.timeout)
        # interleaved (old, new / new, old, ...): a machine whose load changes during the runs
        # slows both trees alike instead of the one that happens to run later
        runs: dict[str, list[dict]] = {name: [] for name in trees}
        for index in range(options.runs):
            for name in ("old", "new") if index % 2 == 0 else ("new", "old"):
                tree, python = trees[name]
                run = start_once(tree, python, mode, options.port, options.timeout)
                runs[name].append(run)
                print(f"  {name}-{mode} #{index + 1}: ready {run['ready_s']} s, reported {run['reported_s']} s, RSS {run['rss_mb']} MB", flush=True)
        for name in trees:
            results[f"{name}-{mode}"] = summarise(runs[name])
    print(f"\n{'tree-mode':<10} {'ready (median)':>15} {'reported':>10} {'RSS':>8}")
    for key, row in results.items():
        print(f"{key:<10} {row['ready_median_s']!s:>13} s {row['reported_median_s']!s:>8} s {row['rss_median_mb']:>5} MB")
    if options.out:
        options.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
