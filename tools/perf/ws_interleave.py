"""Interleaved preview overhead measured with a non-browser WebSocket client.

    REMI_PERF_TRACE=/tmp/trace.jsonl python tools/perf/serve_traced.py --api --port 7861 ...
    REMI_PERF_TRACE=/tmp/trace.jsonl python tools/perf/ws_interleave.py --size 512 --jobs 2 --window 15 [--json out.json] [--no-client]

A diagnostic of the 2026-10-20 verify stage (docs/research/ui-performance.md, "Verify runs of
2026-10-20"): the client connects to the WebSocket hub, receives every frame and discards it, so
the server's cost of encoding and publishing the previews is measured without a browser's
rendering cost. ``--no-client`` connects nothing (the hub then has no client to publish to and
the on/off windows should not differ).

Each job runs a long txt2img generation. While it runs, windows of ``--window`` seconds alternate
on, off, off, on, ... : an "off" window creates ``$REMI_PERF_MUTE`` (``serve_traced.py`` then acts
as if no client were connected), an "on" window removes it. The step times are read from
``$REMI_PERF_TRACE`` (``serve_traced.py``'s ``step`` records); adjacent windows (0-1, 2-3, ...) of
one job form a pair whose relative difference ``mean(on) / mean(off) - 1`` (%) is reported per
job and pooled (mean, standard error, one-sided 95 % upper bound, median).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ORDER = ("on", "off", "off", "on")
"""the window conditions, repeated (each pair has one window of each)"""


@dataclass
class Window:
    condition: str
    start: float
    end: float
    steps_ms: list[float] = field(default_factory=list)


def read_steps(trace: Path, since: float) -> list[dict]:
    """The ``step`` records of ``serve_traced.py``'s trace written at or after ``since``."""

    steps = []
    with trace.open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            if record.get("e") == "step" and record["t"] >= since:
                steps.append(record)
    return steps


def assign_steps(steps: list[dict], windows: list[Window]) -> None:
    """Add every step's duration (ms, consecutive step numbers only) to the window it lies in."""

    for a, b in itertools.pairwise(steps):
        if b["step"] != a["step"] + 1:
            continue
        for window in windows:
            if a["t"] >= window.start and b["t"] <= window.end:
                window.steps_ms.append((b["t"] - a["t"]) * 1000)


def window_pairs(windows: list[Window]) -> list[float]:
    """Relative on/off differences (%) of adjacent windows (0-1, 2-3, ...) of one job."""

    pairs = []
    for i in range(0, len(windows) - 1, 2):
        x, y = windows[i], windows[i + 1]
        if not x.steps_ms or not y.steps_ms:
            continue
        on, off = (x, y) if x.condition == "on" else (y, x)
        pairs.append((statistics.mean(on.steps_ms) / statistics.mean(off.steps_ms) - 1) * 100)
    return pairs


def pooled_stats(pairs: list[float]) -> dict:
    """Mean, standard error, one-sided 95 % upper bound (normal) and median of the pairs."""

    n = len(pairs)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "upper95": None, "median": None}
    mean = statistics.mean(pairs)
    se = statistics.stdev(pairs) / n**0.5 if n > 1 else None
    return {
        "n": n,
        "mean": round(mean, 2),
        "se": None if se is None else round(se, 2),
        "upper95": None if se is None else round(mean + 1.645 * se, 2),
        "median": round(statistics.median(pairs), 2),
    }


def set_muted(mute: Path, muted: bool) -> None:
    if muted:
        mute.touch()
    else:
        mute.unlink(missing_ok=True)


class Api:
    """JSON requests to the WebUI, never through a proxy (the server is local)."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=60) as response:
            return json.loads(response.read())


def job_request(size: int, steps: int) -> dict:
    return {
        "tab": "txt2img",
        "core": {"prompt": "masterpiece, best quality, 1girl, silver hair", "negative_prompt": "worst quality", "width": size, "height": size, "cfg_scale": 1},
        "scripts": {"sampler": {"sampler_name": "Euler", "scheduler": "Simple", "steps": steps}, "seed": {"seed": 12345}},
    }


def start_client(base: str, frames: list[int]) -> None:
    """Receive (and count, then discard) every binary frame of the hub in a daemon thread."""

    import websockets.sync.client

    url = base.replace("http", "ws", 1).rstrip("/") + "/api/v2/ws"

    def receive() -> None:
        with websockets.sync.client.connect(url, origin=base.rstrip("/"), max_size=None) as ws:
            for message in ws:
                if isinstance(message, bytes):
                    frames[0] += 1

    threading.Thread(target=receive, daemon=True).start()


def run_job(api: Callable[..., dict], args: argparse.Namespace, mute: Path, trace: Path) -> list[float]:
    """One generation with alternating windows; its window pairs (%)."""

    set_muted(mute, False)
    job = api("POST", "/api/v2/jobs", job_request(args.size, args.steps))["job_id"]
    started = time.time()
    while not read_steps(trace, started):
        time.sleep(0.5)
    windows: list[Window] = []
    while api("GET", f"/api/v2/jobs/{job}")["status"] in ("queued", "running"):
        condition = ORDER[len(windows) % len(ORDER)]
        set_muted(mute, condition == "off")
        start = time.time()
        while time.time() - start < args.window:
            time.sleep(0.2)
        windows.append(Window(condition, start, time.time()))
    set_muted(mute, False)
    assign_steps(read_steps(trace, started), windows)
    return window_pairs(windows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:7861", help="the WebUI started by serve_traced.py")
    parser.add_argument("--size", type=int, default=512, help="width and height")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=150, help="sampling steps of each job (long enough for many windows)")
    parser.add_argument("--window", type=float, default=15, help="window length in seconds")
    parser.add_argument("--no-client", action="store_true", help="toggle the mute file without connecting a client")
    parser.add_argument("--json", type=Path, help="write the pairs and pooled statistics here")
    args = parser.parse_args(argv)

    mute = Path(os.environ.get("REMI_PERF_MUTE", "/tmp/remi-perf-mute"))
    trace = Path(os.environ.get("REMI_PERF_TRACE", "/tmp/remi-perf-trace.jsonl"))
    api = Api(args.base)
    frames = [0]
    if not args.no_client:
        start_client(args.base, frames)
    time.sleep(2)

    pairs: list[float] = []
    for index in range(args.jobs):
        job_pairs = run_job(api, args, mute, trace)
        pairs += job_pairs
        stats = pooled_stats(job_pairs)
        print(f"job {index + 1}: {stats['n']} pairs mean {stats['mean']} median {stats['median']}; frames so far {frames[0]}", flush=True)
    pooled = pooled_stats(pairs)
    print(f"pooled: n {pooled['n']} mean {pooled['mean']} % se {pooled['se']} upper95 {pooled['upper95']} median {pooled['median']}")
    if args.json:
        args.json.write_text(json.dumps({"size": args.size, "client": not args.no_client, "pairs": pairs, "pooled": pooled}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
