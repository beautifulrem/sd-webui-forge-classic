"""Pool the window pairs of several ``interleaved`` runs of the same configuration.

    python tools/perf/pool_interleaved.py run1/browser-interleaved.json run2/browser-interleaved.json [--json pooled.json]

``webui/tools/perf/browser.mjs --only interleaved --repeat N`` pools the jobs of one run; this
pools runs (the same size, steps, window length and batch count, checked), so a run that ended
with too few pairs for a tight bound can be extended by another instead of being repeated.
Pairs never span two jobs: each job's windows are paired on their own, as in ``browser.mjs``.

Runs written before ``browser.mjs`` recorded each job's window range (``jobs[].firstWindow``)
are split into jobs by search: the split must reproduce the run's recorded ``pairs`` exactly
(and each job's pair count), else the file is rejected. Every run's recomputed statistics are
checked against the ones it recorded.

Output: per run and pooled, the window pairs' mean, two-sided 95 % interval, one-sided 95 %
upper bound and median (Student t), for all steps and with each window's first step dropped -
``pairStats`` / ``trimmedPairStats`` of ``browser.mjs``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

CONFIG = ("size", "steps", "windowS", "batchCount")


def window_pairs(windows: list[dict], trim: bool = False) -> list[float]:
    """Relative on/off differences (%) of adjacent windows (0-1, 2-3, ...) of one job."""

    pairs = []
    need = 2 if trim else 1
    for i in range(0, len(windows) - 1, 2):
        x, y = windows[i], windows[i + 1]
        if len(x["steps"]) < need or len(y["steps"]) < need:
            continue
        on, off = (x, y) if x["condition"] == "on" else (y, x)
        a, b = on["steps"][1:] if trim else on["steps"], off["steps"][1:] if trim else off["steps"]
        pairs.append((statistics.mean(a) / statistics.mean(b) - 1) * 100)
    return pairs


def pair_stats(values: list[float]) -> dict:
    from scipy import stats

    n = len(values)
    if n < 2:
        return {"n": n, "mean": None, "ci95": None, "upper95": None, "median": None}
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, 2),
        "ci95": round(float(stats.t.ppf(0.975, n - 1)) * se, 2),
        "upper95": round(mean + float(stats.t.ppf(0.95, n - 1)) * se, 2),
        "median": round(statistics.median(values), 2),
    }


def _same(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= 0.05 for x, y in zip(a, b, strict=True))


def split_jobs(run: dict) -> list[list[dict]]:
    """The run's windows, per job."""

    windows, jobs = run["windows"], run["jobs"]
    if all("firstWindow" in job for job in jobs):
        return [windows[job["firstWindow"] : job["firstWindow"] + job["windowCount"]] for job in jobs]
    recorded = run["pairs"]
    counts = [job["pairStats"]["n"] for job in jobs]

    def search(job: int, start: int, pair_at: int) -> list[int] | None:
        if job == len(jobs):
            return [] if start == len(windows) and pair_at == len(recorded) else None
        last = job == len(jobs) - 1
        for end in [len(windows)] if last else range(start + 1, len(windows)):
            if windows[start]["condition"] != "on" or (not last and windows[end]["condition"] != "on"):
                continue
            pairs = window_pairs(windows[start:end])
            if len(pairs) != counts[job] or not _same(pairs, recorded[pair_at : pair_at + len(pairs)]):
                continue
            rest = search(job + 1, end, pair_at + len(pairs))
            if rest is not None:
                return [end, *rest]
        return None

    ends = search(0, 0, 0)
    if ends is None:
        raise ValueError("cannot split the windows into the recorded jobs")
    starts = [0, *ends[:-1]]
    return [windows[a:b] for a, b in zip(starts, ends, strict=True)]


def analyse(paths: list[Path]) -> dict:
    runs, pairs, trimmed, config = [], [], [], None
    for path in paths:
        run = json.loads(path.read_text())["interleaved"]
        this = {key: run.get(key) for key in CONFIG}
        if config is None:
            config = this
        elif this != config:
            raise ValueError(f"{path}: {this} differs from {config}")
        jobs = split_jobs(run)
        run_pairs = [p for job in jobs for p in window_pairs(job)]
        run_trimmed = [p for job in jobs for p in window_pairs(job, trim=True)]
        if not _same(run_pairs, run["pairs"]):
            raise ValueError(f"{path}: recomputed pairs differ from the recorded ones")
        summary = {"file": str(path), "jobs": len(jobs), "previewClients": run.get("previewClients"), "paintedShare": run.get("paintedShare"), "pairStats": pair_stats(run_pairs), "trimmedPairStats": pair_stats(run_trimmed)}
        for key in ("pairStats", "trimmedPairStats"):
            recorded = run[key]
            if abs(summary[key]["mean"] - recorded["mean"]) > 0.05:
                raise ValueError(f"{path}: {key} mean {summary[key]['mean']} != recorded {recorded['mean']}")
        runs.append(summary)
        pairs += run_pairs
        trimmed += run_trimmed
    return {"config": config, "runs": runs, "pooled": {"pairStats": pair_stats(pairs), "trimmedPairStats": pair_stats(trimmed)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path, nargs="+", help="browser-interleaved.json files")
    parser.add_argument("--json", type=Path, help="also write the result here")
    options = parser.parse_args()
    result = analyse(options.runs)
    for run in result["runs"]:
        print(f"{run['file']}: {run['jobs']} jobs, pairs {run['pairStats']}, first step dropped {run['trimmedPairStats']}")
    pooled = result["pooled"]
    print(f"pooled {result['config']}: pairs {pooled['pairStats']}; first step dropped {pooled['trimmedPairStats']}")
    if options.json:
        options.json.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
