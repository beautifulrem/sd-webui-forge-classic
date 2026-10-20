"""Where a sampler step's time goes with and without the page served, from a serve_traced.py trace.

    python tools/perf/step_segments.py $PERF/trace.jsonl [--since <unix time>] [--until <unix time>] [--json out.json]
    python tools/perf/step_segments.py $PERF/trace.jsonl --run il512/browser-interleaved.json [--run ...]

Each step interval (one ``step`` record to the next) is split at the live preview assignment
(``image``) inside it:

- *before the preview*: the step callback waits for the GPU to finish the step (the preview reads
  the latent back), so this part is GPU-bound;
- *after the preview*: the preview is encoded and published (served condition only) and Python
  dispatches the next step's work - the part where server-side work on the sampler's thread, or
  a thread that holds the GIL, would show up.

A step counts as served ("on") when a preview frame was published inside it (an ``interleaved``
run mutes the hub in its "off" windows). Per job and pooled, the medians and means of both parts
are printed. The primary figure of the UI's overhead (docs/research/ui-performance.md) is
``serverCost``: each job's served - muted difference of the "after" medians, with its mean, 95 %
interval and one-sided 95 % upper bound over the jobs (Student t), in ms and in % of the mean step
- the server-side cost per step that the whole-step comparison cannot resolve on a noisy machine.
``gpuPart`` compares the GPU-bound part the same way (on / off median, % per job): where the
page's drawing competed with the sampler for the GPU it would show there; it is as noisy as the
whole step.

``--run browser-interleaved.json`` takes the time range from an ``interleaved`` run of
``webui/tools/perf/browser.mjs`` (``traceSince`` / ``traceUntil``; runs written before those were
recorded: from its start time, the run's number of jobs); repeated, it pools the runs' jobs.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import itertools
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path


def _pair_stats(values: list[float]) -> dict:
    """Mean, two-sided 95 % interval, one-sided 95 % upper bound and median (pool_interleaved.py's)."""

    spec = importlib.util.spec_from_file_location("remi_perf_pool_interleaved", Path(__file__).with_name("pool_interleaved.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pair_stats(values)


def run_range(run: dict, records: list[dict]) -> tuple[float, float]:
    """The trace time range (unix s) of a ``browser.mjs`` ``interleaved`` result file's jobs."""

    interleaved = run["interleaved"]
    if interleaved.get("traceSince") is not None and interleaved.get("traceUntil") is not None:
        return interleaved["traceSince"], interleaved["traceUntil"]
    since = datetime.fromisoformat(run["date"].replace("Z", "+00:00")).timestamp()
    begins = sorted(r["t"] for r in records if r["e"] == "job" and r.get("phase") == "begin" and r["t"] >= since)
    count = len(interleaved["jobs"])
    if len(begins) < count:
        raise ValueError(f"the trace has {len(begins)} jobs after {run['date']}, the run {count}")
    # until the job after the run's last one (the next run's first job), else the end of the trace
    return since, begins[count] if len(begins) > count else float("inf")


def segments(records: list[dict], since: float = 0.0, until: float = float("inf")) -> dict[tuple[int, str], list[tuple[float, float]]]:
    """``{(job index, "on" | "off"): [(ms before the preview, ms after it), ...]}``."""

    jobs = [r["t"] for r in records if r["e"] == "job" and r.get("phase") == "begin"]
    steps = [r for r in records if r["e"] == "step" and since <= r["t"] <= until]
    images = sorted(r["t"] for r in records if r["e"] == "image")
    frames = sorted(r["t"] for r in records if r["e"] == "frame")
    out: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for a, b in itertools.pairwise(steps):
        if b["step"] != a["step"] + 1:
            continue
        i = bisect.bisect(images, a["t"])
        if i >= len(images) or images[i] > b["t"]:
            continue
        job = bisect.bisect(jobs, a["t"]) - 1
        served = bisect.bisect(frames, b["t"]) - bisect.bisect(frames, a["t"]) > 0
        out.setdefault((job, "on" if served else "off"), []).append(((images[i] - a["t"]) * 1000, (b["t"] - images[i]) * 1000))
    return out


def summary(found: dict[tuple[int, str], list[tuple[float, float]]]) -> dict:
    def part(values: list[tuple[float, float]], index: int) -> dict:
        xs = [v[index] for v in values]
        return {"n": len(xs), "median": round(statistics.median(xs), 2), "mean": round(statistics.mean(xs), 2)}

    jobs = sorted({job for job, _ in found})
    per_job = []
    for job in jobs:
        on, off = found.get((job, "on"), []), found.get((job, "off"), [])
        if len(on) < 5 or len(off) < 5:
            continue
        per_job.append({"job": job, "on": {"before": part(on, 0), "after": part(on, 1)}, "off": {"before": part(off, 0), "after": part(off, 1)}})
    after_diffs = [round(j["on"]["after"]["median"] - j["off"]["after"]["median"], 2) for j in per_job]
    gpu_part = [(j["on"]["before"]["median"] / j["off"]["before"]["median"] - 1) * 100 for j in per_job]
    steps = [v[0] + v[1] for values in found.values() for v in values]
    mean_step = statistics.mean(steps) if steps else None
    cost = _pair_stats(after_diffs)

    def share(value: float | None) -> float | None:
        return None if value is None or not mean_step else round(value / mean_step * 100, 3)

    return {
        "jobs": per_job,
        "afterMedianDiffMs": {"per_job": after_diffs, "mean": round(statistics.mean(after_diffs), 2) if after_diffs else None},
        "meanStepMs": round(mean_step, 1) if mean_step else None,
        # the primary figure: what serving the page costs the sampling loop per step, measured directly
        "serverCost": {"ms": cost, "pct": {"mean": share(cost["mean"]), "upper95": share(cost["upper95"])}},
        "gpuPart": {"pct": _pair_stats(gpu_part), "per_job": [round(v, 2) for v in gpu_part]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace", type=Path, help="serve_traced.py's trace (.jsonl, or .jsonl.gz)")
    parser.add_argument("--since", type=float, default=0.0, help="ignore steps before this unix time")
    parser.add_argument("--until", type=float, default=float("inf"), help="ignore steps after this unix time")
    parser.add_argument("--run", type=Path, action="append", help="an interleaved run's browser-interleaved.json: its time range (repeat to pool runs)")
    parser.add_argument("--json", type=Path, help="also write the summary here")
    options = parser.parse_args()
    if options.trace.suffix == ".gz":
        import gzip

        text = gzip.decompress(options.trace.read_bytes()).decode()
    else:
        text = options.trace.read_text()
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    ranges = [run_range(json.loads(path.read_text()), records) for path in options.run] if options.run else [(options.since, options.until)]
    found: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for since, until in ranges:  # job indices are the trace's: the runs' jobs stay apart
        for key, values in segments(records, since, until).items():
            found.setdefault(key, []).extend(values)
    result = summary(found)
    result["ranges"] = [{"since": since, "until": until if until != float("inf") else None} for since, until in ranges]
    for job in result["jobs"]:
        on, off = job["on"], job["off"]
        print(f"job {job['job']}: before the preview on {on['before']['median']} / off {off['before']['median']} ms, after it on {on['after']['median']} / off {off['after']['median']} ms (median; n {on['after']['n']} / {off['after']['n']})")
    cost, gpu = result["serverCost"], result["gpuPart"]["pct"]
    if cost["ms"]["mean"] is not None and result["meanStepMs"]:
        print(f"server cost per step (served - muted, after the preview; {cost['ms']['n']} jobs): {cost['ms']['mean']} ± {cost['ms']['ci95']} ms, one-sided upper 95 % {cost['ms']['upper95']} ms = {cost['pct']['mean']} % (upper {cost['pct']['upper95']} %) of a {result['meanStepMs']:.0f} ms step")
        print(f"GPU-bound part, served / muted: {gpu['mean']} ± {gpu['ci95']} % (upper {gpu['upper95']} %)")
    if options.json:
        options.json.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
