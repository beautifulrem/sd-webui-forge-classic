"""Where a sampler step's time goes with and without the page served, from a serve_traced.py trace.

    python tools/perf/step_segments.py $PERF/trace.jsonl [--since <unix time>] [--until <unix time>] [--json out.json]

Each step interval (one ``step`` record to the next) is split at the live preview assignment
(``image``) inside it:

- *before the preview*: the step callback waits for the GPU to finish the step (the preview reads
  the latent back), so this part is GPU-bound;
- *after the preview*: the preview is encoded and published (served condition only) and Python
  dispatches the next step's work - the part where server-side work on the sampler's thread, or
  a thread that holds the GIL, would show up.

A step counts as served ("on") when a preview frame was published inside it (an ``interleaved``
run mutes the hub in its "off" windows). Per job and pooled, the medians and means of both parts
are printed; the pooled difference of the "after" medians is the server-side cost per step that
the whole-step comparison cannot resolve on a noisy machine.
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import statistics
import sys
from pathlib import Path


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
    after_diffs = [j["on"]["after"]["median"] - j["off"]["after"]["median"] for j in per_job]
    steps = [v[0] + v[1] for values in found.values() for v in values]
    return {
        "jobs": per_job,
        "afterMedianDiffMs": {"per_job": after_diffs, "mean": round(statistics.mean(after_diffs), 2) if after_diffs else None},
        "meanStepMs": round(statistics.mean(steps), 1) if steps else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("--since", type=float, default=0.0, help="ignore steps before this unix time")
    parser.add_argument("--until", type=float, default=float("inf"), help="ignore steps after this unix time")
    parser.add_argument("--json", type=Path, help="also write the summary here")
    options = parser.parse_args()
    records = [json.loads(line) for line in options.trace.read_text().splitlines() if line.strip()]
    result = summary(segments(records, options.since, options.until))
    for job in result["jobs"]:
        on, off = job["on"], job["off"]
        print(f"job {job['job']}: before the preview on {on['before']['median']} / off {off['before']['median']} ms, after it on {on['after']['median']} / off {off['after']['median']} ms (median; n {on['after']['n']} / {off['after']['n']})")
    diff = result["afterMedianDiffMs"]["mean"]
    if diff is not None and result["meanStepMs"]:
        print(f"served - muted, after the preview: {diff} ms per step (mean of the jobs' median differences) = {diff / result['meanStepMs'] * 100:.3f} % of a {result['meanStepMs']:.0f} ms step")
    if options.json:
        options.json.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
