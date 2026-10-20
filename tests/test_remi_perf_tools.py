"""The UI performance tools (docs/research/ui-performance.md): the startup comparison."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def startup():
    spec = importlib.util.spec_from_file_location("remi_perf_startup", ROOT / "tools" / "perf" / "startup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_servers_start_without_proxies(startup):
    env = startup.server_env({"PATH": "/bin", "ALL_PROXY": "socks5://x", "https_proxy": "http://y", "NO_PROXY": "z"})
    assert env["PATH"] == "/bin" and "ALL_PROXY" not in env and "https_proxy" not in env
    assert env["NO_PROXY"] == env["no_proxy"] == "127.0.0.1,localhost" and env["REMI_SKIP_WEBUI_BUILD"] == "1"
    # no Hugging Face lookup may wait on a blocked network
    assert env["HF_HUB_OFFLINE"] == "1"


def test_both_trees_start_without_adetailer_downloads(startup, tmp_path):
    # the old tree's ADetailer downloads its missing stock detectors on every start (minutes when
    # outbound connections are blocked); both trees accept the flag that turns that off
    for mode in ("ui", "api"):
        args = startup.launch_args("py", mode, 7871, tmp_path)
        assert args[:2] == ["py", "launch.py"] and "--ad-no-huggingface" in args
        assert args[args.index("--data-dir") + 1] == str(tmp_path) and args[args.index("--port") + 1] == "7871"
        assert ("--nowebui" in args) == (mode == "api")
    assert '"--ad-no-huggingface"' in (ROOT / "extensions-builtin" / "ADetailer-Neo" / "preload.py").read_text()


def test_summary_of_one_tree_and_mode(startup):
    runs = [{"ready_s": 5.0, "reported_s": 4.9, "rss_mb": 600}, {"ready_s": None, "reported_s": None, "rss_mb": 0}, {"ready_s": 3.0, "reported_s": 2.9, "rss_mb": 640}]
    summary = startup.summarise(runs)
    assert summary["ready_median_s"] == 4.0 and summary["ready_min_s"] == 3.0 and summary["ready_max_s"] == 5.0
    assert summary["reported_median_s"] == 3.9 and summary["rss_median_mb"] == 600 and summary["runs"] is runs


def test_trees_alternate_after_one_warm_up_each(startup, monkeypatch, tmp_path):
    calls = []

    def start_once(tree, python, mode, port, timeout):
        calls.append((python, mode))
        return {"ready_s": 1.0 if python == "old-py" else 0.5, "reported_s": 1.0, "rss_mb": 1}

    monkeypatch.setattr(startup, "start_once", start_once)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(process_iter=lambda attrs: []))
    out = tmp_path / "startup.json"
    monkeypatch.setattr(sys, "argv", ["startup.py", "--old", str(tmp_path), "--old-python", "old-py", "--new-python", "new-py", "--runs", "3", "--modes", "api", "--out", str(out)])
    assert startup.main() == 0
    order = [python for python, _mode in calls]
    # warm-ups, then old/new, new/old, old/new
    assert order == ["old-py", "new-py", "old-py", "new-py", "new-py", "old-py", "old-py", "new-py"]
    results = json.loads(out.read_text())
    assert results["old-api"]["ready_median_s"] == 1.0 and results["new-api"]["ready_median_s"] == 0.5
    assert len(results["old-api"]["runs"]) == len(results["new-api"]["runs"]) == 3


def test_traced_hub_records_how_many_clients_get_a_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("REMI_PERF_TRACE", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("REMI_PERF_MUTE", str(tmp_path / "mute"))
    spec = importlib.util.spec_from_file_location("remi_perf_serve_traced", ROOT / "tools" / "perf" / "serve_traced.py")
    traced = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traced)

    sent = []

    class Hub:
        def __init__(self):
            self.clients = {"page", "other tab"}

        def wants_frames(self):
            return bool(self.clients)

        def _fan_event(self, event):
            sent.append(event)

        def _fan_frame(self, frame):
            sent.append(frame)

        async def serve(self, websocket):
            pass

    traced._patch_ws(SimpleNamespace(Hub=Hub))
    hub = Hub()
    hub._fan_frame("frame 1")
    (tmp_path / "mute").write_text("")
    hub._fan_frame("frame 2")
    assert not hub.wants_frames()
    traced._out.flush()
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert sent == ["frame 1"]
    assert [(record["e"], record["clients"]) for record in records] == [("fan", 2)]
    traced._out.close()


def test_traced_hub_refuses_other_browsers_when_asked(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setenv("REMI_PERF_TRACE", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("REMI_PERF_WS_ONLY", "HeadlessChrome")
    spec = importlib.util.spec_from_file_location("remi_perf_serve_traced_only", ROOT / "tools" / "perf" / "serve_traced.py")
    traced = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traced)
    served, closed = [], []

    class Hub:
        clients = ()

        def wants_frames(self):
            return True

        def _fan_event(self, event):
            pass

        def _fan_frame(self, frame):
            pass

        async def serve(self, websocket):
            served.append(websocket)

    class Socket:
        def __init__(self, agent):
            self.headers = {"user-agent": agent, "origin": "http://127.0.0.1:7861"}

        async def close(self, code):
            closed.append((self, code))

    traced._patch_ws(SimpleNamespace(Hub=Hub))
    page, other = Socket("Mozilla/5.0 HeadlessChrome/156"), Socket("Mozilla/5.0 Chrome/156")
    asyncio.run(Hub().serve(page))
    asyncio.run(Hub().serve(other))
    traced._out.close()
    assert served == [page] and closed == [(other, 1008)]
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [(record["e"], record["refused"]) for record in records] == [("ws", False), ("ws", True)]


def test_step_segments_split_steps_at_the_preview():
    spec = importlib.util.spec_from_file_location("remi_perf_step_segments", ROOT / "tools" / "perf" / "step_segments.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    records = [{"e": "job", "t": 0.0, "phase": "begin"}]
    t = 1.0
    for step in range(20):
        served = step % 2 == 0
        records.append({"e": "step", "t": t, "step": step})
        records.append({"e": "image", "t": t + 2.0})
        if served:
            records.append({"e": "frame", "t": t + 2.001})
        t += 2.0 + (0.030 if served else 0.026)
    found = tool.segments(records)
    assert set(found) == {(0, "on"), (0, "off")}
    result = tool.summary(found)
    assert result["afterMedianDiffMs"]["per_job"] == [4.0]
    assert result["jobs"][0]["on"]["before"]["median"] == 2000.0


def _step_segments_tool():
    spec = importlib.util.spec_from_file_location("remi_perf_step_segments", ROOT / "tools" / "perf" / "step_segments.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _served_jobs(costs_ms: list[float], start: float = 1000.0, steps: int = 20) -> list[dict]:
    """One job per cost: steps alternate served / muted; a served step's part after the preview is
    ``cost`` ms longer, the GPU-bound part the same."""

    records, t = [], start
    for cost in costs_ms:
        records.append({"e": "job", "t": t, "phase": "begin"})
        for step in range(steps):
            served = step % 2 == 0
            records.append({"e": "step", "t": t, "step": step})
            records.append({"e": "image", "t": t + 2.0})
            if served:
                records.append({"e": "frame", "t": t + 2.001})
            t += 2.0 + 0.026 + (cost / 1000 if served else 0)
        t += 30
    return records


def test_step_segments_server_cost_is_bounded_over_the_jobs():
    tool = _step_segments_tool()
    result = tool.summary(tool.segments(_served_jobs([2.0, 2.2, 1.8, 2.0])))
    cost = result["serverCost"]
    assert cost["ms"]["n"] == 4 and cost["ms"]["mean"] == 2.0
    assert cost["ms"]["upper95"] > 2.0 and cost["ms"]["ci95"] > 0
    # the share of a ~2.03 s step
    assert 0.09 < cost["pct"]["mean"] < 0.11 and cost["pct"]["upper95"] > cost["pct"]["mean"]
    # the GPU-bound part did not change
    assert result["gpuPart"]["pct"]["mean"] == 0.0 and result["gpuPart"]["per_job"] == [0.0] * 4


def test_step_segments_takes_the_range_of_an_interleaved_run():
    tool = _step_segments_tool()
    records = _served_jobs([2.0, 2.0, 5.0], start=1_000.0)
    begins = [r["t"] for r in records if r["e"] == "job"]
    # recorded range (browser.mjs since 2026-10-20 verify): used as is
    assert tool.run_range({"interleaved": {"traceSince": 1.0, "traceUntil": 2.0, "jobs": []}}, records) == (1.0, 2.0)
    # older runs: from the start time, as many jobs as the run had (here the first two)
    run = {"date": "1970-01-01T00:16:39.000Z", "interleaved": {"jobs": [{}, {}]}}
    since, until = tool.run_range(run, records)
    assert since == 999.0 and until == begins[2]
    assert tool.summary(tool.segments(records, since, until))["serverCost"]["ms"]["mean"] == 2.0
    with pytest.raises(ValueError):
        tool.run_range({"date": run["date"], "interleaved": {"jobs": [{}] * 4}}, records)


def _pool_tool():
    spec = importlib.util.spec_from_file_location("remi_perf_pool_interleaved", ROOT / "tools" / "perf" / "pool_interleaved.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _interleaved_run(tool, jobs_windows, *, record_ranges):
    windows = [w for job in jobs_windows for w in job]
    jobs, first = [], 0
    for job in jobs_windows:
        entry = {"pairStats": {"n": len(tool.window_pairs(job))}}
        if record_ranges:
            entry.update(firstWindow=first, windowCount=len(job))
        jobs.append(entry)
        first += len(job)
    pairs = [p for job in jobs_windows for p in tool.window_pairs(job)]
    trimmed = [p for job in jobs_windows for p in tool.window_pairs(job, trim=True)]
    return {"size": "512x512", "steps": 150, "windowS": 15, "batchCount": 1, "windows": windows, "jobs": jobs, "pairs": [round(p, 2) for p in pairs], "pairStats": tool.pair_stats(pairs), "trimmedPairStats": tool.pair_stats(trimmed)}


def test_pool_interleaved_splits_old_runs_into_jobs_and_pools(tmp_path):
    tool = _pool_tool()

    def job(seed, count):
        order = ["on", "off", "off", "on"]
        return [{"condition": order[i % 4], "steps": [2000 + (seed * 37 + i * 101 + k * 13) % 97 for k in range(5)]} for i in range(count)]

    # the second job starts where the cycle would have been "on" anyway: only the pairs tell the split
    jobs = [job(1, 7), job(2, 6), job(3, 9)]
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"interleaved": _interleaved_run(tool, jobs, record_ranges=False)}))
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"interleaved": _interleaved_run(tool, [job(4, 8)], record_ranges=True)}))

    assert [len(windows) for windows in tool.split_jobs(json.loads(old.read_text())["interleaved"])] == [7, 6, 9]
    result = tool.analyse([old, new])
    all_pairs = [p for windows in [*jobs, job(4, 8)] for p in tool.window_pairs(windows)]
    assert result["pooled"]["pairStats"] == tool.pair_stats(all_pairs)
    assert [run["jobs"] for run in result["runs"]] == [3, 1]


def test_pool_interleaved_refuses_other_configurations(tmp_path):
    tool = _pool_tool()
    windows = [{"condition": c, "steps": [100, 101, 99]} for c in ["on", "off", "off", "on"]]
    a = _interleaved_run(tool, [windows], record_ranges=True)
    b = {**a, "size": "768x768"}
    (tmp_path / "a.json").write_text(json.dumps({"interleaved": a}))
    (tmp_path / "b.json").write_text(json.dumps({"interleaved": b}))
    with pytest.raises(ValueError, match="differs"):
        tool.analyse([tmp_path / "a.json", tmp_path / "b.json"])


def _ws_tool():
    spec = importlib.util.spec_from_file_location("remi_perf_ws_interleave", ROOT / "tools" / "perf" / "ws_interleave.py")
    tool = importlib.util.module_from_spec(spec)
    # dataclasses resolve the module's (postponed) annotations through sys.modules
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    return tool


def test_ws_interleave_pairs_adjacent_windows_of_consecutive_steps(tmp_path):
    tool = _ws_tool()
    trace = tmp_path / "trace.jsonl"
    records = [{"e": "job", "t": 0.5}, {"e": "step", "t": 1.0, "step": 1}, {"e": "step", "t": 1.2, "step": 2}, {"e": "step", "t": 1.4, "step": 3}, {"e": "step", "t": 2.0, "step": 5}, {"e": "step", "t": 2.1, "step": 6}, {"e": "step", "t": 2.2, "step": 7}]
    trace.write_text("".join(json.dumps(r) + "\n" for r in records))
    steps = tool.read_steps(trace, 1.0)
    assert [s["step"] for s in steps] == [1, 2, 3, 5, 6, 7]
    assert tool.read_steps(trace, 2.05) == steps[-2:]

    windows = [tool.Window("on", 0.9, 1.5), tool.Window("off", 1.5, 2.3), tool.Window("off", 2.3, 3.0)]
    tool.assign_steps(steps, windows)
    # 3 -> 5 skips a step number and spans two windows: dropped
    assert [round(ms) for ms in windows[0].steps_ms] == [200, 200] and [round(ms) for ms in windows[1].steps_ms] == [100, 100]
    assert windows[2].steps_ms == []
    # the last window has no partner; a pair with an empty window is skipped
    assert tool.window_pairs(windows) == [pytest.approx(100.0)]
    assert tool.window_pairs([tool.Window("off", 0, 1, [100.0]), tool.Window("on", 1, 2, [])]) == []


def test_ws_interleave_pooled_statistics():
    tool = _ws_tool()
    assert tool.pooled_stats([]) == {"n": 0, "mean": None, "se": None, "upper95": None, "median": None}
    assert tool.pooled_stats([3.0]) == {"n": 1, "mean": 3.0, "se": None, "upper95": None, "median": 3.0}
    stats = tool.pooled_stats([1.0, 2.0, 3.0, 6.0])
    assert stats["n"] == 4 and stats["mean"] == 3.0 and stats["median"] == 2.5
    assert stats["se"] == 1.08 and stats["upper95"] == 4.78


def test_ws_interleave_measures_the_cost_of_unmuted_windows(tmp_path, monkeypatch, capsys):
    """A simulated server whose steps take 10 % longer while the mute file is absent."""

    tool = _ws_tool()
    mute, trace = tmp_path / "mute", tmp_path / "trace.jsonl"
    trace.write_text("")
    monkeypatch.setenv("REMI_PERF_MUTE", str(mute))
    monkeypatch.setenv("REMI_PERF_TRACE", str(trace))
    clock = {"now": 1000.0, "step": 0}

    def sleep(seconds):
        clock["now"] += seconds * (1.0 if mute.exists() else 1.1)
        clock["step"] += 1
        with trace.open("a") as out:
            out.write(json.dumps({"e": "step", "t": clock["now"], "step": clock["step"]}) + "\n")

    monkeypatch.setattr(tool, "time", SimpleNamespace(time=lambda: clock["now"], sleep=sleep))
    calls = []

    class FakeApi:
        def __init__(self, base):
            self.polls = 0

        def __call__(self, method, path, body=None):
            calls.append((method, path))
            if method == "POST":
                assert body["core"]["width"] == body["core"]["height"] == 768 and body["scripts"]["sampler"]["steps"] == 150
                self.polls = 0
                return {"job_id": "j1"}
            self.polls += 1
            return {"status": "running" if self.polls <= 8 else "done"}

    monkeypatch.setattr(tool, "Api", FakeApi)
    out = tmp_path / "ws.json"
    assert tool.main(["--size", "768", "--jobs", "2", "--window", "2", "--no-client", "--json", str(out)]) == 0
    assert not mute.exists()
    result = json.loads(out.read_text())
    assert result["size"] == 768 and result["client"] is False
    assert len(result["pairs"]) == 8 and all(p == pytest.approx(10.0) for p in result["pairs"])
    assert result["pooled"]["n"] == 8 and result["pooled"]["mean"] == 10.0
    assert calls.count(("POST", "/api/v2/jobs")) == 2
    printed = capsys.readouterr().out
    assert "job 1: 4 pairs mean 10.0" in printed and "pooled: n 8 mean 10.0 %" in printed
