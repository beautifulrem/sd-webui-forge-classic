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
