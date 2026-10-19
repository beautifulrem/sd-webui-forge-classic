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
