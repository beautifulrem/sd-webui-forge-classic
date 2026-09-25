from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pid_state import batch_value, result_sink


def test_batch_value_prefers_current_batch():
    process = types.SimpleNamespace(
        prompts=["current-0", "current-1"],
        all_prompts=["old-0", "old-1", "current-0", "current-1"],
        iteration=1,
        batch_size=2,
    )
    assert batch_value(process, "prompts", "all_prompts", 1) == "current-1"


def test_batch_value_uses_iteration_offset_as_fallback():
    process = types.SimpleNamespace(
        all_seeds=[1, 2, 3, 4], iteration=0, batch_size=2
    )
    assert [batch_value(process, "seeds", "all_seeds", i) for i in range(2)] == [1, 2]
    process.iteration = 1
    assert [batch_value(process, "seeds", "all_seeds", i) for i in range(2)] == [3, 4]
    process.iteration = 2  # past the end: clamps to the last value
    assert batch_value(process, "seeds", "all_seeds", 0) == 4


def test_result_sink_is_process_local_and_persistent_across_batches():
    first = types.SimpleNamespace()
    second = types.SimpleNamespace()
    sink = result_sink(first)
    sink.append("batch-one")
    first.iteration = 1  # the next batch appends to the same list
    result_sink(first).append("batch-two")
    assert result_sink(first) is sink
    assert result_sink(first) == ["batch-one", "batch-two"]
    assert result_sink(second) == []
    result_sink(second).append("other")
    assert result_sink(first) == ["batch-one", "batch-two"]
