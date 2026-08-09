"""Small process-local state helpers for the PiD postprocessor."""

from __future__ import annotations


def batch_value(process, current_name: str, all_name: str, index: int):
    current = getattr(process, current_name, None)
    if isinstance(current, (list, tuple)) and len(current) > index:
        return current[index]
    values = getattr(process, all_name)
    iteration = int(getattr(process, "iteration", 0))
    offset = iteration * int(getattr(process, "batch_size", len(values)))
    return values[min(offset + index, len(values) - 1)]


def result_sink(process) -> list:
    if not hasattr(process, "_forge_pid_results"):
        process._forge_pid_results = []
    return process._forge_pid_results
