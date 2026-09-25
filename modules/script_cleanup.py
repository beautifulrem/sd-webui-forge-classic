"""Dependency-free runner for guaranteed script cleanup hooks."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def run_script_cleanup(
    scripts: Iterable,
    process,
    report_error: Callable[[object], None],
) -> None:
    for script in scripts:
        try:
            script_args = process.script_args[script.args_from : script.args_to]
            script.cleanup(process, *script_args)
        except Exception:
            report_error(script)
