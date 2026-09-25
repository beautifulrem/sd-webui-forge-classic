from types import SimpleNamespace

from modules.script_cleanup import run_script_cleanup


def test_cleanup_runs_every_registered_script_with_its_arguments():
    calls = []

    class CleanupScript:
        args_from = 0
        args_to = 1
        filename = "cleanup-test.py"

        def cleanup(self, process, value):
            calls.append((process, value))

    script = CleanupScript()
    process = SimpleNamespace(script_args=["sentinel"])

    run_script_cleanup([script], process, lambda failed: calls.append(failed))

    assert calls == [(process, "sentinel")]


def test_cleanup_continues_after_a_script_raises():
    calls = []

    class BrokenScript:
        args_from = 0
        args_to = 0

        def cleanup(self, process):
            raise RuntimeError("boom")

    class HealthyScript:
        args_from = 0
        args_to = 0

        def cleanup(self, process):
            calls.append("healthy")

    broken = BrokenScript()
    process = SimpleNamespace(script_args=[])
    run_script_cleanup(
        [broken, HealthyScript()],
        process,
        lambda failed: calls.append(failed),
    )

    assert calls == [broken, "healthy"]
