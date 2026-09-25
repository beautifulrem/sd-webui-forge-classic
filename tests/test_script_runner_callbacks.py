from types import SimpleNamespace

import pytest

from modules import errors, scripts


class _Script:
    def __init__(self, name, calls, *, fail=None, args=(0, 0)):
        self.filename = name
        self.args_from, self.args_to = args
        self._calls = calls
        self._fail = fail

    def _run(self, hook, p, *args):
        self._calls.append((hook, self.filename, args))
        if self._fail is not None:
            raise self._fail

    def cleanup(self, p, *args):
        self._run("cleanup", p, *args)

    def before_process_batch(self, p, *args, **kwargs):
        self._run("before_process_batch", p, *args)

    def process_before_every_sampling(self, p, *args, **kwargs):
        self._run("process_before_every_sampling", p, *args)


@pytest.fixture
def runner(monkeypatch):
    reported = []
    monkeypatch.setattr(errors, "report", lambda message, exc_info=False: reported.append(message))
    runner = scripts.ScriptRunner()
    runner.reported = reported
    return runner


def _order(runner, monkeypatch, ordered):
    monkeypatch.setattr(runner, "ordered_scripts", lambda method: ordered)


def test_cleanup_runs_every_script_with_its_arguments_even_after_a_failure(runner, monkeypatch):
    calls = []
    broken = _Script("broken.py", calls, fail=RuntimeError("boom"))
    healthy = _Script("healthy.py", calls, args=(0, 1))
    _order(runner, monkeypatch, [broken, healthy])

    runner.cleanup(SimpleNamespace(script_args=["sentinel"]))

    assert calls == [("cleanup", "broken.py", ()), ("cleanup", "healthy.py", ("sentinel",))]
    assert runner.reported == ["Error running cleanup: broken.py"]


def test_sampling_hook_runs_scripts_in_callback_order(runner, monkeypatch):
    calls = []
    _order(runner, monkeypatch, [_Script("second.py", calls), _Script("first.py", calls)])

    runner.process_before_every_sampling(SimpleNamespace(script_args=[]))

    assert [name for _, name, _ in calls] == ["second.py", "first.py"]


def test_before_process_batch_propagates_script_abort_only(runner, monkeypatch):
    calls = []
    _order(runner, monkeypatch, [_Script("reported.py", calls, fail=ValueError("x")), _Script("after.py", calls)])
    runner.before_process_batch(SimpleNamespace(script_args=[]))
    assert [name for _, name, _ in calls] == ["reported.py", "after.py"]

    _order(runner, monkeypatch, [_Script("abort.py", calls, fail=scripts.ScriptAbort("conflict"))])
    with pytest.raises(scripts.ScriptAbort):
        runner.before_process_batch(SimpleNamespace(script_args=[]))


def test_after_model_load_runs_scripts_and_reports_failures(runner, monkeypatch):
    calls = []

    class Script(_Script):
        def after_model_load(self, p, *args):
            self._run("after_model_load", p, *args)

    _order(runner, monkeypatch, [Script("broken.py", calls, fail=RuntimeError("x")), Script("ok.py", calls)])
    runner.after_model_load(SimpleNamespace(script_args=[]))

    assert [name for _, name, _ in calls] == ["broken.py", "ok.py"]
    assert runner.reported == ["Error running after_model_load: broken.py"]


def test_after_model_load_is_called_after_the_checkpoint_reload():
    # process_images needs a real model to run: check the call order instead
    import ast
    from pathlib import Path

    source = (Path(__file__).parents[1] / "modules" / "processing.py").read_text(encoding="utf-8")
    process_images = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "process_images")
    calls = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)): node.lineno
        for node in ast.walk(process_images)
        if isinstance(node, ast.Call)
    }
    assert calls["after_model_load"] > calls["manage_model_and_prompt_cache"]
