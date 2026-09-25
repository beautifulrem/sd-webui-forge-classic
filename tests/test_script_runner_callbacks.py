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


def _process_images(calls, *, fail_in=None):
    """Run processing.py's real ``process_images`` against stubs: the module
    itself only imports inside an initialised WebUI."""
    import __future__
    import ast
    import contextlib
    from pathlib import Path

    path = Path(__file__).parents[1] / "modules" / "processing.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_images")

    def step(name):
        def run(*_args, **_kwargs):
            calls.append(name)
            if name == fail_in:
                raise RuntimeError(name)
            return name

        return run

    namespace = {
        "opts": SimpleNamespace(data={}, get_default=lambda key: None),
        "sd_models": SimpleNamespace(checkpoint_aliases={}),
        "set_config": lambda *args, **kwargs: None,
        "manage_model_and_prompt_cache": step("load model"),
        "sd_samplers": SimpleNamespace(fix_p_invalid_sampler_and_scheduler=lambda p: None),
        "profiling": SimpleNamespace(Profiler=contextlib.nullcontext),
        "process_images_inner": step("sample"),
    }
    code = compile(ast.Module([function], []), str(path), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)  # noqa: S102 - our own source file

    hooks = SimpleNamespace(**{name: step(name) for name in ("before_process", "after_model_load", "cleanup")})
    p = SimpleNamespace(scripts=hooks, override_settings={}, override_settings_restore_afterwards=True)
    return namespace["process_images"](p)


def test_process_images_runs_after_model_load_after_the_reload_and_cleanup_last():
    calls = []
    assert _process_images(calls) == "sample"
    assert calls == ["before_process", "load model", "after_model_load", "sample", "cleanup"]


@pytest.mark.parametrize("fail_in", ["load model", "after_model_load", "sample"])
def test_process_images_runs_cleanup_even_when_the_run_raises(fail_in):
    calls = []
    with pytest.raises(RuntimeError, match=fail_in):
        _process_images(calls, fail_in=fail_in)
    assert calls[-2:] == [fail_in, "cleanup"]
