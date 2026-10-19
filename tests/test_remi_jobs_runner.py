"""remi.jobs.runner: the tab dispatch of a job, the conversion of what the
WebUI's processing returns into a JobResult, and running on the WebUI main
thread (``modules_forge.main_thread``) with the result or exception handed
back to the worker."""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from remi.core import JobContext
from remi.jobs import runner
from remi.jobs.store import Job


def make_job(tab: str, core=None, scripts=None) -> Job:
    return Job(id="job-1", seq=1, tab=tab, core=core or {"prompt": "a fox"}, scripts=scripts or {"seed": {"seed": 7}})


def image_with_infotext(text: str | None, color=(255, 0, 0)) -> Image.Image:
    image = Image.new("RGB", (8, 8), color)
    if text is not None:
        image.info["parameters"] = text
    return image


@pytest.fixture
def stub_module(monkeypatch):
    """Install ``modules.<name>`` with the given attributes (the real ones
    need the whole WebUI and a model)."""

    def install(name: str, **attributes):
        module = types.ModuleType(f"modules.{name}")
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, f"modules.{name}", module)
        return module

    return install


# region Tab dispatch


@pytest.mark.parametrize(("tab", "module", "entry"), [("txt2img", "txt2img", "run_txt2img"), ("img2img", "img2img", "run_img2img")])
def test_generation_tabs_call_their_named_entry_point(stub_module, tab, module, entry):
    calls = []
    first, second = image_with_infotext(None), image_with_infotext(None, (0, 0, 255))
    processed = SimpleNamespace(images=[first, second], infotexts=["fox\nSteps: 20, Seed: 7", "fox\nSteps: 20, Seed: 8"], info="ignored")

    def run(core, scripts, ctx):
        calls.append((core, scripts, ctx))
        return processed

    stub_module(module, **{entry: run})
    job, ctx = make_job(tab), JobContext(id_task="task-1")
    result = runner.run_job(job, ctx)
    assert calls == [(job.core, job.scripts, ctx)]
    assert result.images == [first, second]
    assert result.infotexts == ["fox\nSteps: 20, Seed: 7", "fox\nSteps: 20, Seed: 8"]
    assert result.info == "fox\nSteps: 20, Seed: 7" and result.files == []


def test_unknown_tabs_are_refused():
    with pytest.raises(ValueError, match="unknown tab 'png-info'"):
        runner.run_job(make_job("png-info"), JobContext())


def test_extras_results_become_images_files_and_infotexts(stub_module):
    own = image_with_infotext("upscaled\nPostprocess upscale by: 2")
    plain = image_with_infotext(None)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)  # a video operation's last frame
    frame[..., 1] = 200
    seen = []

    def run_extras_named(core, scripts, ctx):
        seen.append((core, scripts, ctx))
        return [own, None, "/out/clip.mp4", plain, frame], "Postprocess upscale by: 2, Postprocess upscaler: R-ESRGAN 4x+\n"

    stub_module("postprocessing", run_extras_named=run_extras_named)
    job, ctx = make_job("extras", core={"image": "upload-1"}, scripts={"upscale": {"upscaling_resize": 2}}), JobContext()
    result = runner.run_job(job, ctx)
    assert seen == [(job.core, job.scripts, ctx)]
    assert result.files == ["/out/clip.mp4"]
    assert result.images[:2] == [own, plain]
    converted = result.images[2]
    assert isinstance(converted, Image.Image) and converted.size == (6, 4) and converted.getpixel((0, 0)) == (0, 200, 0)
    info = "Postprocess upscale by: 2, Postprocess upscaler: R-ESRGAN 4x+"
    assert result.info == info
    # an image's own infotext wins, the others get the job's
    assert result.infotexts == ["upscaled\nPostprocess upscale by: 2", info, info]


@pytest.mark.parametrize("info", [None, "", "  \n"])
def test_extras_without_output_or_infotext(stub_module, info):
    stub_module("postprocessing", run_extras_named=lambda core, scripts, ctx: (None, info))
    result = runner.run_job(make_job("extras"), JobContext())
    assert result == runner.JobResult(images=[], infotexts=[], info=None, files=[])


def test_validation_errors_reach_the_caller(stub_module):
    from remi import ui

    def run(core, scripts, ctx):
        raise ui.ValidationError([ui.ParamError("too large", param="width")])

    stub_module("txt2img", run_txt2img=run)
    with pytest.raises(ui.ValidationError):
        runner.run_job(make_job("txt2img"), JobContext())


# endregion

# region Processed results


def test_extra_images_are_converted_and_get_the_infotext_of_their_position():
    detected = np.full((2, 3, 3), 9, dtype=np.uint8)  # e.g. ControlNet's detected map
    own = image_with_infotext("from the file")
    processed = SimpleNamespace(images=[image_with_infotext(None), None, own], extra_images=[detected], infotexts=["first", 42], info="job info")
    result = runner._from_processed(processed)
    assert len(result.images) == 3 and isinstance(result.images[2], Image.Image) and result.images[2].size == (3, 2)
    # strings only; later images fall back to their own infotext, then the first one
    assert result.infotexts == ["first", "from the file", "first"]
    assert result.info == "first"


def test_processed_without_infotexts_uses_its_info_string_only():
    assert runner._from_processed(SimpleNamespace(images=[image_with_infotext(None)], info="the info")).infotexts == ["the info"]
    result = runner._from_processed(SimpleNamespace(images=[], info={"not": "a string"}))
    assert result.info is None and result.images == [] and result.infotexts == []
    assert runner._from_processed(None) == runner.JobResult()


# endregion

# region Main thread


@pytest.fixture(scope="module")
def main_thread_loop():
    """The WebUI's main-thread loop, as webui.py runs it (on a daemon thread
    here; it polls its queue)."""

    from modules_forge import main_thread

    thread = threading.Thread(target=main_thread.loop, name="test-main-thread", daemon=True)
    thread.start()
    return thread


def test_run_on_main_thread_returns_the_result_from_that_thread(main_thread_loop):
    threads = []

    def work(a, b):
        threads.append(threading.current_thread())
        return a + b

    assert runner.run_on_main_thread(work, 2, 3) == 5
    assert threads == [main_thread_loop]


def test_run_on_main_thread_raises_the_exception_on_the_caller(main_thread_loop):
    from remi import ui

    class Interrupted(BaseException):
        pass

    def fail(kind):
        raise kind

    error = ui.ValidationError([ui.ParamError("must be at least 1", param="steps")])
    with pytest.raises(ui.ValidationError) as caught:
        runner.run_on_main_thread(fail, error)
    assert caught.value is error
    # not only Exception: the worker must see KeyboardInterrupt-like errors too
    with pytest.raises(Interrupted):
        runner.run_on_main_thread(fail, Interrupted())
    # the main thread survives and runs the next job
    assert runner.run_on_main_thread(lambda: "next") == "next"


def test_run_on_main_thread_returns_none_for_procedures(main_thread_loop):
    assert runner.run_on_main_thread(lambda: None) is None


def test_run_inline_calls_on_the_calling_thread():
    assert runner.run_inline(lambda *args: (threading.current_thread(), args), 1, 2) == (threading.current_thread(), (1, 2))


# endregion
