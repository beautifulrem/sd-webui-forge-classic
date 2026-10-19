"""Hires. fix an existing txt2img result (A1111's ✨ button): the job's
``upscale_from``, its processing object (``run_txt2img_upscale``) with the
settings ``txt2img_upscale_single_batch`` / ``txt2img_upscale_same_seed``, and
the output roles (grid, extra) and videos a generation returns."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from PIL import Image

from remi.core import JobContext
from remi.jobs import runner
from remi.jobs.store import Job
from remi.server.testing import server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def server(webui, tmp_path):
    with server_app(tmp_path) as server:
        yield server


def _post(server, **body):
    return server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "a cat"}, "scripts": {}, **body})


def _first_output(server) -> str:
    created = _post(server)
    assert created.status_code == 201, created.text
    assert server.services.queue.wait_idle()
    return server.client.get(f"/api/v2/jobs/{created.json()['job_id']}").json()["outputs"][0]["id"]


# region API


def test_a_txt2img_job_can_start_from_an_output(server):
    source = _first_output(server)
    created = _post(server, upscale_from=source)
    assert created.status_code == 201, created.text
    assert server.services.queue.wait_idle()
    job, _ = server.executor.calls[-1]
    assert job.meta["upscale_from"] == source

    # a retry Hires. fixes the same image again
    retried = server.client.post(f"/api/v2/jobs/{job.id}/retry")
    assert retried.status_code == 201
    assert server.services.store.get(retried.json()["job_id"]).meta["upscale_from"] == source


@pytest.mark.parametrize(("tab", "source", "message"), [("img2img", None, "only txt2img"), ("txt2img", "0123456789abcdef", "unknown image")])
def test_the_source_must_be_a_known_image_of_a_txt2img_job(server, tab, source, message):
    source = source or _first_output(server)
    core = {"prompt": "a cat", "init_img": source} if tab == "img2img" else {"prompt": "a cat"}
    response = server.client.post("/api/v2/jobs", json={"tab": tab, "core": core, "upscale_from": source})
    assert response.status_code == 422
    [error] = [error for error in response.json()["errors"] if error["param"] == "upscale_from"]
    assert message in error["message"]


def test_a_queued_upscale_keeps_its_source_from_upload_pruning(server):
    source = _first_output(server)
    server.services.queue.set_paused(True)
    assert _post(server, upscale_from=source).status_code == 201
    assert source in server.services.store.referenced_strings()
    server.services.queue.set_paused(False)


# endregion

# region Runner


def test_the_runner_hands_the_image_and_its_parameters_to_run_txt2img_upscale(monkeypatch):
    source = Image.new("RGB", (8, 4))
    source.info["parameters"] = "a fox\nSteps: 20, Seed: 1234, Size: 8x4"
    calls = []

    def upscale(core, scripts, ctx, image, infotext):
        calls.append((core, scripts, image, infotext))
        return SimpleNamespace(images=[image], infotexts=["a fox\nSeed: 1234, Hires upscale: 2"])

    module = types.ModuleType("modules.txt2img")
    module.run_txt2img = lambda *args: pytest.fail("the first pass must not run")
    module.run_txt2img_upscale = upscale
    monkeypatch.setitem(sys.modules, "modules.txt2img", module)

    resolved = []
    ctx = JobContext(resolve_upload=lambda node, file_id: resolved.append((node.name, file_id)) or source)
    job = Job(id="job-1", seq=1, tab="txt2img", core={"prompt": "a fox"}, scripts={}, meta={"upscale_from": "out-1"})
    result = runner.run_job(job, ctx)
    assert resolved == [("upscale_from", "out-1")]
    assert calls == [({"prompt": "a fox"}, {}, source, "a fox\nSteps: 20, Seed: 1234, Size: 8x4")]
    assert result.images == [source]


def test_a_source_that_went_away_fails_the_job_as_a_validation_error():
    from remi import ui

    def gone(node, file_id):
        raise ui.ParamError(f"unknown upload {file_id!r}")

    job = Job(id="job-1", seq=1, tab="txt2img", core={}, scripts={}, meta={"upscale_from": "out-1"})
    with pytest.raises(ui.ValidationError) as caught:
        runner.run_job(job, JobContext(resolve_upload=gone))
    assert caught.value.to_json()["errors"][0]["param"] == "upscale_from"


def test_grid_extra_images_and_the_video_are_told_apart():
    grid, first, second, detected = (Image.new("RGB", (4, 4), color) for color in ("red", "green", "blue", "white"))
    processed = SimpleNamespace(images=[grid, first, second], extra_images=[detected], index_of_first_image=1, infotexts=["g", "a", "b"], video_path="/out/clip.mp4")
    result = runner._from_processed(processed)
    assert result.images == [grid, first, second, detected]
    assert result.roles == ["grid", None, None, "extra"]
    assert result.files == ["/out/clip.mp4"]


def test_output_roles_reach_the_job_outputs(webui, tmp_path):
    images = [Image.new("RGB", (4, 4), color) for color in ("red", "green")]

    def executor(job, ctx):
        return runner.JobResult(images=images, infotexts=["grid", "image"], info="image", roles=["grid", None])

    with server_app(tmp_path, executor=executor) as server:
        created = _post(server)
        assert server.services.queue.wait_idle()
        outputs = server.client.get(f"/api/v2/jobs/{created.json()['job_id']}").json()["outputs"]
    assert [output.get("role") for output in outputs] == ["grid", None]


# endregion

# region Processing object


@pytest.mark.parametrize("single_batch", [True, False])
@pytest.mark.parametrize("same_seed", [True, False])
def test_the_processing_object_is_the_first_pass_plus_hires_fix(monkeypatch, single_batch, same_seed):
    with webui_state({"txt2img_upscale_single_batch": single_batch, "txt2img_upscale_same_seed": same_seed}):
        from modules import txt2img

        made = []
        monkeypatch.setattr(txt2img, "_process", lambda p, job: made.append(p) or "processed")
        image = Image.new("RGB", (768, 512))
        core = {"prompt": "a fox", "width": 512, "height": 512, "batch_size": 2, "batch_count": 3, "enable_hr": False}
        result = txt2img.run_txt2img_upscale(core, {"seed": {"seed": 99}}, JobContext(), image, "a fox\nSteps: 20, Seed: 1234, Variation seed: 77, Size: 768x512")

    assert result == "processed"
    [p] = made
    assert p.enable_hr and p.txt2img_upscale and p.firstpass_image is image
    assert (p.width, p.height) == (768, 512)
    assert p.extra_generation_params["Original Size"] == "512x512"
    assert p.override_settings["save_images_before_highres_fix"] is False
    assert (p.batch_size, p.n_iter) == ((1, 1) if single_batch else (2, 3))
    assert (p.seed, p.subseed) == ((1234, 77) if same_seed else (99, -1))


# endregion
