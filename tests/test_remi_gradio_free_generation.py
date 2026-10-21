"""The WebUI starts and generates with Gradio unimportable, without the real model.

This starts the real server the way a user does: ``launch.py`` through
``tools/tiny_anima.py serve`` (uvicorn on a port, every built-in extension, the
job queue and its worker, the model loader). An import hook makes every Gradio
package unimportable and records any attempt, and ``-X importtime`` lists every
module the process imports. A txt2img job then goes through
``POST /api/v2/jobs`` over HTTP, and its PNG is downloaded. The tiny Anima
checkpoint (``tools/tiny_anima.py``: Anima's real architecture at toy size, with
seeded random weights) loads in seconds on the CPU, so anyone can reproduce
"generates without Gradio" without the real model::

    python tools/tiny_anima.py serve --data-dir /path/to/data --port 7899 --cpu
    curl -s -X POST http://127.0.0.1:7899/api/v2/jobs -H 'Content-Type: application/json' \\
      -d '{"tab": "txt2img", "core": {"prompt": "a cat", "width": 64, "height": 64}, "scripts": {"sampler": {"steps": 4}}}'
    curl -s http://127.0.0.1:7899/api/v2/jobs/<job_id>          # outputs[0].id when "completed"
    curl -s -o out.png http://127.0.0.1:7899/api/v2/files/<output id>

(the pictures are noise-like, but deterministic: the real model is what makes them pictures).
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
GRADIO = ("gradio", "gradio_client", "gradio_rangeslider")

BLOCKER = f"""
import importlib.abc, os, runpy, sys

GRADIO = {GRADIO!r}

class NoGradio(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in GRADIO:
            with open(os.environ["REMI_GRADIO_ATTEMPTS"], "a", encoding="utf-8") as file:
                file.write(name + "\\n")
            raise ImportError(f"{{name}} is blocked: the WebUI runs without Gradio")
        return None

sys.meta_path.insert(0, NoGradio())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
"""Runs ``sys.argv[1]`` as ``__main__`` with Gradio unimportable (every attempt recorded)."""

CORE = {"prompt": "a girl with red hair, blue sky, garden", "negative_prompt": "lowres", "width": 128, "height": 64, "cfg_scale": 4.0}
SCRIPTS = {"sampler": {"steps": 4, "sampler_name": "Euler", "scheduler": "Simple"}, "seed": {"seed": 1234}}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("gradio-free")
    data = folder / "data"
    data.mkdir()
    # the tiny checkpoint (not the Refiner's, first by name); no browser
    (data / "config.json").write_text(json.dumps({"sd_model_checkpoint": "tiny-anima.safetensors", "auto_launch_browser": "Disable"}), encoding="utf-8")
    attempts = folder / "gradio-attempts.txt"
    attempts.touch()
    log = folder / "server.log"
    port = _free_port()
    env = {**os.environ, "REMI_GRADIO_ATTEMPTS": str(attempts), "REMI_SKIP_WEBUI_BUILD": "1", "SD_WEBUI_RESTARTING": "1", "YOLO_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    command = [sys.executable, "-X", "importtime", "-W", "ignore", "-c", BLOCKER, str(ROOT / "tools" / "tiny_anima.py"), "serve", "--data-dir", str(data), "--port", str(port), "--cpu", "--api", "--skip-torch-cuda-test"]
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    # trust_env=False: a proxy from the environment must not see a loopback request
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=60)
    try:
        deadline = time.monotonic() + 900
        while True:
            if process.poll() is not None:
                pytest.fail(f"the server exited with {process.returncode}:\n" + _tail(log))
            try:
                if client.get("/internal/ping", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail("the server did not start:\n" + _tail(log))
            time.sleep(1)
        yield {"client": client, "log": log, "attempts": attempts}
    finally:
        client.close()
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _tail(log: Path) -> str:
    lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if not line.startswith("import time:")]
    return "\n".join(lines[-60:])


def test_gradio_is_not_installed_here():
    """The gate interpreter is Gradio-free (tools/check.sh env:gradio-free); the
    server below blocks Gradio either way, so it proves the same on any machine."""

    import importlib.util

    if any(importlib.util.find_spec(name) for name in GRADIO):
        pytest.skip("this interpreter can import Gradio (the server test still blocks it)")


def test_a_txt2img_job_generates_through_the_real_server(server):
    from PIL import Image

    client = server["client"]
    app = client.get("/api/v2/app").json()
    assert app.get("not_ported", []) == []
    created = client.post("/api/v2/jobs", json={"tab": "txt2img", "core": CORE, "scripts": SCRIPTS})
    assert created.status_code in (200, 201, 202), created.text
    job_id = created.json()["job_id"]
    deadline = time.monotonic() + 600
    while True:
        job = client.get(f"/api/v2/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            break
        assert time.monotonic() < deadline, job
        time.sleep(0.5)
    assert job["status"] == "completed", (job, _tail(server["log"]))
    assert job["outputs"], job
    output = job["outputs"][0]
    png = client.get(f"/api/v2/files/{output['id']}")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    image = Image.open(io.BytesIO(png.content))
    assert image.size == (CORE["width"], CORE["height"])
    # not a blank canvas: the sampler and the VAE produced pixels
    assert len(image.convert("RGB").getcolors(1 << 24)) > 16
    infotext = output.get("infotext") or job.get("infotext") or ""
    assert "Steps: 4" in infotext and "Seed: 1234" in infotext and "tiny-anima" in infotext, infotext


def test_no_gradio_module_was_imported_or_tried(server):
    # after the job above (module order: the server fixture is shared)
    assert server["attempts"].read_text(encoding="utf-8").split() == []
    imported = [line.rsplit("|", 1)[-1].strip() for line in server["log"].read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("import time:")]
    assert len(imported) > 1000  # -X importtime recorded the whole start and the job
    assert [name for name in imported if name.partition(".")[0] in GRADIO] == []
    assert {"fastapi", "uvicorn", "torch", "remi.server.app", "backend.diffusion_engine.anima"} <= set(imported)
