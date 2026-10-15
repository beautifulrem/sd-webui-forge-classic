"""Real-server smoke test: start the WebUI with the real model, check generation, stop it.

    /tmp/remi-testenv/bin/python tools/smoke_real.py
    python tools/smoke_real.py --port 7879 --out /tmp/remi-smoke --keep-data

What it does (one model load, about three minutes on an M4 with Anima Turbo):

1. Starts ``launch.py --api`` on its own port with an isolated data folder: a copy of the
   repo's ``config.json`` (so the same settings apply), models read from ``models/``, outputs
   and ``remi-data`` in the temporary folder. The SPA is not rebuilt
   (``REMI_SKIP_WEBUI_BUILD=1``). Refuses to start while another ``launch.py`` runs: only one
   process may hold the model.
2. Selects the checkpoint and its additional modules (``PATCH /api/v2/options``).
3. txt2img parity, twice: ``POST /sdapi/v1/txt2img`` with the baseline request and
   ``POST /api/v2/jobs`` with the same values as named parameters (incl. the API's distilled
   CFG 3.5, Anima's shift; the form's default is 3.0). Each image must be
   pixel-identical to the baseline image and its infotext equal to the baseline infotext
   (only ``Version`` may differ; reported).
4. img2img at denoising 0.5 from the baseline image (an image of the requested size that
   follows the source but is not a copy) and inpaint with the right half masked (the left
   half stays the source, the right half changes).
5. Extras: Lanczos upscale by 1.5 (the size and the content follow the source).
6. Stops the server (Ctrl+C, then terminate) and prints a result table; exit 1 when a check
   failed, 2 when the server could not be started.

Baseline: ``--baseline`` folder with ``baseline_1.png`` and ``baseline_txt2img.json`` (the
pre-rewrite ``/sdapi/v1/txt2img`` request and response; default ``/tmp/remi-parity``).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "anima-turbo-v1.1.safetensors"
MODULES = ["models/VAE/qwen_image_vae.safetensors", "models/text_encoder/qwen_3_06b_base.safetensors"]

# Loopback only: never through an HTTP proxy from the environment.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass
class Report:
    rows: list[tuple[str, bool, str]] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{f': {detail}' if detail else ''}", flush=True)
        return ok

    @property
    def failed(self) -> int:
        return sum(not ok for _, ok, _ in self.rows)


class Server:
    def __init__(self, python: str, port: int, data_dir: Path, log: Path) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = log
        args = [python, "launch.py", "--skip-prepare-environment", "--skip-install", "--api", "--port", str(port), "--data-dir", str(data_dir), "--model-ref", str(ROOT / "models")]
        env = {**os.environ, "REMI_SKIP_WEBUI_BUILD": "1", "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
        self.log = open(log, "wb")  # noqa: SIM115 - open for the life of the process
        self.process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def request(self, method: str, path: str, body: object | None = None, timeout: float = 600) -> object:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
        try:
            with _opener.open(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{method} {path}: HTTP {error.code}: {error.read()[:2000].decode(errors='replace')}") from None
        kind = response.headers.get("content-type", "")
        return json.loads(raw) if "json" in kind else raw

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return False
            try:
                self.request("GET", "/api/v2/app", timeout=5)
                return True
            except (OSError, RuntimeError):
                time.sleep(2)
        return False

    def stop(self) -> None:
        if self.process.poll() is None:
            for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.process.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.log.close()


def other_launchers() -> list[str]:
    import psutil

    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if proc.pid != os.getpid() and any(part.endswith("launch.py") for part in cmdline):
            found.append(f"{proc.pid}: {' '.join(cmdline)[:160]}")
    return found


def decode(data: str | bytes):
    from PIL import Image

    raw = base64.b64decode(data.split(",", 1)[-1]) if isinstance(data, str) else data
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image


def b64png(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def diff(a, b) -> tuple[float, int]:
    """Mean and max absolute RGB difference."""

    import numpy as np

    x = np.asarray(a.convert("RGB"), dtype=np.int16)
    y = np.asarray(b.convert("RGB"), dtype=np.int16)
    if x.shape != y.shape:
        return float("inf"), 255
    delta = np.abs(x - y)
    return float(delta.mean()), int(delta.max())


_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')


def parse_infotext(text: str) -> dict[str, str]:
    """Prompt, negative prompt and the ``key: value`` parameter line (the WebUI's format)."""

    lines = text.strip().split("\n")
    params = {match.group(1): match.group(2).strip() for match in _PARAM.finditer(lines[-1])} if len(_PARAM.findall(lines[-1])) >= 3 else {}
    if params:
        lines = lines[:-1]
    negative_at = next((index for index, line in enumerate(lines) if line.startswith("Negative prompt:")), len(lines))
    params["Prompt"] = "\n".join(lines[:negative_at]).strip()
    params["Negative prompt"] = "\n".join(lines[negative_at:]).removeprefix("Negative prompt:").strip()
    return params


def infotext_delta(got: str, want: str) -> list[str]:
    """Differences between two infotexts, the ``Version`` value excluded."""

    a, b = parse_infotext(got), parse_infotext(want)
    keys = sorted((set(a) | set(b)) - {"Version"})
    return [f"{key}: {a.get(key)!r} != {b.get(key)!r}" for key in keys if a.get(key) != b.get(key)]


def version_note(got: str, want: str) -> str:
    a, b = parse_infotext(got).get("Version"), parse_infotext(want).get("Version")
    return "" if a == b else f"Version {b} -> {a}"


def wait_job(server: Server, job_id: str, timeout: float = 900) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = server.request("GET", f"/api/v2/jobs/{job_id}")
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(1)
    raise TimeoutError(f"job {job_id} still running after {timeout} s")


def run(server: Server, baseline_dir: Path, out: Path, report: Report) -> None:
    from PIL import Image

    recorded = json.loads((baseline_dir / "baseline_txt2img.json").read_text())
    baseline = Image.open(baseline_dir / "baseline_1.png")
    baseline.load()
    want_info = recorded.get("info")
    want_info = json.loads(want_info)["infotexts"][0] if isinstance(want_info, str) else want_info["infotexts"][0]
    params = recorded["parameters"]

    print("Selecting the model", flush=True)
    server.request("PATCH", "/api/v2/options", {"sd_model_checkpoint": MODEL, "forge_additional_modules": [str(ROOT / path) for path in MODULES]})

    print("txt2img parity (/sdapi/v1/txt2img)", flush=True)
    started = time.monotonic()
    answer = server.request("POST", "/sdapi/v1/txt2img", params)
    image = decode(answer["images"][0])
    image.save(out / "sdapi_txt2img.png")
    info = json.loads(answer["info"])["infotexts"][0]
    mean, peak = diff(image, baseline)
    report.check("sdapi txt2img pixels == baseline", mean == 0 and peak == 0, f"mean {mean:.4f}, max {peak}, {time.monotonic() - started:.1f} s")
    delta = infotext_delta(info, want_info)
    report.check("sdapi txt2img infotext == baseline", not delta, "; ".join(delta) or version_note(info, want_info))

    print("txt2img parity (/api/v2/jobs)", flush=True)
    body = {
        "tab": "txt2img",
        "core": {"prompt": params["prompt"], "negative_prompt": params["negative_prompt"], "width": params["width"], "height": params["height"], "cfg_scale": params["cfg_scale"], "distilled_cfg_scale": params["distilled_cfg_scale"], "batch_size": 1, "batch_count": 1},
        "scripts": {"sampler": {"sampler_name": params["sampler_name"], "scheduler": params["scheduler"], "steps": params["steps"]}, "seed": {"seed": params["seed"]}},
    }
    job = wait_job(server, server.request("POST", "/api/v2/jobs", body)["job_id"])
    if report.check("v2 txt2img job completed", job["status"] == "completed", job.get("error") or ""):
        output = job["outputs"][0]
        image = decode(server.request("GET", f"/api/v2/files/{output['id']}"))
        image.save(out / "v2_txt2img.png")
        mean, peak = diff(image, baseline)
        report.check("v2 txt2img pixels == baseline", mean == 0 and peak == 0, f"mean {mean:.4f}, max {peak}")
        text = output.get("infotext") or job.get("infotext") or ""
        delta = infotext_delta(text, want_info)
        report.check("v2 txt2img infotext == baseline", not delta, "; ".join(delta) or version_note(text, want_info))

    print("img2img (denoising 0.5)", flush=True)
    common = {"prompt": params["prompt"], "negative_prompt": params["negative_prompt"], "seed": params["seed"], "steps": params["steps"], "cfg_scale": params["cfg_scale"], "sampler_name": params["sampler_name"], "scheduler": params["scheduler"], "width": 768, "height": 768}
    answer = server.request("POST", "/sdapi/v1/img2img", {**common, "init_images": [b64png(baseline)], "denoising_strength": 0.5})
    image = decode(answer["images"][0])
    image.save(out / "img2img.png")
    mean, _ = diff(image, baseline)
    info = json.loads(answer["info"])["infotexts"][0]
    report.check("img2img size and effect", image.size == baseline.size and 1 < mean < 60, f"{image.size[0]}x{image.size[1]}, mean diff to the source {mean:.2f}")
    report.check("img2img infotext", "Denoising strength: 0.5" in info, info.splitlines()[-1][:160])

    print("inpaint (right half masked)", flush=True)
    mask = Image.new("L", baseline.size, 0)
    mask.paste(255, (baseline.width // 2, 0, baseline.width, baseline.height))
    answer = server.request("POST", "/sdapi/v1/img2img", {**common, "init_images": [b64png(baseline)], "mask": b64png(mask), "denoising_strength": 0.75, "mask_blur": 0, "inpainting_fill": 1, "inpaint_full_res": False})
    image = decode(answer["images"][0])
    image.save(out / "inpaint.png")
    half = baseline.width // 2
    left_mean, _ = diff(image.crop((0, 0, half - 16, baseline.height)), baseline.crop((0, 0, half - 16, baseline.height)))
    right_mean, _ = diff(image.crop((half + 16, 0, baseline.width, baseline.height)), baseline.crop((half + 16, 0, baseline.width, baseline.height)))
    report.check("inpaint respects the mask", left_mean < 1.0 and right_mean > 2.0, f"unmasked mean diff {left_mean:.3f}, masked {right_mean:.2f}")

    print("extras (Lanczos x1.5)", flush=True)
    answer = server.request("POST", "/sdapi/v1/extra-single-image", {"image": b64png(baseline), "upscaling_resize": 1.5, "upscaler_1": "Lanczos", "resize_mode": 0})
    image = decode(answer["image"])
    image.save(out / "extras.png")
    expected = (round(baseline.width * 1.5), round(baseline.height * 1.5))
    mean, _ = diff(image.resize(baseline.size, Image.Resampling.LANCZOS), baseline)
    report.check("extras upscale", image.size == expected and mean < 3, f"{image.size[0]}x{image.size[1]} (want {expected[0]}x{expected[1]}), mean diff after downscaling {mean:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter for launch.py (default: this one)")
    parser.add_argument("--port", type=int, default=7879)
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/remi-parity"))
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "remi-smoke", help="images and the server log")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--keep-data", action="store_true", help="keep the temporary data folder")
    options = parser.parse_args()

    for name in ("baseline_1.png", "baseline_txt2img.json"):
        if not (options.baseline / name).is_file():
            print(f"missing {options.baseline / name}", file=sys.stderr)
            return 2
    if busy := other_launchers():
        print("another launch.py is running (only one process may load the model):\n  " + "\n  ".join(busy), file=sys.stderr)
        return 2

    options.out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(tempfile.mkdtemp(prefix="remi-smoke-data-"))
    if (ROOT / "config.json").is_file():
        shutil.copy2(ROOT / "config.json", data_dir / "config.json")
    for name in ("styles.csv", "ui-config.json"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, data_dir / name)

    report = Report()
    print(f"Starting the server on port {options.port} (log: {options.out / 'server.log'}, data: {data_dir})", flush=True)
    started = time.monotonic()
    server = Server(options.python, options.port, data_dir, options.out / "server.log")
    try:
        if not report.check("server starts", server.wait_ready(options.startup_timeout), f"{time.monotonic() - started:.1f} s"):
            print(Path(options.out / "server.log").read_text(errors="replace")[-4000:], file=sys.stderr)
            return 2
        try:
            run(server, options.baseline, options.out, report)
        except Exception as error:  # a failed request ends the run but still stops the server
            report.check("smoke run", False, f"{type(error).__name__}: {error}"[:2000])
    finally:
        print("Stopping the server", flush=True)
        server.stop()
        report.check("server stops", server.process.poll() is not None, f"exit code {server.process.returncode}")
        if not options.keep_data:
            shutil.rmtree(data_dir, ignore_errors=True)

    print(f"\n{'check':<40} result  detail")
    for name, ok, detail in report.rows:
        print(f"{name:<40} {'pass' if ok else 'FAIL':<7} {detail}")
    print(f"\nImages: {options.out}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
