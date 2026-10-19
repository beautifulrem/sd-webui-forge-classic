"""Real-server smoke test: start the WebUI with the real model, check generation, stop it.

    python tools/smoke_real.py      # an interpreter with requirements.txt and without Gradio
    python tools/smoke_real.py --port 7879 --out /tmp/remi-smoke --keep-data
    python tools/smoke_real.py --features   # also the LoRA features, Sparse Attention and paste round-trips

What it does (one model load, about three minutes on an M4 with Anima Turbo):

0. Checks that the server's interpreter (``--python``) cannot import Gradio
   (``gradio``, ``gradio_client``, ``gradio_rangeslider``), so the run proves a Gradio-free start.
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
6. With ``--features``: the Anima features that need LoRA files, against two tiny test LoRAs
   generated for the checkpoint into the temporary data folder (``tools/anima_test_loras.py``,
   passed as ``--lora-dir``), at 512x512 with a fixed seed:

   - a LoRA changes the image; LoRA Layer Weight at block weight 0 undoes it; LoRA Stage
     Scheduler changes it; Anima FreeFuse routes the two subject LoRAs (infotext
     ``Anima FreeFuse: 2 subjects ...``, no ``error:``) - each with its infotext keys;
   - Sparse Attention: on a CUDA GPU with sol_attn its infotext; elsewhere a job warning
     "... was skipped: CUDA device is required" and an unchanged image (not a silent no-op);
   - inpaint with "Masked content: latent noise" finishes in time and keeps the unmasked half;
   - paste round-trips: each feature job's infotext through ``POST /api/v2/infotext/parse``
     (no problems) and queued again gives the same pixels and infotext - also for Prompt
     Anchor, PP Suite and the Refiner;
   - the Refiner with a second, different checkpoint: a perturbed copy of the checkpoint
     (``tools/anima_test_loras.py --refiner``, about 4 GB) written into the temporary data
     folder (``--ckpt-dirs``; ``--no-refiner-checkpoint`` uses an installed one) takes over
     at 0.5 - the image changes, the infotext records ``Refiner: <name> [<hash>]``, and the
     base checkpoint is selected again;
   - runs a feature changed the request of (an Anima Flow sampler's scheduler lock, Dynamic
     Shift's Automatic schedule, the Anima Hires Guard's clamp) record the replacement, and
     their paste restores the request, so the pasted job writes the same infotext.

7. Stops the server (Ctrl+C, then terminate) and prints a result table; exit 1 when a check
   failed, 2 when the server could not be started.

Baseline: ``--baseline`` folder, by default the committed pre-rewrite fixture
``tests/fixtures/parity`` (``anima_turbo_txt2img_seed12345.png`` and its request values and
infotext); a folder with ``baseline_1.png`` and ``baseline_txt2img.json`` (a recorded
``/sdapi/v1/txt2img`` request and response) works too.
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

    def document(self, python: str, gradio: list[str], features: bool) -> dict:
        """The run as committable evidence (``--report``): what ran, on which revision, and each check."""

        def git(*args: str) -> str:
            try:
                return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                return ""

        return {
            "tool": "tools/smoke_real.py",
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "revision": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
            "platform": sys.platform,
            "server_python": python,
            "server_python_gradio_modules": gradio,
            "model": MODEL,
            "features": features,
            "passed": not self.failed,
            "checks": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in self.rows],
        }


class Server:
    def __init__(self, python: str, port: int, data_dir: Path, log: Path, extra_args: list[str] | None = None) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = log
        args = [python, "launch.py", "--skip-prepare-environment", "--skip-install", "--api", "--port", str(port), "--data-dir", str(data_dir), "--model-ref", str(ROOT / "models"), *(extra_args or [])]
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


GRADIO_MODULES = ("gradio", "gradio_client", "gradio_rangeslider")


def gradio_modules_in(python: str) -> list[str]:
    """The Gradio packages ``python`` could import (none for a Gradio-free WebUI)."""

    code = f"import importlib.util; print(' '.join(name for name in {GRADIO_MODULES!r} if importlib.util.find_spec(name)))"
    result = subprocess.run([python, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120, check=True)
    return result.stdout.split()


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


FIXTURE = ROOT / "tests" / "fixtures" / "parity"
FIXTURE_NAME = "anima_turbo_txt2img_seed12345"


def load_baseline(baseline_dir: Path) -> tuple[dict, str, Path]:
    """The baseline's ``/sdapi/v1/txt2img`` request, infotext and image: a recorded
    request/response pair (``baseline_txt2img.json``, ``baseline_1.png``), or the
    committed fixture (``tests/fixtures/parity``: the request's values and infotext
    in ``anima_turbo_txt2img_seed12345.json``, the image next to it)."""

    if (baseline_dir / "baseline_txt2img.json").is_file():
        recorded = json.loads((baseline_dir / "baseline_txt2img.json").read_text())
        info = recorded.get("info")
        info = json.loads(info)["infotexts"][0] if isinstance(info, str) else info["infotexts"][0]
        return recorded["parameters"], info, baseline_dir / "baseline_1.png"
    fixture = json.loads((baseline_dir / f"{FIXTURE_NAME}.json").read_text())
    keys = ("prompt", "negative_prompt", "seed", "steps", "cfg_scale", "width", "height", "sampler_name", "scheduler")
    # The recorded request left the distilled CFG at the API default (written as "Shift: 3.5")
    params = {**{key: fixture[key] for key in keys}, "distilled_cfg_scale": 3.5, "batch_size": 1, "n_iter": 1}
    return params, fixture["infotexts"][0], baseline_dir / f"{FIXTURE_NAME}.png"


def baseline_files(baseline_dir: Path) -> list[Path]:
    if (baseline_dir / "baseline_txt2img.json").is_file():
        return [baseline_dir / "baseline_txt2img.json", baseline_dir / "baseline_1.png"]
    return [baseline_dir / f"{FIXTURE_NAME}.json", baseline_dir / f"{FIXTURE_NAME}.png"]


def run(server: Server, baseline_dir: Path, out: Path, report: Report) -> dict:
    """Steps 2-5; the baseline request's values (the feature checks reuse them)."""

    from PIL import Image

    params, want_info, image_path = load_baseline(baseline_dir)
    baseline = Image.open(image_path)
    baseline.load()

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
    return params


# region Features (--features)

FEATURE_PROMPT = "1girl, red hair, standing in a garden, masterpiece, best quality"
FREEFUSE_PROMPT = "a girl with red hair and a boy with blue hair standing in a garden, masterpiece"


def script_owner(schema: dict, *args: str) -> str:
    """The owner id of the script whose params include ``args`` (ids are slugs of titles,
    which some panels translate; their param names are stable)."""

    for entry in schema["scripts"]:
        if set(args) <= set(entry.get("args") or ()):
            return entry["id"]
    raise LookupError(f"no script with the params {args}")


def run_v2(server: Server, tab: str, core: dict, scripts: dict, timeout: float = 900) -> tuple[dict, object | None, str]:
    """Queue a ``/api/v2/jobs`` job and wait: (job, first image, its infotext)."""

    job = wait_job(server, server.request("POST", "/api/v2/jobs", {"tab": tab, "core": core, "scripts": scripts})["job_id"], timeout)
    if job["status"] != "completed" or not job.get("outputs"):
        return job, None, ""
    output = job["outputs"][0]
    image = decode(server.request("GET", f"/api/v2/files/{output['id']}"))
    return job, image, output.get("infotext") or job.get("infotext") or ""


def check_round_trip(server: Server, report: Report, name: str, tab: str, image, infotext: str) -> None:
    """Paste ``infotext`` (``POST /api/v2/infotext/parse``) and queue the pasted values:
    the same pixels and infotext."""

    parsed = server.request("POST", "/api/v2/infotext/parse", {"text": infotext, "tab": tab})
    if not report.check(f"{name}: paste has no problems", not parsed["problems"], json.dumps(parsed["problems"])[:300]):
        return
    job, again, text = run_v2(server, tab, parsed["core"], parsed["scripts"])
    if not report.check(f"{name}: pasted job completed", again is not None, job.get("error") or ""):
        return
    mean, peak = diff(again, image)
    delta = infotext_delta(text, infotext)
    report.check(f"{name}: paste regenerates it", mean == 0 and not delta, f"mean {mean:.4f}, max {peak}; " + ("; ".join(delta)[:300] or "same infotext"))


def run_features(server: Server, out: Path, report: Report, base_values: dict) -> None:
    """The feature checks (see the module docstring, step 6)."""

    schema = server.request("GET", "/api/v2/schema/txt2img")
    layer = script_owner(schema, "lora_enabled", "lora_block_weights")
    stage = script_owner(schema, "enabled", "base_timing")
    freefuse = script_owner(schema, "subject1_selector", "subject2_concept")
    sparse = script_owner(schema, "tau", "min_tokens", "dense_blocks")
    anchor = script_owner(schema, "anchor", "separator")
    pp_suite = script_owner(schema, "vignette_enabled", "vignette_amount")
    sampler = {"sampler_name": base_values["sampler_name"], "scheduler": base_values["scheduler"], "steps": base_values["steps"]}
    common_scripts = {"sampler": sampler, "seed": {"seed": base_values["seed"]}}

    def core(prompt: str) -> dict:
        return {"prompt": prompt, "negative_prompt": base_values["negative_prompt"], "width": 512, "height": 512, "cfg_scale": base_values["cfg_scale"], "distilled_cfg_scale": base_values["distilled_cfg_scale"]}

    def generate(name: str, prompt: str, scripts: dict | None = None) -> tuple[dict, object | None, str]:
        print(f"feature: {name}", flush=True)
        job, image, info = run_v2(server, "txt2img", core(prompt), {**common_scripts, **(scripts or {})})
        report.check(f"{name}: job completed", image is not None, job.get("error") or "")
        if image is not None:
            image.save(out / f"feature_{name}.png")
        return job, image, info

    lora = f"{FEATURE_PROMPT}, <lora:remi-test-a:1>"
    _, plain, _ = generate("no_lora", FEATURE_PROMPT)
    # The prompt the model sees once the tag is taken out ("..., best quality, "): a
    # LoRA at weight 0 must give exactly this image (the trailing ", " is a token too)
    _, untagged, _ = generate("no_lora_tag_removed", re.sub(r"<(\w+):([^>]+)>", "", lora))
    _, with_lora, lora_info = generate("lora", lora)
    if plain is None or untagged is None or with_lora is None:
        return
    lora_effect, _ = diff(with_lora, untagged)
    report.check("LoRA changes the image", lora_effect > 1.0 and "remi-test-a" in lora_info, f"mean diff {lora_effect:.2f}; " + lora_info.splitlines()[-1][:200])

    # LoRA Layer Weight: every block at 0 undoes the LoRA
    _, zeroed, info = generate("layer_weight", lora, {layer: {"lora_enabled": True, "lora_block_enabled": True, "lora_block_weights": "0-27=0"}})
    if zeroed is not None:
        to_plain, peak = diff(zeroed, untagged)
        report.check("Layer Weight 0 removes the LoRA's effect", to_plain == 0 and peak == 0, f"mean diff to no LoRA {to_plain:.3f}, max {peak} (the LoRA: {lora_effect:.2f})")
        report.check("Layer Weight infotext", "Anima LoRA layer lora enabled: True" in info and "Anima LoRA layer lora blocks: " in info, info.splitlines()[-1][:300])
        check_round_trip(server, report, "Layer Weight", "txt2img", zeroed, info)

    # LoRA Stage Scheduler (default template and auto timing)
    _, staged, info = generate("stage_scheduler", lora, {stage: {"enabled": True}})
    if staged is not None:
        to_lora, _ = diff(staged, with_lora)
        report.check("Stage Scheduler changes the LoRA's effect", to_lora > 0.2, f"mean diff to the unscheduled LoRA {to_lora:.2f}")
        report.check("Stage Scheduler infotext", "Anima LoRA stage enabled: True" in info, info.splitlines()[-1][:300])
        check_round_trip(server, report, "Stage Scheduler", "txt2img", staged, info)

    # Anima FreeFuse: two subject LoRAs routed to their trigger phrases
    two = f"{FREEFUSE_PROMPT}, <lora:remi-test-a:1>, <lora:remi-test-b:1>"
    _, unrouted, _ = generate("two_loras", two)
    subjects = {"enable": True, "subject1_active": True, "subject1_selector": "remi-test-a", "subject1_concept": "a girl with red hair", "subject2_active": True, "subject2_selector": "remi-test-b", "subject2_concept": "a boy with blue hair"}
    _, routed, info = generate("freefuse", two, {freefuse: subjects})
    if routed is not None and unrouted is not None:
        to_unrouted, _ = diff(routed, unrouted)
        summary = parse_infotext(info).get("Anima FreeFuse", "").strip('"')  # quoted: it holds a comma
        report.check("FreeFuse infotext", summary.startswith("2 subjects") and "error:" not in info, summary or info.splitlines()[-1][:300])
        report.check("FreeFuse changes the image", to_unrouted > 0.5, f"mean diff to the unrouted LoRAs {to_unrouted:.2f}")
        check_round_trip(server, report, "FreeFuse", "txt2img", routed, info)

    # Sparse Attention: runs (infotext) or says why it did not (job warning)
    job, sparse_image, info = generate("sparse_attention", FEATURE_PROMPT, {sparse: {"enable": True, "min_tokens": 256}})
    if sparse_image is not None:
        notes = [json.dumps(note.get("values")) for note in job.get("warnings") or []]
        if "Sparse attention tau" in info:
            report.check("Sparse Attention ran (infotext)", True, info.splitlines()[-1][:300])
            check_round_trip(server, report, "Sparse Attention", "txt2img", sparse_image, info)
        else:
            unchanged, _ = diff(sparse_image, plain)
            skipped = any("Sparse Attention Integrated was skipped" in note for note in notes)
            report.check("Sparse Attention reports why it did not run", skipped and unchanged == 0, f"warnings {notes}; mean diff to without it {unchanged:.3f}")

    # Paste round-trips of the other panels that rewrite the run
    _, anchored, info = generate("prompt_anchor", "masterpiece, 1girl", {anchor: {"enabled": True, "anchor": "red hair"}})
    if anchored is not None:
        report.check("Prompt Anchor infotext", info.startswith("red hair, masterpiece, 1girl") and 'Anchor prompt: "red hair"' in info, info.splitlines()[0][:200])
        check_round_trip(server, report, "Prompt Anchor", "txt2img", anchored, info)
    _, vignetted, info = generate("pp_suite", FEATURE_PROMPT, {pp_suite: {"enable": True, "vignette_enabled": True, "vignette_amount": 0.8}})
    if vignetted is not None:
        report.check("PP Suite infotext", "PP Suite: " in info, info.splitlines()[-1][:300])
        check_round_trip(server, report, "PP Suite", "txt2img", vignetted, info)

    # Runs a feature changed the request of: the infotext records the result and
    # what it replaced, and the paste restores the request, so the pasted run
    # replaces it again (the same infotext, not only the same pixels)
    dynshift = script_owner(schema, "shift_start", "shift_end", "curve")
    guard = script_owner(schema, "policy", "base_mp", "hires_mp", "max_upscale")
    simple = {**sampler, "scheduler": "Simple"}
    hires = {"enable_hr": True, "hr_scale": 2.0, "hr_upscaler": "Lanczos", "hr_second_pass_steps": 4, "denoising_strength": 0.5}
    replacements = {
        "anima_flow_lock": ({}, {"sampler": {**simple, "sampler_name": "Anima Flow Euler"}}, "Anima Flow replaced scheduler: Simple"),
        "dynshift_schedule": ({}, {"sampler": simple, dynshift: {"enabled": True}}, "DynShift replaced scheduler: Simple"),
        "hires_guard_clamp": (hires, {"sampler": simple, guard: {"enabled": True}}, "Anima Hires adjustments: hires 1024x1024 -> 896x896"),
    }
    for name, (core_values, scripts, record) in replacements.items():
        print(f"feature: {name}", flush=True)
        job, image, info = run_v2(server, "txt2img", {**core(FEATURE_PROMPT), **core_values}, {**common_scripts, **scripts})
        if not report.check(f"{name}: job completed", image is not None, job.get("error") or ""):
            continue
        image.save(out / f"feature_{name}.png")
        report.check(f"{name}: infotext records the replacement", record in info, info.splitlines()[-1][:300])
        check_round_trip(server, report, name, "txt2img", image, info)

    # Refiner: the second checkpoint (by default the perturbed copy written into the data
    # folder, see main) takes over at 0.5 - the image changes, the infotext records it, the
    # base checkpoint is selected again afterwards, and the paste regenerates it
    from anima_test_loras import REFINER_NAME

    models = [model["title"] for model in server.request("GET", "/sdapi/v1/sd-models")]
    other = next((title for title in models if REFINER_NAME in title), None) or next((title for title in models if not title.startswith(MODEL)), None)
    if other is None:
        report.check("Refiner (needs a second checkpoint)", False, "no second checkpoint installed")
    else:
        refiner = script_owner(schema, "checkpoint", "switch_at")
        name = other.split(" [")[0]  # a Checkpoint choice; the paste must map "Refiner: <short title>" back onto one
        before = server.request("GET", "/sdapi/v1/options")["sd_model_checkpoint"]
        _, refined, info = generate("refiner", FEATURE_PROMPT, {refiner: {"enable": True, "checkpoint": name, "switch_at": 0.5}})
        if refined is not None:
            effect, _ = diff(refined, plain)
            report.check("Refiner changes the image", effect > 0.5, f"{name}: mean diff to without the refiner {effect:.2f}")
            report.check("Refiner infotext", f"Refiner: {name} [" in info and "Refiner switch at: 0.5" in info, info.splitlines()[-1][:300])
            after = server.request("GET", "/sdapi/v1/options")["sd_model_checkpoint"]
            report.check("Refiner restores the base checkpoint", after == before, f"before {before}, after {after}")
            check_round_trip(server, report, "Refiner", "txt2img", refined, info)

    # Inpaint, "Masked content: latent noise" (5D Anima latents): finishes, keeps the unmasked half
    from PIL import Image

    print("feature: inpaint latent noise", flush=True)
    mask = Image.new("L", plain.size, 0)
    mask.paste(255, (plain.width // 2, 0, plain.width, plain.height))
    started = time.monotonic()
    request = {**common_scripts["sampler"], "prompt": FEATURE_PROMPT, "negative_prompt": base_values["negative_prompt"], "seed": base_values["seed"], "cfg_scale": base_values["cfg_scale"], "width": 512, "height": 512}
    answer = server.request("POST", "/sdapi/v1/img2img", {**request, "init_images": [b64png(plain)], "mask": b64png(mask), "denoising_strength": 1.0, "mask_blur": 0, "inpainting_fill": 2, "inpaint_full_res": False}, timeout=600)
    elapsed = time.monotonic() - started
    image = decode(answer["images"][0])
    image.save(out / "feature_inpaint_latent_noise.png")
    half = plain.width // 2
    kept, _ = diff(image.crop((0, 0, half - 16, plain.height)), plain.crop((0, 0, half - 16, plain.height)))
    changed, _ = diff(image.crop((half + 16, 0, plain.width, plain.height)), plain.crop((half + 16, 0, plain.width, plain.height)))
    info = json.loads(answer["info"])["infotexts"][0]
    report.check("inpaint latent noise", image.size == plain.size and kept < 1.0 and changed > 2.0 and "Masked content: latent noise" in info and "Inpaint area: Whole picture" in info, f"{elapsed:.1f} s, unmasked mean diff {kept:.3f}, masked {changed:.2f}")


# endregion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter for launch.py (default: this one)")
    parser.add_argument("--port", type=int, default=7879)
    parser.add_argument("--baseline", type=Path, default=FIXTURE, help="a recorded request/response pair (baseline_txt2img.json, baseline_1.png) or the committed fixture folder (default)")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "remi-smoke", help="images and the server log")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--keep-data", action="store_true", help="keep the temporary data folder")
    parser.add_argument("--features", action="store_true", help="also check the LoRA features, Sparse Attention and paste round-trips (step 6)")
    parser.add_argument("--no-refiner-checkpoint", action="store_true", help="with --features: do not write the perturbed second checkpoint (the Refiner check then needs one installed)")
    parser.add_argument("--report", type=Path, help="also write the result as JSON here (evidence of a Gradio-free start and generation)")
    options = parser.parse_args()

    for path in baseline_files(options.baseline):
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
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
    found = gradio_modules_in(options.python)
    report.check("server interpreter has no Gradio", not found, f"{options.python} can import {', '.join(found)}" if found else options.python)
    print(f"Starting the server on port {options.port} (log: {options.out / 'server.log'}, data: {data_dir})", flush=True)
    started = time.monotonic()
    extra_args = []
    if options.features:
        from anima_test_loras import REFINER_NAME, write_refiner_checkpoint, write_test_loras

        lora_dir = data_dir / "Lora"
        write_test_loras(ROOT / "models" / "Stable-diffusion" / MODEL, lora_dir)
        extra_args = ["--lora-dir", str(lora_dir)]
        if not options.no_refiner_checkpoint:
            print("Writing a second checkpoint for the Refiner (a perturbed copy)", flush=True)
            refiner_dir = data_dir / "Stable-diffusion"
            write_refiner_checkpoint(ROOT / "models" / "Stable-diffusion" / MODEL, refiner_dir / f"{REFINER_NAME}.safetensors")
            extra_args += ["--ckpt-dirs", str(refiner_dir)]
    server = Server(options.python, options.port, data_dir, options.out / "server.log", extra_args)
    started_ok = False
    try:
        started_ok = report.check("server starts", server.wait_ready(options.startup_timeout), f"{time.monotonic() - started:.1f} s")
        if not started_ok:
            print(Path(options.out / "server.log").read_text(errors="replace")[-4000:], file=sys.stderr)
        else:
            try:
                values = run(server, options.baseline, options.out, report)
                if options.features:
                    run_features(server, options.out, report, values)
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
    if options.report is not None:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report.document(options.python, found, options.features), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Report: {options.report}")
    if not started_ok:
        return 2
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
