"""The checks tools/smoke_real.py --features / --models runs against the real server:
their helpers, and the installed model files they rely on (skipped when not installed)."""

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "tools" / "smoke_real.py"
_spec = importlib.util.spec_from_file_location("smoke_real", _TOOL)
smoke = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("smoke_real", smoke)  # its dataclasses look their module up
_spec.loader.exec_module(smoke)

BASELINE = ROOT / "tests" / "fixtures" / "parity" / "anima_turbo_txt2img_seed12345.png"


def test_pick_upscaler_takes_an_installed_file_never_a_download(tmp_path):
    folder = tmp_path / "ESRGAN"
    folder.mkdir()
    for name in ("4x_plain.pth", "RealESRGAN_x4plus_anime_6B.pth"):
        (folder / name).write_bytes(b"x")
    elsewhere = tmp_path / "other.pth"
    elsewhere.write_bytes(b"x")
    upscalers = [
        {"name": "None", "model_path": None},
        {"name": "Lanczos", "model_path": None},
        {"name": "R-ESRGAN 4x+ Anime6B", "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"},
        {"name": "missing anime", "model_path": str(folder / "gone_anime.pth")},
        {"name": "other anime", "model_path": str(elsewhere)},
        {"name": "4x_plain", "model_path": str(folder / "4x_plain.pth")},
        {"name": "RealESRGAN_x4plus_anime_6B", "model_path": str(folder / "RealESRGAN_x4plus_anime_6B.pth")},
    ]
    assert smoke.pick_upscaler(upscalers, folder)["name"] == "RealESRGAN_x4plus_anime_6B"
    assert smoke.pick_upscaler(upscalers[:-1], folder)["name"] == "4x_plain"
    assert smoke.pick_upscaler(upscalers[:4], folder) is None


def test_pick_model():
    assert smoke.pick_model(["None", "canny [1]", "anima-lllite-lineart-1 [abc]"], "lineart") == "anima-lllite-lineart-1 [abc]"
    assert smoke.pick_model(["None", "canny [1]"], "lineart") == "canny [1]"
    assert smoke.pick_model(["None"], "lineart") is None


def test_line_correlation_tells_an_image_that_follows_the_drawing():
    drawing = smoke.control_drawing((384, 384))
    assert drawing.tobytes() == smoke.control_drawing((384, 384)).tobytes()  # deterministic
    # An image with the drawing's lines (inverted colours, softened) and one with other lines
    follows = Image.eval(drawing, lambda value: 255 - value // 2)
    other = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(other).line((0, 384, 384, 0), fill="black", width=6)
    assert smoke.line_correlation(follows, drawing) > 0.5
    assert smoke.line_correlation(other, drawing) < 0.1
    assert smoke.line_correlation(Image.new("RGB", (384, 384), "gray"), drawing) == 0.0


def test_region_diffs_separates_the_box_from_the_rest():
    reference = Image.new("RGB", (200, 100), (10, 20, 30))
    image = reference.copy()
    image.paste((60, 20, 30), (50, 20, 90, 60))  # inside the box
    image.putpixel((52, 15), (255, 20, 30))  # within the margin
    inside, outside, peak = smoke.region_diffs(image, reference, (50, 20, 90, 60), margin=8)
    assert inside == 50 and outside == 0 and peak == 0
    image.putpixel((150, 90), (20, 20, 30))  # far away
    _, outside, peak = smoke.region_diffs(image, reference, (50, 20, 90, 60), margin=8)
    assert 0 < outside < 0.01 and peak == 10


def test_lokr_tab_values_are_the_layer_weight_params():
    assert smoke.lokr_tab("0-13=0") == {"lokr_enabled": True, "lokr_block_enabled": True, "lokr_block_weights": "0-13=0"}


@pytest.mark.skipif(not (ROOT / "models" / "adetailer" / smoke.ADETAILER_MODEL).is_file() or importlib.util.find_spec("ultralytics") is None, reason="needs models/adetailer/face_yolov8n.pt and ultralytics")
def test_the_installed_face_detector_finds_the_baseline_face():
    """--models runs ADetailer on the baseline request, whose first pass is the baseline image."""
    faces = smoke.detect_faces(ROOT / "models" / "adetailer" / smoke.ADETAILER_MODEL, Image.open(BASELINE))
    assert len(faces) == 1
    x1, y1, x2, y2 = faces[0]
    assert 0 <= x1 < x2 <= 768 and 0 <= y1 < y2 <= 768 and (x2 - x1) * (y2 - y1) < 0.5 * 768 * 768  # a face, not the whole image


def _esrgan_models():
    return sorted(path for path in (ROOT / "models" / "ESRGAN").glob("*") if path.suffix in {".pth", ".pt", ".safetensors"})


@pytest.mark.skipif(not _esrgan_models(), reason="no model in models/ESRGAN")
def test_the_installed_esrgan_model_loads_and_upscales():
    """The loader the WebUI's ESRGAN upscaler uses (spandrel) reads the installed model, and
    it upscales by its scale while keeping the content."""
    import numpy as np
    import spandrel
    import torch

    from backend.utils import load_torch_file

    path = _esrgan_models()[0]
    model = spandrel.ModelLoader(device=torch.device("cpu")).load_from_state_dict(load_torch_file(str(path), safe_load=True, device=torch.device("cpu"))).eval()
    crop = Image.open(BASELINE).convert("RGB").crop((300, 150, 364, 214))
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        result = model(tensor).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    upscaled = Image.fromarray((result * 255).round().astype(np.uint8))
    assert upscaled.size == (64 * model.scale, 64 * model.scale)
    mean, _ = smoke.diff(upscaled.resize(crop.size, Image.Resampling.LANCZOS), crop)
    assert mean < 8


class _FakeServer:
    """Answers the requests run_models makes the way the WebUI does, with images that pass
    (or, with ``broken``, fail) its checks; records the request bodies."""

    def __init__(self, folder: Path, broken: bool = False):
        self.folder = folder
        self.broken = broken
        self.bodies: dict[str, list] = {}
        self.baseline = Image.open(folder / "baseline" / "baseline_1.png").convert("RGB")
        self.files: dict[str, bytes] = {}

    def png(self, image) -> bytes:
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def request(self, method, path, body=None, timeout=600):
        import json

        from PIL import ImageFilter

        self.bodies.setdefault(f"{method} {path}", []).append(body)
        if path == "/sdapi/v1/upscalers":
            return [{"name": "Lanczos", "model_path": None}, {"name": "RealESRGAN_x4plus_anime_6B", "model_path": str(self.folder / "models" / "ESRGAN" / "RealESRGAN_x4plus_anime_6B.pth")}]
        if path == "/sdapi/v1/extra-single-image":
            size = (self.baseline.width * 2, self.baseline.height * 2)
            image = self.baseline.resize(size, Image.Resampling.LANCZOS)
            if not self.broken:
                image = image.filter(ImageFilter.UnsharpMask(1, 100, 0))
            return {"image": smoke.b64png(image), "html_info": f"<p>Postprocess upscale by: 2.0, Postprocess upscaler: {body['upscaler_1']}</p>"}
        if path == "/controlnet/model_list":
            return {"model_list": ["None", "anima-lllite-lineart-1 [4ac9d1f0]"]}
        if path == "/controlnet/module_list":
            return {"module_list": ["None", smoke.CONTROLNET_MODULE]}
        if path == "/sdapi/v1/txt2img":
            unit = body["alwayson_scripts"]["ControlNet"]["args"][0]
            drawing = smoke.decode(unit["image"]).convert("RGB")
            image = self.baseline if self.broken else Image.blend(self.baseline, drawing, 0.5)
            info = f'{body["prompt"]}\nSteps: 8, Seed: {body["seed"]}, ControlNet 0: "Module: {unit["module"]}, Model: {unit["model"]}, Weight: 1.0"'
            return {"images": [smoke.b64png(image), smoke.b64png(drawing)], "info": json.dumps({"infotexts": [info]})}
        if path == "/api/v2/schema/txt2img":
            return {"scripts": [{"id": "sampler", "args": ["sampler_name"]}, {"id": "adetailer", "args": ["ad_enable", "ad_model", "ad_confidence"]}]}
        if path == "/api/v2/jobs":
            image = self.baseline.copy()
            if not self.broken:
                image.paste((200, 120, 120), (84, 50, 166, 134))
            self.files["out"] = self.png(image)
            return {"job_id": "job"}
        if path == "/api/v2/jobs/job":
            info = "1girl\nSteps: 8, Seed: 12345, ADetailer model: face_yolov8n.pt, ADetailer Version: 25.3.0"
            return {"status": "completed", "outputs": [{"id": "out", "infotext": info}]}
        if path == "/api/v2/files/out":
            return self.files["out"]
        if path == "/api/v2/infotext/parse":
            return {"problems": [], "core": {"prompt": "1girl"}, "scripts": {"adetailer": {"ad_enable": True}}}
        raise AssertionError(f"unexpected {method} {path}")


def _small_baseline(folder: Path) -> Path:
    """The parity baseline at 256x256 as a recorded request/response pair (fast checks)."""
    import json

    params, infotext, _ = smoke.load_baseline(smoke.FIXTURE)
    folder.mkdir(parents=True)
    Image.open(BASELINE).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS).save(folder / "baseline_1.png")
    request = {**params, "width": 256, "height": 256}
    (folder / "baseline_txt2img.json").write_text(json.dumps({"parameters": request, "info": json.dumps({"infotexts": [infotext]})}))
    return folder


def _run_models(tmp_path, monkeypatch, broken=False):
    baseline = _small_baseline(tmp_path / "baseline")
    for folder, name in (("ESRGAN", "RealESRGAN_x4plus_anime_6B.pth"), ("adetailer", smoke.ADETAILER_MODEL)):
        (tmp_path / "models" / folder).mkdir(parents=True)
        (tmp_path / "models" / folder / name).write_bytes(b"x")
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke, "detect_faces", lambda model, image: [(80.0, 47.0, 170.0, 137.0)])
    server = _FakeServer(tmp_path, broken)
    report = smoke.Report()
    params, _, _ = smoke.load_baseline(baseline)
    (tmp_path / "out").mkdir()
    smoke.run_models(server, baseline, tmp_path / "out", report, params)
    return server, report, params


def test_run_models_requests_and_checks(tmp_path, monkeypatch):
    server, report, params = _run_models(tmp_path, monkeypatch)
    assert [name for name, ok, _ in report.rows if not ok] == []
    assert len(report.rows) == 14
    extras = server.bodies["POST /sdapi/v1/extra-single-image"][0]
    assert extras["upscaler_1"] == "RealESRGAN_x4plus_anime_6B" and extras["upscaling_resize"] == 2
    request = server.bodies["POST /sdapi/v1/txt2img"][0]
    assert {key: request[key] for key in params} == params  # the baseline request
    unit = request["alwayson_scripts"]["ControlNet"]["args"][0]
    assert (unit["enabled"], unit["module"], unit["model"]) == (True, smoke.CONTROLNET_MODULE, "anima-lllite-lineart-1 [4ac9d1f0]")
    job = server.bodies["POST /api/v2/jobs"][0]
    assert job["tab"] == "txt2img" and job["scripts"]["adetailer"] == {"ad_enable": True, "ad_model": smoke.ADETAILER_MODEL}
    assert job["scripts"]["seed"] == {"seed": params["seed"]} and job["core"]["width"] == params["width"]
    assert (tmp_path / "out" / "models_controlnet_map_1.png").is_file()


def test_run_models_fails_a_silent_fallback(tmp_path, monkeypatch):
    """A Lanczos resize instead of the model, ControlNet without effect, ADetailer without a pass."""
    _, report, _ = _run_models(tmp_path, monkeypatch, broken=True)
    failed = {name for name, ok, _ in report.rows if not ok}
    assert failed == {"Extras ESRGAN ran the model (not a plain resize)", "ControlNet changes the image", "ControlNet follows the control image", "ADetailer redoes the face"}


def test_run_models_reports_missing_models(tmp_path, monkeypatch):
    baseline = _small_baseline(tmp_path / "baseline")
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    server = _FakeServer(tmp_path)
    server.request = lambda method, path, body=None, timeout=600: {"/sdapi/v1/upscalers": [], "/controlnet/model_list": {"model_list": ["None"]}, "/controlnet/module_list": {"module_list": []}}[path]
    report = smoke.Report()
    params, _, _ = smoke.load_baseline(baseline)
    smoke.run_models(server, baseline, tmp_path, report, params)
    assert [(name, ok) for name, ok, _ in report.rows] == [
        ("Extras ESRGAN (needs a model in models/ESRGAN)", False),
        ("ControlNet (needs a model in models/ControlNet)", False),
        (f"ADetailer (needs models/adetailer/{smoke.ADETAILER_MODEL})", False),
    ]


def test_run_lokr_features_compares_the_tab_with_the_plain_tags(monkeypatch):
    """The LoKR tab runs: each check compares the right two images, and a tab that does
    nothing (the LoKr loaded at the tag's weight whatever the tab) fails them."""

    def image(shade):
        return Image.new("RGB", (64, 64), (shade, shade, shade))

    def run(tab_works):
        calls = []

        def generate(name, prompt, scripts=None):
            calls.append((name, prompt, scripts))
            weight = prompt.rsplit(":", 1)[1].rstrip(">")
            values = (scripts or {}).get("layer", {})
            # The tab's weight when it works, else the tag's
            by_tab = {"all=1.0": 100, "all=0": 0, "0-13=0": 50}
            shade = by_tab[values["lokr_block_weights"]] if values.get("lokr_enabled") and tab_works else round(float(weight) * 100)
            info = f"x <lora:remi-test-lokr:{weight}>\nSteps: 8" + (f", Anima LoRA layer lokr enabled: True, Anima LoRA layer lokr blocks: {values['lokr_block_weights']}" if values.get("lokr_enabled") else "")
            return {}, image(shade), info

        report = smoke.Report()
        monkeypatch.syspath_prepend(str(ROOT / "tools"))  # as when the script runs
        monkeypatch.setattr(smoke, "check_round_trip", lambda server, report, name, *args, **kwargs: report.check(f"{name}: paste regenerates it", True))
        smoke.run_lokr_features(None, report, generate, "layer", image(0))
        return calls, report

    calls, report = run(True)
    assert [name for name, ok, _ in report.rows if not ok] == []
    assert [name for name, _, _ in calls] == ["lokr", "lokr_weight_0", "lokr_layer_weight_1", "lokr_layer_weight_0", "lokr_layer_weight_blocks", "lokr_under_lora_tab"]
    assert calls[2][2] == {"layer": {"lokr_enabled": True, "lokr_block_enabled": True, "lokr_block_weights": "all=1.0"}}
    assert calls[5][2] == {"layer": {"lora_enabled": True, "lora_block_enabled": True, "lora_block_weights": "all=0"}}

    _, report = run(False)
    assert {name for name, ok, _ in report.rows if not ok} == {"LoKR tab all=1 == <lora:lokr:1>", "LoKR tab all=0 == <lora:lokr:0>", "LoKR tab block range changes part of the effect"}
