"""End-to-end runs of the built-in extensions on a tiny Anima model.

Not a test module: ``tests/test_remi_extensions_e2e.py`` runs this file in a fresh
interpreter (the WebUI is initialised for real, which a test process must not be)::

    python tests/remi_extensions_e2e.py DATA_DIR OUT.json [--only dave,refiner]

It writes the tiny models (``tools/tiny_anima.write_models``: the checkpoint, a
second one for the Refiner, test LoRAs and a LoKr, a Control-LLLite, an ADetailer
detector) into ``DATA_DIR/models``, starts the WebUI on the CPU (``--cpu``) with
every built-in extension, as ``remi.server.launch.serve`` does, and serves the
real app (``remi.server.create_app``) with the job queue running jobs through
the real runner (``remi.jobs.run_job``). For each :data:`CASES` entry it
queues, through ``POST /api/v2/jobs``:

1. the request without the extension (its params at their defaults), and
2. the same request with the extension on,

and records for the second: the job's status, warnings and error, the scripts'
errors (``modules.errors.report``), the infotext, the pixels' difference to the
first run, the hooks of the extension's script that ran with the arguments they
got (named by the script's params, as the schema converted them), and the paste
round trip: the infotext through ``POST /api/v2/infotext/parse``, queued again,
must give the same pixels and infotext.

The cases are plain data, so the test reads them without the WebUI.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PROMPT = "a girl with red hair, blue sky, garden"
NEGATIVE = "lowres, watermark"
SIZE = 64
CORE = {"prompt": PROMPT, "negative_prompt": NEGATIVE, "width": SIZE, "height": SIZE, "cfg_scale": 4.0}
SCRIPTS = {"sampler": {"steps": 6, "sampler_name": "Euler", "scheduler": "Simple"}, "seed": {"seed": 1234}}
LORA_A, LORA_B = "remi-test-a", "remi-test-b"
LORA_NETWORK = "extensions-builtin/sd_forge_lora"
REFINER = "tiny-anima-refiner.safetensors"
CONTROL_MODEL = "tiny-anima-lllite"
DETECTOR = "tiny-face.pt"
"""the names ``tools/tiny_anima.write_models`` writes"""
HOOKS = ("run", "setup", "before_process", "after_model_load", "before_process_init_images", "process", "before_process_batch", "after_extra_networks_activate", "process_batch", "process_before_every_sampling", "post_sample", "on_mask_blend", "postprocess_batch", "postprocess_batch_list", "postprocess_image", "postprocess_maskoverlay", "postprocess_image_after_composite", "before_hr", "postprocess")
WITH_OBJECT = {"before_process_init_images", "post_sample", "on_mask_blend", "postprocess_batch_list", "postprocess_image", "postprocess_maskoverlay", "postprocess_image_after_composite", "postprocess"}
"""hooks called as ``hook(p, <object>, *script args)``; the others as ``hook(p, *script args)``"""


@dataclass(frozen=True)
class Case:
    """One extension (``owner``: its script id) switched on by ``params``."""

    name: str
    owner: str
    params: dict[str, Any] = field(default_factory=dict)
    tab: str = "txt2img"
    core: dict[str, Any] = field(default_factory=dict)
    """core values of both runs (on top of :data:`CORE`)"""
    scripts: dict[str, dict[str, Any]] = field(default_factory=dict)
    """other scripts' params of both runs (on top of :data:`SCRIPTS`)"""
    on_core: dict[str, Any] = field(default_factory=dict)
    """core values of the run with the extension only (an extension driven by prompt syntax, a selectable script)"""
    options: dict[str, Any] = field(default_factory=dict)
    """settings for both runs (``PATCH /api/v2/options``, restored afterwards)"""
    infotext: tuple[str, ...] = ()
    """keys the infotext has with the extension on, and not without"""
    contains: tuple[str, ...] = ()
    """texts the infotext has with the extension on (a key's value)"""
    covers: tuple[str, ...] = ()
    """other built-in extensions the run goes through (a ControlNet preprocessor or model type, the LoRA network)"""
    changes: bool = True
    """whether the image (its pixels or its size) differs from the run without the extension"""
    same_as: dict[str, Any] | None = None
    """core values of a run without the extension that must give the same pixels"""
    differs_from: dict[str, Any] | None = None
    """core values of a run without the extension that must give other pixels"""
    hooks: tuple[str, ...] = ()
    """hooks of the extension's script that must run, with ``params`` among their arguments"""
    received: dict[str, Any] = field(default_factory=dict)
    """the arguments the hooks get for params the schema converts (a choice's index)"""
    warnings: tuple[str, ...] = ()
    """texts the job's warnings must contain (no other warning is allowed)"""
    outputs: int = 1
    round_trip: bool = True
    """paste the infotext into a fresh form (``POST /api/v2/infotext/parse``) and queue it:
    the same pixels and infotext"""
    keep: tuple[str, ...] = ()
    """values the paste keeps from the form, as ``owner[.param[.field]]`` paths: inputs an
    infotext cannot carry (images)"""
    paste_drops: tuple[str, ...] = ()
    """infotext keys the pasted run does not write again (a record of what the run did, not a setting)"""
    note: str = ""


IMG2IMG = {"mode": "img2img", "init_img": "@source", "denoising_strength": 0.6}
INPAINT = {"mode": "Inpaint upload", "init_img": "@source", "mask_upload": "@mask", "denoising_strength": 0.8, "inpaint_full_res": "Whole picture"}
WITH_LORA = {"prompt": f"{PROMPT}, <lora:{LORA_A}:1>"}
WITH_LOKR = {"prompt": f"{PROMPT}, <lora:remi-test-lokr:1>"}
EXTRAS_OFF = {"upscale": {"upscale_enabled": False}, "degrid": {"degrid_enabled": False}}
TAB_CORE = {"txt2img": CORE, "img2img": {**IMG2IMG, **CORE}, "extras": {"mode": "Single Image", "image": "@grid"}}
GUIDANCE = "anima_guidance_corrections"
SAMPLING = ("process_before_every_sampling",)

CASES: tuple[Case, ...] = (
    # Anima model patches
    Case("dave", "anima_dave_diversity", {"enabled": True, "strength": 0.8}, infotext=("Anima DAVE enabled", "Anima DAVE strength"), hooks=SAMPLING),
    Case("cross_attention_boost", "anima_cross_attention_boost", {"enabled": True, "strength": 3.0}, infotext=("Anima xattn boost enabled", "Anima xattn boost strength"), hooks=SAMPLING),
    Case("regional", "anima_regional_conditioning", {"enable": True, "region1_prompt": "a red ball", "region2_prompt": "a blue cube"}, infotext=("Anima regional enabled", "Anima region 1 prompt"), hooks=("process", *SAMPLING)),
    Case("artist_mixer", "anima_artist_scheduled_mixer", {"enable": True, "base_rows": [{"artist": "@wlop"}, {"artist": "@guweiz"}]}, infotext=("Anima Artist Mixer", "Anima Artist Mixer base"), hooks=("process", *SAMPLING)),
    Case("negpip", "negpip", core={"prompt": PROMPT}, on_core={"prompt": f"{PROMPT}, (red hair:-1.0)"}, infotext=("NegPiP",), hooks=("process_batch",), note="driven by negative weights in the prompt"),
    Case("prompt_rescaler", "anima_prompt_schedule_rescaler", {"auto_enable": True, "auto_source": 35}, tab="img2img", core={"prompt": "[a girl:a cat:15], garden"}, scripts={"sampler": {"steps": 20, "sampler_name": "Euler", "scheduler": "Simple"}}, infotext=("Anima schedule rescale",), hooks=("before_process",), paste_drops=("Anima schedule rescale",), note="the infotext keeps the rewritten prompt; pasting it does not turn auto-rescale on again (that would rescale twice)"),
    # guidance and sampling
    Case("guidance_smc_cfg", GUIDANCE, {"enable": True, "guidance_mode": "SMC-CFG"}, infotext=("Anima guidance enabled", "Anima SMC alpha"), hooks=("process", *SAMPLING)),
    Case("guidance_cfg_zero", GUIDANCE, {"enable": True, "guidance_mode": "CFG-Zero*"}, infotext=("Anima guidance enabled", "Anima CFG-Zero zero-init steps"), hooks=("process", *SAMPLING)),
    Case("guidance_apg", GUIDANCE, {"enable": True, "guidance_mode": "APG"}, infotext=("Anima guidance enabled", "Anima APG eta"), hooks=("process", *SAMPLING)),
    Case("guidance_tcfg", GUIDANCE, {"enable": True, "guidance_mode": "TCFG"}, infotext=("Anima guidance enabled", "Anima CFG guidance mode"), hooks=("process", *SAMPLING)),
    Case("guidance_fdg", GUIDANCE, {"enable": True, "guidance_mode": "FDG (experimental)"}, infotext=("Anima guidance enabled", "Anima FDG detail guidance"), hooks=("process", *SAMPLING)),
    Case("guidance_nag", GUIDANCE, {"enable": True, "nag_enable": True}, infotext=("Anima guidance enabled", "Anima NAG scale"), hooks=("process", *SAMPLING)),
    Case("guidance_momentum", GUIDANCE, {"enable": True, "momentum_enable": True}, infotext=("Anima guidance enabled", "Anima Momentum strength"), hooks=("process", *SAMPLING)),
    Case("guidance_range", GUIDANCE, {"enable": True, "guidance_range_enable": True, "guidance_range_start": 0.3, "guidance_range_end": 0.7}, infotext=("Anima guidance enabled", "Anima CFG range start"), hooks=("process", *SAMPLING)),
    Case("guidance_skim", GUIDANCE, {"enable": True, "skim_enable": True}, infotext=("Anima guidance enabled", "Anima skim CFG"), hooks=("process", *SAMPLING)),
    Case("guidance_dcw", GUIDANCE, {"enable": True, "dcw_enable": True, "dcw_lambda": -0.05}, infotext=("Anima guidance enabled", "Anima DCW lambda"), hooks=("process", *SAMPLING)),
    Case("detail_daemon", "detail_daemon", {"enabled": True, "d1_active": True, "d1_amount": 0.5}, infotext=("Detail Daemon",), hooks=("process",)),
    Case("dynamic_shift", "dynamic_shift", {"enabled": True}, infotext=("DynShift enabled", "DynShift replaced scheduler"), hooks=("process",), note="its schedule replaces the Simple scheduler, as the infotext records"),
    Case("spectrum", "spectrum_integrated", {"enable": True, "warmup": 2, "window": 2, "tail_actual": 2}, scripts={"sampler": {"steps": 24, "sampler_name": "Euler", "scheduler": "Simple"}}, infotext=("spec_enabled", "spec_warmup_steps"), hooks=SAMPLING, note="forecasts need a few steps of history: 24 steps"),
    Case("sparse_attention", "sparse_attention_integrated", {"enable": True}, changes=False, infotext=("Sparse attention tau", "Sparse attention range", "Sparse attention min tokens", "Sparse attention extra tokens"), hooks=SAMPLING, warnings=("Sparse Attention Integrated used the reference implementation (CUDA device is required): slower than regular attention",), note="no Sol-Attn kernels on the CPU: the reference implementation runs and the job says it is slower; the tiny model's 32-wide heads are not Sol-Attn's 128 (and its tokens are under the threshold), so every call stays dense and the pixels match - the sparse result itself: extensions-builtin/sd_forge_sparse/tests"),
    Case("never_oom", "never_oom_integrated", {"unet_enabled": True, "vae_enabled": True}, changes=False, hooks=("setup",), note="memory handling only: the UNet is offloaded after use and the VAE decodes in tiles (one tile at 64 px)"),
    Case("mahiro", "mahiro", {"enable": True}, infotext=("MaHiRo",), hooks=("after_extra_networks_activate",)),
    Case("rescale_cfg", "rescalecfg", {"cfg": 0.7}, infotext=("Rescale CFG",), hooks=("after_extra_networks_activate",)),
    Case("comments", "comments", core={"prompt": PROMPT}, on_core={"prompt": f"{PROMPT}\n# a comment the model never sees"}, changes=False, hooks=("process",), note="the comment is removed before the prompt is encoded: the same pixels as without it"),
    # LoRA features (tiny test LoRAs of the tiny model)
    Case("lora_layer_weight", "anima_lora_layer_weight", {"lora_enabled": True, "lora_block_weights": "all=0"}, core=WITH_LORA, same_as={"prompt": f"{PROMPT}, <lora:{LORA_A}:0>"}, infotext=("Anima LoRA layer lora enabled", "Anima LoRA layer lora blocks"), hooks=("before_process_batch",), note="block weight 0 undoes the LoRA: the pixels of <lora:...:0>", covers=(LORA_NETWORK,)),
    Case("lokr_layer_weight", "anima_lora_layer_weight", {"lokr_enabled": True, "lokr_block_weights": "all=0"}, core=WITH_LOKR, same_as={"prompt": f"{PROMPT}, <lora:remi-test-lokr:0>"}, infotext=("Anima LoRA layer lokr enabled", "Anima LoRA layer lokr blocks"), hooks=("before_process_batch",), note="the LoKR tab: block weight 0 undoes the LoKr", covers=(LORA_NETWORK,)),
    Case("lora_stage_scheduler", "anima_lora_stage_scheduler", {"enabled": True}, core=WITH_LORA, infotext=("Anima LoRA stage enabled", "Anima LoRA stage base template"), hooks=("process", *SAMPLING), covers=(LORA_NETWORK,)),
    Case("freefuse", "anima_freefuse_multi_lora", {"enable": True, "subject1_selector": LORA_A, "subject1_concept": "a girl", "subject2_selector": LORA_B, "subject2_concept": "blue sky"}, core={"prompt": f"{PROMPT}, <lora:{LORA_A}:1>, <lora:{LORA_B}:1>"}, infotext=("Anima FreeFuse enabled", "Anima FreeFuse subject 1 LoRA"), hooks=("process", *SAMPLING), covers=(LORA_NETWORK,)),
    Case("lora_control", "lora_control_integrated", core=WITH_LORA, on_core={"prompt": f"{PROMPT}, <lora:{LORA_A}:[1@0,0@0.5]>"}, differs_from={"prompt": f"{PROMPT}, <lora:{LORA_A}:0>"}, options={"forge_unet_storage_dtype": "Automatic (fp16 LoRA)"}, hooks=("before_process",), note="a weight schedule in the prompt; it needs on-the-fly LoRA (a setting); the image is neither the LoRA at 1 nor at 0", covers=(LORA_NETWORK,)),
    # prompt and image extensions
    Case("prompt_anchor", "prompt_anchor", {"enabled": True, "anchor": "masterpiece, best quality"}, infotext=("Anchor prompt",), hooks=("before_process",)),
    Case("prompt_blacklist", "prompt_blacklist", {"blacklist": "red hair", "remove_dupes": True}, changes=False, note="acts only when its button is pressed (it rewrites the prompt in the form)"),
    Case("image_stitch", "imagestitch_integrated", {"enable": True, "references": ["@source"]}, options={"anima_do_reference": True}, hooks=("process",), keep=("imagestitch_integrated",), note="Anima uses references with the [Anima] Enable Reference setting; the reference images are inputs, not infotext"),
    Case("post_processing_suite", "post_processing_suite", {"enable": True, "vignette_enabled": True, "vignette_amount": 0.8}, infotext=("PP Suite",), hooks=("postprocess_image_after_composite",)),
    Case("hires_guard", "anima_hires_guard", {"enabled": True, "preset": "Custom"}, core={"enable_hr": True, "hr_scale": 2.0, "hr_second_pass_steps": 2, "denoising_strength": 0.5}, infotext=("Anima Hires guard enabled", "Anima Hires adjustments"), hooks=("after_model_load",), note="the 2x Hires request is above the 1.75x limit: clamped onto the 64 px grid"),
    Case("random_resolution", "anima_random_resolution", {"randomize": True}, scripts={"sampler": {"steps": 2, "sampler_name": "Euler", "scheduler": "Simple"}}, hooks=("before_process",), note="a random preset size; the infotext's Size reproduces it"),
    Case("refiner", "refiner", {"enable": True, "checkpoint": REFINER, "switch_at": 0.5}, infotext=("Refiner", "Refiner switch at"), hooks=("setup",), note="a second tiny checkpoint takes over at half the steps"),
    # extensions with their own models (tiny ones: a random YOLOv8n, a Control-LLLite of the tiny DiT)
    Case("adetailer", "adetailer", {"ad_enable": True, "ad_model": DETECTOR, "ad_confidence": 0.0, "ad_mask_k": 1, "ad_denoising_strength": 0.6, "ad_use_inpaint_width_height": True, "ad_inpaint_width": 64, "ad_inpaint_height": 64}, infotext=("ADetailer model", "ADetailer confidence"), hooks=("postprocess_image",), note="the detector's best box is inpainted again"),
    Case("controlnet", "controlnet", {"unit_0": {"enabled": True, "module": "None", "model": CONTROL_MODEL, "image": "@source"}}, outputs=2, infotext=("ControlNet 0",), contains=(f"Model: {CONTROL_MODEL}",), covers=("extensions-builtin/sd_forge_controlllite",), hooks=("process", *SAMPLING), note="the second output is the control image; a paste keeps the unit's image and Enable, as the old paste did"),
    Case("controlnet_canny", "controlnet", {"unit_0": {"enabled": True, "module": "canny", "model": CONTROL_MODEL, "image": "@source"}}, outputs=2, infotext=("ControlNet 0",), contains=("Module: canny",), covers=("extensions-builtin/sd_forge_controlllite", "extensions-builtin/forge_legacy_preprocessors"), hooks=("process", *SAMPLING), note="the canny preprocessor's map is the second output"),
    Case("controlnet_tile", "controlnet", {"unit_0": {"enabled": True, "module": "tile_colorfix", "model": CONTROL_MODEL, "image": "@source"}}, outputs=2, infotext=("ControlNet 0",), contains=("Module: tile_colorfix",), covers=("extensions-builtin/sd_forge_controlllite", "extensions-builtin/forge_preprocessor_tile"), hooks=("process", *SAMPLING), note="forge_preprocessor_tile's tile_colorfix (it also steers the colours from the control image's latent)"),
    Case("controlnet_inpaint", "controlnet", {"unit_0": {"enabled": True, "module": "inpaint_only", "model": CONTROL_MODEL}}, tab="img2img", core=INPAINT, outputs=3, infotext=("ControlNet 0",), contains=("Module: inpaint_only",), covers=("extensions-builtin/sd_forge_controlllite", "extensions-builtin/forge_preprocessor_inpaint"), hooks=("process", *SAMPLING), note="forge_preprocessor_inpaint's inpaint_only on the img2img image and mask (it keeps the unmasked latents); the control image and the mask are the other outputs"),
    # img2img only
    Case("soft_inpainting", "soft_inpainting", {"enabled": True}, tab="img2img", core=INPAINT, infotext=("Soft inpainting enabled",), hooks=("process", "on_mask_blend")),
    Case("multidiffusion", "multidiffusion_integrated", {"enabled": True, "tile_width": 256, "tile_height": 256, "tile_overlap": 64}, tab="img2img", core={"width": 384, "height": 384}, infotext=("multidiffusion_enabled", "multidiffusion_tile_width"), hooks=SAMPLING),
    # selectable scripts (scripts/)
    Case("prompt_matrix", "prompt_matrix", {}, on_core={"script": "Prompt Matrix", "prompt": "a girl|red hair|blue sky"}, outputs=5, hooks=("run",), round_trip=False, note="the grid and its 4 images"),
    Case("prompts_from_file", "prompts_from_file_or_textbox", {"prompt_txt": "a cat\na dog"}, on_core={"script": "Prompts from File or Textbox"}, outputs=2, hooks=("run",), round_trip=False),
    Case("xyz_plot", "x_y_z_plot", {"x_type": "Seed", "x_values": "1,2"}, on_core={"script": "X/Y/Z plot"}, infotext=("Script", "X Type"), hooks=("run",), received={"x_type": 1}, round_trip=False, note="the grid (lone images are off); run() gets the axis type's index, as the Gradio dropdown gave"),
    Case("loopback", "loopback", {"loops": 2}, tab="img2img", on_core={"script": "Loopback"}, outputs=3, infotext=("Final Denoising Strength",), hooks=("run",), round_trip=False, note="the grid and its 2 loops"),
    Case("sd_upscale", "sd_upscale", {"upscaler_index": "Lanczos", "scale_factor": 2.0, "overlap": 16}, tab="img2img", on_core={"script": "SD Upscale"}, infotext=("SD Upscale - Upscaler",), hooks=("run",), received={"upscaler_index": 1}, round_trip=False, note="run() gets the upscaler's index, as the Gradio radio gave"),
    # Extras operations (scripts/)
    Case("extras_upscale", "upscale", {"upscale_enabled": True, "upscaler_1_name": "Lanczos", "upscale_by": 2.0}, tab="extras", scripts=EXTRAS_OFF, infotext=("Postprocess upscale by", "Postprocess upscaler"), round_trip=False),
    Case("extras_degrid", "degrid", {"degrid_enabled": True}, tab="extras", scripts=EXTRAS_OFF, round_trip=False, note="removes the 8 px grid of the source"),
)


# region Driver (runs inside the WebUI)


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _source_image():
    """A deterministic 64x64 picture for img2img (a gradient with a disc)."""

    import numpy as np
    from PIL import Image, ImageDraw

    y, x = np.mgrid[0:SIZE, 0:SIZE]
    array = np.stack([x * 4, y * 4, (x + y) * 2], axis=-1).clip(0, 255).astype("uint8")
    image = Image.fromarray(array, "RGB")
    ImageDraw.Draw(image).ellipse((16, 16, 48, 48), fill=(240, 220, 200))
    return image


def _grid_image():
    """The source with an 8 px grid of faint lines (what DeGrid removes)."""

    import numpy as np
    from PIL import Image

    array = np.asarray(_source_image(), dtype="int16").copy()
    array[::8, :, :] += 12
    array[:, ::8, :] += 12
    return Image.fromarray(array.clip(0, 255).astype("uint8"), "RGB")


def _mask_image():
    """The right half white (inpainted), the left half black (kept)."""

    from PIL import Image

    mask = Image.new("L", (SIZE, SIZE), 0)
    mask.paste(255, (SIZE // 2, 0, SIZE, SIZE))
    return mask


class Recorder:
    """The hooks each script ran in the current job, with their named arguments."""

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []

    def reset(self) -> None:
        self.calls = {}
        self.errors = []

    def instrument(self, runner, names: dict[str, list[str]]) -> None:
        for script in runner.scripts:
            owner = getattr(script, "remi_id", None) or getattr(script, "script_id", None)
            if owner is None:
                continue
            for hook in HOOKS:
                function = getattr(script, hook, None)
                if function is None or getattr(function, "remi_e2e", False):
                    continue
                setattr(script, hook, self._wrap(owner, hook, function, names.get(owner, [])))

    def _wrap(self, owner: str, hook: str, function, names: list[str]):
        def wrapper(p, *args, **kwargs):
            values = args[1:] if hook in WITH_OBJECT else args
            named = dict(zip(names, values, strict=False)) if len(values) == len(names) else {"*": [repr(value)[:80] for value in values]}
            self.calls.setdefault(owner, {}).setdefault(hook, {key: _jsonable(value) for key, value in named.items()})
            return function(p, *args, **kwargs)

        wrapper.remi_e2e = True
        return wrapper


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)[:200]


def _start(data_dir: Path):
    sys.argv = ["launch.py", "--skip-prepare-environment", "--skip-version-check", "--skip-torch-cuda-test", "--cpu", "--data-dir", str(data_dir)]
    sys.path[:0] = [str(ROOT), str(ROOT / "modules_forge" / "packages"), str(ROOT / "tools")]
    import tiny_anima

    checkpoint = tiny_anima.write_models(data_dir / "models")["checkpoint"]
    # MaHiRo and Rescale CFG are shown only with these settings (read when the runners are built)
    config = data_dir / "config.json"
    if not config.is_file():
        config.write_text(json.dumps({"show_mahiro": True, "show_rescale_cfg": True}), encoding="utf-8")

    import webui  # noqa: F401 - initialises the WebUI

    tiny_anima.install()
    from modules import script_callbacks
    from remi.server import runtime

    # as remi.server.launch.serve starts the app
    script_callbacks.before_ui_callback()
    runtime.prepare()
    return checkpoint


def run(data_dir: Path, out: Path, only: set[str] | None = None) -> dict:
    started = time.monotonic()
    checkpoint = _start(data_dir)

    from fastapi.testclient import TestClient
    from PIL import Image

    from modules import errors, extension_route_guard, script_callbacks, scripts, shared
    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth

    recorder = Recorder()
    report = errors.report

    def recording_report(message: str, *, exc_info: bool = False) -> None:
        recorder.errors.append(message)
        report(message, exc_info=exc_info)

    errors.report = recording_report
    services = Services.create(data_dir / "remi", dispatch=run_inline)
    app = create_app(services=services, auth=Auth(), sdapi=True, start_services=True)
    routes_before = extension_route_guard.snapshot_routes(app)
    script_callbacks.app_started_callback(None, app)
    extension_route_guard.guard_new_routes(app, routes_before)
    results: dict[str, Any] = {"checkpoint": checkpoint.name, "selected": shared.opts.sd_model_checkpoint, "cases": {}, "schema": {}}
    with TestClient(app, base_url="http://127.0.0.1:7860") as client:

        def call(method: str, path: str, **kwargs):
            response = client.request(method, path, **kwargs)
            if response.status_code >= 400:
                raise RuntimeError(f"{method} {path}: {response.status_code} {response.text[:500]}")
            return response.json() if "json" in response.headers.get("content-type", "") else response.content

        # the checkpoint, not the Refiner's (the first in the list)
        call("PATCH", "/api/v2/options", json={"sd_model_checkpoint": checkpoint.name})
        results["selected"] = shared.opts.sd_model_checkpoint
        schemas = {tab: call("GET", f"/api/v2/schema/{tab}") for tab in ("txt2img", "img2img", "extras")}
        names = {tab: {entry["id"]: list(entry.get("args") or []) for entry in schema["scripts"]} for tab, schema in schemas.items()}
        defaults = {tab: {entry["id"]: entry.get("defaults") or {} for entry in schema["scripts"]} for tab, schema in schemas.items()}

        def complete(tab: str, owner: str, values: dict) -> dict:
            """``values`` with object params (a ControlNet unit) filled from their defaults."""

            own = defaults[tab].get(owner, {})
            return {key: {**own[key], **value} if isinstance(value, dict) and isinstance(own.get(key), dict) else value for key, value in values.items()}

        results["owners"] = _owners(scripts)
        recorder.instrument(scripts.scripts_txt2img, names["txt2img"])
        recorder.instrument(scripts.scripts_img2img, names["img2img"])
        uploads = {
            "@source": call("POST", "/api/v2/uploads", files={"file": ("source.png", _png(_source_image()), "image/png")})["id"],
            "@mask": call("POST", "/api/v2/uploads", files={"file": ("mask.png", _png(_mask_image()), "image/png")})["id"],
            "@grid": call("POST", "/api/v2/uploads", files={"file": ("grid.png", _png(_grid_image()), "image/png")})["id"],
        }

        def resolve(values: Any) -> Any:
            if isinstance(values, dict):
                return {key: resolve(value) for key, value in values.items()}
            if isinstance(values, list):
                return [resolve(value) for value in values]
            return uploads.get(values, values) if isinstance(values, str) else values

        def queue(tab: str, core: dict, script_values: dict) -> dict:
            recorder.reset()
            job_id = call("POST", "/api/v2/jobs", json={"tab": tab, "core": resolve(core), "scripts": resolve(script_values)})["job_id"]
            deadline = time.monotonic() + 300
            while True:
                job = call("GET", f"/api/v2/jobs/{job_id}")
                if job["status"] not in ("queued", "running") or time.monotonic() > deadline:
                    break
                time.sleep(0.05)
            image = None
            if job.get("outputs"):
                image = Image.open(io.BytesIO(call("GET", f"/api/v2/files/{job['outputs'][0]['id']}"))).convert("RGB")
            infotext = (job["outputs"][0].get("infotext") if job.get("outputs") else None) or job.get("infotext") or ""
            return {"job": job, "image": image, "infotext": infotext, "calls": recorder.calls, "script_errors": list(recorder.errors)}

        baselines: dict[str, dict] = {}
        for case in CASES:
            if only and case.name not in only:
                continue
            saved = {key: shared.opts.data.get(key, shared.opts.get_default(key)) for key in case.options}
            if case.options:
                call("PATCH", "/api/v2/options", json=case.options)
            try:
                core = {**TAB_CORE[case.tab], **case.core}
                others = {**(SCRIPTS if case.tab in ("txt2img", "img2img") else {}), **case.scripts}
                key = json.dumps([case.tab, core, others, case.options], sort_keys=True)
                if key not in baselines:
                    baselines[key] = queue(case.tab, core, others)
                off = baselines[key]
                on = queue(case.tab, {**core, **case.on_core}, {**others, case.owner: complete(case.tab, case.owner, {**others.get(case.owner, {}), **case.params})})
                entry = _summarize(case, off, on)
                if case.same_as is not None:
                    reference = queue(case.tab, {**core, **case.same_as}, others)
                    entry["same_as_difference"] = _difference(reference["image"], on["image"])
                if case.differs_from is not None:
                    reference = queue(case.tab, {**core, **case.differs_from}, others)
                    entry["differs_from_difference"] = _difference(reference["image"], on["image"])
                if case.round_trip and on["image"] is not None:
                    form = {**others, case.owner: complete(case.tab, case.owner, {**others.get(case.owner, {}), **case.params})}
                    entry["round_trip"] = _round_trip(call, queue, case, {**core, **case.on_core}, form, on, complete)
            finally:
                if saved:
                    call("PATCH", "/api/v2/options", json=saved)
            results["cases"][case.name] = entry
    results["seconds"] = round(time.monotonic() - started, 1)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    return results


def _owners(scripts) -> dict[str, dict[str, str]]:
    """``{tab: {owner: source file relative to the repo}}`` of every script and Extras operation."""

    from remi.core.extras import assign_args

    def relative(filename: str) -> str:
        try:
            return Path(filename).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(filename)

    owners = {tab: {script.script_id: relative(script.filename) for script in runner.scripts if getattr(script, "script_id", None)} for tab, runner in (("txt2img", scripts.scripts_txt2img), ("img2img", scripts.scripts_img2img))}
    owners["extras"] = {layout.owner: relative(getattr(operation, "filename", "") or type(operation).__module__) for operation, layout in assign_args(scripts.scripts_postproc)}
    return owners


def _warning_text(warning: Any) -> str:
    """A job warning ``{"message", "values"}`` with its ``{{name}}`` placeholders filled."""

    if not isinstance(warning, dict):
        return str(warning)
    text = str(warning.get("message") or "")
    for key, value in (warning.get("values") or {}).items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def _difference(first, second) -> float | None:
    import numpy as np

    if first is None or second is None or first.size != second.size:
        return None
    return float(np.abs(np.asarray(first, dtype="float32") - np.asarray(second, dtype="float32")).mean())


def _digest(image) -> str | None:
    return None if image is None else hashlib.sha256(image.tobytes()).hexdigest()[:16]


def _summarize(case: Case, off: dict, on: dict) -> dict:
    job = on["job"]
    return {
        "status": job["status"],
        "error": job.get("error"),
        "warnings": [_warning_text(warning) for warning in job.get("warnings") or []],
        "script_errors": on["script_errors"],
        "infotext": on["infotext"],
        "infotext_off": off["infotext"],
        "status_off": off["job"]["status"],
        "difference": _difference(off["image"], on["image"]),
        "size": list(on["image"].size) if on["image"] is not None else None,
        "size_off": list(off["image"].size) if off["image"] is not None else None,
        "digest": _digest(on["image"]),
        "outputs": len(job.get("outputs") or []),
        "new_keys": sorted(set(_infotext_keys(on["infotext"])) - set(_infotext_keys(off["infotext"]))),
        "calls": on["calls"].get(case.owner, {}),
        "owners_called": sorted(on["calls"]),
        "image_png": base64.b64encode(_png(on["image"])).decode("ascii") if on["image"] is not None else None,
    }


def _infotext_keys(text: str) -> list[str]:
    from modules.infotext_utils import parse_generation_parameters

    return list(parse_generation_parameters(text)) if text else []


PASTE_MARKER = "__paste__"
IMAGE_INPUTS = ("mode", "init_img", "mask_upload", "inpaint_full_res", "resize_mode")
"""img2img inputs an infotext does not carry (the images, the mode)"""


def _round_trip(call, queue, case: Case, core: dict, form: dict, on: dict, complete) -> dict:
    """The infotext pasted into a fresh form (``POST /api/v2/infotext/parse``) and
    queued again, with the images of the first run (``IMAGE_INPUTS``, ``case.keep``
    taken from ``form``)."""

    parsed = call("POST", "/api/v2/infotext/parse", json={"text": on["infotext"], "tab": case.tab})
    pasted = {**{key: core[key] for key in IMAGE_INPUTS if key in core}, **(parsed.get("core") or {})}
    scripts = {}
    for owner, values in (parsed.get("scripts") or {}).items():
        # a value marked "__paste__" (a ControlNet unit) is merged into the form's, as the SPA
        # does (webui/src/widgets/controlnet-unit/paste.ts): images and "Enable" are kept
        merged = {name: {**form.get(owner, {}).get(name, {}), **{key: field for key, field in value.items() if key != PASTE_MARKER}} if isinstance(value, dict) and value.get(PASTE_MARKER) else value for name, value in values.items()}
        scripts[owner] = complete(case.tab, owner, merged)
    for path in case.keep:
        owner, *rest = path.split(".")
        if not rest:
            scripts[owner] = form[owner]
            continue
        target = scripts.setdefault(owner, {})
        source = form[owner]
        for name in rest[:-1]:
            source = source[name]
            target = target.setdefault(name, {})
        target[rest[-1]] = source[rest[-1]]
    again = queue(case.tab, pasted, scripts)
    expected = _without_keys(on["infotext"], case.paste_drops)
    return {
        "problems": parsed.get("problems") or [],
        "status": again["job"]["status"],
        "error": again["job"].get("error"),
        "same_pixels": _digest(again["image"]) == _digest(on["image"]),
        "difference": _difference(on["image"], again["image"]),
        "same_infotext": again["infotext"] == expected,
        "infotext": again["infotext"],
        "pasted": _jsonable((parsed.get("scripts") or {}).get(case.owner)),
    }


def _without_keys(infotext: str, keys: tuple[str, ...]) -> str:
    """``infotext`` without the ``Key: value`` entries of ``keys`` (simple values)."""

    for key in keys:
        infotext = re.sub(rf", {re.escape(key)}: [^,\n]*", "", infotext)
    return infotext


# endregion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", default="", help="comma-separated case names")
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args.data_dir.resolve(), args.out.resolve(), {name for name in args.only.split(",") if name} or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

