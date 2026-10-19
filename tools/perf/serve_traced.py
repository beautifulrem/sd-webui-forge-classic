"""Run ``launch.py`` with timing probes for the UI performance measurements.

    REMI_PERF_TRACE=/tmp/trace.jsonl python tools/perf/serve_traced.py --api --port 7861 ...

Arguments are passed to ``launch.py`` unchanged. The probes are installed by wrapping
functions when their modules are imported (no change to the server code) and append one
JSON object per line to ``$REMI_PERF_TRACE``; times are ``time.time()`` (the wall clock the
browser's ``Date.now()`` also reads on the same host):

- ``{"e": "step", "t", "step"}``: the sampler's step callback (``it/s``);
- ``{"e": "image", "t", "id"}``: a live preview image was assigned (``State.assign_current_image``);
- ``{"e": "frame", "t0", "t", "seq", "step", "id", "image_t", "bytes"}``: the preview encoder
  published a frame (``t0`` tick start, ``t`` when it was handed to the hubs, after encoding;
  ``image_t`` when its image was assigned);
- ``{"e": "job", "t", "phase"}``: ``State.begin`` / ``State.end``.

While the file ``$REMI_PERF_MUTE`` (default ``/tmp/remi-perf-mute``) exists, the WebSocket hub
acts as if no client were connected (no preview encoded, nothing sent).

``REMI_PERF_FAKE_SAMPLER=<blocks>`` replaces ``processing.process_images`` with a synthetic
sampler that loads no model: every step runs ``<blocks>`` DiT-like transformer blocks (1,024
tokens, width 2,048, fp16) on the GPU (MPS / CUDA, else CPU), then makes a live preview from the
"latent" as the real callback does (``.cpu()``, RGB, ``assign_current_image``). 28 blocks take
about 3 s per step on an M4 (Anima Turbo at 512×512: 3.1 s). It measures the UI's overhead on a
GPU-bound sampling loop without the model (``browser.mjs --only interleaved``); ``REMI_PERF_DIST``
serves another SPA build (to compare two frontends against the same server).
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_out = open(os.environ.get("REMI_PERF_TRACE", "/tmp/remi-perf-trace.jsonl"), "a", buffering=1)  # noqa: SIM115 - process lifetime
_images: dict[int, tuple[float, int]] = {}
"""assignment time and id of the live previews, by ``id_live_preview``"""


def emit(**record) -> None:
    line = json.dumps(record)
    with _lock:
        _out.write(line + "\n")


def _patch_shared_state(module) -> None:
    state_cls = module.State
    assign, begin, end = state_cls.assign_current_image, state_cls.begin, state_cls.end

    @functools.wraps(assign)
    def assign_current_image(self, image):
        # timed and keyed before the call: the encoder may publish the preview inside it (its
        # listener encodes small previews on this thread); assign increments id_live_preview
        now = time.time()
        _images.pop(self.id_live_preview + 1, None)  # the id restarts with every job: newest last
        _images[self.id_live_preview + 1] = (now, self.id_live_preview + 1)
        if len(_images) > 256:
            _images.pop(next(iter(_images)))
        assign(self, image)
        emit(e="image", t=now, id=self.id_live_preview)

    @functools.wraps(begin)
    def begin_(self, *args, **kwargs):
        emit(e="job", t=time.time(), phase="begin")
        return begin(self, *args, **kwargs)

    @functools.wraps(end)
    def end_(self, *args, **kwargs):
        emit(e="job", t=time.time(), phase="end")
        return end(self, *args, **kwargs)

    state_cls.assign_current_image, state_cls.begin, state_cls.end = assign_current_image, begin_, end_


def _patch_samplers(module) -> None:
    callback = module.Sampler.callback_state

    @functools.wraps(callback)
    def callback_state(self, d):
        emit(e="step", t=time.time(), step=int(d["i"]))
        return callback(self, d)

    module.Sampler.callback_state = callback_state


def _patch_realtime(module) -> None:
    tick, publish = module.PreviewEncoder.tick, module.EventBus.publish_frame
    local = threading.local()

    @functools.wraps(publish)
    def publish_frame(self, frame):
        local.frame, local.published = frame, time.time()
        return publish(self, frame)

    @functools.wraps(tick)
    def tick_(self, last):
        local.frame = None
        t0 = time.time()
        preview = getattr(self.state(), "id_live_preview", None)
        result = tick(self, last)
        frame = local.frame
        if frame is not None:
            image_t, image_id = _images.get(preview, (None, None))
            emit(e="frame", t0=t0, t=local.published, seq=frame.job_seq, step=frame.step, id=image_id, image_t=image_t, bytes=len(frame.data))
        return result

    module.PreviewEncoder.tick, module.EventBus.publish_frame = tick_, publish_frame


MUTE_FILE = os.environ.get("REMI_PERF_MUTE", "/tmp/remi-perf-mute")


def _patch_ws(module) -> None:
    """While ``$REMI_PERF_MUTE`` exists the hub behaves as if no client were connected: no
    preview is encoded and nothing is sent (the overhead measurement toggles it mid-job)."""

    hub = module.Hub
    wants, fan_event, fan_frame = hub.wants_frames, hub._fan_event, hub._fan_frame

    def muted() -> bool:
        return os.path.exists(MUTE_FILE)

    hub.wants_frames = lambda self: False if muted() else wants(self)
    hub._fan_event = lambda self, event: None if muted() else fan_event(self, event)
    hub._fan_frame = lambda self, frame: None if muted() else fan_frame(self, frame)


class _FakeSampler:
    """The synthetic sampling loop of ``REMI_PERF_FAKE_SAMPLER`` (weights shared by the blocks)."""

    TOKENS, WIDTH, MLP, HEADS = 1024, 2048, 8192, 16

    def __init__(self, blocks: int) -> None:
        import torch

        self.torch, self.blocks = torch, blocks
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        generator = torch.Generator(device="cpu").manual_seed(0)
        shapes = {"q": (self.WIDTH, self.WIDTH), "k": (self.WIDTH, self.WIDTH), "v": (self.WIDTH, self.WIDTH), "o": (self.WIDTH, self.WIDTH), "up": (self.WIDTH, self.MLP), "down": (self.MLP, self.WIDTH)}
        self.weights = {key: (torch.randn(*shape, generator=generator) * 0.02).to(self.device, dtype) for key, shape in shapes.items()}
        self.dtype = dtype

    def block(self, x):
        functional, w = self.torch.nn.functional, self.weights
        tokens, width, heads = self.TOKENS, self.WIDTH, self.HEADS
        h = functional.layer_norm(x, (width,))
        q, k, v = (h @ w[key] for key in ("q", "k", "v"))
        attention = functional.scaled_dot_product_attention(*(t.view(1, tokens, heads, width // heads).transpose(1, 2) for t in (q, k, v)))
        x = x + attention.transpose(1, 2).reshape(1, tokens, width) @ w["o"]
        h = functional.layer_norm(x, (width,))
        return x + functional.gelu(h @ w["up"]) @ w["down"]

    def __call__(self, p):
        import numpy as np
        from PIL import Image

        from modules import sd_models, shared

        torch, state = self.torch, shared.state
        # as after a real load: the selected checkpoint counts as loaded (no "model.loading" for the
        # whole job, which would keep the model bar's spinner turning)
        data = getattr(sd_models, "model_data", None)
        if getattr(data, "forge_loading_parameters", None):
            data.forge_hash = str(data.forge_loading_parameters)
        steps = max(1, int(getattr(p, "steps", 20) or 20))
        width, height = int(getattr(p, "width", 512) or 512), int(getattr(p, "height", 512) or 512)
        images = max(1, int(getattr(p, "n_iter", 1) or 1))  # batch count: images one after another
        state.job_count, state.job_no, state.sampling_steps = images, 0, steps
        side = max(8, min(width, height) // 8)  # a latent-sized preview, as the RGB approximation makes
        with torch.inference_mode():
            for job_no in range(images):
                if state.interrupted:
                    break
                state.job_no, state.sampling_step, state.skipped = job_no, 0, False
                x = torch.randn(1, self.TOKENS, self.WIDTH, device=self.device, dtype=self.dtype) * 0.1
                for step in range(steps):
                    if state.interrupted or state.skipped:
                        break
                    for _ in range(self.blocks):
                        x = self.block(x) * 0.5
                    # the k-diffusion callback: step counter, then the live preview (Sampler.callback_state)
                    emit(e="step", t=time.time(), step=step)
                    state.sampling_step = step
                    if shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps > 0:
                        grid = int(self.TOKENS**0.5)
                        pixels = x[0, : grid * grid, :3].float().reshape(grid, grid, 3)
                        pixels = ((pixels - pixels.amin()) / (pixels.amax() - pixels.amin() + 1e-6) * 255).to(torch.uint8).cpu().numpy()
                        state.assign_current_image(Image.fromarray(np.ascontiguousarray(pixels)).resize((side, side)))
                        state.current_image_sampling_step = step
        info = f"{getattr(p, 'prompt', '')}\nSteps: {steps}, Sampler: synthetic ({self.blocks} blocks), Seed: {getattr(p, 'seed', -1)}, Size: {width}x{height}"
        image = Image.new("RGB", (width, height), (96, 96, 128))
        from types import SimpleNamespace

        return SimpleNamespace(images=[image], infotexts=[info], info=info, index_of_first_image=0, extra_images=[], js=lambda: "{}")


def _patch_processing(module) -> None:
    blocks = os.environ.get("REMI_PERF_FAKE_SAMPLER")
    if not blocks:
        return
    sampler = None

    def process_images(p):
        nonlocal sampler
        if sampler is None:
            sampler = _FakeSampler(int(blocks))
        return sampler(p)

    module.process_images = process_images


def _patch_spa(module) -> None:
    dist = os.environ.get("REMI_PERF_DIST")
    if dist:
        module.DIST = Path(dist).resolve()


PATCHES = {
    "modules.shared_state": _patch_shared_state,
    "modules.sd_samplers_common": _patch_samplers,
    "modules.processing": _patch_processing,
    "remi.realtime": _patch_realtime,
    "remi.server.ws": _patch_ws,
    "remi.server.spa": _patch_spa,
}


class _PostImport(importlib.abc.MetaPathFinder):
    """Runs a patch right after its module is executed."""

    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            PATCHES[name](module)

        spec.loader.exec_module = exec_and_patch
        return spec


def main() -> None:
    sys.meta_path.insert(0, _PostImport())
    emit(e="process", t=time.time(), argv=sys.argv[1:])
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / "launch.py"), *sys.argv[1:]]
    runpy.run_path(str(ROOT / "launch.py"), run_name="__main__")


if __name__ == "__main__":
    main()
