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


def emit(**record) -> None:
    line = json.dumps(record)
    with _lock:
        _out.write(line + "\n")


def _patch_shared_state(module) -> None:
    state_cls = module.State
    assign, begin, end = state_cls.assign_current_image, state_cls.begin, state_cls.end

    @functools.wraps(assign)
    def assign_current_image(self, image):
        assign(self, image)
        now = time.time()
        _images[id(self.current_image)] = (now, self.id_live_preview)
        if len(_images) > 256:
            _images.pop(next(iter(_images)))
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
        image = getattr(self.state(), "current_image", None)
        result = tick(self, last)
        frame = local.frame
        if frame is not None:
            image_t, image_id = _images.get(id(image), (None, None))
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


PATCHES = {"modules.shared_state": _patch_shared_state, "modules.sd_samplers_common": _patch_samplers, "remi.realtime": _patch_realtime, "remi.server.ws": _patch_ws}


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
