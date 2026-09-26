from __future__ import annotations

import importlib.util
import logging
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

sys.argv = sys.argv[:1]
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "pid.py"


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load():
    # These need the full WebUI install (k_diffusion, huggingface_guess);
    # stub only what the script imports, and only while it is imported.
    stubs = {
        "modules.processing": _module(
            "modules.processing",
            Processed=object,
            StableDiffusionProcessing=object,
            StableDiffusionProcessingImg2Img=object,
            apply_color_correction=None,
            decode_latent_batch=None,
            logger=logging.getLogger("pid-test"),
            process_images=None,
        ),
        "modules.sd_models": _module("modules.sd_models", checkpoint_tiles=lambda use_short=True: []),
        "modules.sd_vae": _module("modules.sd_vae", vae_dict={}),
        "modules.images": _module("modules.images", save_image=None),
        "modules_forge.main_entry": _module("modules_forge.main_entry", module_list={}),
    }
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("pid_script_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


pid = _load()


def _processed():
    return SimpleNamespace(images=["main"], infotexts=["main info"], extra_images=[])


def test_postprocess_moves_collected_results_into_processed_once():
    p = SimpleNamespace(_forge_pid_results=[("pid-0", "info-0"), ("pid-1", "info-1")])
    processed = _processed()
    pid.PiDForForge.postprocess(None, p, processed)
    assert processed.extra_images == ["pid-0", "pid-1"]
    assert processed.infotexts == ["main info", "info-0", "info-1"]
    assert not hasattr(p, "_forge_pid_results")

    pid.PiDForForge.postprocess(None, p, processed)  # nothing collected any more
    assert processed.extra_images == ["pid-0", "pid-1"]


def test_postprocess_without_pid_results_is_a_no_op():
    processed = _processed()
    pid.PiDForForge.postprocess(None, SimpleNamespace(), processed)
    assert processed.extra_images == [] and processed.infotexts == ["main info"]


class _Img2Img:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.main_prompt = kwargs["prompt"]
        self.closed = False

    def close(self):
        self.closed = True


def test_every_batch_sample_inner_result_reaches_postprocess(monkeypatch):
    runs = []

    def process_images(i2i):
        runs.append(i2i)
        return SimpleNamespace(images=[f"image-{i2i.seed}"], infotexts=[f"info-{i2i.seed}"])

    monkeypatch.setattr(pid, "StableDiffusionProcessingImg2Img", _Img2Img)
    monkeypatch.setattr(pid, "process_images", process_images)
    monkeypatch.setattr(pid.images, "save_image", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(pid, "opts", SimpleNamespace(samples_format="png"))

    p = SimpleNamespace(
        outpath_samples="out",
        outpath_grids="grids",
        all_prompts=["a", "b", "c", "d"],
        all_seeds=[1, 2, 3, 4],
        all_subseeds=[5, 6, 7, 8],
        subseed_strength=0.0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        width=64,
        height=64,
        batch_size=2,
    )
    for iteration in range(2):  # two batches of one generation
        p.iteration = iteration
        for b in range(2):
            pid.PiDForForge._sample_inner(p, b, None, 0.0, "", {}, {}, None)

    assert [run.seed for run in runs] == [1, 2, 3, 4]
    assert all(run.closed for run in runs)
    processed = _processed()
    pid.PiDForForge.postprocess(None, p, processed)
    assert processed.extra_images == ["image-1", "image-2", "image-3", "image-4"]
    assert processed.infotexts == ["main info", "info-1", "info-2", "info-3", "info-4"]
