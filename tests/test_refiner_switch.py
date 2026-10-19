"""The refiner switches checkpoints mid-sampling (``apply_refiner``, the
path without "fast" state-dict swapping), shown with two tiny synthetic
checkpoints that really differ: two safetensors files registered as
checkpoints, each holding the weight of a toy denoiser.

A sampling loop calls ``apply_refiner`` before every model call, as
``CFGDenoiser.forward`` does, and denoises with whatever model is loaded
(``shared.sd_model``). Only the loader is a stand-in: Forge's
``refresh_model_loading_parameters`` / ``forge_model_reload`` read the
selected checkpoint's file into a toy model instead of a diffusion engine.
Everything else (the switch point, the checkpoint change and its restore,
the infotext) is the WebUI's own code."""

from types import SimpleNamespace

import pytest
import torch

STEPS = 8


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


class _ToyModel:
    """A denoiser made from a checkpoint file: ``x * weight / (1 + sigma)``."""

    def __init__(self, info):
        from safetensors.torch import load_file

        self.sd_checkpoint_info = info
        self.weight = load_file(info.filename)["weight"]
        self.forge_objects = SimpleNamespace(unet=SimpleNamespace(model=None))

    def denoise(self, x, sigma):
        return x * self.weight / (1.0 + sigma)


@pytest.fixture
def checkpoints(webui, monkeypatch, tmp_path):
    """Two registered checkpoints with different weights, the first loaded."""
    from safetensors.torch import save_file

    from modules import hashes, sd_models, shared

    monkeypatch.setattr(sd_models, "checkpoints_list", {})
    monkeypatch.setattr(sd_models, "checkpoint_aliases", {})
    # no hash or metadata caches written for the test files
    monkeypatch.setattr(sd_models.cache, "cached_data_for_file", lambda subsection, title, filename, func: func())
    monkeypatch.setattr(hashes, "sha256_from_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(hashes, "sha256", lambda filename, title, use_addnet_hash=False: hashes.calculate_sha256_real(filename))
    monkeypatch.setattr(sd_models, "model_path", str(tmp_path))

    infos = []
    for name, weight in (("anima-base.safetensors", 0.5), ("anima-refiner.safetensors", 0.9)):
        save_file({"weight": torch.tensor(weight)}, str(tmp_path / name))
        info = sd_models.CheckpointInfo(str(tmp_path / name))
        info.register()
        infos.append(info)
    base, refiner = infos

    loaded = {}

    def refresh_model_loading_parameters(*, refresh=True):
        if refresh:
            loaded["selected"] = sd_models.select_checkpoint()

    def forge_model_reload():
        shared.sd_model = _ToyModel(loaded["selected"])
        return shared.sd_model, True

    from modules import sd_samplers_common

    monkeypatch.setitem(webui.opts.data, "sd_model_checkpoint", base.title)
    monkeypatch.setattr(shared, "sd_model", _ToyModel(base), raising=False)
    monkeypatch.setattr(sd_samplers_common.main_entry, "refresh_model_loading_parameters", refresh_model_loading_parameters)
    monkeypatch.setattr(sd_samplers_common.sd_models, "forge_model_reload", forge_model_reload)
    monkeypatch.setattr(sd_samplers_common, "sampling_cleanup", lambda unet: None)
    monkeypatch.setattr(sd_samplers_common, "sampling_prepare", lambda unet, x: None)
    monkeypatch.setattr(sd_samplers_common, "ORIGINAL_CHECKPOINT", None)
    # the module's own opts (other suites may swap shared.opts)
    monkeypatch.setattr(sd_samplers_common, "opts", webui.opts)
    monkeypatch.setitem(webui.opts.data, "refiner_use_steps", True)
    monkeypatch.setitem(webui.opts.data, "refiner_fast_sd", False)
    return base, refiner


class _Denoiser:
    """What ``apply_refiner`` reads and calls of ``CFGDenoiser``."""

    def __init__(self, p):
        self.p = p
        self.step = 0
        self.total_steps = STEPS
        self.model_wrap = None
        self.models: list[str] = []

    def update_inner_model(self):
        self.model_wrap = None

    def __call__(self, x, sigma):
        from modules import sd_samplers_common, shared

        sd_samplers_common.apply_refiner(self, x, sigma)
        self.models.append(shared.sd_model.sd_checkpoint_info.name_for_extra)
        return shared.sd_model.denoise(x, sigma)


def _sample(refiner_info, switch_at=0.5):
    """Euler sampling of a fixed latent; the processing object and the
    names of the checkpoint used at every step."""

    from modules import sd_samplers_common

    # a job starts by loading the selected checkpoint (processing's
    # manage_model_and_prompt_cache), so a refined job leaves the next one as is
    sd_samplers_common.main_entry.refresh_model_loading_parameters()
    sd_samplers_common.sd_models.forge_model_reload()
    conds = []
    p = SimpleNamespace(
        refiner_switch_at=switch_at if refiner_info is not None else None,
        refiner_checkpoint_info=refiner_info,
        enable_hr=False,
        disable_extra_networks=True,
        extra_generation_params={},
        setup_conds=lambda: conds.append(True),
    )
    denoiser = _Denoiser(p)
    sigmas = torch.linspace(1.0, 0.0, STEPS + 1)
    x = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)
    for i in range(STEPS):
        denoised = denoiser(x, sigmas[i])
        x = x + (x - denoised) / sigmas[i] * (sigmas[i + 1] - sigmas[i])
        denoiser.step += 1
    return x, p, denoiser.models, conds


def test_a_different_refiner_checkpoint_changes_the_image_from_the_switch_on(checkpoints):
    from modules import sd_samplers_common, shared

    base, refiner = checkpoints
    plain, _, plain_models, _ = _sample(None)
    refined, p, models, conds = _sample(refiner, switch_at=0.5)

    assert plain_models == ["anima-base"] * STEPS
    # the second half of the steps ran on the refiner checkpoint
    assert models == ["anima-base"] * 4 + ["anima-refiner"] * 4
    assert not torch.equal(refined, plain)
    # the conds were rebuilt and the run recorded the refiner (with its hash)
    assert conds == [True]
    assert refiner.shorthash and p.extra_generation_params == {"Refiner": f"anima-refiner [{refiner.shorthash}]", "Refiner switch at": 0.5}
    # the selected checkpoint is the base again (the refiner is loaded for this job only)
    assert shared.opts.sd_model_checkpoint == base.title
    assert sd_samplers_common.ORIGINAL_CHECKPOINT is None


def test_the_switch_point_decides_where_the_refiner_takes_over(checkpoints):
    _, refiner = checkpoints
    early = _sample(refiner, switch_at=0.25)
    late = _sample(refiner, switch_at=0.75)
    assert early[2] == ["anima-base"] * 2 + ["anima-refiner"] * 6
    assert late[2] == ["anima-base"] * 6 + ["anima-refiner"] * 2
    assert not torch.equal(early[0], late[0])


def test_the_loaded_checkpoint_as_refiner_is_a_no_op(checkpoints):
    """As apply_refiner always did: the same checkpoint has nothing to switch to."""
    base, _ = checkpoints
    plain = _sample(None)
    same = _sample(base)
    assert torch.equal(same[0], plain[0]) and same[2] == plain[2]
    assert same[1].extra_generation_params == {}
