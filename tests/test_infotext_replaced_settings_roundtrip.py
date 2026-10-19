"""Infotext round-trips of runs where a feature replaced what was asked:
an Anima Flow sampler's scheduler lock, Dynamic Shift's Automatic schedule
and the Anima Hires Guard's clamped sizes. The infotext records the result
plus the replaced request; pasting it (``POST /api/v2/infotext/parse``) must
give back the request, so the pasted run makes the same replacement and
writes the same infotext, byte for byte (pasting the result made the same
image without the record).

No model: each run goes through the API's processing object
(``prepare_txt2img``), the scripts' real hooks, ``p.init``, the sampler's
``get_sigmas`` and ``create_infotext``, with a stand-in Anima model."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DYNSHIFT = ROOT / "extensions-builtin" / "sd-forge-dynamic-shift" / "scripts" / "dynamic_shift.py"
HIRES_GUARD = ROOT / "extensions-builtin" / "sd_forge_anima_hires_guard" / "scripts" / "anima_hires_guard.py"

BASE = {"prompt": "1girl, silver hair", "negative_prompt": "worst quality", "width": 512, "height": 512, "cfg_scale": 1.0}
SAMPLER = {"steps": 8, "sampler_name": "Euler", "scheduler": "Simple"}
HIRES = {"enable_hr": True, "hr_scale": 2.0, "hr_upscaler": "Lanczos", "hr_second_pass_steps": 4, "denoising_strength": 0.5, "hr_cfg": 1.0}


@pytest.fixture(scope="module")
def webui():
    import remi_script_harness as harness

    from modules import script_callbacks
    from remi.testing import webui_state

    dynshift = harness.load_script(DYNSHIFT, "dynamic_shift_roundtrip_under_test")
    guard = harness.load_script(HIRES_GUARD, "anima_hires_guard_roundtrip_under_test")
    try:
        with webui_state({"add_version_to_infotext": False}, script_classes=[dynshift.DynamicShiftScript, guard.AnimaHiresGuardScript]) as state:
            yield state
    finally:
        for module in (dynshift, guard):
            script_callbacks.remove_callbacks_for_function(module._on_infotext_pasted)


@pytest.fixture(autouse=True)
def _anima(webui, monkeypatch):
    from modules import processing, shared

    monkeypatch.setattr(shared, "sd_upscalers", [SimpleNamespace(name="Lanczos")], raising=False)
    monkeypatch.setattr(shared, "sd_model", _Model(), raising=False)
    # the job's step count (a progress total) is already known
    state = SimpleNamespace(processing_has_refined_job_count=True, interrupted=False)
    monkeypatch.setattr(processing, "state", state)
    monkeypatch.setattr(shared, "state", state, raising=False)
    monkeypatch.setattr(processing.sd_samplers, "create_sampler", _stop_at_the_sampler)


class _Model:
    """What processing and the scripts read of an Anima model."""

    is_anima_engine = True
    use_shift = True
    use_distilled_cfg_scale = False
    is_sd1 = False
    tiling_enabled = False
    sd_model_hash = "fba1195327"
    sd_checkpoint_info = None
    forge_objects = SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(predictor=SimpleNamespace(shift=3.0))))

    def is_webui_legacy_model(self):
        return False

    def set_shift(self, **kwargs):
        pass


class _HiresSampler(Exception):
    """sample_hr_pass got to sampling: everything it records is recorded."""


def _stop_at_the_sampler(name, model):
    raise _HiresSampler(name)


def _sigmas(p, hires: bool):
    """``KDiffusionSampler.get_sigmas`` of the pass's sampler, which records
    the schedule (and an Anima Flow lock's replacement) in the infotext."""

    from modules import sd_samplers
    from modules.sd_samplers_kdiffusion import KDiffusionSampler

    name = (p.hr_sampler_name or p.sampler_name) if hires else p.sampler_name
    model_wrap = SimpleNamespace(sigmas=torch.linspace(0.03, 1.0, 1000), inner_model=SimpleNamespace(use_shift=False), get_sigmas=lambda steps: torch.linspace(1.0, 0.0, steps + 1))
    sampler = SimpleNamespace(config=sd_samplers.all_samplers_map[name], model_wrap=model_wrap)
    p.is_hr_pass = hires
    KDiffusionSampler.get_sigmas(sampler, p, p.hr_second_pass_steps if hires else p.steps)


def _run(core: dict, scripts: dict) -> str:
    """The infotext of a txt2img job with these request values."""

    from modules import processing, sd_samplers
    from modules.txt2img import prepare_txt2img

    # process_images' order; the scripts' setup ran when the args were set
    p = prepare_txt2img(core, scripts)
    p.sd_model_name = "anima-turbo-v1.1"
    p.scripts.before_process(p)
    p.scripts.after_model_load(p)
    sd_samplers.fix_p_invalid_sampler_and_scheduler(p)
    p.setup_prompts()
    p.all_seeds, p.all_subseeds = [int(p.seed)], [int(p.subseed)]
    p.scripts.process(p)
    p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
    _sigmas(p, hires=False)
    if p.enable_hr:
        # what sample_hr_pass records, up to creating the second pass's sampler
        with pytest.raises(_HiresSampler):
            p.sample_hr_pass(None, None, p.all_seeds, p.all_subseeds, 0.0, p.all_prompts)
        _sigmas(p, hires=True)
    return processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, index=0)


def _request(core=None, sampler=None, scripts=None) -> tuple[dict, dict]:
    return {**BASE, **(core or {})}, {"sampler": {**SAMPLER, **(sampler or {})}, "seed": {"seed": 999}, **(scripts or {})}


def _round_trip(core, scripts):
    from remi.core.infotext import parse_infotext

    text = _run(core, scripts)
    pasted = parse_infotext(text, "txt2img")
    assert pasted["problems"] == []
    return text, pasted, _run(pasted["core"], pasted["scripts"])


GUARD = {"anima_hires_guard": {"enabled": True}}

CASES = {
    "anima-flow-replaced-scheduler": (_request(sampler={"sampler_name": "Anima Flow Euler", "scheduler": "Simple"}), "Anima Flow replaced scheduler: Simple", {("sampler", "scheduler"): "Simple"}),
    "anima-flow-replaced-karras": (_request(sampler={"sampler_name": "Anima Flow UniPC2", "scheduler": "Karras"}), "Anima Flow replaced scheduler: Karras", {("sampler", "scheduler"): "Karras"}),
    "anima-flow-replaced-hires-scheduler": (
        _request(core={**HIRES, "hr_scale": 1.5, "hr_sampler_name": "Anima Flow Euler"}),
        "Anima Flow replaced hires scheduler: Simple",
        {("core", "hr_sampler_name"): "Anima Flow Euler", ("core", "hr_scheduler"): "Use same scheduler"},
    ),
    "anima-flow-replaced-own-hires-scheduler": (
        _request(core={**HIRES, "hr_scale": 1.5, "hr_scheduler": "Karras"}, sampler={"sampler_name": "Anima Flow Euler"}),
        "Anima Flow replaced hires scheduler: Karras",
        {("sampler", "scheduler"): "Simple", ("core", "hr_scheduler"): "Karras"},
    ),
    "dynshift-replaced-scheduler": (_request(scripts={"dynamic_shift": {"enabled": True}}), "DynShift replaced scheduler: Simple", {("sampler", "scheduler"): "Simple"}),
    "hires-guard-clamped-scale": (
        _request(core=HIRES, scripts=GUARD),
        "Anima Hires adjustments: hires 1024x1024 -> 896x896",
        {("core", "hr_scale"): 2.0, ("core", "hr_resize_x"): 0, ("core", "hr_resize_y"): 0},
    ),
    "hires-guard-unchanged-scale": (
        _request(core={**HIRES, "hr_scale": 1.5}, scripts=GUARD),
        "Hires upscale: 1.5",
        {("core", "hr_scale"): 1.5, ("core", "hr_resize_x"): 0, ("core", "hr_resize_y"): 0},
    ),
    "hires-guard-clamped-base-and-resize": (
        _request(core={**HIRES, "width": 1536, "height": 1024, "hr_resize_x": 2000, "hr_resize_y": 0}, scripts=GUARD),
        "Anima Hires requested resize: 2000x0",
        {("core", "width"): 1536, ("core", "height"): 1024, ("core", "hr_resize_x"): 2000, ("core", "hr_resize_y"): 0},
    ),
}


@pytest.mark.parametrize(("request_values", "record", "restored"), CASES.values(), ids=CASES.keys())
def test_the_pasted_run_writes_the_same_infotext(webui, request_values, record, restored):
    original, pasted, again = _round_trip(*request_values)
    assert record in original
    assert again == original
    # ...because the paste restores what was asked, not what the feature made of it
    for (owner, name), value in restored.items():
        values = pasted["core"] if owner == "core" else pasted["scripts"][owner]
        assert values[name] == value, (owner, name)


def test_a_guarded_base_records_the_requested_size(webui):
    original, pasted, _ = _round_trip(*CASES["hires-guard-clamped-base-and-resize"][0])
    assert "Size: 1328x880" in original and "base 1536x1024 -> 1328x880" in original
    assert (pasted["core"]["width"], pasted["core"]["height"]) == (1536, 1024)


@pytest.mark.parametrize("policy", ["Report only", "off"])
def test_a_guard_that_changed_nothing_pastes_its_sizes(webui, policy):
    """Report only (and an off panel) keeps the request: nothing to restore."""
    scripts = {"anima_hires_guard": {"enabled": policy != "off", "policy": "Clamp unsafe values" if policy == "off" else policy}}
    original, pasted, again = _round_trip(*_request(core={**HIRES, "width": 1536, "height": 1024}, scripts=scripts))
    assert again == original and "Hires upscale: 2.0" in original and "Hires resize" not in original
    assert (pasted["core"]["width"], pasted["core"]["height"], pasted["core"]["hr_scale"]) == (1536, 1024, 2.0)


def test_a_chosen_flow_schedule_is_not_a_replacement(webui):
    original, pasted, again = _round_trip(*_request(sampler={"sampler_name": "Anima Flow Euler", "scheduler": "Automatic"}))
    assert "replaced" not in original and again == original
    assert pasted["scripts"]["sampler"]["scheduler"] == "Automatic"

