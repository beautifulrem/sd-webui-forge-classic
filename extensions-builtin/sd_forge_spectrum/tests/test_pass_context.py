import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).parents[1] / "lib_spectrum" / "pass_context.py"
SPEC = importlib.util.spec_from_file_location("spectrum_pass_context", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_base_pass_uses_base_sampling_parameters():
    process = SimpleNamespace(
        is_hr_pass=False,
        steps=30,
        sampler_name="Euler",
        cfg_scale=5.0,
        hr_second_pass_steps=12,
        hr_sampler_name="Anima Flow PC3",
        hr_cfg=3.0,
    )

    result = MODULE.resolve_pass_context(process)

    assert (result.steps, result.sampler, result.cfg, result.is_hires) == (
        30,
        "Euler",
        5.0,
        False,
    )


def test_hires_pass_uses_effective_hires_sampling_parameters():
    process = SimpleNamespace(
        is_hr_pass=True,
        steps=30,
        sampler_name="Euler",
        cfg_scale=5.0,
        hr_second_pass_steps=12,
        hr_sampler_name="Anima Flow PC3",
        hr_cfg=3.0,
    )

    result = MODULE.resolve_pass_context(process)

    assert (result.steps, result.sampler, result.cfg, result.is_hires) == (
        12,
        "Anima Flow PC3",
        3.0,
        True,
    )


def test_hires_zero_steps_and_sampler_fall_back_to_base():
    process = SimpleNamespace(
        is_hr_pass=True,
        steps=24,
        sampler_name="Euler",
        cfg_scale=5.0,
        hr_second_pass_steps=0,
        hr_sampler_name=None,
        hr_cfg=1.0,
    )

    result = MODULE.resolve_pass_context(process)

    assert result.steps == 24
    assert result.sampler == "Euler"


def test_img2img_pass_counts_only_the_steps_denoise_runs():
    process = SimpleNamespace(
        is_hr_pass=False,
        steps=30,
        denoising_strength=0.4,
        init_images=[object()],
        sampler_name="Euler",
        cfg_scale=5.0,
    )

    # setup_img2img_steps() returns (schedule steps, t_enc); t_enc + 1 run.
    result = MODULE.resolve_pass_context(process, img2img_steps=lambda p: (30, 12))

    assert result.steps == 13


def test_scheduler_and_shift_follow_the_pass():
    process = SimpleNamespace(
        is_hr_pass=False,
        steps=30,
        sampler_name="Euler",
        scheduler="Simple",
        cfg_scale=5.0,
        distilled_cfg_scale=3.0,
        hr_scheduler="Beta",
        hr_distilled_cfg=5.0,
    )

    base = MODULE.resolve_pass_context(process)
    process.is_hr_pass = True
    hires = MODULE.resolve_pass_context(process)
    process.hr_scheduler = process.hr_distilled_cfg = None
    fallback = MODULE.resolve_pass_context(process)

    assert (base.scheduler, base.shift) == ("Simple", 3.0)
    assert (hires.scheduler, hires.shift) == ("Beta", 5.0)
    # As sd_samplers_kdiffusion: an unset hires schedule is "Automatic".
    assert (fallback.scheduler, fallback.shift) == ("Automatic", 3.0)
