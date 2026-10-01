import math
import sys

import pytest

from lib_anima_hires import GuardLimits, guard_dimensions


def test_safe_dimensions_remain_unchanged():
    result = guard_dimensions(
        1024,
        1024,
        enable_hr=True,
        hr_scale=1.4,
        hr_resize_x=0,
        hr_resize_y=0,
        limits=GuardLimits(1.2, 2.1, 1.75),
    )
    assert (result.base_width, result.base_height) == (1024, 1024)
    assert (result.hires_width, result.hires_height) == (1440, 1440)
    assert not result.changes


def test_base_and_hires_are_clamped_and_aligned():
    result = guard_dimensions(
        1537,
        1537,
        enable_hr=True,
        hr_scale=3.0,
        hr_resize_x=0,
        hr_resize_y=0,
        limits=GuardLimits(1.2, 2.1, 1.75),
    )
    assert result.base_width % 16 == result.base_height % 16 == 0
    assert result.base_width * result.base_height <= 1_200_000
    assert result.hires_width * result.hires_height <= 2_100_000
    assert result.hires_width <= result.base_width * 1.75
    assert len(result.changes) == 2


def test_explicit_single_axis_target_uses_guarded_base_aspect():
    result = guard_dimensions(
        1024,
        768,
        enable_hr=True,
        hr_scale=2.0,
        hr_resize_x=1600,
        hr_resize_y=0,
        limits=GuardLimits(1.2, 2.1, 2.0),
    )
    assert (result.hires_width, result.hires_height) == (1600, 1200)


def test_hires_never_becomes_a_downscale():
    result = guard_dimensions(
        1024,
        1024,
        enable_hr=True,
        hr_scale=1.0,
        hr_resize_x=512,
        hr_resize_y=512,
        limits=GuardLimits(1.2, 2.1, 1.75),
    )
    assert (result.hires_width, result.hires_height) == (1024, 1024)


def test_no_hires_returns_no_target():
    result = guard_dimensions(
        1025,
        1025,
        enable_hr=False,
        hr_scale=2.0,
        hr_resize_x=0,
        hr_resize_y=0,
        limits=GuardLimits(1.2, 2.1, 1.75),
    )
    assert (result.base_width, result.base_height) == (1024, 1024)
    assert result.hires_width is None and result.hires_height is None


def test_hires_ceiling_cannot_fall_below_base_ceiling():
    limits = GuardLimits(1.2, 0.8, 1.75).validated()
    assert limits.hires_megapixels == 1.2


def test_alignment_never_rounds_above_megapixel_ceiling():
    result = guard_dimensions(
        1095,
        1095,
        enable_hr=False,
        hr_scale=1.0,
        hr_resize_x=0,
        hr_resize_y=0,
        limits=GuardLimits(1.2, 2.1, 1.75),
    )

    assert (result.base_width, result.base_height) == (1088, 1088)
    assert result.base_width * result.base_height <= 1_200_000


def test_rotated_hires_target_falls_back_without_combining_axes():
    result = guard_dimensions(
        1200,
        600,
        enable_hr=True,
        hr_scale=1.0,
        hr_resize_x=600,
        hr_resize_y=1200,
        limits=GuardLimits(1.0, 1.0, 2.0),
    )

    assert (result.hires_width, result.hires_height) == (
        result.base_width,
        result.base_height,
    )
    assert result.hires_width * result.hires_height <= 1_000_000


def test_guard_contract_holds_across_aspects_and_targets():
    limits = GuardLimits(1.2, 2.1, 1.75).validated()
    bases = [(127, 4096), (511, 1537), (1095, 1095), (4096, 127)]
    targets = [(0, 0), (600, 1800), (1800, 600), (4096, 4096)]

    for width, height in bases:
        for resize_x, resize_y in targets:
            result = guard_dimensions(
                width,
                height,
                enable_hr=True,
                hr_scale=2.5,
                hr_resize_x=resize_x,
                hr_resize_y=resize_y,
                limits=limits,
            )
            assert result.base_width * result.base_height <= 1_200_000
            assert result.hires_width * result.hires_height <= 2_100_000
            assert result.hires_width >= result.base_width
            assert result.hires_height >= result.base_height
            assert result.hires_width <= result.base_width * 1.75
            assert result.hires_height <= result.base_height * 1.75


def test_fixed_base_keeps_the_image_and_guards_only_the_target():
    # txt2img's Upscale button: the base is a finished 1536x1536 image.
    result = guard_dimensions(
        1536,
        1536,
        enable_hr=True,
        hr_scale=1.5,
        hr_resize_x=0,
        hr_resize_y=0,
        limits=GuardLimits(1.2, 2.1, 1.75),
        fixed_base=True,
    )
    assert (result.base_width, result.base_height) == (1536, 1536)
    # Above the Hires ceiling already: no upscale, but never a downscale
    assert (result.hires_width, result.hires_height) == (1536, 1536)

    result = guard_dimensions(
        1000,
        1000,
        enable_hr=True,
        hr_scale=1.4,
        hr_resize_x=0,
        hr_resize_y=0,
        limits=GuardLimits(0.9, 2.1, 1.75),
        fixed_base=True,
    )
    assert (result.base_width, result.base_height) == (1000, 1000)
    assert (result.hires_width, result.hires_height) == (1408, 1408)


def _guard_script(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    # The WebUI modules parse sys.argv on first import: keep pytest's away
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    path = Path(__file__).parents[1] / "scripts" / "anima_hires_guard.py"
    spec = importlib.util.spec_from_file_location("anima_hires_guard_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._is_anima = lambda process: True
    return module.AnimaHiresGuardScript()


def _process(**fields):
    from types import SimpleNamespace

    defaults = dict(width=1536, height=1536, enable_hr=True, hr_scale=1.5, hr_resize_x=0, hr_resize_y=0, extra_generation_params={})
    return SimpleNamespace(**{**defaults, **fields})


def test_upscale_button_does_not_shrink_the_image(monkeypatch):
    script = _guard_script(monkeypatch)
    args = (True, "Clamp unsafe values", "Balanced", True, 1.2, 2.1, 1.75)

    upscale = _process(txt2img_upscale=True, firstpass_image=object())
    script.after_model_load(upscale, *args)
    assert (upscale.width, upscale.height) == (1536, 1536)
    assert upscale.hr_resize_x >= 1536 and upscale.hr_resize_y >= 1536

    generation = _process()  # a normal run still gets a safe base pass
    script.after_model_load(generation, *args)
    assert generation.width * generation.height <= 1_200_000


def test_report_only_keeps_the_dimensions_and_says_so(monkeypatch):
    script = _guard_script(monkeypatch)
    report = _process()
    script.after_model_load(report, True, "Report only", "Balanced", True, 1.2, 2.1, 1.75)
    assert (report.width, report.height, report.hr_resize_x, report.hr_resize_y) == (1536, 1536, 0, 0)
    note = report.extra_generation_params["Anima Hires adjustments"]

    clamp = _process()
    script.after_model_load(clamp, True, "Clamp unsafe values", "Balanced", True, 1.2, 2.1, 1.75)
    applied = clamp.extra_generation_params["Anima Hires adjustments"]
    # The same changes, but never written as if they had been applied
    assert note == f"report only (not applied): {applied}"
    assert not applied.startswith("report only")


def test_legacy_hires_size_setting_skips_the_guard(monkeypatch):
    from types import SimpleNamespace

    script = _guard_script(monkeypatch)
    globals_ = type(script).after_model_load.__globals__
    monkeypatch.setitem(globals_, "shared", SimpleNamespace(opts=SimpleNamespace(use_old_hires_fix_width_height=True)))
    process = _process()
    script.after_model_load(process, True, "Clamp unsafe values", "Balanced", True, 1.2, 2.1, 1.75)
    # Width/height are the final Hires size here: nothing is guarded or recorded as applied
    assert (process.width, process.height, process.hr_resize_x, process.hr_resize_y) == (1536, 1536, 0, 0)
    assert process.extra_generation_params == {"Anima Hires adjustments": "skipped: the legacy hires width/height setting is on"}

    # Without Hires the setting does not apply and the base pass is guarded
    process = _process(enable_hr=False)
    script.after_model_load(process, True, "Clamp unsafe values", "Balanced", True, 1.2, 2.1, 1.75)
    assert process.width * process.height <= 1_200_000


def test_guard_logger_is_visible(monkeypatch):
    # The WebUI root logger sits at ERROR: an unconfigured logger drops every warning
    script = _guard_script(monkeypatch)
    logger = type(script).after_model_load.__globals__["logger"]
    assert logger.handlers and not logger.propagate


def _forge_hires_target(process, step):
    """Forge's calculate_target_resolution (modules/processing.py) and sRound (modules/ui.py)."""

    def s_round(value):
        return math.floor(value / step + 0.5) * step

    if process.hr_resize_x == 0 and process.hr_resize_y == 0:
        return s_round(process.width * process.hr_scale), s_round(process.height * process.hr_scale)
    if process.hr_resize_y == 0:
        x, y = process.hr_resize_x, process.hr_resize_x * (process.height / process.width)
    elif process.hr_resize_x == 0:
        x, y = process.hr_resize_y * (process.width / process.height), process.hr_resize_y
    else:
        x, y = process.hr_resize_x, process.hr_resize_y
    return s_round(x), s_round(y)


def _with_resolution_step(monkeypatch, step):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "modules.ui", SimpleNamespace(_STEP=step))


@pytest.mark.parametrize("step", [64, 16, 128])
@pytest.mark.parametrize("align", [True, False])
def test_rendered_hires_size_holds_the_caps_after_forge_rounding(monkeypatch, step, align):
    script = _guard_script(monkeypatch)
    _with_resolution_step(monkeypatch, step)
    bases = [(1024, 1024), (1000, 1000), (832, 1216), (896, 1152), (1184, 1744), (1095, 1095), (1056, 1056)]
    targets = [(1.41, 0, 0), (1.75, 0, 0), (2.5, 0, 0), (1.0, 1600, 0), (1.0, 0, 2000), (1.0, 1216, 1728), (1.0, 4096, 4096)]
    for width, height in bases:
        for hr_scale, resize_x, resize_y in targets:
            process = _process(width=width, height=height, hr_scale=hr_scale, hr_resize_x=resize_x, hr_resize_y=resize_y)
            script.after_model_load(process, True, "Clamp unsafe values", "Balanced", align, 1.2, 2.1, 1.75)
            rendered = _forge_hires_target(process, step)
            case = (step, align, width, height, hr_scale, resize_x, resize_y, rendered)
            assert process.width * process.height <= 1_200_000, case
            assert rendered[0] * rendered[1] <= 2_100_000, case
            assert rendered[0] <= process.width * 1.75 and rendered[1] <= process.height * 1.75, case
            # The explicit target is already on Forge's grid: its rounding changes nothing
            assert rendered == (process.hr_resize_x, process.hr_resize_y), case


def test_balanced_scale_that_forge_rounds_over_the_ceiling_is_clamped(monkeypatch):
    # 1024 x 1.41 = 1443.84: 1440 on the 16 px grid (2.07 MP), but Forge's
    # 64 px Resolution Step renders 1472 (2.17 MP, over Balanced's 2.1)
    script = _guard_script(monkeypatch)
    _with_resolution_step(monkeypatch, 64)
    process = _process(width=1024, height=1024, hr_scale=1.41)
    script.after_model_load(process, True, "Clamp unsafe values", "Balanced", True, 1.2, 2.1, 1.75)
    assert (process.hr_resize_x, process.hr_resize_y) == (1408, 1408)
    assert _forge_hires_target(process, 64) == (1408, 1408)
    assert process.extra_generation_params["Anima Hires adjustments"] == "hires 1472x1472 -> 1408x1408"


def test_safe_scale_is_still_written_as_an_explicit_target(monkeypatch):
    script = _guard_script(monkeypatch)
    _with_resolution_step(monkeypatch, 64)
    process = _process(width=1024, height=1024, hr_scale=1.25)
    script.after_model_load(process, True, "Clamp unsafe values", "Balanced", True, 1.2, 2.1, 1.75)
    assert (process.hr_resize_x, process.hr_resize_y) == (1280, 1280)
    assert "Anima Hires adjustments" not in process.extra_generation_params

    report = _process(width=1024, height=1024, hr_scale=1.25)
    script.after_model_load(report, True, "Report only", "Balanced", True, 1.2, 2.1, 1.75)
    assert (report.hr_resize_x, report.hr_resize_y) == (0, 0)


def test_resolution_step_prefers_the_value_forge_loaded(monkeypatch):
    from types import SimpleNamespace

    script = _guard_script(monkeypatch)
    step = type(script).after_model_load.__globals__["_forge_resolution_step"]
    _with_resolution_step(monkeypatch, 128)
    assert step() == 128
    monkeypatch.delitem(sys.modules, "modules.ui")
    monkeypatch.setitem(type(script).after_model_load.__globals__, "shared", SimpleNamespace(opts=SimpleNamespace(res_step=32)))
    assert step() == 32


def test_off_grid_base_fallback_lands_on_the_hires_grid():
    # A downscale target falls back to the base, snapped to the Hires grid
    result = guard_dimensions(
        1056,
        1056,
        enable_hr=True,
        hr_scale=1.0,
        hr_resize_x=512,
        hr_resize_y=512,
        limits=GuardLimits(1.2, 1.2, 1.75),
        hires_alignment=64,
    )
    assert (result.base_width, result.base_height) == (1056, 1056)
    # Nearest (1088^2 = 1.18 MP) fits the 1.2 MP ceiling
    assert (result.hires_width, result.hires_height) == (1088, 1088)

    result = guard_dimensions(
        1088,
        1088,
        enable_hr=True,
        hr_scale=1.0,
        hr_resize_x=512,
        hr_resize_y=512,
        limits=GuardLimits(1.2, 1.2, 1.75),
        hires_alignment=128,
    )
    # Nearest 1152^2 = 1.33 MP would cross the ceiling: floor to 1024
    assert (result.hires_width, result.hires_height) == (1024, 1024)


NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("value", NON_FINITE, ids=repr)
@pytest.mark.parametrize("field", ["base_mp", "hires_mp", "max_upscale"])
def test_a_non_finite_limit_is_refused_before_anything_is_guarded(monkeypatch, field, value):
    # The API reads JSON NaN / Infinity as floats: max(0.1, nan) shrank a
    # 1024x1024 Custom job to 304x304 and recorded it as an ordinary clamp,
    # inf lifted the ceiling, a NaN upscale cap turned the Hires upscale off
    script = _guard_script(monkeypatch)
    _with_resolution_step(monkeypatch, 64)
    limits = {"base_mp": 1.2, "hires_mp": 2.1, "max_upscale": 1.75, field: value}
    process = _process(width=1024, height=1024, hr_scale=1.4)
    with pytest.raises(ValueError, match=rf"^Anima Hires Guard refuses non-finite script arguments: {field}={value!r}$"):
        script.after_model_load(process, True, "Clamp unsafe values", "Custom", True, *limits.values())
    assert (process.width, process.height, process.hr_resize_x, process.hr_resize_y) == (1024, 1024, 0, 0)
    assert process.extra_generation_params == {}


def test_limits_and_scale_that_are_not_finite_numbers_read_as_the_defaults():
    from lib_anima_hires import BALANCED

    nan = float("nan")
    for value in (nan, float("inf"), "nan", None):
        assert GuardLimits(value, value, value).validated() == BALANCED.validated()
    result = guard_dimensions(1024, 1024, enable_hr=True, hr_scale=nan, hr_resize_x=0, hr_resize_y=0, limits=BALANCED)
    assert (result.base_width, result.base_height, result.hires_width, result.hires_height) == (1024, 1024, 1024, 1024)
