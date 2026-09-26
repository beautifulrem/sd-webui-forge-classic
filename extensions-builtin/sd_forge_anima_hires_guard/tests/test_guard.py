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


def _process(**upscale):
    from types import SimpleNamespace

    return SimpleNamespace(width=1536, height=1536, enable_hr=True, hr_scale=1.5, hr_resize_x=0, hr_resize_y=0, extra_generation_params={}, **upscale)


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


def test_guard_logger_is_visible(monkeypatch):
    # The WebUI root logger sits at ERROR: an unconfigured logger drops every warning
    script = _guard_script(monkeypatch)
    logger = type(script).after_model_load.__globals__["logger"]
    assert logger.handlers and not logger.propagate
