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
