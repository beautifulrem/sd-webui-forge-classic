"""The "Alpha Threshold" setting (upstream 2c24de16): saving an RGBA image
(``images.save_image_with_geninfo``, which the Remi gallery's save and the
normal sample saving go through) zeroes alpha below the threshold and blackens
the fully transparent pixels; 0 keeps the alpha as it is."""

import pytest
from PIL import Image

from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


def _rgba(alphas):
    image = Image.new("RGBA", (len(alphas), 1))
    image.putdata([(200, 100, 50, a) for a in alphas])
    return image


def test_setting_is_a_slider_with_the_upstream_default():
    from modules import shared_options
    from remi import ui

    info = shared_options.options_templates["alpha_threshold"]
    assert (info.default, info.component, info.component_args) == (16, ui.Slider, {"minimum": 0, "maximum": 255, "step": 1})


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (16, [(0, 0, 0, 0), (0, 0, 0, 0), (200, 100, 50, 16), (200, 100, 50, 255)]),
        (100, [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (200, 100, 50, 255)]),
        (0, [(200, 100, 50, 0), (200, 100, 50, 15), (200, 100, 50, 16), (200, 100, 50, 255)]),
    ],
)
def test_alpha_below_the_threshold_is_cleared(webui, monkeypatch, threshold, expected):
    from modules import images

    monkeypatch.setattr(images, "opts", webui.opts)  # bound at import (by whichever suite imported it first)
    monkeypatch.setitem(webui.opts.data, "alpha_threshold", threshold)
    result = images._process_alpha(_rgba([0, 15, 16, 255]))
    assert result.mode == "RGBA" and list(result.getdata()) == expected


def test_opaque_rgba_becomes_rgb_and_rgb_is_untouched(webui):
    from modules import images

    assert images._process_alpha(_rgba([255, 255])).mode == "RGB"
    rgb = Image.new("RGB", (2, 2))
    assert images._process_alpha(rgb) is rgb
