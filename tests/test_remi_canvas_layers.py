"""The img2img canvas layers as the editor sends them (webui/src/canvas):
a sketch / mask PNG at the image's size, painted on the image as the
browser shows it (upright, EXIF orientation applied); masks white with
the painted alpha. API clients may send other sizes."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from remi.testing import webui_state


@pytest.fixture(scope="module", autouse=True)
def webui():
    with webui_state() as state:
        yield state


def _upright_needed(size=(100, 50)) -> Image.Image:
    """A landscape JPEG whose EXIF says "rotate 90° clockwise to view"
    (shown 50×100 by browsers)."""

    image = Image.new("RGB", size, (0, 0, 255))
    exif = image.getexif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    buffer.seek(0)
    loaded = Image.open(buffer)
    loaded.load()
    return loaded


def _editor_mask(size, box) -> Image.Image:
    """The editor's mask export: white everywhere, alpha 255 inside `box`."""

    mask = Image.new("RGBA", size, (255, 255, 255, 0))
    mask.paste((255, 255, 255, 255), box)
    return mask


def test_layers_follow_the_upright_image():
    from modules.img2img import compose_inputs

    base = _upright_needed()
    # Painted on the 50×100 picture the browser showed: the top half.
    mask = _editor_mask((50, 100), (0, 0, 50, 50))
    image, out = compose_inputs(2, base, None, mask)
    assert image.size == (50, 100)
    assert out.size == (50, 100)
    assert out.getpixel((25, 10))[0] == 255 and out.getpixel((25, 90))[0] == 0

    sketch = Image.new("RGBA", (50, 100), (0, 0, 0, 0))
    sketch.paste((255, 0, 0, 255), (0, 0, 50, 20))
    image, out = compose_inputs(1, base, sketch, None)
    assert image.size == (50, 100)
    assert image.getpixel((25, 5))[:3] == (255, 0, 0)
    assert out is None


def test_layers_of_another_size_are_scaled_to_the_image():
    from modules.img2img import compose_inputs

    base = Image.new("RGBA", (200, 100), (0, 0, 255, 255))
    small_mask = _editor_mask((100, 50), (0, 0, 50, 50))  # left half
    _, out = compose_inputs(2, base, None, small_mask)
    assert out.size == (200, 100)
    assert out.getpixel((40, 50))[0] == 255 and out.getpixel((160, 50))[0] == 0

    uploaded = Image.new("L", (20, 10), 0)
    uploaded.paste(255, (10, 0, 20, 10))  # right half, luminance-coded
    _, out = compose_inputs(4, base, None, uploaded)
    assert out.size == (200, 100)
    assert out.getpixel((160, 50))[0] == 255 and out.getpixel((40, 50))[0] == 0

    sketch = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    sketch.paste((0, 255, 0, 255), (40, 15, 60, 35))
    image, out = compose_inputs(3, base, sketch, None)
    assert image.size == out.size == (200, 100)
    assert image.getpixel((100, 50))[:3] == (0, 255, 0)
    assert out.getpixel((100, 50))[0] == 255 and out.getpixel((5, 5))[0] == 0


def test_a_soft_painted_mask_is_thresholded_at_half():
    from modules.img2img import compose_inputs

    base = Image.new("RGB", (10, 10))
    mask = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    mask.paste((255, 255, 255, 200), (0, 0, 5, 10))
    mask.paste((255, 255, 255, 100), (5, 0, 10, 10))
    _, out = compose_inputs(2, base, None, mask)
    assert out.getpixel((2, 5))[0] == 255 and out.getpixel((7, 5))[0] == 0


def _source_blocks(layout: dict) -> list[dict]:
    """The img2img form's top-level blocks the Source panel shows: the ones
    holding the canvas layers or the inpaint options."""

    def names(node) -> set[str]:
        found = {node["name"]} if isinstance(node, dict) and "name" in node else set()
        for child in node.get("children") or () if isinstance(node, dict) else ():
            found |= names(child)
        return found

    return [block for block in layout["children"] if names(block) & {"init_img", "mask_blur"}]


def test_the_source_area_is_translated():
    """The mode picker, the batch inputs and the inpaint options are in the
    core catalog (localizations/zh_CN.json); the Source panel shows them."""

    from pathlib import Path

    from remi.core.tabs import tab_schema
    from tests import legacy_ui_contract as contract

    root = Path(__file__).resolve().parent.parent
    blocks = _source_blocks(tab_schema("img2img")["core"]["layout"])
    assert blocks, "no source blocks in the img2img layout"
    strings = set().union(*(contract.schema_strings(block) for block in blocks))
    assert contract.missing_translations(strings, root / "localizations" / "zh_CN.json") == []
