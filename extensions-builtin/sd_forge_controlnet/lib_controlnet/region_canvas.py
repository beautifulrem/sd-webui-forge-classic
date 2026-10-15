"""The "Region" control type's starting canvas: coloured subject shapes on white."""

from __future__ import annotations

import random
import time

import numpy as np
from PIL import Image, ImageDraw

COLORS = ("red", "orange", "yellow", "green", "blue", "violet", "purple")
SHAPES = ("Rectangle", "Oval")


def generate_canvas(width: int, height: int, subjects: int, randomize: float, vertical: float, shape: str, *, seed: float | None = None) -> np.ndarray:
    """An RGB canvas of ``width`` x ``height``: white for 0 subjects, else one
    coloured shape per subject in evenly spaced columns, centred at
    ``vertical`` (0 = bottom, 1 = top), each jittered by ``randomize``."""

    if subjects <= 0:
        return np.full((height, width, 3), 255, dtype=np.uint8)

    image = Image.new("RGB", (width, height), "white")
    rng = random.Random(time.time() if seed is None else seed)
    draw = ImageDraw.Draw(image)
    column = width / subjects

    for i in range(subjects):
        cx = (i + 0.5 + rng.uniform(-randomize, randomize)) * column
        cy = ((1.0 - vertical) + rng.uniform(-randomize, randomize)) * height

        box_w = column * 0.85 * rng.uniform(1.0 - randomize, 1.0 + randomize)
        box_h = height * 0.9 * rng.uniform(1.0 - randomize, 1.0 + randomize)

        x0, x1 = int(cx - box_w / 2), int(cx + box_w / 2)
        y0, y1 = int(cy - box_h / 2), int(cy + box_h / 2)

        if shape == "Rectangle":
            draw.rectangle((x0, y0, x1, y1), fill=COLORS[i % len(COLORS)])
        else:
            draw.ellipse((x0, y0, x1, y1), fill=COLORS[i % len(COLORS)])

    return np.asarray(image, dtype=np.uint8)
