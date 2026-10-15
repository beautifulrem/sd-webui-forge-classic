"""OpenPose JSON (the openpose preprocessors' pose output) for the unit's pose editor."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MAX_CANVAS = 4096  # a side; the rendered RGB pose is at most 48 MB


def check_pose(pose: Any) -> dict:
    """The pose dict of an OpenPose JSON value (a list holds one pose per
    image: the first is used, as the old editor did); ``ValueError`` when it
    is not one."""

    if isinstance(pose, list):
        if not pose:
            raise ValueError("the pose list is empty")
        pose = pose[0]
    if not isinstance(pose, dict):
        raise ValueError("a pose is a JSON object")
    for key in ("canvas_width", "canvas_height"):
        size = pose.get(key)
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or not 1 <= size <= MAX_CANVAS:
            raise ValueError(f"{key} must be a size between 1 and {MAX_CANVAS}")
    for key in ("people", "animals"):
        if not isinstance(pose.get(key, []), list):
            raise ValueError(f"{key} must be a list")
    return pose


def render_pose(pose: Any) -> np.ndarray:
    """The RGB image of an OpenPose JSON value (human poses with body, hands
    and face; animal poses when it holds no human)."""

    from annotator.openpose import decode_json_as_poses, draw_poses
    from annotator.openpose.animalpose import draw_animalposes

    pose = check_pose(pose)
    try:
        poses, animals, height, width = decode_json_as_poses(pose)
        height, width = int(height), int(width)
        if poses:
            return draw_poses(poses, height, width, draw_body=True, draw_hand=True, draw_face=True)
        return draw_animalposes(animals, height, width)
    except (AssertionError, KeyError, TypeError, IndexError) as error:
        raise ValueError(f"not an OpenPose JSON pose: {error}") from None
