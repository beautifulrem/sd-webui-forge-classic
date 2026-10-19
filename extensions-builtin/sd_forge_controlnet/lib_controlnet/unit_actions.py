"""Server side of the unit widget's buttons (the script's ``@ui.action``s).

Each returns ``__outputs__`` keyed by what it made (``preview``, ``pose``,
``canvas``, ``catalog``) with the unit index; the widget merges the result
into that unit's current value, so edits made while an action runs are kept.
Images go to the file store as uploads, like any image the user uploads.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from lib_controlnet import external_code, global_state, openpose, region_canvas, unit_param
from lib_controlnet.logging import logger
from lib_controlnet.utils import judge_image_type

from remi import ui
from remi.ui.nodes import check_finite_json, to_number

MAX_POSE_BYTES = 4 * 1024 * 1024


def file_store(ctx: ui.ActionContext) -> ui.ActionFiles:
    """The action's files (``ctx.files``, filled by the server)."""

    if ctx.files is None:
        raise bad_input(ctx.owner, "this action needs the server's file store")
    return ctx.files


def save_png(ctx: ui.ActionContext, image: np.ndarray, name: str) -> str:
    return file_store(ctx).store_image(image, name)


def load_image(ctx: ui.ActionContext, file_id: str | None):
    """The PIL image of an upload or output id; ``None`` when there is none
    or the file is gone."""

    return file_store(ctx).image(file_id) if file_id else None


def bad_input(owner: str, message: str) -> ui.ValidationError:
    return ui.ValidationError([ui.ParamError(message, owner=owner or None)])


def unit_index(ctx: ui.ActionContext, count: int) -> int:
    index = ctx.extra.get("unit")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise bad_input(ctx.owner, f"unit must be an index below {count}, got {index!r}")
    return index


def core_size(ctx: ui.ActionContext) -> tuple[int, int]:
    """The generation size of the form (``core.width`` / ``core.height``)."""

    def read(name: str) -> int:
        try:
            number = to_number(ctx.core.get(name, 512))
        except ui.ParamError:
            number = 512
        return int(min(max(number, 64), 8192))

    return read("width"), read("height")


def _setting(value: Any, parameter) -> Any:
    """A slider value, or the preprocessor's default for an unset (negative) one."""

    return parameter.slider_kwargs.get("value", 512) if value is None or value < 0 else value


def run_preview(ctx: ui.ActionContext, index: int, value: dict[str, Any]) -> dict[str, Any]:
    """Run the unit's preprocessor on its image (the old "Run preprocessor")."""

    image = load_image(ctx, value["image"])
    if image is None:
        return ui.toast("Upload a control image first", "warning")
    preprocessor = global_state.supported_preprocessors.get(value["module"])
    if preprocessor is None:
        return ui.toast(ui.message("Unknown preprocessor {{name}}", name=value["module"]), "error")

    from modules_forge.utils import HWC3

    img = HWC3(unit_param.rgba(unit_param.with_sketch(image, load_image(ctx, value["sketch"]))))
    mask = None
    painted = load_image(ctx, value["image_fg"])
    if painted is not None:
        alpha = unit_param.mask_layer(painted)[:, :, 3]
        if (alpha > 5).any():
            mask = HWC3(alpha)
            if mask.shape[:2] != img.shape[:2]:
                import cv2

                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    resolution = _setting(value["processor_res"], preprocessor.slider_resolution)
    if value["pixel_perfect"]:
        width, height = core_size(ctx)
        resolution = external_code.pixel_perfect_resolution(img, target_H=height, target_W=width, resize_mode=external_code.resize_mode_from_value(value["resize_mode"]))
    logger.info(f"Preview Resolution = {resolution}")

    poses: list[dict] = []
    try:
        result = preprocessor(
            input_image=img,
            resolution=resolution,
            slider_1=_setting(value["threshold_a"], preprocessor.slider_1),
            slider_2=_setting(value["threshold_b"], preprocessor.slider_2),
            input_mask=mask,
            # Only the openpose preprocessors give a pose; others keep their result cache
            json_pose_callback=poses.append if "openpose" in value["module"] else None,
        )
    except Exception as error:  # a preprocessor's own failure (missing download, bad input)
        logger.exception("Preprocessor preview failed")
        return ui.toast(ui.message("The preprocessor failed: {{error}}", error=str(error)), "error")

    if not judge_image_type(result):
        result = img
    result = external_code.visualize_inpaint_mask(result)
    file_id = save_png(ctx, result, "controlnet-preview.png")
    return {"__outputs__": {"preview": {"unit": index, "image": file_id, "pose": poses[0] if poses else None}}}


def render_pose(ctx: ui.ActionContext, index: int) -> dict[str, Any]:
    """Draw an edited or uploaded OpenPose JSON (the old pose editor's render)."""

    pose = ctx.extra.get("pose")
    if isinstance(pose, str):
        if len(pose) > MAX_POSE_BYTES:
            return ui.toast("The pose file is too large", "error")
        try:
            pose = json.loads(pose)
        except ValueError:
            return ui.toast("The pose is not valid JSON", "error")
    try:
        check_finite_json(pose)
        image = openpose.render_pose(pose)
    except (ValueError, ui.ParamError) as error:
        return ui.toast(ui.message("Not an OpenPose pose: {{error}}", error=str(error)), "error")
    file_id = save_png(ctx, image, "controlnet-pose.png")
    return {"__outputs__": {"pose": {"unit": index, "image": file_id, "pose": pose}}}


def _number(ctx: ui.ActionContext, name: str) -> float:
    control = unit_param.CONTROLS[f"region_{name}"]
    raw = ctx.extra.get(name, control["default"])
    try:
        number = to_number(raw)
    except ui.ParamError:
        raise bad_input(ctx.owner, f"{name}: expected a number, got {raw!r}") from None
    return min(max(number, control["minimum"]), control["maximum"])


def make_region_canvas(ctx: ui.ActionContext, index: int) -> dict[str, Any]:
    """The "Region" type's starting canvas, at the generation size."""

    shape = ctx.extra.get("shape", "Rectangle")
    if shape not in region_canvas.SHAPES:
        raise bad_input(ctx.owner, f"shape is one of {list(region_canvas.SHAPES)}")
    width, height = core_size(ctx)
    subjects = math.floor(_number(ctx, "subjects") + 0.5)
    image = region_canvas.generate_canvas(width, height, subjects, _number(ctx, "randomize"), _number(ctx, "vertical"), shape)
    return {"__outputs__": {"canvas": {"unit": index, "image": save_png(ctx, image, "controlnet-region.png")}}}
