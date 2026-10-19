"""A ControlNet unit as one ``remi.ui`` param.

The processing argument stays one :class:`ControlNetUnit` per unit (what the
Gradio ``gr.State`` held), so ``process()``, ``external_code`` users and the
``/sdapi`` ``alwayson_scripts`` (``{"ControlNet": {"args": [unit dict,
...]}}``) are unchanged. In the form a unit is one JSON object edited by the
``controlnet-unit`` widget (webui/src/widgets/controlnet-unit):

- the unit's settings under their ``ControlNetUnit`` names (``enabled``,
  ``module``, ``model``, ``weight``, ``guidance_start`` / ``guidance_end``,
  ``processor_res``, ``threshold_a`` / ``threshold_b``, ``pixel_perfect``,
  ``control_mode``, ``resize_mode``, ``hr_option``, ``type_filter``,
  ``save_detected_map``, ``use_preview_as_input``);
- upload ids: ``image`` (the control image), ``sketch`` (a colour layer
  painted over it, merged into the image), ``image_fg`` (a mask painted over
  it), ``mask_image`` / ``mask_image_fg`` (the "Use Mask" image and the mask
  painted over it, read while ``mask_upload`` is on) and ``generated_image``
  (the preprocessor preview);
- ``pose`` (the OpenPose JSON of the preview, for the pose editor) and, in
  img2img, ``upload_independent`` (without it the unit reads the img2img
  image, as before).

Masks are uploaded alpha-coded (schema-format.md §1.3) and reach the unit the
way the old canvas sent them: RGBA arrays whose alpha is the mask.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from lib_controlnet import global_state
from lib_controlnet.enums import HiResFixOption
from lib_controlnet.external_code import ControlMode, ControlNetUnit, ResizeMode, control_mode_from_value, resize_mode_from_value
from lib_controlnet.infotext import parse_unit_fields

from remi import ui
from remi.ui.nodes import check_finite_json, mask_channel, to_number

WIDGET = "controlnet-unit"
PASTE_MARKER = "__paste__"
"""Key of a pasted (partial) unit: the widget merges it into the unit it
replaces, keeping the images (pasted infotext never carries them)."""

SETTINGS = (
    "enabled",
    "type_filter",
    "module",
    "model",
    "weight",
    "guidance_start",
    "guidance_end",
    "processor_res",
    "threshold_a",
    "threshold_b",
    "pixel_perfect",
    "control_mode",
    "resize_mode",
    "hr_option",
    "save_detected_map",
    "use_preview_as_input",
)
IMAGES = ("image", "sketch", "image_fg", "mask_image", "mask_image_fg", "generated_image")
BOOLEANS = frozenset({"enabled", "pixel_perfect", "save_detected_map", "use_preview_as_input", "mask_upload", "upload_independent"})
TEXTS = frozenset({"type_filter", "module", "model"})

# The unit's controls (labels, bounds, choices) as the legacy ui() declared
# them; the widget renders from these, the parity test compares them.
CONTROLS: dict[str, dict[str, Any]] = {
    "enabled": {"label": "Enable"},
    "pixel_perfect": {"label": "Pixel Perfect"},
    "mask_upload": {"label": "Use Mask"},
    "use_preview_as_input": {"label": "Preview as Input"},
    "upload_independent": {"label": "Upload independent control image"},
    "type_filter": {"label": "Control Type"},
    "module": {"label": "Preprocessor"},
    "model": {"label": "Model"},
    "weight": {"label": "Control Weight", "minimum": 0.0, "maximum": 2.0, "step": 0.05},
    "timestep_range": {"label": "Timestep Range", "minimum": 0.0, "maximum": 1.0, "step": 0.01},
    "processor_res": {"label": "Preprocessor resolution"},
    "threshold_a": {"label": "Threshold A"},
    "threshold_b": {"label": "Threshold B"},
    "control_mode": {"label": "Control Mode", "choices": [mode.value for mode in ControlMode]},
    "resize_mode": {"label": "Resize Mode", "choices": [mode.value for mode in ResizeMode]},
    "hr_option": {"label": "Hires-Fix Option", "choices": [option.value for option in HiResFixOption]},
    "region_subjects": {"label": "Number of Subjects", "minimum": 0, "maximum": 7, "step": 1, "default": 0, "info": "0 for Blank Canvas"},
    "region_vertical": {"label": "Vertical Position", "minimum": 0.2, "maximum": 0.8, "step": 0.05, "default": 0.4},
    "region_randomize": {"label": "Randomize Bias", "minimum": 0.0, "maximum": 0.5, "step": 0.025, "default": 0.075},
    "region_shape": {"label": "Mask Shape", "choices": ["Rectangle", "Oval"], "default": "Rectangle"},
}
INVERT_HINT = "Set the preprocessor to [invert] If your image has white background and black lines."


def _plain(value: Any) -> Any:
    return value.value if hasattr(value, "value") and not isinstance(value, (bool, int, float, str)) else value


def default_value(is_img2img: bool) -> dict[str, Any]:
    """A fresh unit: the legacy UI's default ``ControlNetUnit`` (disabled,
    no preprocessor, no model), no images."""

    unit = ControlNetUnit(enabled=False, module="None", model="None")
    value = {name: _plain(getattr(unit, name)) for name in SETTINGS}
    value.update(dict.fromkeys(IMAGES))
    value.update(mask_upload=False, pose=None)
    if is_img2img:
        value["upload_independent"] = False
    return value


# region Value rules


def _boolean(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ui.ParamError(f"expected true or false, got {value!r}")


def _clamped(value: Any, low: float, high: float) -> float:
    return float(min(max(to_number(value), low), high))


def _choice(enum, convert, value: Any) -> str:
    # API units are written to the infotext as str(member): "ResizeMode.INNER_FIT"
    if isinstance(value, str) and value.startswith(f"{enum.__name__}."):
        member = enum.__members__.get(value.partition(".")[2])
        if member is not None:
            return member.value
    try:
        return convert(value).value
    except (ValueError, KeyError, IndexError, AssertionError, AttributeError, TypeError):
        raise ui.ParamError(f"{value!r} is not one of the choices") from None


def coerce_field(name: str, value: Any) -> Any:
    """One unit field's canonical value, or ``ParamError``."""

    if name in BOOLEANS:
        return _boolean(value)
    if name in TEXTS:
        if not isinstance(value, str):
            raise ui.ParamError(f"expected text, got {type(value).__name__}")
        return value
    if name == "weight":
        return _clamped(value, 0.0, 2.0)
    if name in ("guidance_start", "guidance_end"):
        return _clamped(value, 0.0, 1.0)
    if name == "processor_res":
        number = to_number(value)
        return -1 if number < 0 else math.floor(number + 0.5)
    if name in ("threshold_a", "threshold_b"):
        number = to_number(value)
        return -1 if number < 0 else number
    if name == "control_mode":
        return _choice(ControlMode, control_mode_from_value, value)
    if name == "resize_mode":
        return _choice(ResizeMode, resize_mode_from_value, value)
    if name == "hr_option":
        return _choice(HiResFixOption, HiResFixOption.from_value, value)
    if name in IMAGES:
        if value is None or (isinstance(value, str) and value):
            return value
        raise ui.ParamError(f"expected an upload id, got {type(value).__name__}")
    if name == "pose":
        if value is None or isinstance(value, (dict, list)):
            return value
        raise ui.ParamError("a pose is an OpenPose JSON object")
    raise ui.ParamError("unknown field")


def normalize(value: Any, is_img2img: bool) -> dict[str, Any]:
    """The canonical unit for ``value``: every field present (defaults for
    the missing ones), unknown keys (and the paste marker) dropped."""

    if not isinstance(value, Mapping):
        raise ui.ParamError(f"expected a ControlNet unit object, got {type(value).__name__}")
    check_finite_json(value)
    result = default_value(is_img2img)
    for key, raw in value.items():
        if key not in result:
            continue
        try:
            result[key] = coerce_field(key, raw)
        except ui.ParamError as error:
            raise ui.ParamError(f"{key}: {error}") from None
    if result["guidance_start"] > result["guidance_end"]:
        result["guidance_start"], result["guidance_end"] = result["guidance_end"], result["guidance_start"]
    return result


def pasted_fields(text: str) -> dict[str, Any]:
    """The unit fields an infotext entry (``"Module: canny, Weight: 1, ..."``)
    sets. Like the old paste (a full ``ControlNetUnit`` parsed from the
    entry), every infotext field is set: one the entry leaves out takes the
    ``ControlNetUnit`` default (the infotext omits the unset -1 resolution
    and thresholds). A field whose value does not read is left out, as a
    Gradio control kept its value then."""

    try:
        fields = parse_unit_fields(text)
    except ValueError as error:
        raise ui.ParamError(str(error)) from None
    if not fields:
        return {}
    defaults = ControlNetUnit()
    result = {}
    for key in ControlNetUnit.infotext_fields():
        try:
            result[key] = coerce_field(key, fields[key] if key in fields else _plain(getattr(defaults, key)))
        except ui.ParamError:
            continue
    return result


# endregion

# region Images


def _is_pil(value: Any) -> bool:
    from PIL import Image as PILImage

    return isinstance(value, PILImage.Image)


def rgba(image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image.convert("RGBA"), dtype=np.uint8))


def with_sketch(image, sketch):
    """``image`` with the colour ``sketch`` layer painted over it (RGBA)."""

    from PIL import Image as PILImage

    base = image.convert("RGBA")
    if sketch is None:
        return base
    layer = sketch.convert("RGBA")
    if layer.size != base.size:
        layer = layer.resize(base.size, PILImage.Resampling.NEAREST)
    return PILImage.alpha_composite(base, layer)


def mask_layer(mask) -> np.ndarray:
    """A painted (or uploaded) mask as the old canvas foreground: white, the
    mask in the alpha channel."""

    channel = np.asarray(mask_channel(mask), dtype=np.uint8)
    layer = np.full((*channel.shape, 4), 255, dtype=np.uint8)
    layer[:, :, 3] = channel
    return layer


def mask_background(mask) -> np.ndarray:
    """An uploaded mask image as the old mask canvas background: the mask
    (alpha, else luminance) in the colour channels, opaque."""

    channel = np.asarray(mask_channel(mask), dtype=np.uint8)
    return np.dstack([channel, channel, channel, np.full_like(channel, 255)])


def _picture(value: Any):
    """A unit picture as processing held it (an array, a PIL image, or the
    base64 text of an API unit) as a PIL image; ``None`` for anything else."""

    from PIL import Image as PILImage

    if _is_pil(value):
        return value
    if isinstance(value, np.ndarray):
        array = value[:, :, 0] if value.ndim == 3 and value.shape[2] == 1 else value
        if array.ndim == 2 or (array.ndim == 3 and array.shape[2] in (3, 4)):
            return PILImage.fromarray(np.ascontiguousarray(array.astype(np.uint8)))
        return None
    if isinstance(value, str) and value and not value.startswith(("http://", "https://")):
        try:
            from modules.api import api

            return api.decode_base64_to_image(value)
        except Exception:  # not an image: dropped like any other unreadable value
            return None
    return None


def _alpha_layer(picture):
    """A painted layer (the mask in its alpha) as an ``L`` mask, so that it
    reads back the same whatever its colour channels hold."""

    return picture.getchannel("A") if "A" in picture.getbands() else picture


def imported_images(source: Mapping[str, Any], store_upload, is_img2img: bool) -> dict[str, Any]:
    """The upload ids of a processing unit's pictures (``source``: its
    fields) stored with ``store_upload``, plus the switches that make the
    unit read them again (``mask_upload``; ``upload_independent`` in img2img,
    where processing read the unit's own image whenever it had one)."""

    image, mask_image = source.get("image"), source.get("mask_image")
    if isinstance(image, Mapping):  # the old Gradio {image, mask} pair
        image, mask_image = image.get("image"), image.get("mask")
    pictures = {"image": image, "image_fg": source.get("image_fg"), "mask_image": mask_image, "mask_image_fg": source.get("mask_image_fg"), "generated_image": source.get("generated_image")}
    result: dict[str, Any] = {}
    for key, value in pictures.items():
        picture = _picture(value)
        if picture is not None:
            result[key] = store_upload(_alpha_layer(picture) if key.endswith("_fg") else picture)
    if "mask_image" in result or "mask_image_fg" in result:
        result["mask_upload"] = True
    if is_img2img and ("image" in result or (source.get("use_preview_as_input") and "generated_image" in result)):
        result["upload_independent"] = True
    return result


# endregion

# region Catalog (widget props)


def _slider(parameter) -> dict[str, Any]:
    kwargs = parameter.slider_kwargs
    entry = {key: kwargs[key] for key in ("label", "minimum", "maximum", "step", "value") if key in kwargs}
    entry["visible"] = bool(kwargs.get("visible", False))
    return {key: value for key, value in entry.items() if not (isinstance(value, float) and not math.isfinite(value))}


def default_choice(names: list[str], tag: str) -> str:
    """The legacy "Control Type" switch: the second entry of a filtered list
    (its first is "None"), the first for "All" or a one-entry list."""

    return names[1] if tag != "All" and len(names) > 1 else names[0]


def catalog() -> dict[str, Any]:
    """Preprocessors (sliders, flags), models and the control types' lists,
    for the widget (also the ``refresh_models`` action's output)."""

    preprocessors = []
    for name, preprocessor in global_state.get_sorted_preprocessors().items():
        preprocessors.append(
            {
                "name": name,
                "tags": list(preprocessor.tags),
                "resolution": _slider(preprocessor.slider_resolution),
                "slider_1": _slider(preprocessor.slider_1),
                "slider_2": _slider(preprocessor.slider_2),
                "control_mode": bool(preprocessor.show_control_mode),
                "model": not preprocessor.do_not_need_model,
            }
        )
    types = []
    for tag in global_state.get_all_preprocessor_tags():
        modules = global_state.get_all_preprocessor_names() if tag == "All" else global_state.get_filtered_preprocessor_names(tag)
        models = list(global_state.get_filtered_controlnet_names(tag))
        types.append({"name": tag, "modules": modules, "models": models, "default_module": default_choice(modules, tag), "default_model": default_choice(models, tag)})
    return {"preprocessors": preprocessors, "models": list(global_state.get_all_controlnet_names()), "types": types}


# endregion


class ControlNetUnitParam(ui.Custom):
    """One ControlNet unit (``index``) of a tab. The wire value is the unit
    JSON (see the module docstring); the processing argument a
    :class:`ControlNetUnit`. ``infotext`` is the unit's entry (``ControlNet
    <index>``): a paste sets the infotext fields (see :func:`pasted_fields`)
    and keeps the rest (images, "Enable")."""

    def __init__(self, name: str, index: int, *, is_img2img: bool, tab_titles: list[str], paste: bool = True, **common):
        self.index = index
        self.is_img2img = is_img2img
        controls = {key: control for key, control in CONTROLS.items() if is_img2img or key != "upload_independent"}

        def props() -> dict[str, Any]:  # resolved once per request
            # The catalog ships once per tab, with the first unit; the others read it from there
            extra = {"catalog": catalog()} if index == 0 else {}
            return {"unit": index, "img2img": is_img2img, "tabs": list(tab_titles), "controls": controls, "invert_hint": INVERT_HINT, **extra}

        super().__init__(name, WIDGET, default_value(is_img2img), props=props, infotext=f"ControlNet {index}" if paste else None, **common)

    # region Value protocol

    def coerce(self, value: Any, ctx: ui.Context) -> dict[str, Any]:
        return normalize(value, self.is_img2img)

    def from_paste(self, raw: Any, ctx: ui.Context) -> dict[str, Any]:
        if not isinstance(raw, str):
            raise ui.ParamError(f"expected the unit's infotext entry, got {type(raw).__name__}")
        return {PASTE_MARKER: True, **pasted_fields(raw)}

    def to_arg(self, value: dict[str, Any], ctx: ui.Context) -> ControlNetUnit:
        unit = ControlNetUnit(**{name: value[name] for name in SETTINGS})
        if not unit.enabled:
            return unit  # a disabled unit never reads (or blocks a job on) its uploads
        independent = not self.is_img2img or value.get("upload_independent", False)
        preview = independent and value["use_preview_as_input"] and value["generated_image"] is not None
        unit.use_preview_as_input = preview
        self._check(value, preview)
        if not independent:
            return unit  # the img2img image (and mask) are the input, as before

        image = self._load(ctx, value["image"])
        if image is not None:
            unit.image = rgba(with_sketch(image, self._load(ctx, value["sketch"])))
        fg = self._load(ctx, value["image_fg"])
        unit.image_fg = mask_layer(fg) if fg is not None else None
        if value["mask_upload"]:
            mask = self._load(ctx, value["mask_image"])
            unit.mask_image = mask_background(mask) if mask is not None else None
            mask_fg = self._load(ctx, value["mask_image_fg"])
            unit.mask_image_fg = mask_layer(mask_fg) if mask_fg is not None else None
        generated = self._load(ctx, value["generated_image"])
        unit.generated_image = rgba(generated) if generated is not None else None
        return unit

    def from_arg(self, arg: Any, ctx: ui.Context) -> Any:
        """The unit value of a ``ControlNetUnit`` (or an API unit dict): its
        settings and, when ``ctx.store_upload`` stores images (job
        importers), its pictures as uploads; without it they are dropped."""

        source = vars(arg) if isinstance(arg, ControlNetUnit) else arg
        if not isinstance(source, Mapping):
            return arg
        value = {name: _plain(source[name]) for name in SETTINGS if name in source}
        if ctx.store_upload is not None:
            value.update(imported_images(source, ctx.store_upload, self.is_img2img))
        return value

    # endregion

    def _load(self, ctx: ui.Context, file_id: str | None):
        """The upload's PIL image; ``None`` without one, or while only
        validating (the resolver then just checks the id)."""

        if not file_id or ctx.resolve_upload is None:
            return None
        image = ctx.resolve_upload(self, file_id)
        return image if _is_pil(image) else None

    def _check(self, value: dict[str, Any], preview: bool) -> None:
        """What processing would fail on, reported before the job runs."""

        module = "None" if preview else value["module"]
        preprocessor = global_state.supported_preprocessors.get(module)
        if preprocessor is None:
            raise ui.ParamError(f"unknown preprocessor {module!r}")
        if not preprocessor.do_not_need_model:
            if value["model"] == "None":
                raise ui.ParamError("select a ControlNet model")
            if value["model"] not in global_state.controlnet_filename_dict:
                raise ui.ParamError(f"ControlNet model {value['model']!r} was not found (refresh the model list)")
        if not self.is_img2img and not preview and not value["image"]:
            raise ui.ParamError("upload a control image")
