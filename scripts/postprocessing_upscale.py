import re

import numpy as np
from PIL import Image

from modules import scripts_postprocessing, shared
from modules.processing import apply_color_correction, setup_color_correction
from remi import ui

upscale_cache = {}


def limit_size_by_one_dimension(w: int, h: int, limit: int) -> tuple[int, int]:
    if h > w and h > limit:
        w = limit * w // h
        h = limit
    elif w > limit:
        h = limit * h // w
        w = limit

    return (int(w), int(h))


def upscaler_names() -> list[str]:
    return [x.name for x in shared.sd_upscalers]


def scale_from_name(name: str) -> int | None:
    """The scale an upscaler's name states (``4x-UltraSharp``, ``RealESRGAN_x2``)."""

    match = re.search(r"(\d)[xX]|[xX](\d)", name or "")
    return int(match.group(1) or match.group(2)) if match else None


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Upscale"
    order = 1000

    def params(self):
        # A None default is the first upscaler
        return ui.Section(
            "Upscale",
            enable=ui.Toggle("upscale_enabled", True, label="Upscale"),
            open=True,
            children=[
                # The selected tab: 0 = scale by, 1 = scale to (Gradio kept it in a hidden number)
                ui.Choice("upscale_mode", [("Scale by", 0), ("Scale to", 1)], 0, label="Resize mode", hidden=True),
                ui.Tabs(
                    ui.Tab(
                        "Scale by",
                        ui.Slider("upscale_by", 1.0, 8.0, 4.0, 0.05, label="Resize"),
                        ui.Slider("max_side_length", 0, 8192, 0, 8, label="Max Side Length"),
                        value=0,
                    ),
                    ui.Tab(
                        "Scale to",
                        ui.Row(
                            ui.Column(
                                ui.Slider("upscale_to_width", 64, 8192, 1024, 8, label="Width"),
                                ui.Slider("upscale_to_height", 64, 8192, 1024, 8, label="Height"),
                                width=9,
                            ),
                            ui.Column(
                                ui.Button("Switch width/height", "swap_size", "tool"),
                                ui.Toggle("upscale_crop", True, label="Crop to fit"),
                                width=3,
                            ),
                        ),
                        value=1,
                    ),
                    bind="upscale_mode",
                ),
                # One column per upscaler: "Upscaler 2 visibility" belongs to "Upscaler 2" and stays
                # under it, also when a narrow window stacks the columns (each then spans the row);
                # "Color Correction" fills the room under "Upscaler 1"; Refresh reloads both lists.
                ui.Row(
                    ui.Column(
                        ui.Choice("upscaler_1_name", upscaler_names, None, label="Upscaler 1", on_change="upscaler_scale"),
                        ui.Toggle("upscale_cc", False, label="Color Correction"),
                    ),
                    ui.Column(
                        ui.Choice(
                            "upscaler_2_name",
                            upscaler_names,
                            None,
                            label="Upscaler 2",
                            enabled_if={"param": "upscaler_2_visibility", "gt": 0.04},
                        ),
                        ui.Slider("upscaler_2_visibility", 0.0, 1.0, 0.0, 0.05, label="Upscaler 2 visibility"),
                    ),
                ),
                ui.Button("Refresh", "refresh_upscalers", "tool"),
            ],
        )

    @ui.action
    def swap_size(self, ctx, values):
        width, height = values.get("upscale_to_width", 1024), values.get("upscale_to_height", 1024)
        return {"upscale_to_width": ui.update(value=height), "upscale_to_height": ui.update(value=width)}

    @ui.action
    def refresh_upscalers(self, ctx, values):
        from modules.modelloader import load_upscalers

        load_upscalers()
        if ctx.request is not None:
            # Other tabs' upscaler choices (Hires. fix, SD upscale) live in their schemas, which
            # clients refetch on options.changed, as after POST /models/refresh.
            from remi.server.deps import get_services, publish_options_changed

            publish_options_changed(get_services(ctx.request), ["upscalers"])
        choices = upscaler_names()
        return {"upscaler_1_name": ui.update(choices=choices), "upscaler_2_name": ui.update(choices=choices)}

    @ui.action
    def upscaler_scale(self, ctx, values):
        """With ``set_scale_by_when_changing_upscaler``, "Resize" follows the scale in the upscaler's name."""

        scale = scale_from_name(values.get("upscaler_1_name")) if shared.opts.set_scale_by_when_changing_upscaler else None
        return {} if scale is None else {"upscale_by": ui.update(value=scale)}

    def _upscale(
        self,
        image: Image.Image,
        info: dict,
        upscaler,
        upscale_mode: int,
        upscale_by: float,
        max_side_length: int,
        upscale_to_width: int,
        upscale_to_height: int,
        upscale_crop: bool,
    ):
        if upscale_mode == 1:
            upscale_by = max(upscale_to_width / image.width, upscale_to_height / image.height)
            info["Postprocess upscale to"] = f"{upscale_to_width}x{upscale_to_height}"
        else:
            info["Postprocess upscale by"] = upscale_by
            if max_side_length != 0 and max(*image.size) * upscale_by > max_side_length:
                upscale_mode = 1
                upscale_crop = False
                upscale_to_width, upscale_to_height = limit_size_by_one_dimension(image.width * upscale_by, image.height * upscale_by, max_side_length)
                upscale_by = max(upscale_to_width / image.width, upscale_to_height / image.height)
                info["Max side length"] = max_side_length

        cache_key = (hash(np.array(image.getdata()).tobytes()), upscaler.name, upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop)
        cached_image = upscale_cache.pop(cache_key, None)

        if cached_image is not None:
            image = cached_image
        else:
            image = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)

        upscale_cache[cache_key] = image
        if len(upscale_cache) > shared.opts.upscaling_max_images_in_cache:
            upscale_cache.pop(next(iter(upscale_cache), None), None)

        if upscale_mode == 1 and upscale_crop:
            cropped = Image.new("RGB", (upscale_to_width, upscale_to_height))
            cropped.paste(image, box=(upscale_to_width // 2 - image.width // 2, upscale_to_height // 2 - image.height // 2))
            image = cropped
            info["Postprocess crop to"] = f"{image.width}x{image.height}"

        return image

    def process_firstpass(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        upscale_enabled=True,
        upscale_cc=False,
        upscale_mode=1,
        upscale_by=2.0,
        max_side_length=0,
        upscale_to_width=None,
        upscale_to_height=None,
        upscale_crop=False,
        upscaler_1_name=None,
        upscaler_2_name=None,
        upscaler_2_visibility=0.0,
    ):
        if upscale_mode == 1:
            pp.shared.target_width = upscale_to_width
            pp.shared.target_height = upscale_to_height
        else:
            pp.shared.target_width = int(pp.image.width * upscale_by)
            pp.shared.target_height = int(pp.image.height * upscale_by)

            pp.shared.target_width, pp.shared.target_height = limit_size_by_one_dimension(pp.shared.target_width, pp.shared.target_height, max_side_length)

        if upscale_cc:
            upscale_cache["cc"] = setup_color_correction(pp.image)

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        upscale_enabled=True,
        upscale_cc=False,
        upscale_mode=1,
        upscale_by=2.0,
        max_side_length=0,
        upscale_to_width=None,
        upscale_to_height=None,
        upscale_crop=False,
        upscaler_1_name=None,
        upscaler_2_name=None,
        upscaler_2_visibility=0.0,
    ):
        if not upscale_enabled:
            return

        _upscaler_1_name = None if upscaler_1_name == "None" else upscaler_1_name

        upscaler1 = next(iter([x for x in shared.sd_upscalers if x.name == _upscaler_1_name]), None)
        assert upscaler1 or (_upscaler_1_name is None), f"could not find upscaler named {_upscaler_1_name}"

        if not upscaler1:
            return

        _upscaler_2_name = None if upscaler_2_name == "None" else upscaler_2_name

        upscaler2 = next(iter([x for x in shared.sd_upscalers if x.name == _upscaler_2_name and x.name != "None"]), None)
        assert upscaler2 or (_upscaler_2_name is None), f"could not find upscaler named {_upscaler_2_name}"

        upscaled_image = self._upscale(pp.image, pp.info, upscaler1, upscale_mode, upscale_by, max_side_length, upscale_to_width, upscale_to_height, upscale_crop)
        pp.info["Postprocess upscaler"] = upscaler1.name

        if upscaler2 and upscaler_2_visibility > 0:
            second_upscale = self._upscale(pp.image, pp.info, upscaler2, upscale_mode, upscale_by, max_side_length, upscale_to_width, upscale_to_height, upscale_crop)
            if upscaled_image.mode != second_upscale.mode:
                second_upscale = second_upscale.convert(upscaled_image.mode)
            upscaled_image = Image.blend(upscaled_image, second_upscale, upscaler_2_visibility)

            pp.info["Postprocess upscaler 2"] = upscaler2.name

        if upscale_cc and "cc" in upscale_cache:
            pp.image = apply_color_correction(upscale_cache["cc"], upscaled_image)
        else:
            pp.image = upscaled_image

    def image_changed(self):
        upscale_cache.clear()
