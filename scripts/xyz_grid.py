import csv
import logging
import os.path
import random
import re
from collections import namedtuple
from copy import copy
from io import StringIO
from itertools import chain, permutations
from math import sqrt
from typing import TypeVar

import numpy as np
from PIL import Image

import modules.scripts as scripts
import modules.sd_models
import modules.shared as shared
from backend.logging import setup_logger
from modules import (
    errors,
    images,
    processing,
    sd_models,
    sd_samplers,
    sd_schedulers,
    sd_vae,
)
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
    process_images,
)
from modules.sd_models import model_data, select_checkpoint
from modules.shared import cmd_opts, opts, state
from remi import ui

logger = logging.getLogger("xyz")
setup_logger(logger)


T = TypeVar("T")

AxisInfo = namedtuple("AxisInfo", ["axis", "values"])


# region Apply


def apply_field(field: str):
    def fun(p: StableDiffusionProcessing, x: T, xs: list[T]):
        setattr(p, field, x)

    return fun


def apply_prompt(p: StableDiffusionProcessing, x: str, xs: list[str]):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f'Prompt S/R did not find "{xs[0]}" in either prompt or negative prompt...')

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p: StableDiffusionProcessing, x: list[str], _):
    token_order = []

    # Initially grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token) :]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]

    p.prompt = prompt_tmp + p.prompt


def apply_checkpoint(p: StableDiffusionProcessing, x: str, _):
    if (info := modules.sd_models.get_closet_checkpoint_match(x)) is None:
        raise ValueError(f'Unknown Checkpoint: "{x}"')

    if info.name == p.override_settings.get("sd_model_checkpoint", None):
        return

    orig_ckpt = getattr(opts, "sd_model_checkpoint", None)

    opts.set("sd_model_checkpoint", info.name)
    refresh_loading_params_for_xyz_grid()
    opts.set("sd_model_checkpoint", orig_ckpt)


def apply_size(p: StableDiffusionProcessing, x: str, _):
    try:
        width, height = x.split("x")
        p.width = int(width.strip())
        p.height = int(height.strip())
    except ValueError:
        logger.error(f'Invalid Size "{x}" for X/Y/Z Plot')


def apply_vae(p: StableDiffusionProcessing, x: str, _):
    p.override_settings["sd_vae"] = find_vae(x)


def apply_styles(p: StableDiffusionProcessing, x: str, _):
    p.styles.extend(x.split(","))


def apply_uni_pc_order(p: StableDiffusionProcessing, x: int, _):
    p.override_settings["uni_pc_order"] = min(x, p.steps - 1)


def apply_face_restore(p: StableDiffusionProcessing, x: str, _):
    opt = x.lower()
    if opt == "codeformer":
        is_active = True
        p.face_restoration_model = "CodeFormer"
    elif opt == "gfpgan":
        is_active = True
        p.face_restoration_model = "GFPGAN"
    else:
        is_active = opt in ("true", "yes", "y", "1")

    p.restore_faces = is_active


def apply_override(field: str, boolean: bool = False):
    def fun(p: StableDiffusionProcessing, x: T, xs: list[T]):
        if boolean:
            p.override_settings[field] = True if x.lower() == "true" else False
        else:
            p.override_settings[field] = x

    return fun


# region Validation


def confirm_samplers(p: StableDiffusionProcessing, xs: list[str]):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            raise ValueError(f'Unknown Sampler: "{x}"')


def confirm_checkpoints(p: StableDiffusionProcessing, xs: list[str]):
    for x in xs:
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f'Unknown Checkpoint: "{x}"')


def confirm_checkpoints_or_none(p: StableDiffusionProcessing, xs: list[str]):
    for x in xs:
        if x in (None, "None", "none", ""):
            continue

        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f'Unknown Checkpoint: "{x}"')


def confirm_range(min_val: float, max_val: float, axis_label: str):
    """Generates a AxisOption.confirm() function that checks all values are within the specified range."""

    def confirm_range_fun(p: StableDiffusionProcessing, xs: list[float]):
        for x in xs:
            if not (max_val >= x >= min_val):
                raise ValueError(f'"{axis_label}" value "{x}" is out of range [{min_val}, {max_val}]')

    return confirm_range_fun


def refresh_loading_params_for_xyz_grid():
    """
    Refreshes the loading parameters for the model,
    causes a reload in sd_models.forge_model_reload()
    """
    from modules_forge.main_entry import forge_loading_parameters

    checkpoint_info = select_checkpoint()

    model_data.forge_loading_parameters = forge_loading_parameters(checkpoint_info, shared.opts.forge_unet_storage_dtype)


def find_vae(name: str) -> str:
    if name in (None, "None"):
        return "None"
    else:
        return sd_vae.vae_dict[name]


def boolean_choice(reverse: bool = False):
    def choice():
        return ["False", "True"] if reverse else ["True", "False"]

    return choice


def format_value_add_label(p: StableDiffusionProcessing, opt: "AxisOption", x: T) -> str:
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value(p: StableDiffusionProcessing, opt: "AxisOption", x: T) -> T:
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p: StableDiffusionProcessing, opt: "AxisOption", x: str) -> str:
    return ", ".join(x)


def do_nothing(*args, **kwargs):
    pass


def format_nothing(*args, **kwargs) -> str:
    return ""


def format_remove_path(p: StableDiffusionProcessing, opt: "AxisOption", x: os.PathLike) -> str:
    return os.path.basename(x)


def str_permutations(x: str) -> str:
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


def list_to_csv_string(data_list: list[str]) -> str:
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()


def csv_string_to_list_strip(data_str: str) -> list[str]:
    return list(map(str.strip, chain.from_iterable(csv.reader(StringIO(data_str), skipinitialspace=True))))


# region Options


class AxisOption:
    def __init__(self, label, type, apply, format_value=format_value_add_label, confirm=None, cost=0.0, choices=None, prepare=None):
        self.label = label
        self.type = type
        self.apply = apply
        self.format_value = format_value
        self.confirm = confirm
        self.cost = cost
        self.prepare = prepare
        self.choices = choices


class AxisOptionImg2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True


class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False


axis_options = [
    AxisOption("Nothing", str, do_nothing, format_value=format_nothing),
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOptionTxt2Img("Hires steps", int, apply_field("hr_second_pass_steps")),
    AxisOption("Size", str, apply_size),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOption("Distilled CFG Scale", float, apply_field("distilled_cfg_scale")),
    AxisOption("Shift", float, apply_field("distilled_cfg_scale")),
    AxisOption("Rescale CFG", float, apply_field("rescale_cfg")),
    AxisOption("MaHiRo", str, apply_field("mahiro"), choices=boolean_choice(reverse=True)),
    AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value),
    AxisOption("Prompt order", str_permutations, apply_order, format_value=format_value_join_list),
    AxisOptionTxt2Img("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers if x.name not in opts.hide_samplers]),
    AxisOptionTxt2Img("Hires sampler", str, apply_field("hr_sampler_name"), confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    AxisOptionImg2Img("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    AxisOption("Schedule type", str, apply_field("scheduler"), choices=lambda: [x.label for x in sd_schedulers.schedulers]),
    AxisOption("Checkpoint name", str, apply_checkpoint, format_value=format_remove_path, confirm=confirm_checkpoints, cost=1.0, choices=lambda: sorted(sd_models.checkpoints_list, key=str.casefold)),
    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ["None", *sorted(sd_vae.vae_dict.keys())]),
    AxisOption("Clip skip", int, apply_override("CLIP_stop_at_last_layers")),
    AxisOption("Denoising", float, apply_field("denoising_strength")),
    AxisOption("Initial noise multiplier", float, apply_field("initial_noise_multiplier")),
    AxisOption("Extra noise", float, apply_override("img2img_extra_noise")),
    AxisOptionTxt2Img("Hires upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    AxisOption("Styles", str, apply_styles, choices=lambda: list(shared.prompt_styles.styles)),
    AxisOption("Face restore", str, apply_face_restore, format_value=format_value),
    AxisOption("Negative Guidance minimum sigma", float, apply_field("s_min_uncond")),
    AxisOption("Token merging ratio", float, apply_override("token_merging_ratio")),
    AxisOption("Token merging ratio high-res", float, apply_override("token_merging_ratio_hr")),
    AxisOption("Refiner checkpoint", str, apply_field("refiner_checkpoint"), format_value=format_remove_path, confirm=confirm_checkpoints_or_none, cost=1.0, choices=lambda: ["None"] + sorted(sd_models.checkpoints_list, key=str.casefold)),
    AxisOption("Refiner switch at", float, apply_field("refiner_switch_at")),
    AxisOption("RNG source", str, apply_override("randn_source"), choices=lambda: ["GPU", "CPU", "NV"]),
]

if cmd_opts.adv_samplers:
    axis_options.extend(
        [
            AxisOption("Var. seed", int, apply_field("subseed")),
            AxisOption("Var. strength", float, apply_field("subseed_strength")),
            AxisOptionImg2Img("Image CFG Scale", float, apply_field("image_cfg_scale")),
            AxisOption("Sigma Churn", float, apply_field("s_churn")),
            AxisOption("Sigma min", float, apply_field("s_tmin")),
            AxisOption("Sigma max", float, apply_field("s_tmax")),
            AxisOption("Sigma noise", float, apply_field("s_noise")),
            AxisOption("Schedule min sigma", float, apply_override("sigma_min")),
            AxisOption("Schedule max sigma", float, apply_override("sigma_max")),
            AxisOption("Schedule rho", float, apply_override("rho")),
            AxisOption("Beta schedule alpha", float, apply_override("beta_dist_alpha")),
            AxisOption("Beta schedule beta", float, apply_override("beta_dist_beta")),
            AxisOption("Eta", float, apply_field("eta")),
            AxisOptionImg2Img("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight")),
            AxisOption("UniPC Order", int, apply_uni_pc_order, cost=0.5),
            AxisOption("Always discard next-to-last sigma", str, apply_override("always_discard_next_to_last_sigma", boolean=True), choices=boolean_choice(reverse=True)),
            AxisOption("SGM noise multiplier", str, apply_override("sgm_noise_multiplier", boolean=True), choices=boolean_choice(reverse=True)),
        ]
    )


# region Main


def draw_xyz_grid(p, xs, ys, zs, x_labels, y_labels, z_labels, cell, draw_legend, include_lone_images, include_sub_grids, first_axes_processed, second_axes_processed, margin_size):
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    title_texts = [[images.GridAnnotation(z)] for z in z_labels]

    list_size = len(xs) * len(ys) * len(zs)

    processed_result = None

    state.job_count = list_size * p.n_iter

    def process_cell(x, y, z, ix, iy, iz):
        nonlocal processed_result

        def index(ix, iy, iz):
            return ix + iy * len(xs) + iz * len(xs) * len(ys)

        state.job = f"{index(ix, iy, iz) + 1} out of {list_size}"

        processed: Processed = cell(x, y, z, ix, iy, iz)

        if processed_result is None:
            # Use our first processed result object as a template container to hold our full results
            processed_result = copy(processed)
            processed_result.images = [None] * list_size
            processed_result.all_prompts = [None] * list_size
            processed_result.all_seeds = [None] * list_size
            processed_result.infotexts = [None] * list_size
            processed_result.index_of_first_image = 1

        idx = index(ix, iy, iz)
        if processed.images:
            # Non-empty list indicates some degree of success.
            processed_result.images[idx] = processed.images[0]
            processed_result.all_prompts[idx] = processed.prompt
            processed_result.all_seeds[idx] = processed.seed
            processed_result.infotexts[idx] = processed.infotexts[0]
        else:
            cell_mode = "P"
            cell_size = (processed_result.width, processed_result.height)
            if processed_result.images[0] is not None:
                cell_mode = processed_result.images[0].mode
                # This corrects size in case of batches:
                cell_size = processed_result.images[0].size
            processed_result.images[idx] = Image.new(cell_mode, cell_size)

    if first_axes_processed == "x":
        for ix, x in enumerate(xs):
            if second_axes_processed == "y":
                for iy, y in enumerate(ys):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == "y":
        for iy, y in enumerate(ys):
            if second_axes_processed == "x":
                for ix, x in enumerate(xs):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == "z":
        for iz, z in enumerate(zs):
            if second_axes_processed == "x":
                for ix, x in enumerate(xs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iy, y in enumerate(ys):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)

    if not processed_result:
        # Should never happen, I've only seen it on one of four open tabs and it needed to refresh.
        logger.error("Processing could not begin, you may need to refresh the page or restart the WebUI")
        return Processed(p, [])
    elif not any(processed_result.images):
        logger.error("draw_xyz_grid failed to return any processed image...")
        return Processed(p, [])

    z_count = len(zs)

    for i in range(z_count):
        start_index = (i * len(xs) * len(ys)) + i
        end_index = start_index + len(xs) * len(ys)
        grid = images.image_grid(processed_result.images[start_index:end_index], rows=len(ys))
        if draw_legend:
            grid_max_w, grid_max_h = map(max, zip(*(img.size for img in processed_result.images[start_index:end_index])))
            grid = images.draw_grid_annotations(grid, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)
        processed_result.images.insert(i, grid)
        processed_result.all_prompts.insert(i, processed_result.all_prompts[start_index])
        processed_result.all_seeds.insert(i, processed_result.all_seeds[start_index])
        processed_result.infotexts.insert(i, processed_result.infotexts[start_index])

    z_grid = images.image_grid(processed_result.images[:z_count], rows=1)
    z_sub_grid_max_w, z_sub_grid_max_h = map(max, zip(*(img.size for img in processed_result.images[:z_count])))
    if draw_legend:
        z_grid = images.draw_grid_annotations(z_grid, z_sub_grid_max_w, z_sub_grid_max_h, title_texts, [[images.GridAnnotation()]])
    processed_result.images.insert(0, z_grid)
    # TODO: Deeper aspects of the program rely on grid info being misaligned between metadata arrays, which is not ideal.
    # processed_result.all_prompts.insert(0, processed_result.all_prompts[0])
    # processed_result.all_seeds.insert(0, processed_result.all_seeds[0])
    processed_result.infotexts.insert(0, processed_result.infotexts[0])

    return processed_result


class SharedSettingsStackHelper:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*])?\s*")


# region Script


AXES = ("x", "y", "z")
"""The axis prefixes of the params (``x_type``, ``x_values``, ``x_values_dropdown``, ...)."""


def axis_choices(axis: AxisOption) -> list[str] | None:
    """The values an axis offers (None: free text only)."""

    return None if axis.choices is None else list(axis.choices())


def read_choices(axis: AxisOption) -> list[str] | None:
    """:func:`axis_choices` for the UI: an axis whose list fails (an
    extension's model scan, ...) is logged and offers no values, without
    breaking the other axes."""

    try:
        return axis_choices(axis)
    except Exception:
        errors.report(f"X/Y/Z plot: could not list the values of {axis.label!r}", exc_info=True)
        return None


CHOICES_OUTPUT = "axis_choices"
"""Action output: ``{axis label: values | None}`` of every list axis; a
single axis' fresh list is the output ``axis_choices/<label>``."""


def select_axis(choices: list[str] | None, values: str, values_dropdown: list[str], csv_mode: bool) -> tuple[str, list[str]]:
    """``(values, values_dropdown)`` after choosing an axis or switching the
    text mode: an axis with choices moves the entries that are among them
    between the text box and the dropdown."""

    values_dropdown = list(values_dropdown or [])
    if choices is not None:
        if csv_mode:
            if values_dropdown:
                values = list_to_csv_string([x for x in values_dropdown if x in choices])
                values_dropdown = []
        elif values:
            values_dropdown = [x for x in csv_string_to_list_strip(values) if x in choices]
            values = ""
    return values, values_dropdown


def _value_list(value):
    """A dropdown value: a list of strings (numbers are written as text)."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("expected a list of values")
    if not all(isinstance(item, (str, int, float)) and not isinstance(item, bool) for item in value):
        raise ValueError("expected a list of strings")
    return [str(item) for item in value]


def _paste_dropdown(axis: str):
    def paste(params):
        values = params.get(f"{axis.upper()} Values")
        return None if values is None else csv_string_to_list_strip(str(values))

    return paste


class Script(scripts.Script):
    # The run() arguments: the axes first, then the options (not their visual order)
    param_order = (
        *(f"{axis}_{part}" for axis in AXES for part in ("type", "values", "values_dropdown")),
        "draw_legend",
        "include_lone_images",
        "include_sub_grids",
        "no_fixed_seeds",
        "vary_seeds_x",
        "vary_seeds_y",
        "vary_seeds_z",
        "row_count",
        "margin_size",
        "csv_mode",
    )

    listed_axes: frozenset[str] | None = None
    """Labels of the list axes when the layout was built: their dropdown
    visibility is fixed then. A list axis added later (after before_ui)
    is treated as a text axis, in the actions and in run()."""

    def title(self):
        return "X/Y/Z plot"

    @property
    def current_axis_options(self) -> list[AxisOption]:
        """This tab's axes; read when needed, so axes other extensions add to ``axis_options`` count."""

        return [x for x in axis_options if type(x) == AxisOption or x.is_img2img == self.is_img2img]

    def _axis(self, label) -> AxisOption | None:
        return next((x for x in self.current_axis_options if x.label == label), None)

    def is_listed(self, axis: AxisOption) -> bool:
        """Whether ``axis`` takes its values from the dropdown (else the text box)."""

        return axis.choices is not None and (self.listed_axes is None or axis.label in self.listed_axes)

    def params(self, is_img2img):
        options = [x for x in axis_options if type(x) == AxisOption or x.is_img2img == is_img2img]

        def labels():
            return [x.label for x in self.current_axis_options]

        # Conditions are static JSON: axes must be registered before the layout is built
        with_choices = [x.label for x in options if x.choices is not None]
        self.listed_axes = frozenset(with_choices)
        csv_mode = ui.param("csv_mode")

        def axis_row(axis: str, default: str):
            chooses = ui.param(f"{axis}_type", in_=with_choices)
            return ui.Row(
                # The script receives the axis index (Gradio's type="index"); the wire value is its label
                ui.Choice(f"{axis}_type", labels, default, type="index", label=f"{axis.upper()} type", infotext=f"{axis.upper()} Type", on_change=f"axis_{axis}", width=3),
                ui.Text(f"{axis}_values", "", label=f"{axis.upper()} values", infotext=f"{axis.upper()} Values", visible_if=ui.any_of(ui.not_(chooses), csv_mode), width="fill"),
                ui.Custom(
                    f"{axis}_values_dropdown",
                    "xyz-values",
                    [],
                    props={"axis": f"{axis}_type", "action": "axis_choices"},
                    label=f"{axis.upper()} values",
                    validate=_value_list,
                    paste=_paste_dropdown(axis),
                    visible_if=ui.all_of(chooses, ui.not_(csv_mode)),
                    width="fill",
                ),
                ui.Button("Fill values", f"fill_{axis}", "tool", visible_if=chooses),
            )

        return ui.Column(
            ui.Group(axis_row("x", options[1].label), axis_row("y", options[0].label), axis_row("z", options[0].label)),
            ui.Row(
                ui.Slider("row_count", 0, 8, 0, 1, label="Row Count", info="(set to 0 for auto)"),
                ui.Slider("margin_size", 0, 500, 0, 2, label="Grid Margins", info="(in pixels)"),
            ),
            ui.Row(
                ui.Column(
                    ui.Toggle("draw_legend", True, label="Draw legend"),
                    ui.Toggle("no_fixed_seeds", False, label="Keep -1 for seeds"),
                    ui.Row(
                        ui.Toggle("vary_seeds_x", False, label="Vary seeds for X", info="Use different seeds for images along X axis."),
                        ui.Toggle("vary_seeds_y", False, label="Vary seeds for Y", info="Use different seeds for images along Y axis."),
                        ui.Toggle("vary_seeds_z", False, label="Vary seeds for Z", info="Use different seeds for images along Z axis."),
                    ),
                    width=7,
                ),
                ui.Column(
                    ui.Toggle("include_lone_images", False, label="Include Sub Images"),
                    ui.Toggle("include_sub_grids", False, label="Include Sub Grids"),
                    ui.Toggle("csv_mode", False, label="Use text inputs instead of dropdowns", on_change="text_mode"),
                    width=5,
                ),
            ),
            ui.Row(
                ui.Button("Swap X/Y axes", "swap_xy"),
                ui.Button("Swap Y/Z axes", "swap_yz"),
                ui.Button("Swap X/Z axes", "swap_xz"),
            ),
        )

    # region Actions

    def _values(self, values: dict) -> dict:
        """The action's values over the defaults (a client may send only some)."""

        return {**(self.layout.defaults() if self.layout is not None else {}), **values}

    def _choices(self, values: dict, axis: str) -> tuple[AxisOption | None, list[str] | None]:
        """The selected axis of ``axis`` and its fresh list (None: a text axis or a failed list)."""

        option = self._axis(values.get(f"{axis}_type"))
        return option, (read_choices(option) if option is not None and self.is_listed(option) else None)

    def _fresh(self, option: AxisOption | None, choices: list[str] | None) -> dict:
        """Outputs refreshing the dropdown options of a list axis (the old UI re-read them on every axis change)."""

        return {f"{CHOICES_OUTPUT}/{option.label}": choices} if option is not None and self.is_listed(option) else {}

    def _select(self, values: dict, axis: str, csv_mode: bool, outputs: dict) -> dict:
        option, choices = self._choices(values, axis)
        outputs.update(self._fresh(option, choices))
        text, listed = select_axis(choices, values.get(f"{axis}_values") or "", values.get(f"{axis}_values_dropdown") or [], csv_mode)
        return {f"{axis}_values": ui.update(value=text), f"{axis}_values_dropdown": ui.update(value=listed)}

    def _select_axes(self, values: dict, axes: tuple[str, ...]) -> dict:
        values = self._values(values)
        csv_mode = bool(values.get("csv_mode"))
        outputs: dict = {}
        patch = {key: entry for axis in axes for key, entry in self._select(values, axis, csv_mode, outputs).items()}
        return {**patch, "__outputs__": outputs} if outputs else patch

    @ui.action
    def axis_x(self, ctx, values):
        return self._select_axes(values, ("x",))

    @ui.action
    def axis_y(self, ctx, values):
        return self._select_axes(values, ("y",))

    @ui.action
    def axis_z(self, ctx, values):
        return self._select_axes(values, ("z",))

    @ui.action
    def text_mode(self, ctx, values):
        return self._select_axes(values, AXES)

    @ui.action
    def axis_choices(self, ctx, values):
        """The values of every list axis, for the values dropdowns (``xyz-values``); a failing list is None."""

        choices = {x.label: read_choices(x) for x in self.current_axis_options if self.is_listed(x)}
        return {"__outputs__": {CHOICES_OUTPUT: choices}}

    def _fill(self, values: dict, axis: str) -> dict:
        option, choices = self._choices(values, axis)
        fresh = self._fresh(option, choices)
        patch = {"__outputs__": fresh} if fresh else {}
        if choices is None:
            return patch
        if values.get("csv_mode"):
            return {**patch, f"{axis}_values": ui.update(value=list_to_csv_string(choices))}
        return {**patch, f"{axis}_values_dropdown": ui.update(value=choices)}

    @ui.action
    def fill_x(self, ctx, values):
        values = self._values(values)
        return self._fill(values, "x")

    @ui.action
    def fill_y(self, ctx, values):
        values = self._values(values)
        return self._fill(values, "y")

    @ui.action
    def fill_z(self, ctx, values):
        values = self._values(values)
        return self._fill(values, "z")

    @staticmethod
    def _swap(values: dict, first: str, second: str) -> dict:
        patch = {}
        for part in ("type", "values", "values_dropdown"):
            patch[f"{first}_{part}"] = ui.update(value=values.get(f"{second}_{part}"))
            patch[f"{second}_{part}"] = ui.update(value=values.get(f"{first}_{part}"))
        return patch

    @ui.action
    def swap_xy(self, ctx, values):
        values = self._values(values)
        return self._swap(values, "x", "y")

    @ui.action
    def swap_yz(self, ctx, values):
        values = self._values(values)
        return self._swap(values, "y", "z")

    @ui.action
    def swap_xz(self, ctx, values):
        values = self._values(values)
        return self._swap(values, "x", "z")

    # endregion

    def run(self, p, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z, row_count, margin_size, csv_mode):
        x_type, y_type, z_type = x_type or 0, y_type or 0, z_type or 0  # if axle type is None set to 0

        if not no_fixed_seeds:
            modules.processing.fix_seed(p)

        if not opts.return_grid:
            p.batch_size = 1

        def process_axis(opt, vals, vals_dropdown):
            if opt.label == "Nothing":
                return [0]

            if self.is_listed(opt) and not csv_mode:
                valslist = vals_dropdown
            elif opt.prepare is not None:
                valslist = opt.prepare(vals)
            else:
                valslist = csv_string_to_list_strip(vals)

            if opt.type == int:
                valslist_ext = []

                for val in valslist:
                    if val.strip() == "":
                        continue
                    m = re_range.fullmatch(val)
                    mc = re_range_count.fullmatch(val)
                    if m is not None:
                        start = int(m.group(1))
                        end = int(m.group(2)) + 1
                        step = int(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += list(range(start, end, step))
                    elif mc is not None:
                        start = int(mc.group(1))
                        end = int(mc.group(2))
                        num = int(mc.group(3)) if mc.group(3) is not None else 1

                        valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []

                for val in valslist:
                    if val.strip() == "":
                        continue
                    m = re_range_float.fullmatch(val)
                    mc = re_range_count_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        step = float(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None:
                        start = float(mc.group(1))
                        end = float(mc.group(2))
                        num = int(mc.group(3)) if mc.group(3) is not None else 1

                        valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == str_permutations:
                valslist = list(permutations(valslist))

            valslist = [opt.type(x) for x in valslist]

            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)

            return valslist

        x_opt = self.current_axis_options[x_type]
        if self.is_listed(x_opt) and not csv_mode:
            x_values = list_to_csv_string(x_values_dropdown)
        xs = process_axis(x_opt, x_values, x_values_dropdown)

        y_opt = self.current_axis_options[y_type]
        if self.is_listed(y_opt) and not csv_mode:
            y_values = list_to_csv_string(y_values_dropdown)
        ys = process_axis(y_opt, y_values, y_values_dropdown)

        z_opt = self.current_axis_options[z_type]
        if self.is_listed(z_opt) and not csv_mode:
            z_values = list_to_csv_string(z_values_dropdown)
        zs = process_axis(z_opt, z_values, z_values_dropdown)

        # this could be moved to common code, but unlikely to be ever triggered anywhere else
        Image.MAX_IMAGE_PIXELS = None  # disable check in Pillow and rely on check below to allow large custom image sizes
        grid_mp = round(len(xs) * len(ys) * len(zs) * p.width * p.height / 1000000)
        assert grid_mp < opts.img_max_size_mp, f"Error: Resulting grid would be too large ({grid_mp} MPixels) (max configured size is {opts.img_max_size_mp} MPixels)"

        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ["Seed", "Var. seed"]:
                return [int(random.randrange(4294967294)) if val is None or val == "" or val == -1 else val for val in axis_list]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)
            zs = fix_axis_seeds(z_opt, zs)

        if x_opt.label == "Steps":
            total_steps = sum(xs) * len(ys) * len(zs)
        elif y_opt.label == "Steps":
            total_steps = sum(ys) * len(xs) * len(zs)
        elif z_opt.label == "Steps":
            total_steps = sum(zs) * len(xs) * len(ys)
        else:
            total_steps = p.steps * len(xs) * len(ys) * len(zs)

        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            if x_opt.label == "Hires steps":
                total_steps += sum(xs) * len(ys) * len(zs)
            elif y_opt.label == "Hires steps":
                total_steps += sum(ys) * len(xs) * len(zs)
            elif z_opt.label == "Hires steps":
                total_steps += sum(zs) * len(xs) * len(ys)
            elif p.hr_second_pass_steps:
                total_steps += p.hr_second_pass_steps * len(xs) * len(ys) * len(zs)
            else:
                total_steps *= 2

        total_steps *= p.n_iter

        image_cell_count = p.n_iter * p.batch_size
        cell_console_text = f"; {image_cell_count} images per cell" if image_cell_count > 1 else ""
        plural_s = "s" if len(zs) > 1 else ""
        logger.info(f"X/Y/Z plot will create {len(xs) * len(ys) * len(zs) * image_cell_count} images on {len(zs)} {len(xs)}x{len(ys)} grid{plural_s}{cell_console_text}. (Total steps to process: {total_steps})")
        shared.total_tqdm.updateTotal(total_steps)

        state.xyz_plot_x = AxisInfo(x_opt, xs)
        state.xyz_plot_y = AxisInfo(y_opt, ys)
        state.xyz_plot_z = AxisInfo(z_opt, zs)

        # If one of the axes is very slow to change between (like SD model
        # checkpoint), then make sure it is in the outer iteration of the nested
        # `for` loop.
        first_axes_processed = "z"
        second_axes_processed = "y"
        if x_opt.cost > y_opt.cost and x_opt.cost > z_opt.cost:
            first_axes_processed = "x"
            if y_opt.cost > z_opt.cost:
                second_axes_processed = "y"
            else:
                second_axes_processed = "z"
        elif y_opt.cost > x_opt.cost and y_opt.cost > z_opt.cost:
            first_axes_processed = "y"
            if x_opt.cost > z_opt.cost:
                second_axes_processed = "x"
            else:
                second_axes_processed = "z"
        elif z_opt.cost > x_opt.cost and z_opt.cost > y_opt.cost:
            first_axes_processed = "z"
            if x_opt.cost > y_opt.cost:
                second_axes_processed = "x"
            else:
                second_axes_processed = "y"

        grid_infotext = [None] * (1 + len(zs))

        def cell(x, y, z, ix, iy, iz):
            if shared.state.interrupted or state.stopping_generation:
                return Processed(p, [], p.seed, "")

            pc = copy(p)
            pc.styles = pc.styles[:]
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)
            z_opt.apply(pc, z, zs)

            xdim = len(xs) if vary_seeds_x else 1
            ydim = len(ys) if vary_seeds_y else 1

            if vary_seeds_x:
                pc.seed += ix
            if vary_seeds_y:
                pc.seed += iy * xdim
            if vary_seeds_z:
                pc.seed += iz * xdim * ydim

            try:
                res = process_images(pc)
            except Exception as e:
                errors.display(e, "generating image for xyz plot")

                res = Processed(p, [], p.seed, "")

            # Sets subgrid infotexts
            subgrid_index = 1 + iz
            if grid_infotext[subgrid_index] is None and ix == 0 and iy == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)
                pc.extra_generation_params["Script"] = self.title()

                if x_opt.label != "Nothing":
                    pc.extra_generation_params["X Type"] = x_opt.label
                    pc.extra_generation_params["X Values"] = x_values
                    if x_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed X Values"] = ", ".join([str(x) for x in xs])

                if y_opt.label != "Nothing":
                    pc.extra_generation_params["Y Type"] = y_opt.label
                    pc.extra_generation_params["Y Values"] = y_values
                    if y_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Y Values"] = ", ".join([str(y) for y in ys])

                grid_infotext[subgrid_index] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)

            # Sets main grid infotext
            if grid_infotext[0] is None and ix == 0 and iy == 0 and iz == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)

                if z_opt.label != "Nothing":
                    pc.extra_generation_params["Z Type"] = z_opt.label
                    pc.extra_generation_params["Z Values"] = z_values
                    if z_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Z Values"] = ", ".join([str(z) for z in zs])

                grid_infotext[0] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)

            return res

        with SharedSettingsStackHelper():
            processed = draw_xyz_grid(p, xs=xs, ys=ys, zs=zs, x_labels=[x_opt.format_value(p, x_opt, x) for x in xs], y_labels=[y_opt.format_value(p, y_opt, y) for y in ys], z_labels=[z_opt.format_value(p, z_opt, z) for z in zs], cell=cell, draw_legend=draw_legend, include_lone_images=include_lone_images, include_sub_grids=include_sub_grids, first_axes_processed=first_axes_processed, second_axes_processed=second_axes_processed, margin_size=margin_size)

        # reset loading params to previous state
        refresh_loading_params_for_xyz_grid()

        if not processed.images:
            # It broke, no further handling needed.
            return processed

        z_count = len(zs)

        # Set the grid infotexts to the real ones with extra_generation_params (1 main grid + z_count sub-grids)
        processed.infotexts[: 1 + z_count] = grid_infotext[: 1 + z_count]

        if not include_lone_images:
            # Don't need sub-images anymore, drop from list:
            processed.images = processed.images[: z_count + 1]

        def rearrange_image(original_image: Image.Image, target_count: int):
            width, height = original_image.size
            width_per_image = int(width // len(xs))

            imgs_per_row = (len(xs) // target_count) + int(len(xs) % target_count != 0)
            logger.info(f"Converting to {imgs_per_row}x{target_count} grid...")

            new_width = width_per_image * imgs_per_row
            new_height = height * target_count

            new_image = Image.new("RGB", (new_width, new_height), "white")

            for y in range(target_count):
                row = original_image.crop(
                    (
                        y * (width_per_image * imgs_per_row),
                        0,
                        min(width, ((y + 1) * (width_per_image * imgs_per_row))),
                        height,
                    )
                )
                new_image.paste(row, (0, height * y))

            return new_image

        if row_count == 0:  # Automatic
            if y_type + z_type > 0:
                target_count = 1
            else:
                target_count = int(sqrt(len(xs)))
                if target_count > 1:
                    logger.debug(f"Automatically split into {target_count} rows")
        else:
            if y_type + z_type == 0:
                target_count = row_count
            else:
                target_count = 1
                if row_count > 1:
                    logger.error("Row Count currently only supports X-Axis...")

        if target_count > 1:
            processed.images[0] = rearrange_image(processed.images[0], target_count)

        if opts.grid_save:
            # Auto-save main and sub-grids:
            grid_count = z_count + 1 if z_count > 1 else 1
            for g in range(grid_count):
                # TODO: See previous comment about intentional data misalignment.
                adj_g = g - 1 if g > 0 else g
                images.save_image(processed.images[g], p.outpath_grids, "xyz_grid", info=processed.infotexts[g], extension=opts.grid_format, prompt=processed.all_prompts[adj_g], seed=processed.all_seeds[adj_g], grid=True, p=processed)
                if not include_sub_grids:  # if not include_sub_grids then skip saving after the first grid
                    break

        if not include_sub_grids:
            # Done with sub-grids, drop all related information:
            for _ in range(z_count):
                del processed.images[1]
                del processed.all_prompts[1]
                del processed.all_seeds[1]
                del processed.infotexts[1]

        return processed
