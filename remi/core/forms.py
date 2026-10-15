"""The generation forms (owner ``core``) of the txt2img, img2img and extras tabs.

Derived from what ``modules/ui.py`` and ``modules/ui_postprocessing.py``
built: the same params, labels, bounds, defaults and infotext keys, laid out
in the user's category order (``opts.ui_reorder_list``) with a
:class:`~remi.ui.Slot` where the scripts of each category go. Param names
are the processing parameter names (see ``modules/txt2img.py`` /
``modules/img2img.py`` ``*_processing``).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from remi import ui

if TYPE_CHECKING:
    from modules.scripts import ScriptRunner

TABS = ("txt2img", "img2img", "extras")

PROMPT_PLACEHOLDER = "Prompt\n(Ctrl+Enter to Generate ; Alt+Enter to Skip ; Esc to Interrupt)"
NEGATIVE_PLACEHOLDER = "Negative Prompt\n(Ctrl+Enter to Generate ; Alt+Enter to Skip ; Esc to Interrupt)"

IMG2IMG_MODES = ("img2img", "Sketch", "Inpaint", "Inpaint sketch", "Inpaint upload", "Batch")
"""Values of the img2img ``mode`` param; the processing argument is the index."""
INPAINT_MODES = ("Inpaint", "Inpaint sketch", "Inpaint upload")
RESIZE_MODES = ("Just Resize", "Crop and Resize", "Resize and Fill", "Latent Upscale", "Preserve Aspect Ratio")
RESIZE_TABS = ("Resize to", "Resize by")
MASK_MODES = ("Inpaint masked", "Inpaint not masked")
MASKED_CONTENT = ("fill", "original", "latent noise", "latent nothing")
INPAINT_AREAS = ("Whole picture", "Only masked")
BATCH_SOURCES = (("Upload", "upload"), ("From directory", "from dir"))
PNG_INFO_PROPS = ("Prompt", "Negative prompt", "Seed", "CFG scale", "Sampler", "Steps", "Model hash", "Filename")
EXTRAS_MODES = ("Single Image", "Batch Process", "Batch from Directory", "Single Video")

SAME_CHECKPOINT = "Use same checkpoint"
SAME_MODULES = "Use same choices"
SAME_SAMPLER = "Use same sampler"
SAME_SCHEDULER = "Use same scheduler"


# region Resolution helpers (Gradio-free equivalents of modules.ui's)


def resolution_step() -> int:
    from modules import shared

    return int(getattr(shared.opts, "res_step", 64) or 64)


def snap_resolution(value: float) -> int:
    """``modules.ui.sRound``: the nearest multiple of the resolution step."""

    step = resolution_step()
    return math.floor(value / step + 0.5) * step


# endregion

# region Dynamic choices


def _style_names() -> list[str]:
    from modules import shared

    return list(shared.prompt_styles.styles) if shared.prompt_styles is not None else []


def _upscaler_names() -> list[str]:
    from modules import shared

    return [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]


def _hr_checkpoints() -> list[str]:
    from modules import shared

    return [SAME_CHECKPOINT, *sorted(shared.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short))]


def _hr_modules() -> list[str]:
    from modules import shared

    # The module list main_entry scanned, without importing it (it loads Gradio)
    main_entry = sys.modules.get("modules_forge.main_entry")
    if main_entry is None:
        return [SAME_MODULES]
    modules = main_entry.module_list.keys() if shared.opts.sd_modules_dropdown_use_short else main_entry.module_map.values()
    return [SAME_MODULES, *sorted(modules)]


def _hr_samplers() -> list[str]:
    from modules import sd_samplers

    return [SAME_SAMPLER, *sd_samplers.visible_sampler_names()]


def _hr_schedulers() -> list[str]:
    from modules import sd_schedulers

    return [SAME_SCHEDULER, *[x.label for x in sd_schedulers.schedulers]]


# endregion

# region Paste functions


def _paste_styles(params: dict):
    styles = params.get("Styles array")
    return styles if isinstance(styles, list) else None


def _paste_enable_hr(params: dict) -> bool:
    return "Denoising strength" in params and ("Hires upscale" in params or "Hires upscaler" in params or "Hires resize-1" in params)


def _paste_hr_sampler(params: dict):
    from modules import sd_samplers

    return sd_samplers.get_hr_sampler_from_infotext(params)


def _paste_hr_scheduler(params: dict):
    from modules import sd_samplers

    return sd_samplers.get_hr_scheduler_from_infotext(params)


# endregion

# region Shared pieces


def _prompts(tab: str) -> ui.Node:
    return ui.Group(
        ui.Text("prompt", "", lines=3, prompt=True, label="Prompt", placeholder=PROMPT_PLACEHOLDER, infotext="Prompt"),
        ui.Text("negative_prompt", "", lines=3, prompt=True, label="Negative Prompt", placeholder=NEGATIVE_PLACEHOLDER, infotext="Negative prompt", enabled_if=ui.param("cfg_scale", gt=1)),
        ui.MultiChoice("styles", _style_names, [], label="Styles", paste=_paste_styles),
    )


def _batch() -> list[ui.Node]:
    return [
        ui.Slider("batch_count", 1, 128, 1, 1, label="Batch Count"),
        ui.Slider("batch_size", 1, 8, 1, 1, label="Batch Size", infotext="Batch size"),
    ]


def _cfg(tab: str) -> list[ui.Node]:
    return [
        ui.Slider("distilled_cfg_scale", 0.0 if tab == "img2img" else 1.0, 24.0, 3.0, 0.5, label="Distilled CFG Scale", infotext="Distilled CFG Scale", preset="native.distilled_cfg"),
        ui.Slider("cfg_scale", 1.0, 24.0, 6.0, 0.5, label="CFG Scale", infotext="CFG scale", preset="native.cfg"),
    ]


def _dimensions(tab: str) -> list[ui.Input]:
    step = resolution_step()
    # txt2img: the Hires. fix target resolution follows the size
    on_change = "hr_resolution" if tab == "txt2img" else None
    return [
        ui.Slider("width", 64, 2048, 1024, step, label="Width", infotext="Size-1", on_change=on_change),
        ui.Slider("height", 64, 2048, 1024, step, label="Height", infotext="Size-2", on_change=on_change),
    ]


def _size(tab: str) -> list[ui.Node]:
    return [*_dimensions(tab), ui.Button("Switch width/height", "swap_dimensions", "tool")]


def _enable_hr() -> ui.Toggle:
    return ui.Toggle("enable_hr", False, label="Hires. fix", paste=_paste_enable_hr, preset="native.hires", exclusive=("pid_hires", "hires"), on_change="hr_resolution")


def _hr_target() -> list[ui.Input]:
    step = resolution_step()
    return [
        ui.Slider("hr_scale", 1.0, 4.0, 2.0, 0.05, label="Upscale by", infotext="Hires upscale", on_change="hr_resolution"),
        ui.Slider("hr_resize_x", 0, 4096, 0, step, label="Resize width to", infotext="Hires resize-1", on_change="hr_resolution"),
        ui.Slider("hr_resize_y", 0, 4096, 0, step, label="Resize height to", infotext="Hires resize-2", on_change="hr_resolution"),
    ]


def dimensions_layout(tab: str) -> ui.ScriptLayout:
    """Just ``width``/``height`` of ``tab``'s form (for the core actions)."""

    return ui.ScriptLayout(ui.Group(*_dimensions(tab)), owner="core")


def hires_resolution_layout() -> ui.ScriptLayout:
    """The txt2img params the Hires. fix resolution line reads, declared as
    in the form, without building the whole form."""

    return ui.ScriptLayout(ui.Group(_enable_hr(), *_dimensions("txt2img"), *_hr_target()), owner="core")


def _override_settings() -> ui.Node:
    # Filled by infotext paste (settings the infotext differs in); shown when not empty
    return ui.MultiChoice("override_settings", [], [], allow_custom=True, label="Override settings", visible_if=ui.param("override_settings"))


def _script_selector(runner: ScriptRunner | None) -> ui.Node:
    choices = runner.selector_choices if runner is not None else ["None"]
    return ui.Group(ui.Choice("script", choices, "None", label="Script", paste=lambda params: params.get("Script")), ui.Slot("scripts"))


def _hires_fix() -> ui.Node:
    from modules import shared

    show_sampler = bool(getattr(shared.opts, "hires_fix_show_sampler", False))
    show_prompts = bool(getattr(shared.opts, "hires_fix_show_prompts", False))
    # Shown by the option, or when a pasted infotext sets them
    sampler_rows = None if show_sampler else ui.any_of(ui.param("hr_checkpoint_name", ne=SAME_CHECKPOINT), ui.param("hr_sampler_name", ne=SAME_SAMPLER), ui.param("hr_scheduler", ne=SAME_SCHEDULER))
    prompt_rows = None if show_prompts else ui.any_of(ui.param("hr_prompt"), ui.param("hr_negative_prompt"))
    return ui.Section(
        "Hires. fix",
        enable=_enable_hr(),
        children=[
            ui.Output("hr_final_resolution", "html", "Upscaled resolution"),
            ui.Row(
                ui.Choice("hr_upscaler", _upscaler_names, "Latent", label="Upscaler", infotext="Hires upscaler"),
                ui.Slider("hr_second_pass_steps", 0, 150, 0, 1, label="Hires steps", infotext="Hires steps"),
                ui.Slider("denoising_strength", 0.0, 1.0, 0.6, 0.05, label="Denoising strength", infotext="Denoising strength"),
            ),
            ui.Row(*_hr_target()),
            ui.Row(
                ui.Slider("hr_distilled_cfg", 1.0, 24.0, 3.0, 0.5, label="Hires Distilled CFG Scale", infotext="Hires Distilled CFG Scale"),
                ui.Slider("hr_cfg", 1.0, 24.0, 6.0, 0.5, label="Hires CFG Scale", infotext="Hires CFG Scale"),
            ),
            ui.Group(
                ui.Row(
                    ui.Choice("hr_checkpoint_name", _hr_checkpoints, SAME_CHECKPOINT, label="Hires Checkpoint", infotext="Hires checkpoint"),
                    ui.MultiChoice("hr_additional_modules", _hr_modules, [SAME_MODULES], label="Hires VAE / Text Encoder", infotext="Hires VAE/TE"),
                ),
                ui.Row(
                    ui.Choice("hr_sampler_name", _hr_samplers, SAME_SAMPLER, label="Hires sampling method", paste=_paste_hr_sampler),
                    ui.Choice("hr_scheduler", _hr_schedulers, SAME_SCHEDULER, label="Hires schedule type", paste=_paste_hr_scheduler),
                ),
                visible_if=sampler_rows,
            ),
            ui.Row(
                ui.Text("hr_prompt", "", lines=3, prompt=True, label="Hires prompt", placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.", infotext="Hires prompt"),
                ui.Text("hr_negative_prompt", "", lines=3, prompt=True, label="Hires negative prompt", placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.", infotext="Hires negative prompt", enabled_if=ui.param("hr_cfg", gt=1)),
                visible_if=prompt_rows,
            ),
        ],
    )


def ordered_categories() -> list[str]:
    from modules.scripts import ordered_ui_categories

    return list(ordered_ui_categories())


# endregion

# region Forms


def txt2img_form(runner: ScriptRunner | None = None) -> ui.Node:
    from modules import shared

    together = bool(getattr(shared.opts, "dimensions_and_batch_together", True))
    sections: list[ui.Node] = []
    for category in ordered_categories():
        if category == "prompt":
            sections.append(_prompts("txt2img"))
        elif category == "dimensions":
            sections.append(ui.Column(ui.Row(*_size("txt2img")), ui.Row(*_batch()) if together else None, ui.Slot(category)))
        elif category == "cfg":
            sections.append(ui.Row(*_cfg("txt2img"), ui.Slot(category)))
        elif category == "accordions":
            # stacked, like the Gradio accordions row (each panel spans the form)
            sections.append(ui.Column(_hires_fix(), ui.Slot(category)))
        elif category == "batch":
            sections.append(ui.Row(*(() if together else _batch()), ui.Slot(category)))
        elif category == "override_settings":
            sections.append(ui.Row(_override_settings(), ui.Slot(category)))
        elif category == "scripts":
            sections.append(_script_selector(runner))
        elif category in ("image", "inpaint", "denoising"):
            continue  # img2img only
        else:
            sections.append(ui.Slot(category))
    return ui.Column(*sections)


def _img2img_image() -> ui.Node:
    from modules import shared

    hide_dirs = bool(shared.cmd_opts.hide_ui_dir_config)
    not_batch = ui.param("mode", ne="Batch")
    directory_help = (
        "Process images in a directory on the same machine where the server is running.\n"
        "Use an empty output directory to save pictures normally instead of writing to the output directory.\n"
        "Add inpaint batch mask directory to enable inpaint batch processing."
        + ("\nDisabled when launched with --hide-ui-dir-config." if hide_dirs else "")
    )
    batch = ui.Tab(
        "Batch",
        ui.Choice("batch_source", BATCH_SOURCES, "upload", label="Batch source", hidden=True),
        ui.Tabs(
            ui.Tab("Upload", ui.ImageList("batch_upload", label="Files"), value="upload"),
            ui.Tab(
                "From directory",
                ui.Markdown(directory_help),
                ui.Text("batch_input_dir", "", label="Input directory", hidden=hide_dirs),
                ui.Text("batch_output_dir", "", label="Output directory", hidden=hide_dirs),
                ui.Text("batch_inpaint_mask_dir", "", label="Inpaint batch mask directory (required for inpaint batch processing only)", hidden=hide_dirs),
                value="from dir",
            ),
            bind="batch_source",
        ),
        ui.Section(
            "Append PNG Info",
            enable=ui.Toggle("batch_use_png_info", False, label="Append PNG Info"),
            children=[
                ui.Text("batch_png_info_dir", "", label="PNG info directory", placeholder="Leave empty to use input directory", hidden=hide_dirs),
                ui.MultiChoice("batch_png_info_props", PNG_INFO_PROPS, [], style="checkboxes", label="Parameters to take from png info", info="Prompts from png info will be appended to prompts set in ui."),
            ],
        ),
        value="Batch",
    )
    return ui.Group(
        ui.Choice("mode", IMG2IMG_MODES, "img2img", type="index", label="Mode", hidden=True),
        ui.Tabs(*[ui.Tab(mode, value=mode) for mode in IMG2IMG_MODES[:-1]], batch, bind="mode"),
        # One canvas for every mode: the image, the color sketch over it, the
        # mask painted over it; "Inpaint upload" takes a mask image instead
        ui.Image("init_img", "image", mode="RGBA", label="Image", visible_if=not_batch),
        ui.Image("sketch", "sketch", "init_img", mode="RGBA", label="Sketch", visible_if=ui.param("mode", in_=["Sketch", "Inpaint sketch"])),
        ui.Image("mask", "mask", "init_img", label="Mask", visible_if=ui.param("mode", eq="Inpaint")),
        ui.Image("mask_upload", "mask", label="Mask", visible_if=ui.param("mode", eq="Inpaint upload")),
        ui.Choice("resize_mode", RESIZE_MODES, "Crop and Resize", type="index", style="radio", label="Resize Mode"),
    )


def _img2img_inpaint() -> ui.Node:
    return ui.Group(
        ui.Slider("mask_blur", 0, 64, 4, 4, label="Mask blur", infotext="Mask blur"),
        ui.Choice("inpainting_mask_invert", MASK_MODES, "Inpaint masked", type="index", style="radio", label="Mask mode", infotext="Mask mode"),
        ui.Choice("inpainting_fill", MASKED_CONTENT, "original", type="index", style="radio", label="Masked content", infotext="Masked content"),
        ui.Row(
            ui.Choice("inpaint_full_res", INPAINT_AREAS, "Only masked", type="index", style="radio", label="Inpaint area", infotext="Inpaint area"),
            ui.Slider("inpaint_full_res_padding", 0, 256, 32, 8, label="Only masked padding, pixels", infotext="Masked area padding"),
        ),
        visible_if=ui.param("mode", in_=list(INPAINT_MODES)),
    )


def img2img_form(runner: ScriptRunner | None = None) -> ui.Node:
    from modules import shared

    together = bool(getattr(shared.opts, "dimensions_and_batch_together", True))
    sections: list[ui.Node] = []
    for category in ordered_categories():
        if category == "prompt":
            sections.append(_prompts("img2img"))
        elif category == "image":
            sections.append(ui.Group(_img2img_image(), ui.Slot(category)))
        elif category == "inpaint":
            sections.append(ui.Group(_img2img_inpaint(), ui.Slot(category)))
        elif category == "dimensions":
            sections.append(
                ui.Column(
                    ui.Choice("resize_tab", RESIZE_TABS, "Resize to", type="index", label="Resize", hidden=True),
                    ui.Tabs(
                        ui.Tab("Resize to", ui.Row(*_size("img2img")), value="Resize to"),
                        ui.Tab("Resize by", ui.Slider("scale_by", 0.5, 4.0, 1.0, 0.05, label="Scale"), ui.Output("scale_by_resolution", "html"), value="Resize by"),
                        bind="resize_tab",
                    ),
                    ui.Row(*_batch()) if together else None,
                    ui.Slot(category),
                )
            )
        elif category == "cfg":
            sections.append(ui.Row(*_cfg("img2img"), ui.Slider("image_cfg_scale", 0, 3.0, 1.5, 0.05, label="Image CFG Scale", infotext="Image CFG scale", hidden=True), ui.Slot(category)))
        elif category == "denoising":
            sections.append(ui.Group(ui.Slider("denoising_strength", 0.0, 1.0, 0.6, 0.01, label="Denoising Strength", infotext="Denoising strength"), ui.Slot(category)))
        elif category == "accordions":
            sections.append(ui.Column(ui.Slot(category)))
        elif category == "batch":
            sections.append(ui.Row(*(() if together else _batch()), ui.Slot(category)))
        elif category == "override_settings":
            sections.append(ui.Row(_override_settings(), ui.Slot(category)))
        elif category == "scripts":
            sections.append(_script_selector(runner))
        else:
            sections.append(ui.Slot(category))
    return ui.Column(*sections)


def extras_form() -> ui.Node:
    from modules import shared

    hide_dirs = bool(shared.cmd_opts.hide_ui_dir_config)
    return ui.Column(
        ui.Choice("mode", EXTRAS_MODES, "Single Image", type="index", label="Mode", hidden=True),
        ui.Tabs(
            ui.Tab("Single Image", ui.Image("image", mode="RGBA", label="Source"), value="Single Image"),
            ui.Tab("Batch Process", ui.ImageList("image_batch", label="Batch Process"), value="Batch Process"),
            ui.Tab(
                "Batch from Directory",
                ui.Text("input_dir", "", label="Input directory", placeholder="A directory on the same machine where the server is running.", hidden=hide_dirs),
                ui.Text("output_dir", "", label="Output directory", placeholder="Leave blank to save images to the default path.", hidden=hide_dirs),
                ui.Toggle("show_extras_results", True, label="Show result images"),
                value="Batch from Directory",
            ),
            ui.Tab(
                "Single Video",
                ui.Text("video_input", "", label="Input Clip", info='For video longer than 1 minute, please extract the frames and use "Batch from Directory" instead', placeholder="Path to a video file", hidden=hide_dirs),
                value="Single Video",
            ),
            bind="mode",
        ),
        ui.Slot("scripts"),
    )


def core_form(tab: str, runner: ScriptRunner | None = None) -> ui.Node:
    if tab == "txt2img":
        return txt2img_form(runner)
    if tab == "img2img":
        return img2img_form(runner)
    if tab == "extras":
        return extras_form()
    raise KeyError(f"unknown tab {tab!r} (one of {TABS})")


def core_layout(tab: str, runner: ScriptRunner | None = None) -> ui.ScriptLayout:
    """The ``core`` owner's layout of ``tab``. Built per call: it follows
    the current options (category order, resolution step, ...) and the UI
    preset (its defaults, labels and bounds, see ``forge_preset_patch``)."""

    from remi.core.presets import apply_form_patch, forge_preset_patch

    layout = ui.ScriptLayout(core_form(tab, runner), owner="core")
    if tab != "extras":
        apply_form_patch(layout, forge_preset_patch(tab))
    return layout


# Which canvas layer each img2img mode needs: (param, message)
_REQUIRED_IMAGES: dict[str, tuple[tuple[str, str], ...]] = {
    "img2img": (("init_img", "an image is required"),),
    "Sketch": (("init_img", "an image is required"),),
    "Inpaint": (("init_img", "an image is required"), ("mask", "paint a mask over the image")),
    "Inpaint sketch": (("init_img", "an image is required"), ("sketch", "paint the area to inpaint")),
    "Inpaint upload": (("init_img", "an image is required"), ("mask_upload", "a mask image is required")),
}


def check_core(tab: str, values: Mapping[str, Any]) -> list[ui.ParamError]:
    """Rules across params of a coerced core form: the images each mode
    needs (img2img), the source of an Extras run."""

    problems: list[ui.ParamError] = []
    if tab == "img2img":
        for name, message in _REQUIRED_IMAGES.get(values.get("mode"), ()):
            if values.get(name) is None:
                problems.append(ui.ParamError(message, owner="core", param=name))
        if values.get("mode") == "Batch" and values.get("batch_source") == "upload" and not values.get("batch_upload"):
            problems.append(ui.ParamError("upload the images to process", owner="core", param="batch_upload"))
    elif tab == "extras":
        mode = values.get("mode")
        if mode == "Single Image" and values.get("image") is None:
            problems.append(ui.ParamError("an image is required", owner="core", param="image"))
        elif mode == "Batch Process" and not values.get("image_batch"):
            problems.append(ui.ParamError("upload the images to process", owner="core", param="image_batch"))
        elif mode == "Batch from Directory" and not str(values.get("input_dir") or "").strip():
            problems.append(ui.ParamError("an input directory is required", owner="core", param="input_dir"))
        elif mode == "Single Video" and not str(values.get("video_input") or "").strip():
            problems.append(ui.ParamError("a video file is required", owner="core", param="video_input"))
    return problems


# endregion
