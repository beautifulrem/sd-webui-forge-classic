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
from remi.core import hints

if TYPE_CHECKING:
    from modules.scripts import ScriptRunner

TABS = ("txt2img", "img2img", "extras")

# The SPA fills {{generate}} / {{skip}} / {{interrupt}} with the platform's shortcut text (⌘Enter on macOS).
PROMPT_PLACEHOLDER = "Prompt\n({{generate}} to Generate ; {{skip}} to Skip ; {{interrupt}} to Interrupt)"
NEGATIVE_PLACEHOLDER = "Negative Prompt\n({{generate}} to Generate ; {{skip}} to Skip ; {{interrupt}} to Interrupt)"

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


# region Resolution helpers


def resolution_step() -> int:
    from modules import shared

    return int(getattr(shared.opts, "res_step", 64) or 64)


def snap_resolution(value: float) -> int:
    """The nearest multiple of the resolution step (``processing.sRound``)."""

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

    # The module list main_entry scanned, without importing it (it loads the processing pipeline)
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


def _paste_inpaint_area(params: dict):
    """``Inpaint area``; an inpaint infotext without it ran on the whole
    picture: "Only masked" always wrote the key, and before it was written for
    "Whole picture" too, its absence was the only trace of that choice."""

    if "Inpaint area" in params:
        return params["Inpaint area"]
    if any(key in params for key in ("Mask blur", "Mask mode", "Masked content")):
        return "Whole picture"
    return None


def _paste_hr_sampler(params: dict):
    from modules import sd_samplers

    return sd_samplers.get_hr_sampler_from_infotext(params)


def _paste_hr_scheduler(params: dict):
    from modules import sd_samplers

    return sd_samplers.get_hr_scheduler_from_infotext(params)


# endregion

# region Shared pieces


def _prompts(tab: str, runner: ScriptRunner | None = None) -> ui.Node:
    from modules.scripts import PROMPT_BEFORE_SLOT, script_slot

    # Scripts of the prompt category: slot "prompt_before" (prompt anchor)
    # above the main prompt, section "prompt" (blacklist) below the prompts
    slots = {script_slot(script) for script in runner.alwayson_scripts} if runner is not None else set()
    return ui.Group(
        ui.Slot(PROMPT_BEFORE_SLOT) if PROMPT_BEFORE_SLOT in slots else None,
        ui.Text("prompt", "", lines=3, prompt=True, label="Prompt", placeholder=PROMPT_PLACEHOLDER, infotext="Prompt"),
        ui.Text("negative_prompt", "", lines=3, prompt=True, label="Negative Prompt", placeholder=NEGATIVE_PLACEHOLDER, infotext="Negative prompt", enabled_if=ui.param("cfg_scale", gt=1)),
        ui.MultiChoice("styles", _style_names, [], label="Styles", paste=_paste_styles),
        ui.Slot("prompt") if "prompt" in slots else None,
    )


def _batch() -> list[ui.Node]:
    return [
        ui.Slider("batch_count", 1, 128, 1, 1, label="Batch Count", hint=hints.BATCH_COUNT),
        ui.Slider("batch_size", 1, 8, 1, 1, label="Batch Size", hint=hints.BATCH_SIZE, infotext="Batch size"),
    ]


def _cfg(tab: str) -> list[ui.Node]:
    return [
        ui.Slider("distilled_cfg_scale", 0.0 if tab == "img2img" else 1.0, 24.0, 3.0, 0.5, label="Distilled CFG Scale", infotext="Distilled CFG Scale", preset="native.distilled_cfg"),
        ui.Slider("cfg_scale", 1.0, 24.0, 6.0, 0.5, label="CFG Scale", hint=hints.CFG_SCALE, infotext="CFG scale", preset="native.cfg"),
    ]


def _dimensions(tab: str) -> list[ui.Input]:
    step = resolution_step()
    # txt2img: the Hires. fix target resolution follows the size
    on_change = "hr_resolution" if tab == "txt2img" else None
    # as upstream's ui.py (be6fa5d4): txt2img goes up to 4096, img2img stays at 2048
    maximum = 4096 if tab == "txt2img" else 2048
    return [
        ui.Slider("width", 64, maximum, 1024, step, label="Width", infotext="Size-1", on_change=on_change),
        ui.Slider("height", 64, maximum, 1024, step, label="Height", infotext="Size-2", on_change=on_change),
    ]


def _size(tab: str) -> list[ui.Node]:
    return [*_dimensions(tab), ui.Button("Switch width/height", "swap_dimensions", "tool")]


def _enable_hr() -> ui.Toggle:
    return ui.Toggle("enable_hr", False, label="Hires. fix", hint=hints.HIRES_FIX, paste=_paste_enable_hr, preset="native.hires", exclusive=("pid_hires", "hires"), on_change="hr_resolution")


def _hr_target() -> list[ui.Input]:
    step = resolution_step()
    return [
        # A target size wins over the factor; a size left at 0 follows the other one (the
        # Gradio UI greyed out whichever does not apply): dimmed, still editable
        ui.Slider("hr_scale", 1.0, 4.0, 2.0, 0.05, label="Upscale by", hint=hints.UPSCALE_BY, infotext="Hires upscale", on_change="hr_resolution", inactive_if=ui.any_of(ui.param("hr_resize_x", gt=0), ui.param("hr_resize_y", gt=0))),
        ui.Slider("hr_resize_x", 0, 4096, 0, step, label="Resize width to", hint=hints.RESIZE_WIDTH_TO, infotext="Hires resize-1", on_change="hr_resolution", inactive_if=ui.param("hr_resize_x", lte=0)),
        ui.Slider("hr_resize_y", 0, 4096, 0, step, label="Resize height to", hint=hints.RESIZE_HEIGHT_TO, infotext="Hires resize-2", on_change="hr_resolution", inactive_if=ui.param("hr_resize_y", lte=0)),
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
            ui.Output("hr_final_resolution", "text", "Upscaled resolution"),
            ui.Row(
                ui.Choice("hr_upscaler", _upscaler_names, "Latent", label="Upscaler", infotext="Hires upscaler"),
                ui.Slider("hr_second_pass_steps", 0, 150, 0, 1, label="Hires steps", hint=hints.HIRES_STEPS, infotext="Hires steps"),
                ui.Slider("denoising_strength", 0.0, 1.0, 0.6, 0.05, label="Denoising strength", hint=hints.DENOISING_STRENGTH, infotext="Denoising strength"),
            ),
            ui.Row(*_hr_target()),
            ui.Row(
                ui.Slider("hr_distilled_cfg", 1.0, 24.0, 3.0, 0.5, label="Hires Distilled CFG Scale", infotext="Hires Distilled CFG Scale"),
                ui.Slider("hr_cfg", 1.0, 24.0, 6.0, 0.5, label="Hires CFG Scale", infotext="Hires CFG Scale"),
            ),
            ui.Group(
                ui.Row(
                    ui.Choice("hr_checkpoint_name", _hr_checkpoints, SAME_CHECKPOINT, same=ui.without_model_hash, label="Hires Checkpoint", infotext="Hires checkpoint"),
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
            sections.append(_prompts("txt2img", runner))
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
        ui.Choice("resize_mode", RESIZE_MODES, "Crop and Resize", type="index", style="radio", label="Resize Mode", hints=hints.RESIZE_MODES),
    )


def _img2img_inpaint() -> ui.Node:
    return ui.Group(
        ui.Slider("mask_blur", 0, 64, 4, 4, label="Mask blur", hint=hints.MASK_BLUR, infotext="Mask blur"),
        ui.Choice("inpainting_mask_invert", MASK_MODES, "Inpaint masked", type="index", style="radio", label="Mask mode", infotext="Mask mode"),
        ui.Choice("inpainting_fill", MASKED_CONTENT, "original", type="index", style="radio", label="Masked content", hints=hints.MASKED_CONTENT, infotext="Masked content"),
        ui.Row(
            ui.Choice("inpaint_full_res", INPAINT_AREAS, "Only masked", type="index", style="radio", label="Inpaint area", infotext="Inpaint area", paste=_paste_inpaint_area),
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
            sections.append(_prompts("img2img", runner))
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
                        ui.Tab("Resize by", ui.Slider("scale_by", 0.5, 4.0, 1.0, 0.05, label="Scale"), ui.Output("scale_by_resolution", "text"), value="Resize by"),
                        bind="resize_tab",
                    ),
                    ui.Row(*_batch()) if together else None,
                    ui.Slot(category),
                )
            )
        elif category == "cfg":
            sections.append(ui.Row(*_cfg("img2img"), ui.Slider("image_cfg_scale", 0, 3.0, 1.5, 0.05, label="Image CFG Scale", infotext="Image CFG scale", hidden=True), ui.Slot(category)))
        elif category == "denoising":
            sections.append(ui.Group(ui.Slider("denoising_strength", 0.0, 1.0, 0.6, 0.01, label="Denoising Strength", hint=hints.DENOISING_STRENGTH, infotext="Denoising strength"), ui.Slot(category)))
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


# The inputs each core form needs: (param, message, when). ``when`` is a
# condition (schema-format.md §2) on the coerced core values; while it holds,
# the param must not be empty. The tab schema sends them as ``core.required``
# so the client checks before submitting (no request that must fail).
_IMG2IMG_IMAGE_MODES = ("img2img", "Sketch", "Inpaint", "Inpaint sketch", "Inpaint upload")
REQUIRED_INPUTS: dict[str, tuple[tuple[str, str, ui.conditions.Condition], ...]] = {
    "img2img": (
        ("init_img", "an image is required", ui.param("mode", in_=list(_IMG2IMG_IMAGE_MODES))),
        ("mask", "paint a mask over the image", ui.param("mode", eq="Inpaint")),
        ("sketch", "paint the area to inpaint", ui.param("mode", eq="Inpaint sketch")),
        ("mask_upload", "a mask image is required", ui.param("mode", eq="Inpaint upload")),
        ("batch_upload", "upload the images to process", ui.all_of(ui.param("mode", eq="Batch"), ui.param("batch_source", eq="upload"))),
    ),
    "extras": (
        ("image", "an image is required", ui.param("mode", eq="Single Image")),
        ("image_batch", "upload the images to process", ui.param("mode", eq="Batch Process")),
        ("input_dir", "an input directory is required", ui.param("mode", eq="Batch from Directory")),
        ("video_input", "a video file is required", ui.param("mode", eq="Single Video")),
    ),
}


def required_inputs(tab: str) -> list[dict[str, Any]]:
    """``core.required`` of the tab schema: ``[{"param", "message", "when"}]``."""

    return [{"param": name, "message": message, "when": when} for name, message, when in REQUIRED_INPUTS.get(tab, ())]


def _empty(value: Any) -> bool:
    """No value: ``None``, a blank string or an empty list."""

    return value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, list | tuple) and not value)


def check_core(tab: str, values: Mapping[str, Any]) -> list[ui.ParamError]:
    """Rules across params of a coerced core form: the images each img2img
    mode needs, the source of an Extras run (:data:`REQUIRED_INPUTS`)."""

    return [
        ui.ParamError(message, owner="core", param=name)
        for name, message, when in REQUIRED_INPUTS.get(tab, ())
        if ui.evaluate(when, values) and _empty(values.get(name))
    ]


# endregion
