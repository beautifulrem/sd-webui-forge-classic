from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

import modules.processing as processing
import modules.scripts
import modules.shared as shared
from modules import images
from modules.infotext_utils import (
    create_override_settings_dict,
    parse_generation_parameters,
)
from modules.processing import (
    Processed,
    StableDiffusionProcessingImg2Img,
    process_images,
)
from modules.sd_models import get_closet_checkpoint_match, model_data
from modules.shared import opts, state
from modules_forge import main_thread
from remi.core.forms import resolution_step, snap_resolution
from remi.ui.nodes import mask_channel
from remi.ui.legacy_gradio import gr  # lazy: only the legacy handlers below use Gradio

if TYPE_CHECKING:
    from remi.core import JobContext

BATCH_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".avif")


def _batch_name(image, index: int) -> Path:
    """The path naming a batch input: its file, or an uploaded image's file name."""

    if isinstance(image, Image.Image):
        return Path(getattr(image, "filename", "") or f"image-{index:05}.png")
    return Path(image)


def process_batch(p: StableDiffusionProcessingImg2Img, input, output_dir, inpaint_mask_dir, args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
    """``input``: a directory, or a list of uploads (PIL images, paths, or
    Gradio's named temp files)."""

    output_dir = output_dir.strip()
    processing.fix_seed(p)
    step = resolution_step()

    if isinstance(input, str):
        batch_images = list(shared.walk_files(input, allowed_extensions=BATCH_EXTENSIONS))
    else:
        batch_images = [x if isinstance(x, Image.Image) else os.path.abspath(getattr(x, "name", x)) for x in input]

    is_inpaint_batch = False
    if inpaint_mask_dir:
        inpaint_masks = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = bool(inpaint_masks)

        if is_inpaint_batch:
            print(f"\nInpaint batch is enabled. {len(inpaint_masks)} masks found.")

    print(f"Will process {len(batch_images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    state.job_count = len(batch_images) * p.n_iter

    # extract "default" params to use in case getting png info fails
    prompt = p.prompt
    negative_prompt = p.negative_prompt
    seed = p.seed
    cfg_scale = p.cfg_scale
    sampler_name = p.sampler_name
    steps = p.steps
    width = p.width
    height = p.height
    override_settings = p.override_settings
    sd_model_checkpoint_override = get_closet_checkpoint_match(override_settings.get("sd_model_checkpoint", None))
    batch_results = None
    discard_further_results = False
    for i, image in enumerate(batch_images):
        state.job = f"{i+1} out of {len(batch_images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted or state.stopping_generation:
            break

        try:
            img = image if isinstance(image, Image.Image) else images.read(image)
        except UnidentifiedImageError as e:
            print(e)
            continue
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img)

        if to_scale:
            p.width = snap_resolution(img.width * scale_by)
            p.height = snap_resolution(img.height * scale_by)
        else:
            p.width = width
            p.height = height

        _w, _h = img.size
        if p.resize_mode < 4 and not (_w % step == 0 and _h % step == 0):
            img = images.resize_image(1, img, snap_resolution(_w), snap_resolution(_h))

        p.init_images = [img] * p.batch_size

        image_path = _batch_name(image, i)
        if is_inpaint_batch:
            # try to find corresponding mask for an image using simple filename matching
            if len(inpaint_masks) == 1:
                mask_image_path = inpaint_masks[0]
            else:
                # try to find corresponding mask for an image using simple filename matching
                mask_image_dir = Path(inpaint_mask_dir)
                masks_found = list(mask_image_dir.glob(f"{image_path.stem}.*"))

                if len(masks_found) == 0:
                    print(f"Warning: mask is not found for {image_path} in {mask_image_dir}. Skipping it.")
                    continue

                # it should contain only 1 matching mask
                # otherwise user has many masks with the same name but different extensions
                mask_image_path = masks_found[0]

            mask_image = images.read(mask_image_path)
            p.image_mask = mask_image

        if use_png_info:
            try:
                info_img = img
                if png_info_dir:
                    info_img_path = os.path.join(png_info_dir, image_path.name)
                    info_img = images.read(info_img_path)
                geninfo, _ = images.read_info_from_image(info_img)
                _orig: str = shared.opts.infotext_styles
                shared.opts.infotext_styles = "Ignore"
                parsed_parameters = parse_generation_parameters(geninfo)
                shared.opts.infotext_styles = _orig
                parsed_parameters = {k: v for k, v in parsed_parameters.items() if k in (png_info_props or [])}
            except Exception:
                parsed_parameters = {}

            if "Filename" in (png_info_props or []):
                filename = image_path.stem
                parsed_parameters["Filename"] = filename.replace("(", "\\(").replace(")", "\\)")

            p.prompt = "".join(
                [
                    prompt,
                    (" " + parsed_parameters["Prompt"] if "Prompt" in parsed_parameters else ""),
                    (" " + parsed_parameters["Filename"] if "Filename" in parsed_parameters else ""),
                ]
            )
            p.negative_prompt = negative_prompt + (" " + parsed_parameters["Negative prompt"] if "Negative prompt" in parsed_parameters else "")
            p.seed = int(parsed_parameters.get("Seed", seed))
            p.cfg_scale = float(parsed_parameters.get("CFG scale", cfg_scale))
            p.sampler_name = parsed_parameters.get("Sampler", sampler_name)
            p.steps = int(parsed_parameters.get("Steps", steps))

            model_info = get_closet_checkpoint_match(parsed_parameters.get("Model hash", None))
            if model_info is not None:
                p.override_settings["sd_model_checkpoint"] = model_info.name
            elif sd_model_checkpoint_override:
                p.override_settings["sd_model_checkpoint"] = sd_model_checkpoint_override
            else:
                p.override_settings.pop("sd_model_checkpoint", None)

        if output_dir:
            p.outpath_samples = output_dir
            p.override_settings["save_to_dirs"] = False

        if opts.img2img_batch_use_original_name:
            filename_pattern = f"{image_path.stem}-[generation_number]" if p.n_iter > 1 or p.batch_size > 1 else f"{image_path.stem}"
            p.override_settings["samples_filename_pattern"] = filename_pattern

        proc = modules.scripts.scripts_img2img.run(p, *args)

        if proc is None:
            proc = process_images(p)

        if not discard_further_results and proc:
            if batch_results:
                batch_results.images.extend(proc.images)
                batch_results.infotexts.extend(proc.infotexts)
            else:
                batch_results = proc

            if 0 <= shared.opts.img2img_batch_show_results_limit < len(batch_results.images):
                discard_further_results = True
                batch_results.images = batch_results.images[: int(shared.opts.img2img_batch_show_results_limit)]
                batch_results.infotexts = batch_results.infotexts[: int(shared.opts.img2img_batch_show_results_limit)]

    return batch_results


def _rgba(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGBA" else image.convert("RGBA")


def _over(image: Image.Image, layer: Image.Image | None) -> Image.Image:
    """``layer`` (a color sketch) composited over ``image``."""

    if layer is None:
        return image
    layer = _rgba(layer)
    if layer.size != image.size:
        layer = layer.resize(image.size)
    return Image.alpha_composite(_rgba(image), layer)


def _opaque(mask: Image.Image | None) -> Image.Image | None:
    """An ``L`` mask as the opaque RGBA image the processing expects."""

    if mask is None:
        return None
    return Image.merge("RGBA", (mask, mask, mask, Image.new("L", mask.size, 255)))


def compose_inputs(mode: int, init_img, sketch, mask) -> tuple[Image.Image | None, Image.Image | None]:
    """``(image, mask)`` the processing gets in ``mode`` (an index of
    ``remi.core.forms.IMG2IMG_MODES``) from the canvas layers: the image,
    the color sketch painted over it and the mask (painted over it in
    "Inpaint", the uploaded mask image in "Inpaint upload"). Masks are read
    by ``remi.ui.nodes.mask_channel``: alpha, or luminance when opaque."""

    image = None
    out_mask = None

    if mode == 0:  # img2img
        image = init_img
    elif mode == 1:  # img2img sketch
        image = _over(init_img, sketch)
    elif mode == 2:  # inpaint: the painted mask
        image = init_img
        out_mask = _opaque(mask_channel(mask))
    elif mode == 3:  # inpaint sketch: the sketch's alpha, dilated
        image = _over(init_img, sketch)
        if sketch is not None:
            out_mask = _rgba(sketch).getchannel("A").convert("L")
            short_side = min(out_mask.size)
            dilation_size = int(0.015 * short_side) * 2 + 1
            out_mask = _opaque(out_mask.filter(ImageFilter.MaxFilter(dilation_size)))
    elif mode == 4:  # inpaint upload: the uploaded mask image
        image = init_img
        out_mask = _opaque(mask_channel(mask))

    if out_mask and isinstance(out_mask, Image.Image):
        out_mask = out_mask.point(lambda v: 255 if v > 128 else 0)

    return images.fix_image(image), images.fix_image(out_mask)


def img2img_processing(core: dict[str, Any], script_args, *, user: str | None = None) -> tuple[StableDiffusionProcessingImg2Img, dict[str, Any]]:
    """The processing object for named core processing args (the core form's
    names, choice params as indices, images as PIL) and the full script
    args; also returns the batch settings :func:`run_img2img_processing` needs."""

    override_settings = create_override_settings_dict(core.get("override_settings") or [])

    mode = core["mode"]
    is_batch = mode == 5
    height, width = int(core["height"]), int(core["width"])
    resize_by = core["resize_tab"] == 1
    scale_by = core["scale_by"]
    resize_mode = core["resize_mode"]
    inpaint_full_res = core["inpaint_full_res"]

    image, mask = compose_inputs(mode, core.get("init_img"), core.get("sketch"), core.get("mask_upload") if mode == 4 else core.get("mask"))

    if "pid" in model_data.forge_loading_parameters["checkpoint_info"].filename.lower() and not (resize_by and scale_by == 4.0):
        processing.logger.warning("Resize by 4x is recommended for PiD")

    if resize_by and not is_batch:
        assert image, 'Failed to "Resize by" because no input image is provided'
        assert resize_mode < 4, '"Preserve Aspect Ratio" does not support "Resize by"'
        if mode in (2, 3, 4):
            assert not inpaint_full_res, '"Only masked" does not support "Resize by"'

        width = snap_resolution(image.width * scale_by)
        height = snap_resolution(image.height * scale_by)

    assert 0.0 <= core["denoising_strength"] <= 1.0, "Denoising Strength only supports [0.0, 1.0]"

    p = StableDiffusionProcessingImg2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=core["prompt"],
        negative_prompt=core["negative_prompt"],
        styles=core["styles"],
        batch_size=core["batch_size"],
        n_iter=core["batch_count"],
        cfg_scale=core["cfg_scale"],
        width=width,
        height=height,
        init_images=[image],
        mask=mask,
        mask_blur=core["mask_blur"],
        inpainting_fill=core["inpainting_fill"],
        resize_mode=resize_mode,
        denoising_strength=core["denoising_strength"],
        image_cfg_scale=core["image_cfg_scale"],
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=core["inpaint_full_res_padding"],
        inpainting_mask_invert=core["inpainting_mask_invert"],
        override_settings=override_settings,
        distilled_cfg_scale=core["distilled_cfg_scale"],
    )

    p.scripts = modules.scripts.scripts_img2img
    p.script_args = script_args

    p.user = user

    if shared.opts.enable_console_prompts:
        print(f"\nimg2img: {core['prompt']}", file=shared.progress_print_out)

    batch = {key: core.get(key) for key in ("batch_source", "batch_upload", "batch_input_dir", "batch_output_dir", "batch_inpaint_mask_dir", "batch_use_png_info", "batch_png_info_props", "batch_png_info_dir")}
    batch.update(is_batch=is_batch, to_scale=resize_by, scale_by=scale_by)
    return p, batch


def run_img2img_processing(p: StableDiffusionProcessingImg2Img, batch: dict[str, Any]) -> Processed:
    args = p.script_args
    with closing(p):
        if batch["is_batch"]:
            common = dict(to_scale=batch["to_scale"], scale_by=batch["scale_by"], use_png_info=batch["batch_use_png_info"], png_info_props=batch["batch_png_info_props"])
            if batch["batch_source"] == "upload":
                assert isinstance(batch["batch_upload"], list) and batch["batch_upload"], "no images uploaded for the batch"
                png_info_dir = batch["batch_png_info_dir"] if not shared.cmd_opts.hide_ui_dir_config else ""
                processed = process_batch(p, batch["batch_upload"], "", "", args, png_info_dir=png_info_dir, **common)
            else:  # "from dir"
                assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
                processed = process_batch(p, batch["batch_input_dir"], batch["batch_output_dir"], batch["batch_inpaint_mask_dir"], args, png_info_dir=batch["batch_png_info_dir"], **common)

            if processed is None:
                processed = Processed(p, [], p.seed, "")
        else:
            processed = modules.scripts.scripts_img2img.run(p, *args)
            if processed is None:
                processed = process_images(p)

    shared.total_tqdm.clear()
    return processed


def prepare_img2img(core: dict | None, scripts: dict | None = None, job: JobContext | None = None) -> tuple[StableDiffusionProcessingImg2Img, dict[str, Any]]:
    """Validate named request values (``core``: the img2img form with upload
    ids, ``scripts``: ``{script_id: {param: value}}``) and build the
    processing object. Raises ``remi.ui.ValidationError``."""

    from remi.core import JobContext
    from remi.core.tabs import processing_args

    job = job or JobContext()
    core_args, script_args = processing_args("img2img", core, scripts, job.context())
    p, batch = img2img_processing(core_args, script_args, user=job.user)
    for name, value in job.extra.items():
        setattr(p, name, value)
    return p, batch


def run_img2img(core: dict | None, scripts: dict | None = None, job: JobContext | None = None) -> Processed:
    """Generate from named values; returns the ``Processed`` result. Runs in
    the calling thread, which must be the only one generating (the jobs
    worker runs jobs one at a time on the main thread); ``shared.state`` and
    the progress registry are set up for ``job`` here."""

    from remi.core import JobContext
    from remi.core.job import job_state

    job = job or JobContext()
    p, batch = prepare_img2img(core, scripts, job)
    with job_state(job):
        processed = run_img2img_processing(p, batch)
    if opts.samples_log_stdout:
        print(processed.js())
    return processed


# region Legacy Gradio entry points


def _legacy_core(mode: int, values: dict[str, Any]) -> dict[str, Any]:
    """The named core args of the legacy positional inputs: each mode's own
    canvas becomes the shared image / sketch / mask layers."""

    layers = {
        0: (values["init_img"], None, None),
        1: (values["sketch"], values["sketch_fg"], None),
        2: (values["init_img_with_mask"], None, values["init_img_with_mask_fg"]),
        3: (values["inpaint_color_sketch"], values["inpaint_color_sketch_fg"], None),
        4: (values["init_img_inpaint"], None, values["init_mask_inpaint"]),
    }.get(mode, (None, None, None))
    core = {
        "mode": mode,
        "init_img": layers[0],
        "sketch": layers[1],
        "mask": layers[2] if mode != 4 else None,
        "mask_upload": layers[2] if mode == 4 else None,
        "resize_tab": values["selected_scale_tab"],
        "batch_count": values["n_iter"],
        "styles": values["prompt_styles"],
        "override_settings": values["override_settings_texts"],
        "batch_input_dir": values["img2img_batch_input_dir"],
        "batch_output_dir": values["img2img_batch_output_dir"],
        "batch_inpaint_mask_dir": values["img2img_batch_inpaint_mask_dir"],
        "batch_use_png_info": values["img2img_batch_use_png_info"],
        "batch_png_info_props": values["img2img_batch_png_info_props"],
        "batch_png_info_dir": values["img2img_batch_png_info_dir"],
        "batch_source": values["img2img_batch_source_type"],
        "batch_upload": values["img2img_batch_upload"],
    }
    for name in ("prompt", "negative_prompt", "mask_blur", "inpainting_fill", "batch_size", "cfg_scale", "distilled_cfg_scale", "image_cfg_scale", "denoising_strength", "height", "width", "scale_by", "resize_mode", "inpaint_full_res", "inpaint_full_res_padding", "inpainting_mask_invert"):
        core[name] = values[name]
    return core


def img2img_function(id_task: str, request: gr.Request, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, sketch_fg, init_img_with_mask, init_img_with_mask_fg, inpaint_color_sketch, inpaint_color_sketch_fg, init_img_inpaint, init_mask_inpaint, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, img2img_batch_source_type: str, img2img_batch_upload: list, *args):
    from modules.ui_common import plaintext_to_html

    values = dict(locals())
    p, batch = img2img_processing(_legacy_core(mode, values), args, user=getattr(request, "username", None))
    processed = run_img2img_processing(p, batch)

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    if processed.video_path is None:
        gallery_arg = gr.update(value=processed.images + processed.extra_images, visible=True)
        video_arg = gr.update(value=None, visible=False)
    else:
        gallery_arg = gr.update(value=None, visible=False)
        video_arg = gr.update(value=processed.video_path, visible=True)

    return gallery_arg, video_arg, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def img2img(id_task: str, request: gr.Request, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, sketch_fg, init_img_with_mask, init_img_with_mask_fg, inpaint_color_sketch, inpaint_color_sketch_fg, init_img_inpaint, init_mask_inpaint, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, img2img_batch_source_type: str, img2img_batch_upload: list, *args):
    return main_thread.run_and_wait_result(img2img_function, id_task, request, mode, prompt, negative_prompt, prompt_styles, init_img, sketch, sketch_fg, init_img_with_mask, init_img_with_mask_fg, inpaint_color_sketch, inpaint_color_sketch_fg, init_img_inpaint, init_mask_inpaint, mask_blur, mask_alpha, inpainting_fill, n_iter, batch_size, cfg_scale, distilled_cfg_scale, image_cfg_scale, denoising_strength, selected_scale_tab, height, width, scale_by, resize_mode, inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, override_settings_texts, img2img_batch_use_png_info, img2img_batch_png_info_props, img2img_batch_png_info_dir, img2img_batch_source_type, img2img_batch_upload, *args)


# endregion
