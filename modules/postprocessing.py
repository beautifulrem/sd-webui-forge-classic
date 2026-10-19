import os

import av
import numpy as np
from PIL import Image
from tqdm import tqdm

from modules import devices, images, infotext_utils, scripts, scripts_postprocessing, shared
from modules.shared import opts


def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, _, *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin(job="extras")

    outputs = []

    if isinstance(image, dict):
        image = image["composite"]

    def get_images(extras_mode, image, image_folder, input_dir):
        if extras_mode == 1:
            for img in image_folder:
                if isinstance(img, Image.Image):
                    # An upload carries its original file name (remi.core.JobContext)
                    fn = os.path.splitext(getattr(img, "filename", "") or "")[0] or None
                    image = images.fix_image(img)
                else:
                    image = images.read(os.path.abspath(img.name))
                    fn = os.path.splitext(img.name)[0]
                yield image, fn
        elif extras_mode == 2:
            assert not shared.cmd_opts.hide_ui_dir_config, "--hide-ui-dir-config option must be disabled"
            assert input_dir, "input directory not selected"

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                yield filename, filename
        else:
            assert image, "image not selected"
            yield image, None

    if extras_mode == 2 and output_dir != "":
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ""

    data_to_process = list(get_images(extras_mode, image, image_folder, input_dir))
    shared.state.job_count = len(data_to_process)

    for image_placeholder, name in data_to_process:
        image_data: Image.Image

        shared.state.nextjob()
        shared.state.textinfo = name
        shared.state.skipped = False

        if shared.state.interrupted or shared.state.stopping_generation:
            break

        if isinstance(image_placeholder, str):
            try:
                image_data = images.read(image_placeholder)
            except Exception:
                continue
        else:
            image_data = image_placeholder

        image_data = image_data if image_data.mode in ("RGBA", "RGB") else image_data.convert("RGB")

        parameters, existing_pnginfo = images.read_info_from_image(image_data)
        if parameters:
            existing_pnginfo["parameters"] = parameters

        initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

        scripts.scripts_postproc.run(initial_pp, args)

        if shared.state.skipped:
            continue

        used_suffixes = {}
        for pp in [initial_pp, *initial_pp.extra_images]:
            suffix = pp.get_suffix(used_suffixes)

            if opts.use_original_name_batch and name is not None:
                basename = os.path.splitext(os.path.basename(name))[0]
                forced_filename = basename + suffix
            else:
                basename = ""
                forced_filename = None

            infotext = ", ".join([k if k == v else f"{k}: {infotext_utils.quote(v)}" for k, v in pp.info.items() if v is not None])

            if opts.enable_pnginfo:
                pp.image.info = existing_pnginfo

            shared.state.assign_current_image(pp.image)

            if save_output:
                fullfn, _ = images.save_image(pp.image, path=outpath, basename=basename, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="postprocessing", existing_info=existing_pnginfo, forced_filename=forced_filename, suffix=suffix)

            if extras_mode != 2 or show_extras_results:
                outputs.append(pp.image)

    devices.torch_gc()
    shared.state.end()
    return outputs, infotext


def run_postprocessing_video(_mode, _img, _folder, _in_dir, _out_dir, _show, video_input, *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin(job="extras")

    outputs: list[np.ndarray] = []

    container = av.open(video_input)

    video_stream = container.streams.best("video")
    frames = video_stream.frames

    def get_frames():
        for frame in tqdm(container.decode(video=0), desc="Processing Video", total=frames, unit="frame"):
            yield frame.to_image()

    infotext = None

    shared.state.job_count = frames

    for i, image_data in enumerate(get_frames()):

        shared.state.nextjob()
        shared.state.textinfo = str(i)
        shared.state.skipped = False

        if shared.state.interrupted or shared.state.stopping_generation:
            break

        initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

        scripts.scripts_postproc.run(initial_pp, args)

        if shared.state.skipped:
            continue

        if infotext is None:
            infotext = ", ".join([k if k == v else f"{k}: {infotext_utils.quote(v)}" for k, v in initial_pp.info.items() if v is not None])

        shared.state.assign_current_image(initial_pp.image)

        outputs.append(np.array(initial_pp.image, dtype=np.uint8))

    if not (shared.state.interrupted or shared.state.stopping_generation):
        images.save_video(
            os.path.splitext(os.path.basename(video_input))[0],
            outputs,
            fps=round(float(container.streams.video[0].average_rate)),
            basename=None,
            info=infotext,
            audio_copy=video_input,
        )

    container.close()
    devices.torch_gc()
    shared.state.end()
    return outputs[-1:], infotext or ""


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True, max_side_length: int = 0):
    # Handler for API (does not support video)

    args = scripts.scripts_postproc.create_args_for_run(
        {
            "Upscale": {
                "upscale_enabled": True,
                "upscale_mode": resize_mode,
                "upscale_by": upscaling_resize,
                "max_side_length": max_side_length,
                "upscale_to_width": upscaling_resize_w,
                "upscale_to_height": upscaling_resize_h,
                "upscale_crop": upscaling_crop,
                "upscaler_1_name": extras_upscaler_1,
                "upscaler_2_name": extras_upscaler_2,
                "upscaler_2_visibility": extras_upscaler_2_visibility,
            },
            "GFPGAN": {
                "enable": True,
                "gfpgan_visibility": gfpgan_visibility,
            },
            "CodeFormer": {
                "enable": True,
                "codeformer_visibility": codeformer_visibility,
                "codeformer_weight": codeformer_weight,
            },
        }
    )

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, "", *args, save_output=save_output)


def run_extras_named(core: dict | None, scripts_values: dict | None = None, job=None, *, save_output: bool = True):
    """Extras from named values (``core``: the extras form with upload ids,
    ``scripts_values``: ``{operation id: {param: value}}``); returns
    ``(images, infotext)`` (the operations' parameters as plain text).
    Raises ``remi.ui.ValidationError``."""

    from remi.core import JobContext
    from remi.core.job import job_state
    from remi.core.tabs import processing_args

    job = job or JobContext()
    core_args, script_args = processing_args("extras", core, scripts_values, job.context())
    run_args = (core_args["mode"], core_args["image"], core_args["image_batch"], core_args["input_dir"], core_args["output_dir"], core_args["show_extras_results"], core_args["video_input"], *script_args)
    run = run_postprocessing_video if core_args["mode"] == 3 else run_postprocessing
    with job_state(job):
        images_out, infotext = run(*run_args, save_output=save_output)
    return images_out, infotext
