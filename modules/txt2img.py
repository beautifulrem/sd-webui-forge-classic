from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any

import modules.scripts
import modules.shared as shared
from modules import processing
from modules.infotext_utils import (
    create_override_settings_dict,
)
from modules.shared import opts

if TYPE_CHECKING:
    from remi.core import JobContext

def txt2img_processing(core: dict[str, Any], script_args, *, user: str | None = None) -> processing.StableDiffusionProcessingTxt2Img:
    """The processing object for named core processing args (the core form's
    param names, see remi/core/forms.py) and the full script args."""

    override_settings = create_override_settings_dict(core.get("override_settings") or [])

    p = processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=core["prompt"],
        styles=core["styles"],
        negative_prompt=core["negative_prompt"],
        batch_size=core["batch_size"],
        n_iter=core["batch_count"],
        cfg_scale=core["cfg_scale"],
        distilled_cfg_scale=core["distilled_cfg_scale"],
        width=core["width"],
        height=core["height"],
        enable_hr=core["enable_hr"],
        denoising_strength=core["denoising_strength"],
        hr_scale=core["hr_scale"],
        hr_upscaler=core["hr_upscaler"],
        hr_second_pass_steps=core["hr_second_pass_steps"],
        hr_resize_x=core["hr_resize_x"],
        hr_resize_y=core["hr_resize_y"],
        hr_checkpoint_name=None if core["hr_checkpoint_name"] == "Use same checkpoint" else core["hr_checkpoint_name"],
        hr_additional_modules=core["hr_additional_modules"],
        hr_sampler_name=None if core["hr_sampler_name"] == "Use same sampler" else core["hr_sampler_name"],
        hr_scheduler=None if core["hr_scheduler"] == "Use same scheduler" else core["hr_scheduler"],
        hr_prompt=core["hr_prompt"],
        hr_negative_prompt=core["hr_negative_prompt"],
        hr_cfg=core["hr_cfg"],
        hr_distilled_cfg=core["hr_distilled_cfg"],
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = script_args

    p.user = user

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {core['prompt']}", file=shared.progress_print_out)

    return p


def prepare_txt2img(core: dict | None, scripts: dict | None = None, job: JobContext | None = None) -> processing.StableDiffusionProcessingTxt2Img:
    """Validate named request values (``core``: the txt2img form,
    ``scripts``: ``{script_id: {param: value}}``) and build the processing
    object. Raises ``remi.ui.ValidationError`` listing every bad value."""

    from remi.core import JobContext
    from remi.core.tabs import processing_args

    job = job or JobContext()
    core_args, script_args = processing_args("txt2img", core, scripts, job.context())
    p = txt2img_processing(core_args, script_args, user=job.user)
    for name, value in job.extra.items():
        setattr(p, name, value)
    return p


def run_txt2img(core: dict | None, scripts: dict | None = None, job: JobContext | None = None) -> processing.Processed:
    """Generate from named values; returns the ``Processed`` result. Runs in
    the calling thread, which must be the only one generating (the jobs
    worker runs jobs one at a time on the main thread); ``shared.state`` and
    the progress registry are set up for ``job`` here."""

    from remi.core import JobContext
    from remi.core.job import job_state

    job = job or JobContext()
    p = prepare_txt2img(core, scripts, job)
    with job_state(job), closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    if opts.samples_log_stdout:
        print(processed.js())

    return processed
