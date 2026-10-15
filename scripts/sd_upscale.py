import math
import re

from PIL import Image

import modules.scripts as scripts
from modules import devices, images, processing, shared
from modules.processing import Processed
from modules.shared import opts, state
from remi import ui


def upscaler_names() -> list[str]:
    return [x.name for x in shared.sd_upscalers]


class SDUpscale(scripts.Script):
    # The tile overlap comes first in the arguments, after the upscaler in the layout
    param_order = ("overlap", "upscaler_index", "scale_factor", "override")

    def title(self):
        return "SD Upscale"

    def show(self, is_img2img):
        return is_img2img

    def params(self, is_img2img):
        return ui.Column(
            ui.Markdown("Upscale the image by the selected **Scale Factor**; use the **Width** and **Height** to set the tile size"),
            ui.Row(
                # The script receives the index (Gradio's type="index"); a None default is the first upscaler
                ui.Choice("upscaler_index", upscaler_names, None, type="index", label="Upscaler"),
                ui.Slider("scale_factor", 1.0, 8.0, 2.0, 0.05, label="Scale Factor"),
            ),
            ui.Row(
                ui.Slider("overlap", 0, 256, 64, 16, label="Tile Overlap"),
                ui.Toggle("override", False, label="Save to Extras folder instead"),
            ),
        )

    def run(self, p, overlap: int, upscaler_index: str | int, scale_factor: float, override: bool):
        if isinstance(upscaler_index, str):
            upscaler = next(
                (x for x in shared.sd_upscalers if x.name == upscaler_index),
                None,
            )
            assert upscaler is not None
        else:
            assert isinstance(upscaler_index, int)
            upscaler = shared.sd_upscalers[upscaler_index]

        processing.fix_seed(p)

        p.extra_generation_params["SD Upscale - Overlap"] = overlap
        p.extra_generation_params["SD Upscale - Upscaler"] = upscaler.name

        initial_info = None
        seed = p.seed

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)

        if upscaler.name != "None":
            img = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path)
        else:
            img = init_img

        devices.torch_gc()

        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=overlap)

        batch_size = p.batch_size
        upscale_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []

        for _, _, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / batch_size)
        state.job_count = batch_count * upscale_count

        print(
            f"""
[SD Upscale]
- Processing {len(grid.tiles[0][2])}x{len(grid.tiles)} tiles
- totaling {len(work)} images at a batch size of {batch_size}
- resulting in {state.job_count} iterations
            """
        )

        result_images = []
        for n in range(upscale_count):
            start_seed = seed + n
            p.seed = start_seed

            work_results = []
            for i in range(batch_count):
                p.batch_size = batch_size
                p.init_images = work[i * batch_size : (i + 1) * batch_size]

                state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                processed = processing.process_images(p)

                if initial_info is None:
                    initial_info = processed.info

                p.seed = processed.seed + 1
                work_results += processed.images

            image_index = 0
            for _, _, row in grid.tiles:
                for tiledata in row:
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                    image_index += 1

            combined_image = images.combine_grid(grid)
            result_images.append(combined_image)

            if opts.samples_save:
                if override:
                    images.save_image(
                        combined_image,
                        path=opts.outdir_samples or opts.outdir_extras_samples,
                        basename="",
                        extension=opts.samples_format,
                        info=initial_info,
                        short_filename=True,
                        no_prompt=True,
                        grid=False,
                        pnginfo_section_name="extras",
                        existing_info=None,
                        forced_filename=None,
                        suffix="",
                    )
                else:
                    images.save_image(
                        combined_image,
                        p.outpath_samples,
                        "",
                        start_seed,
                        p.prompt,
                        opts.samples_format,
                        info=initial_info,
                        p=p,
                    )

        new_w, new_h = img.size
        pattern = r"Size: (\d+)x(\d+)"
        initial_info = re.sub(pattern, f"Size: {new_w}x{new_h}", initial_info)

        return Processed(p, result_images, seed, initial_info)
