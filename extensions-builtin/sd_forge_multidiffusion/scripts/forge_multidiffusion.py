from lib_multidiffusion.tiled_diffusion import TiledDiffusion

from modules import scripts
from remi import ui

TILE_MIN, TILE_MAX = 256, 2048


def source_image_size(ctx: ui.ActionContext) -> tuple[int, int] | None:
    """Width and height of the img2img source image (core ``init_img``), as
    the server indexed it (``ctx.files``, which the server fills); None
    without an indexed image."""

    return ctx.files.image_size(ctx.core.get("init_img"))


class MultiDiffusionForForge(scripts.Script):
    sorting_priority = 16

    def title(self):
        return "MultiDiffusion Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else None

    def params(self, is_img2img):
        # The infotext keys are the ones process_before_every_sampling writes, so
        # pasting an image's parameters restores the panel (the Gradio panel
        # wrote them but did not read them back)
        return ui.Section(
            self.title(),
            enable=ui.Toggle("enabled", False, label=self.title(), infotext="multidiffusion_enabled"),
            children=[
                ui.Choice("method", ("MultiDiffusion", "Mixture of Diffusers"), "Mixture of Diffusers", style="radio", label="Method", infotext="multidiffusion_method"),
                ui.Row(
                    ui.Slider("tile_width", TILE_MIN, TILE_MAX, 768, 64, label="Tile Width", infotext="multidiffusion_tile_width"),
                    ui.Slider("tile_height", TILE_MIN, TILE_MAX, 768, 64, label="Tile Height", infotext="multidiffusion_tile_height"),
                ),
                ui.Button("Use source image size", "detect_size", "tool"),
                ui.Row(
                    ui.Slider("tile_overlap", 0, 1024, 64, 16, label="Tile Overlap", infotext="multidiffusion_tile_overlap"),
                    ui.Slider("tile_batch_size", 1, 8, 1, 1, label="Tile Batch Size", infotext="multidiffusion_tile_batch_size"),
                ),
            ],
        )

    @ui.action("detect_size")
    def detect_size(self, ctx: ui.ActionContext, values: dict) -> dict:
        """Tile size from the source image's size (the former 📐 button)."""

        size = source_image_size(ctx)
        if size is None:
            return ui.toast("Add a source image to read its size", "warning")
        width, height = (min(max(int(value), TILE_MIN), TILE_MAX) for value in size)
        return {"tile_width": {"value": width}, "tile_height": {"value": height}}

    def process_before_every_sampling(self, p, enabled: bool, method: str, tile_width: int, tile_height: int, tile_overlap: int, tile_batch_size: int, **kwargs):
        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet
        unet = TiledDiffusion.apply(unet, method, tile_width, tile_height, tile_overlap, tile_batch_size)
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(
            {
                "multidiffusion_enabled": enabled,
                "multidiffusion_method": method,
                "multidiffusion_tile_width": tile_width,
                "multidiffusion_tile_height": tile_height,
                "multidiffusion_tile_overlap": tile_overlap,
                "multidiffusion_tile_batch_size": tile_batch_size,
            }
        )
