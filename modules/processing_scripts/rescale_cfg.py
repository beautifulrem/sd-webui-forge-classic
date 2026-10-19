# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy_extras/nodes_model_advanced.py#L274

import torch

from modules import scripts
from modules.shared import opts
from remi import ui
from remi.core import hints


class ScriptRescaleCFG(scripts.ScriptBuiltinUI):
    section = "cfg"

    def title(self):
        return "RescaleCFG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if opts.show_rescale_cfg else None

    def params(self, is_img2img):
        return ui.Slider("cfg", 0.0, 1.0, 0.0, 0.05, label="Rescale CFG", hint=hints.RESCALE_CFG, infotext="Rescale CFG")

    def after_extra_networks_activate(self, p, cfg, *args, **kwargs):
        if cfg > 0.0:
            p.extra_generation_params.update({"Rescale CFG": cfg})
            p.rescale_cfg = cfg

    def process_before_every_sampling(self, p, *args, **kwargs):
        if getattr(p, "rescale_cfg", 0.0) < 0.05:
            return
        if p.is_hr_pass:
            return

        self.apply_rescale_cfg(p, p.rescale_cfg)

    @staticmethod
    def apply_rescale_cfg(p, cfg: float):

        @torch.inference_mode()
        def rescale_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x_orig = args["input"]

            x = x_orig / (sigma * sigma + 1.0)
            cond = ((x - (x_orig - cond)) * (sigma**2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_orig - uncond)) * (sigma**2 + 1.0) ** 0.5) / (sigma)

            x_cfg = uncond + cond_scale * (cond - uncond)
            ro_pos = torch.std(cond, dim=(1, 2, 3), keepdim=True)
            ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)

            x_rescaled = x_cfg * (ro_pos / ro_cfg)
            x_final = cfg * x_rescaled + (1.0 - cfg) * x_cfg

            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)

        unet = p.sd_model.forge_objects.unet.clone()
        unet.set_model_sampler_cfg_function(rescale_cfg)
        p.sd_model.forge_objects.unet = unet

        print(f"rescale_cfg = {cfg}")
