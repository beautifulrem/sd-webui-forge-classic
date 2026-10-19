# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy_extras/nodes_mahiro.py

import torch
import torch.nn.functional as F

from modules import scripts
from modules.shared import opts
from remi import ui
from remi.core import hints


class ScriptMahiro(scripts.ScriptBuiltinUI):
    section = "cfg"
    sorting_priority = 1

    def title(self):
        return "MaHiRo"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if opts.show_mahiro else None

    def params(self, is_img2img):
        return ui.Toggle("enable", False, label="MaHiRo", hint=hints.MAHIRO, infotext="MaHiRo")

    def after_extra_networks_activate(self, p, enable, *args, **kwargs):
        if enable:
            p.extra_generation_params.update({"MaHiRo": enable})
            p.mahiro = True

    def process_before_every_sampling(self, p, *args, **kwargs):
        if str(getattr(p, "mahiro", False)) != "True":
            return

        @torch.inference_mode()
        def mahiro_normd(args: dict):
            scale: float = args["cond_scale"]
            cond_p: torch.Tensor = args["cond_denoised"]
            uncond_p: torch.Tensor = args["uncond_denoised"]
            leap = cond_p * scale
            u_leap = uncond_p * scale
            cfg: torch.Tensor = args["denoised"]
            merge = (leap + cfg) / 2
            normu = torch.sqrt(u_leap.abs()) * u_leap.sign()
            normm = torch.sqrt(merge.abs()) * merge.sign()
            sim = F.cosine_similarity(normu, normm).mean()
            simsc = 2 * (sim + 1)
            wm = (simsc * cfg + (4 - simsc) * leap) / 4
            return wm

        unet = p.sd_model.forge_objects.unet.clone()
        unet.set_model_sampler_post_cfg_function(mahiro_normd)
        p.sd_model.forge_objects.unet = unet

        print("using MaHiRo")
