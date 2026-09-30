"""Child process for test_dcw: DCW through Forge's real sampling stack.

Runs k-diffusion's ``sample_euler`` over the real ``CFGDenoiser.forward``,
``cfg_denoiser_callback`` and ``sampling_function``, with this panel's DCW
callback and post-CFG hook installed by ``_install_dcw``; only the DiT is a
closed-form stand-in. It needs the WebUI's module globals (``shared.opts``,
the ``modules.processing`` import chain), so it runs in its own process.

Usage: ``python dcw_sampler_run.py OUTPUT.pt``; saves the noise, sigmas and
final latents with DCW off and on.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

sys.argv, output_path = sys.argv[:1], sys.argv[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "modules_forge" / "packages"))  # k_diffusion, as at startup

import torch  # noqa: E402

from modules import shared  # noqa: E402  (first: options imports it circularly)
from modules import options, shared_options, shared_state  # noqa: E402

# What shared_init.initialize() sets, without loading the user's config.
shared.options_templates = shared_options.options_templates
shared.restricted_opts = shared_options.restricted_opts
shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
shared.state = shared_state.State()

from modules import processing  # noqa: E402,F401  (the WebUI's import order)
from modules import prompt_parser  # noqa: E402
from modules.sd_samplers_cfg_denoiser import CFGDenoiser  # noqa: E402
from backend.patcher.base import set_model_options_post_cfg_function  # noqa: E402
from k_diffusion.sampling import sample_euler  # noqa: E402

spec = importlib.util.spec_from_file_location("anima_guidance_dcw_run", Path(__file__).parents[1] / "scripts" / "anima_guidance.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)  # registers the DCW cfg_denoiser callback

COND_SCALE = 3.0


class Model:
    """x0 = factor * x: 0.7 for the prompt (context of ones), 0.9 for the
    negative (zeros), so CFG 3 gives 0.3 * x."""

    def memory_required(self, shape):
        return 0

    def apply_model(self, x, t, c_crossattn, **kwargs):
        context = c_crossattn.mean(dim=(1, 2)).reshape(-1, *([1] * (x.ndim - 1)))
        return x * (0.9 - 0.2 * context)


class Unet:
    def __init__(self):
        self.model = Model()
        self.controlnet_linked_list = None
        self.extra_concat_condition = None
        self.model_options = {}

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(self.model_options, fn, disable_cfg1_optimization)


class Process:
    def __init__(self):
        self.seeds = [0]
        self.refiner_switch_at = None
        self.is_hr_pass = False
        self.scripts = None
        self.extra_generation_params = {}


def run(noise, sigmas, dcw):
    unet, p = Unet(), Process()
    if dcw:
        script._install_dcw(p, types.SimpleNamespace(dcw_lambda=-0.2, dcw_schedule="one_minus_sigma", dcw_bands="LL"), unet)

    class Denoiser(CFGDenoiser):
        @property
        def inner_model(self):
            engine = types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=unet), is_inpaint=False)
            return types.SimpleNamespace(inner_model=engine)

    denoiser = Denoiser(types.SimpleNamespace())
    denoiser.p = p
    denoiser.steps = denoiser.total_steps = len(sigmas) - 1

    def schedule(value):
        return [prompt_parser.ScheduledPromptConditioning(1000, torch.full((3, 8), value))]

    cond = prompt_parser.MulticondLearnedConditioning(
        shape=(1,), batch=[[prompt_parser.ComposableScheduledPromptConditioning(schedule(1.0), 1.0)]]
    )
    extra_args = {"cond": cond, "uncond": [schedule(0.0)], "cond_scale": COND_SCALE, "s_min_uncond": 0.0, "image_cond": None}
    return sample_euler(denoiser, noise.clone(), sigmas, extra_args=extra_args, disable=True)


noise = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(0))
sigmas = torch.tensor([1.0, 0.8, 0.55, 0.3, 0.1, 0.0])
torch.save({"noise": noise, "sigmas": sigmas, "off": run(noise, sigmas, False), "on": run(noise, sigmas, True)}, output_path)
