"""Regenerates ``anima_flow_upstream_reference.pt`` from KeithZ117/Comfyui-anima-sampler
(MIT) at effba3c5dc232938aef0a8d19f2c67d4598b46cc, run on the CPU with ComfyUI stubbed:

    gh api repos/KeithZ117/Comfyui-anima-sampler/tarball/effba3c5dc232938aef0a8d19f2c67d4598b46cc > u.tgz && tar xzf u.tgz
    python tests/fixtures/make_anima_flow_upstream_reference.py KeithZ117-Comfyui-anima-sampler-effba3c/anima_sampler

The closed-form Gaussian denoiser, float64, upstream's full sampling loop
(``sample_anima_flow_corrective``) with static CFG and no endpoint refresh."""

import pathlib
import sys
import types

import torch

MU, STD = 0.7, 0.4


def shifted(steps):
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return 3 * t / (1 + 2 * t)  # Anima's shift 3


SCHEDULES = {
    "shift3_6": shifted(6),
    "shift3_16": shifted(16),
    "shift3_30": shifted(30),
    "uneven_10": torch.tensor([1.0, 0.93, 0.82, 0.66, 0.47, 0.3, 0.17, 0.09, 0.045, 0.02, 0.0], dtype=torch.float64),
}
# name -> (upstream solver, upstream kwargs, our sampler, our kwargs)
VARIANTS = {
    "unipc2_bh2": ("flow_unipc2_diffusers_x0", {}, "unipc2", {}),
    "unipc2_bh1": ("flow_unipc2_diffusers_x0", {"flow_unipc_solver_type": "bh1"}, "unipc2", {"flow_unipc_solver_type": "bh1"}),
    "unipc2_no_first_correctors": ("flow_unipc2_diffusers_x0", {"flow_unipc_disable_corrector_first": 2}, "unipc2", {"flow_unipc_disable_corrector_first": 2}),
    "unipc2_thresholding": ("flow_unipc2_diffusers_x0", {"flow_unipc_thresholding": True}, "unipc2", {"flow_unipc_thresholding": True}),
    "unipc2_thresholding_quantile": (
        "flow_unipc2_diffusers_x0",
        {"flow_unipc_thresholding": True, "flow_unipc_dynamic_thresholding_ratio": 0.9, "flow_unipc_sample_max_value": 1.5},
        "unipc2",
        {"flow_unipc_thresholding": True, "flow_unipc_dynamic_thresholding_ratio": 0.9, "flow_unipc_sample_max_value": 1.5},
    ),
    "pc3": ("flow_pc3_diffusers_damped", {}, "pc3", {}),
    "pc3_gamma_half": ("flow_pc3_diffusers_damped", {"flow_pc3_gamma": 0.5, "flow_pc3_tolerance": 0.02}, "pc3", {"flow_pc3_gamma": 0.5, "flow_pc3_tolerance": 0.02}),
}


def gaussian_denoiser(calls):
    def model(x, sigma, **_kwargs):
        calls.append(float(sigma.flatten()[0]))
        t = sigma.view(-1, *([1] * (x.ndim - 1)))
        return MU + (1 - t) * STD**2 / ((1 - t) ** 2 * STD**2 + t**2) * (x - (1 - t) * MU)

    return model


def main(upstream_package):
    sampling = types.ModuleType("comfy.k_diffusion.sampling")
    sampling.trange = lambda n, disable=None: range(n)
    sys.modules.update({"comfy": types.ModuleType("comfy"), "comfy.k_diffusion": types.ModuleType("comfy.k_diffusion"), "comfy.k_diffusion.sampling": sampling})
    package = types.ModuleType("anima_sampler")  # skips the package __init__ (ComfyUI nodes)
    package.__path__ = [str(upstream_package)]
    sys.modules["anima_sampler"] = package
    from anima_sampler.sampling_loop import sample_anima_flow_corrective

    static = dict(
        flow_schedule="flow_rf_linear_shift", final_clean_pass=False, flow_er_order=2, flow_pc3_gamma=1.0, flow_pc3_tolerance=0.005,
        base_cfg=1.0, cfg_schedule_domain="progress", cfg_schedule_mode="constant", early_cfg_boost=0.0, early_cfg_until=0.0,
        late_cfg_scale=1.0, late_cfg_start=1.0, cfg_early_scale=1.0, cfg_early_ramp_end=0.0, cfg_peak_boost=0.0, cfg_bump_start=0.0,
        cfg_bump_end=1.0, cfg_beta_alpha=2.0, cfg_beta_beta=2.0, cfg_interval_start=0.0, cfg_interval_rise_end=0.0,
        cfg_interval_fall_start=1.0, cfg_interval_end=1.0, rf_endpoint_noise_refresh_enabled=False,
        rf_endpoint_noise_refresh_strength=0.0, rf_endpoint_noise_refresh_until=0.0,
    )
    noise = torch.randn(1, 4, 6, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    cases = {}
    for schedule, sigmas in SCHEDULES.items():
        for variant, (solver, upstream_kwargs, _, _) in VARIANTS.items():
            calls = []
            x = sample_anima_flow_corrective(gaussian_denoiser(calls), noise.clone(), sigmas, {"seed": 0}, disable=True, **{**static, "flow_solver": solver, **upstream_kwargs})
            cases[f"{schedule}/{variant}"] = {"x": x, "calls": torch.tensor(calls, dtype=torch.float64)}
    target = pathlib.Path(__file__).with_name("anima_flow_upstream_reference.pt")
    torch.save({"noise": noise, "cases": cases}, target)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]).resolve())
