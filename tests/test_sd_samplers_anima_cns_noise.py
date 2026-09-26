import k_diffusion.sampling
import torch

import modules.sd_samplers_anima_cns as cns


def _run(monkeypatch, recolorer, **kwargs):
    """Run the sampler with Forge's ER-SDE replaced by a probe of its noise sampler."""
    monkeypatch.setattr(cns, "require_anima_flow_denoiser", lambda _model, _feature: None)
    monkeypatch.setattr(cns.CNSRecolorer, "try_calibrated", staticmethod(lambda strength, process: recolorer))
    seen = {}

    def sample_er_sde(model, x, sigmas, *, noise_sampler, **options):
        seen["noise"] = None if noise_sampler is None else noise_sampler(sigmas[0], sigmas[1])
        seen["options"] = options
        return x

    monkeypatch.setattr(k_diffusion.sampling, "sample_er_sde", sample_er_sde)
    cns.sample_anima_er_sde_cns(object(), torch.zeros(1, 2, 2), torch.tensor([0.8, 0.4, 0.0]), **kwargs)
    return seen


class _Recolorer:
    def __init__(self):
        self.sigmas = []

    def recolor(self, noise, sigma):
        self.sigmas.append(sigma)
        return noise * 2


def test_each_noise_draw_is_recolored_at_its_sigma(monkeypatch):
    recolorer = _Recolorer()
    white = torch.ones(1, 2, 2)

    seen = _run(monkeypatch, recolorer, noise_sampler=lambda sigma, sigma_next: white, max_stage=2)

    assert torch.equal(seen["noise"], white * 2)
    # the schedule's own sigma, not float(sigma): that would sync the device every step
    assert len(recolorer.sigmas) == 1 and torch.is_tensor(recolorer.sigmas[0]) and float(recolorer.sigmas[0]) == 0.800000011920929
    assert seen["options"]["max_stage"] == 2


def test_without_calibration_the_solver_keeps_its_own_noise(monkeypatch):
    seen = _run(monkeypatch, None)
    assert seen["noise"] is None  # ER-SDE falls back to its default sampler

    seen = _run(monkeypatch, _Recolorer(), cns_strength=0.0)
    assert seen["noise"] is None
