import pathlib
import sys
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_regional.regional import apply_regional_patch


class _Patcher:
    def __init__(self, dit, previous=None):
        self.model = SimpleNamespace(diffusion_model=dit)
        self.model_options = {} if previous is None else {"model_function_wrapper": previous}

    def clone(self):
        return _Patcher(self.model.diffusion_model, self.model_options.get("model_function_wrapper"))

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


class _State:
    blocks = {0}
    current_masks = None

    def active(self, sigma):
        return True

    def prepare_masks(self, input_x, dit):
        self.current_masks = []


def _dit():
    block = torch.nn.Module()
    block.cross_attn = torch.nn.Identity()
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([block])
    return dit


def _args():
    return {"input": torch.zeros(1), "timestep": torch.ones(1), "c": {"transformer_options": {}}}


def test_wrapper_runs_with_and_without_previous_wrapper_and_restores_forward():
    dit = _dit()
    seen = []

    def model_function(x, t, **c):
        seen.append(c["transformer_options"]["forge_spectrum_force_actual"])
        assert "forward" in dit.blocks[0].cross_attn.__dict__
        return x

    patched = apply_regional_patch(_Patcher(dit), _State())
    # Spectrum is the outer wrapper, so the flag must live on the patcher.
    assert patched.model_options["transformer_options"]["forge_spectrum_force_actual"] == "regional_conditioning"
    patched.model_options["model_function_wrapper"](model_function, _args())

    previous_calls = []

    def previous(apply_model, args):
        previous_calls.append(True)
        return apply_model(args["input"], args["timestep"], **args["c"])

    wrapper = apply_regional_patch(_Patcher(dit, previous), _State()).model_options["model_function_wrapper"]
    wrapper(model_function, _args())

    assert seen == ["regional_conditioning", "regional_conditioning"]
    assert previous_calls == [True]
    assert "forward" not in dit.blocks[0].cross_attn.__dict__
