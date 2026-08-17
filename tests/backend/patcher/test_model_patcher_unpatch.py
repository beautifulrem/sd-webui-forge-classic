import torch

from backend.patcher.base import ModelPatcher


class PatchableLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameters_manual_cast = True
        self.prev_parameters_manual_cast = False
        self.weight_function = [object()]
        self.bias_function = [object()]


class PatchableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = PatchableLayer()


def test_unpatch_clears_weight_functions_after_full_vram_load():
    model = PatchableModel()
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    assert model.model_lowvram is False

    patcher.unpatch_model()

    assert model.layer.weight_function == []
    assert model.layer.bias_function == []
    assert model.layer.parameters_manual_cast is False
    assert not hasattr(model.layer, "prev_parameters_manual_cast")
