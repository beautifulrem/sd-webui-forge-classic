import torch
from torch import nn

from backend.nn.anima import keep_sensitive_weights_precise


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 4)
        self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(4, 4))
        self.norm = nn.RMSNorm(4)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(4, 4)
        self.final_layer = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])


def test_sensitive_parts_leave_fp8_storage():
    model = _Model().to(torch.float8_e4m3fn)

    kept = keep_sensitive_weights_precise(model, torch.bfloat16)

    dtypes = {name: p.dtype for name, p in model.named_parameters()}
    for name in ("x_embedder.weight", "final_layer.bias", "blocks.0.mlp.weight", "blocks.1.adaln_modulation_mlp.1.weight"):
        assert dtypes[name] == torch.bfloat16, name
    for block in range(3):
        assert dtypes[f"blocks.{block}.norm.weight"] == torch.bfloat16
    # Bulk of the network stays in FP8.
    assert dtypes["blocks.1.mlp.weight"] == torch.float8_e4m3fn
    assert dtypes["blocks.2.adaln_modulation_mlp.1.weight"] == torch.float8_e4m3fn
    assert kept == sum(1 for dtype in dtypes.values() if dtype == torch.bfloat16)
