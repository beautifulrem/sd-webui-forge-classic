import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from modules_forge.packages.comfy.lora import model_lora_keys_unet

_SOURCE = Path(__file__).parents[1] / "extensions-builtin" / "sd_forge_lora" / "networks.py"
_tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_ns = {"torch": torch, "logger": logging.getLogger("test")}
exec(compile(ast.Module([n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_anima"], []), str(_SOURCE), "exec"), _ns)
process_anima = _ns["process_anima"]


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.diffusion_model.blocks[0].mlp = torch.nn.Linear(2, 2, bias=False)
        self.diffusion_model.config = {}
        self.config = SimpleNamespace(huggingface_repo="circlestone-labs/Anima")


def test_trainer_prefixes_map_to_anima_weights():
    key_map = model_lora_keys_unet(_FakeModel(), {})
    target = "diffusion_model.blocks.0.mlp.weight"
    for alias in (
        "diffusion_model.blocks.0.mlp",
        "lora_unet_blocks_0_mlp",
        "transformer.blocks.0.mlp",
        "net.blocks.0.mlp",
        "lycoris_blocks_0_mlp",
        "lora_transformer_blocks_0_mlp",
        "blocks.0.mlp",
    ):
        assert key_map.get(alias) == target, alias


def test_block_remap_and_adapter_move_work_for_every_prefix():
    for prefix, sep in (("transformer.blocks.", "."), ("lycoris_blocks_", "_"), ("diffusion_model.blocks.", ".")):
        lora = {f"{prefix}{i}{sep}mlp.lora_down.weight": torch.full((1,), float(i)) for i in range(28)}
        lora["transformer.llm_adapter.x.lora_down.weight"] = torch.zeros(1)
        assert process_anima(lora, 40)
        assert f"{prefix}39{sep}mlp.lora_down.weight" in lora, prefix
        assert "text_encoders.qwen3_06b.llm_adapter.x.lora_down.weight" in lora
