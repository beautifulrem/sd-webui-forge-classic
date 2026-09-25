import ast
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, quants

from backend.operations_gguf import ParameterGGUF
from backend.utils import weight_dtype

# backend.loader imports the whole WebUI; only its Anima helper is needed.
_SOURCE = Path(__file__).parents[1] / "backend" / "loader.py"
_tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_ns = {"torch": torch}
exec(compile(ast.Module([n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_anima"], []), str(_SOURCE), "exec"), _ns)
process_anima = _ns["process_anima"]


def test_gguf_adapter_is_dequantized_when_moved_to_the_text_encoder():
    weight = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    raw = torch.from_numpy(quants.quantize(weight, GGMLQuantizationType.Q8_0))
    dit = {
        "llm_adapter.proj.weight": ParameterGGUF(raw, tensor_type=GGMLQuantizationType.Q8_0, tensor_shape=torch.Size((64, 64))),
        "blocks.0.mlp.weight": ParameterGGUF(raw, tensor_type=GGMLQuantizationType.Q8_0, tensor_shape=torch.Size((64, 64))),
    }
    enc = {"model.embed_tokens.weight": torch.zeros(8, 4, dtype=torch.bfloat16)}

    process_anima(dit, enc)

    adapter = enc["llm_adapter.proj.weight"]
    assert not hasattr(adapter, "gguf_cls") and adapter.shape == (64, 64)
    assert torch.allclose(adapter.float(), torch.from_numpy(weight), atol=0.05)
    assert weight_dtype(enc) != "gguf" and list(dit) == ["blocks.0.mlp.weight"]
