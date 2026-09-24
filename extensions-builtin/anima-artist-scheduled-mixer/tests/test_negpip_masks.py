import ast
from pathlib import Path
from types import SimpleNamespace

import torch

# The script imports gradio/WebUI modules at import time; load only the pure
# context/mask helpers.
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_artist_scheduled_mixer.py"
_HELPERS = {"_ensure_3d", "_to_context", "_broadcast_batch", "_artist_mask_like", "_joined_mask", "_with_negpip_mask"}
_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
_nodes = [node for node in _tree.body if isinstance(node, ast.FunctionDef) and node.name in _HELPERS]
_ns = {"torch": torch, "NEGPIP_MASK_KEY": "negpip_mask"}
exec(compile(ast.Module(_nodes, []), str(SCRIPT), "exec"), _ns)


def test_artist_mask_is_aligned_to_the_artist_context_batch():
    artist = SimpleNamespace(negpip_mask=torch.tensor([[[1.0], [-1.0]]]))
    context = torch.zeros(4, 2, 3)

    mask = _ns["_artist_mask_like"](artist, context)

    assert mask.shape == (4, 2, 1)
    assert _ns["_artist_mask_like"](SimpleNamespace(negpip_mask=None), context) is None


def test_joined_mask_fills_parts_without_a_mask_with_ones():
    base = torch.zeros(2, 3, 4)
    artist = torch.zeros(2, 2, 4)
    artist_mask = torch.tensor([[[-1.0], [1.0]]]).expand(2, -1, -1)

    joined = _ns["_joined_mask"]([(base, None), (artist, artist_mask)], 1)

    assert joined.shape == (2, 5, 1)
    assert joined[0, :, 0].tolist() == [1.0, 1.0, 1.0, -1.0, 1.0]
    assert _ns["_joined_mask"]([(base, None), (artist, None)], 1) is None


def test_foreign_context_options_replace_the_base_mask():
    base_mask = torch.ones(1, 3, 1)
    options = _ns["_with_negpip_mask"]({"negpip_mask": base_mask, "other": 1}, None)

    assert options == {"negpip_mask": None, "other": 1}
