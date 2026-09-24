import torch

from backend.nn.anima import Anima


class _RecordingBlock(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def forward(self, x, t_embedding, context, transformer_options=None, **kwargs):
        self.seen.append(transformer_options.get("block_index"))
        return x


def _stub_anima(num_blocks, seen):
    # Anima's layers need Forge's patched nn ops; only the block loop matters.
    model = Anima.__new__(Anima)
    torch.nn.Module.__init__(model)
    model.patch_temporal, model.patch_spatial = 1, 1
    model.blocks = torch.nn.ModuleList([_RecordingBlock(seen) for _ in range(num_blocks)])
    model.prepare_embedded_sequence = lambda x, padding_mask=None: (x.permute(0, 2, 3, 4, 1), torch.zeros(1, 1), None)
    model.t_embedder = [lambda t: t, lambda t: (t.float(), None)]
    model.t_embedding_norm = torch.nn.Identity()
    model.final_layer = lambda x, t, adaln_lora_B_T_3D=None: x
    model.unpatchify = lambda x: x.permute(0, 4, 1, 2, 3)
    return model


def test_blocks_see_their_index_and_caller_options_stay_untouched():
    seen = []
    model = _stub_anima(3, seen)
    options = {"other": 1}

    model(torch.randn(1, 4, 1, 2, 2), torch.tensor([0.5]), torch.randn(1, 5, 8), transformer_options=options)

    assert seen == [0, 1, 2]
    assert options == {"other": 1}
