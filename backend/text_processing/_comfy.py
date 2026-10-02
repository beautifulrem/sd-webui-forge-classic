# https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/sd1_clip.py
# Shared encoder core adapted from Forge Neo's TextProcessingEngine rewrite.
# Tokenization stays in the Remi engines to preserve extension interfaces.

import numbers
from typing import Final, TypeAlias

import torch

from backend import memory_management

EMBEDDINGS: TypeAlias = list[int]
TOKEN_WEIGHTS: TypeAlias = list[list[tuple[int, float]]]

INF: Final[int] = 99999999

def gen_empty_tokens(special_tokens: dict[str, int], length: int) -> list[int]:
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output

class ClipTokenWeightEncoder:
    def encode_token_weights(self: "SDClipModel", token_weight_pairs: TOKEN_WEIGHTS):
        to_encode = []
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)
        sections = len(to_encode)
        if has_weights or sections == 0:
            if hasattr(self, "gen_empty_tokens"):
                to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
            else:
                to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        o = self.forward(to_encode)
        out, pooled = o[:2]
        if pooled is not None:
            first_pooled = pooled[0:1].to(device=memory_management.intermediate_device())
        else:
            first_pooled = pooled
        output = []
        for k in range(0, sections):
            z = out[k : k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)
        if len(output) == 0:
            r = (out[-1:].to(device=memory_management.intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(device=memory_management.intermediate_device()), first_pooled)
        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0).to(device=memory_management.intermediate_device())
                extra[k] = v

            r = r + (extra,)
        return r


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    def __init__(self, model: torch.nn.Module, max_length=77, layer="last", layer_idx=None, special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True, enable_attention_masks=False, zero_out_masked=False, return_projected_pooled=True, return_attention_masks=False):
        super().__init__()

        self.transformer = model
        self.freeze()
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.zero_out_masked = zero_out_masked
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.return_attention_masks = return_attention_masks

        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            if isinstance(self.layer, list) or self.layer == "all":
                pass
            elif isinstance(layer_idx, list):
                self.layer = layer_idx
            elif layer_idx is None or abs(layer_idx) > self.num_layers:
                self.layer = "last"
            else:
                self.layer = "hidden"
                self.layer_idx = layer_idx
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def process_tokens(self, tokens, device):
        end_token = self.special_tokens.get("end", None)
        pad_token = self.special_tokens.get("pad", -1)
        if end_token is None:
            cmp_token = pad_token
        else:
            cmp_token = end_token

        embeds_out = []
        attention_masks = []
        num_tokens = []
        for x in tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0
            left_pad = False
            for y in x:
                if isinstance(y, numbers.Integral):
                    token = int(y)
                    if index == 0 and token == pad_token:
                        left_pad = True
                    if eos or (left_pad and token == pad_token):
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)
                        left_pad = False
                    tokens_temp += [token]
                    if not eos and token == cmp_token and not left_pad:
                        if end_token is None:
                            attention_mask[-1] = 0
                        eos = True
                else:
                    other_embeds.append((index, y))
                index += 1
            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.transformer.get_input_embeddings()(tokens_embed).to(dtype=torch.float32)
            index = 0
            pad_extra = 0
            embeds_info = []
            for o in other_embeds:
                emb = o[1]
                if torch.is_tensor(emb):
                    emb = {"type": "embedding", "data": emb}
                extra = None
                emb_type = emb.get("type", None)
                if emb_type == "embedding":
                    emb = emb.get("data", None)
                else:
                    if hasattr(self.transformer, "preprocess_embed"):
                        emb, extra = self.transformer.preprocess_embed(emb, device=device)
                    else:
                        emb = None

                if emb is None:
                    index += -1
                    continue
                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]
                if emb.shape[-1] == tokens_embed.shape[-1]:
                    tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                    attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                    index += emb_shape - 1
                    embeds_info.append({"type": emb_type, "index": ind, "size": emb_shape, "extra": extra})
                else:
                    index += -1
                    pad_extra += emb_shape
                    memory_management.logger.warning("Shape mismatch when applying embedding, embedding will be ignored ({} != {})".format(emb.shape[-1], tokens_embed.shape[-1]))
            if pad_extra > 0:
                padd_embed = self.transformer.get_input_embeddings()(torch.tensor([[self.special_tokens["pad"]] * pad_extra], device=device, dtype=torch.long)).to(dtype=torch.float32)
                tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
                attention_mask = attention_mask + [0] * pad_extra

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))
        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens, embeds_info

    def forward(self, tokens):
        # Offloaded weights may reside on CPU while the configured encoder runs
        # on an accelerator. Match the model patcher's execution device.
        device = memory_management.text_encoder_device()

        embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(tokens, device)

        attention_mask_model = None
        if self.enable_attention_masks:
            attention_mask_model = attention_mask
        if isinstance(self.layer, list):
            intermediate_output = self.layer
        elif self.layer == "all":
            intermediate_output = "all"
        else:
            intermediate_output = self.layer_idx

        outputs = self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.float32, embeds_info=embeds_info)
        if self.layer == "last":
            z = outputs[0].float()
        else:
            z = outputs[1].float()

        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        extra = {}
        if self.return_attention_masks:
            extra["attention_mask"] = attention_mask

        if len(extra) > 0:
            return z, pooled_output, extra

        return z, pooled_output
