"""Tensor helpers of the Anima Artist Scheduled Mixer: aligning artist
contexts and NegPiP masks with the base context, and blending outputs."""

from __future__ import annotations

import torch

from modules.anima_support import NEGPIP_CONTEXT_MASK_KEY, branch_rows


def ensure_3d(tensor):
    if tensor is None:
        return None
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    if tensor.dim() == 3:
        return tensor
    return tensor.reshape(1, tensor.shape[-2], tensor.shape[-1])


def to_context(tensor, context):
    tensor = ensure_3d(tensor)
    return tensor.to(device=context.device, dtype=context.dtype, non_blocking=True)


def broadcast_batch(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1, -1)
    if batch_size % tensor.shape[0] == 0:
        return tensor.repeat(batch_size // tensor.shape[0], 1, 1)
    return tensor[:1].expand(batch_size, -1, -1)


def to_context_like(tensor, context):
    artist = broadcast_batch(to_context(tensor, context), context.shape[0])
    if artist.shape[-1] != context.shape[-1]:
        raise RuntimeError(f"Artist context dim {artist.shape[-1]} does not match base context dim {context.shape[-1]}")
    while artist.dim() < context.dim():
        artist = artist.unsqueeze(1)
    if artist.dim() != context.dim():
        raise RuntimeError(f"Artist context rank {artist.dim()} does not match base context rank {context.dim()}")
    expand_shape = list(artist.shape)
    for dim in range(1, context.dim() - 2):
        if artist.shape[dim] == context.shape[dim]:
            continue
        if artist.shape[dim] != 1:
            raise RuntimeError(f"Cannot align artist context shape {tuple(artist.shape)} to base context shape {tuple(context.shape)}")
        expand_shape[dim] = context.shape[dim]
    return artist.expand(*expand_shape)


def context_token_dim(context):
    return max(1, context.dim() - 2)


def concat_contexts(base_context, artist_context):
    return torch.cat([base_context, artist_context], dim=context_token_dim(base_context))


def artist_mask_like(mask, artist_context):
    """An artist's NegPiP V-mask aligned with ``artist_context``, or None."""

    if mask is None:
        return None
    return broadcast_batch(to_context(mask, artist_context), artist_context.shape[0])


def joined_mask(parts, dim):
    """NegPiP mask for contexts joined along ``dim``; ones fill parts without one."""

    if all(mask is None for _, mask in parts):
        return None
    masks = [
        mask if mask is not None else torch.ones_like(part[..., :1])
        for part, mask in parts
    ]
    return torch.cat(masks, dim=dim)


def with_negpip_mask(transformer_options, mask):
    """Options for a foreign context: its own mask replaces the base prompt's."""

    options = dict(transformer_options) if isinstance(transformer_options, dict) else {}
    options[NEGPIP_CONTEXT_MASK_KEY] = mask
    return options


def norm_limited_delta(base_out, artist_out, strength, max_ratio=0.6):
    delta = artist_out - base_out
    if delta.numel() == 0:
        return base_out
    with torch.no_grad():
        flat_delta = delta.detach().float().flatten(1)
        flat_base = base_out.detach().float().flatten(1)
        delta_norm = flat_delta.norm(dim=1).clamp_min(1e-6)
        base_norm = flat_base.norm(dim=1).clamp_min(1e-6)
        limit = base_norm * float(max_ratio)
        scale = torch.minimum(torch.ones_like(delta_norm), limit / delta_norm)
        shape = [scale.shape[0]] + [1] * (delta.dim() - 1)
    return base_out + delta * scale.to(device=delta.device, dtype=delta.dtype).view(*shape) * float(strength)


def resolve_row_mask(cond_or_uncond, batch_size, apply_uncond):
    """Rows to mix into: the positive-prompt rows, or every row with
    ``apply_uncond`` or when the markers do not describe the batch."""
    rows = branch_rows(cond_or_uncond, batch_size)
    if rows is None or apply_uncond:
        return [True] * batch_size
    rows = set(rows)
    return [row in rows for row in range(batch_size)]
