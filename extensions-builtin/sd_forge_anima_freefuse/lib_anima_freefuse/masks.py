"""FreeFuse balanced assignment and mask cleanup, adapted for Anima grids."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize(value: torch.Tensor) -> torch.Tensor:
    low = value.amin(dim=-1, keepdim=True)
    high = value.amax(dim=-1, keepdim=True)
    return (value - low) / (high - low).clamp_min(1e-8)


def fit_mask_batch(mask: torch.Tensor, batch: int) -> torch.Tensor:
    """Repeat per-image masks across CFG branches without mixing images."""

    mask_batch = int(mask.shape[0])
    if mask_batch == batch:
        return mask
    if mask_batch <= 0 or batch % mask_batch:
        raise RuntimeError(
            f"Anima FreeFuse mask batch {mask_batch} does not match model batch {batch}"
        )
    return mask.repeat(batch // mask_batch, 1, 1)


def stabilized_balanced_argmax(
    logits: torch.Tensor, height: int, width: int, iterations: int
) -> torch.Tensor:
    """Official FreeFuse-style balanced assignment with spatial regularity."""

    batch, concepts, tokens = logits.shape
    if tokens != height * width:
        raise ValueError(
            f"FreeFuse mask grid {height}x{width} does not match {tokens} tokens"
        )
    dtype, device = logits.dtype, logits.device
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.3, 1.3, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    flat_y = grid_y.reshape(1, 1, tokens)
    flat_x = grid_x.reshape(1, 1, tokens)
    running = _normalize(logits)
    bias = torch.zeros((batch, concepts, 1), device=device, dtype=dtype)
    logit_scale = max(float((logits.max() - logits.min()).item()), 1e-4)
    effective_lr = 1e-5 * logit_scale
    kernel = torch.ones((concepts, 1, 3, 3), device=device, dtype=torch.float32) / 8.0
    kernel[:, :, 1, 1] = 0.0
    current = logits.clone()
    target = tokens / concepts

    for index in range(max(1, int(iterations))):
        probabilities = _normalize(current - bias)
        running = 0.8 * probabilities + 0.2 * running
        mass = running.sum(dim=2, keepdim=True).clamp_min(1e-6)
        center_y = (running * flat_y).sum(dim=2, keepdim=True) / mass
        center_x = (running * flat_x).sum(dim=2, keepdim=True) / mass
        distance = (flat_y - center_y).square() + (flat_x - center_x).square()
        hard = (current - bias).argmax(dim=1)
        counts = F.one_hot(hard, num_classes=concepts).float().sum(dim=1)
        step = effective_lr * (0.95**index)
        bias = (bias + torch.sign(counts - target).unsqueeze(2) * step).clamp(
            -10.0 * logit_scale,
            10.0 * logit_scale,
        )
        neighbors = F.conv2d(
            running.float().view(batch, concepts, height, width),
            kernel,
            padding=1,
            groups=concepts,
        )
        current = (
            logits
            - bias
            + neighbors.to(dtype).view(batch, concepts, tokens) * 4e-5
            - distance * 1e-5
        )
    return current.argmax(dim=1)


def _morphological_clean(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    image = mask.float().view(mask.shape[0], 1, height, width)

    def dilate(value):
        result = F.max_pool2d(value, 2, stride=1, padding=1)
        if result.shape[-2:] != (height, width):
            result = F.interpolate(result, size=(height, width), mode="nearest")
        return result

    def erode(value):
        return 1.0 - dilate(1.0 - value)

    return erode(dilate(dilate(erode(image)))).view(mask.shape[0], -1).to(mask.dtype)


def generate_masks(
    similarity_maps: dict[str, torch.Tensor],
    background_map: torch.Tensor,
    *,
    height: int,
    width: int,
    bg_scale: float,
    iterations: int,
    feather: int,
) -> dict[str, torch.Tensor]:
    names = list(similarity_maps)
    if len(names) < 2:
        raise ValueError("FreeFuse requires at least two collected concepts")
    batches = {int(similarity_maps[name].shape[0]) for name in names}
    batches.add(int(background_map.shape[0]))
    if len(batches) != 1:
        raise ValueError("FreeFuse similarity maps have inconsistent batch sizes")
    batch = batches.pop()
    maps = torch.stack(
        [similarity_maps[name].reshape(batch, -1) for name in names], dim=1
    )
    tokens = height * width
    if maps.shape[-1] != tokens or background_map.numel() != batch * tokens:
        raise ValueError("FreeFuse similarity map has the wrong Anima spatial size")

    foreground_logits = torch.cat(
        [maps, background_map.reshape(batch, 1, tokens) * float(bg_scale)],
        dim=1,
    )
    foreground = (foreground_logits.argmax(dim=1) != len(names)).to(maps.dtype)
    foreground = _morphological_clean(foreground, height, width)
    assignment = stabilized_balanced_argmax(maps, height, width, iterations)

    result = {}
    for index, name in enumerate(names):
        mask = ((assignment == index).to(maps.dtype) * foreground).view(
            batch, 1, height, width
        )
        radius = max(0, int(feather))
        if radius:
            kernel = radius * 2 + 1
            mask = F.avg_pool2d(mask, kernel, stride=1, padding=radius)
        result[name] = mask[:, 0].clamp(0.0, 1.0)
    return result
