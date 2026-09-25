"""Normalized Attention Guidance for Anima cross-attention."""

from __future__ import annotations

import torch

from backend.nn.anima_attention import ANIMA_PROJECT_KV
from modules.spectrum_force import in_sigma_window


def _select_batch(value: torch.Tensor | None, indices: list[int]):
    if value is None or value.ndim == 0 or value.shape[0] == 1:
        return value
    if value.shape[0] > max(indices, default=-1):
        return value[indices]
    return value


class NAGAttentionModifier:
    """Apply NAG only to positive cross-attention rows in a CFG batch."""

    def __init__(
        self,
        *,
        scale: float = 2.0,
        tau: float = 2.5,
        alpha: float = 0.5,
        sigma_start: float,
        sigma_end: float,
        on_unbatched=None,
    ):
        self.scale = float(scale)
        self.tau = max(float(tau), 1e-6)
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self.sigma_low, self.sigma_high = sorted(
            (float(sigma_start), float(sigma_end))
        )
        self.on_unbatched = on_unbatched
        self._reported_unbatched = False
        # The negative prompt's text context for the current step, set by the
        # CFG denoiser callback. Lets NAG run on positive-only model calls
        # (CFG 1, e.g. Anima-Turbo) without a full unconditional pass.
        self.negative_context: torch.Tensor | None = None

    def set_negative_context(self, context) -> None:
        if isinstance(context, dict):
            context = context.get("crossattn", context.get("c_crossattn"))
        self.negative_context = context if torch.is_tensor(context) else None

    def _report_unbatched(self) -> None:
        if self._reported_unbatched:
            return
        self._reported_unbatched = True
        if self.on_unbatched is not None:
            self.on_unbatched()

    def _active(self, transformer_options: dict) -> bool:
        if transformer_options.get("anima_nag_skip"):
            return False
        sigmas = transformer_options.get("sigmas")
        if sigmas is None:
            return True
        sigma = float(sigmas.detach().flatten()[0]) if torch.is_tensor(sigmas) else float(sigmas)
        return in_sigma_window(sigma, self.sigma_low, self.sigma_high)

    @torch.no_grad()
    def __call__(
        self,
        next_attention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask=None,
        transformer_options: dict,
        is_self_attention: bool,
    ) -> torch.Tensor:
        base = next_attention(q, k, v, mask=mask)
        if self.scale == 0.0 or self.alpha == 0.0 or is_self_attention or not self._active(transformer_options):
            return base

        markers = tuple(transformer_options.get("cond_or_uncond", ()))
        if 0 in markers and 1 not in markers and q.shape[0] % len(markers) == 0:
            projected = self._negative_kv(transformer_options, q)
            if projected is not None:
                group = q.shape[0] // len(markers)
                positive = [i * group + j for i, mark in enumerate(markers) if mark == 0 for j in range(group)]
                k_negative, v_negative = projected
                rows = [index % k_negative.shape[0] for index in range(len(positive))]
                negative_attention = next_attention(q[positive], k_negative[rows], v_negative[rows], mask=None)
                return self._blend(base, positive, negative_attention)
        if 0 not in markers or 1 not in markers or q.shape[0] % len(markers) != 0:
            self._report_unbatched()
            return base

        group = q.shape[0] // len(markers)
        positive = [i * group + j for i, mark in enumerate(markers) if mark == 0 for j in range(group)]
        negative_group = next((i for i, mark in enumerate(markers) if mark == 1), None)
        if negative_group is None:
            return base
        negative = [negative_group * group + (index % group) for index in range(len(positive))]

        negative_attention = next_attention(
            q[positive],
            k[negative],
            v[negative],
            mask=_select_batch(mask, negative),
        )
        return self._blend(base, positive, negative_attention)

    def _negative_kv(self, transformer_options: dict, q: torch.Tensor):
        project = transformer_options.get(ANIMA_PROJECT_KV)
        context = self.negative_context
        if project is None or context is None:
            return None
        return project(context.to(device=q.device, dtype=q.dtype))

    def _blend(self, base: torch.Tensor, positive: list[int], negative_attention: torch.Tensor) -> torch.Tensor:
        positive_attention = base[positive]
        guided = positive_attention + self.scale * (positive_attention - negative_attention)
        norm_positive = torch.linalg.vector_norm(
            positive_attention, ord=1, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        norm_guided = torch.linalg.vector_norm(
            guided, ord=1, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        ratio = norm_guided / norm_positive
        normalized = torch.minimum(ratio, torch.full_like(ratio, self.tau)) / ratio * guided
        blended = positive_attention.lerp(normalized, self.alpha)

        result = base.clone()
        result[positive] = blended
        return result
