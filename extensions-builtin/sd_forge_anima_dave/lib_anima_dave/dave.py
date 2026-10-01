"""DAVE (DC Attenuation for diVersity Enhancement, ICML 2026, arXiv:2606.06813).

Early in sampling, the spatial mean (the DC component) of the DiT block
features becomes nearly identical across seeds and locks every seed into the
same layout. DAVE removes a fraction of that mean from selected block outputs
during the first ``tau`` of the schedule::

    h <- h - strength * mean_tokens(h)

leaving the seed-specific residual ``h - mean`` (the AC part) untouched.
Ported from ComfyUI-Anima-DAVE (MIT), whose block band (2B blocks 8-18) and
defaults come from the author's per-block probe of Anima 2B.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from modules.anima_lora_support import REFERENCE_BLOCKS, anima_model_block, sigma_progress
from modules.anima_support import clamp, generated_token_frames, parse_block_list, to_float, windowed_override_patch
from modules.forward_override import cached_overrides

# The reference band and defaults are written for the REFERENCE_BLOCKS (2B) layout.
DEFAULT_BLOCKS = "8-18"
DEFAULT_STRENGTH = 0.30
DEFAULT_TAU = 0.10

_PATCH_LOCK = threading.RLock()


def attenuate_dc(x: torch.Tensor, strength: float, frames: int | None = None) -> torch.Tensor:
    """Remove ``strength`` of the per-sample, per-channel token mean of a
    ``(B, T, H, W, D)`` block output.

    Only the first ``frames`` token frames (the generated image) are averaged
    and edited; reference-latent frames an edit model appends after them keep
    their features. Anima's sequence holds no text or padding-only tokens."""

    target = x if frames is None or frames >= x.shape[1] else x[:, :frames]
    dims = tuple(range(1, target.ndim - 1)) or (1,)
    # fp32 accumulation without an fp32 copy of the block output
    mean = target.mean(dim=dims, keepdim=True, dtype=torch.float32)
    edited = target - (float(strength) * mean).to(x.dtype)
    return edited if target is x else torch.cat((edited, x[:, frames:]), dim=1)


def window_steps(total_steps: int, tau: float) -> int:
    """Number of leading schedule steps DAVE runs on (at least one)."""

    tau = clamp(to_float(tau, DEFAULT_TAU), 0.0, 1.0)
    if tau >= 1.0:
        return total_steps
    return max(1, min(total_steps, round(tau * total_steps)))


def in_window(sigma, sigmas, tau: float) -> bool:
    """Whether the model call at ``sigma`` is among the first ``tau`` of the
    full schedule ``sigmas``. Hires. fix and img2img start part-way through
    it, so they only get DAVE when their denoise reaches the window."""

    progress = sigma_progress(sigma, sigmas)
    if progress is None:
        return False
    total_steps = len(sigmas) - 1
    return round(progress * total_steps) < window_steps(total_steps, tau)


def model_blocks(text: str, total_blocks: int) -> tuple[int, ...]:
    """Model block indices for a block list written in the 2B layout.

    On the depth-expanded 2.9B/3.8B models each 2B block maps to its
    original (not an inserted copy), so the number of attenuated blocks and
    hence the compounded dose stay those the reference was tuned for."""

    selected = parse_block_list(str(text), REFERENCE_BLOCKS, feature="Anima DAVE")
    mapped = {anima_model_block(index, REFERENCE_BLOCKS, total_blocks) for index in selected}
    blocks = tuple(sorted(index for index in mapped if index < total_blocks))
    if not blocks:
        raise ValueError("Anima DAVE block selection is empty")
    return blocks


@dataclass
class DaveState:
    blocks: tuple[int, ...]
    strength: float
    tau: float
    # Full schedule of the pass, read from the model calls' options
    schedule: object = None
    # Generated-image token frames of the current call
    frames: int | None = None

    def active(self, sigma) -> bool:
        return self.strength > 0.0 and self.schedule is not None and in_window(sigma, self.schedule, self.tau)


def _make_block_forward(original_forward, state: DaveState):
    def forward(*args, **kwargs):
        return attenuate_dc(original_forward(*args, **kwargs), state.strength, state.frames)

    return forward


def apply_dave_patch(model, state: DaveState):
    dit = model.model.diffusion_model  # shared by the clone
    override = cached_overrides(lambda index: dit.blocks[index], lambda original, _index: _make_block_forward(original, state))

    def gate(args):
        options = args.get("c", {}).get("transformer_options", {})
        state.schedule = options.get("sampling_sigmas", state.schedule)
        # Stateless and gated by its own sigma, so a PC3 endpoint probe sees
        # the same model as the step it predicts: no probe special case.
        return args["input"] if state.active(args["timestep"]) else None

    def begin(input_x):
        state.frames = generated_token_frames(input_x, getattr(dit, "patch_temporal", 1))

    def end():
        state.frames = None

    # Spectrum may ask before the first call has published the schedule:
    # force until the window is known.
    def force_actual(sigma):
        return state.schedule is None or state.active(sigma)

    # Outermost: DAVE attenuates the whole block output, including what
    # another block override adds (the cross-attention boost's norm
    # matching), whichever of the two wrappers runs first. A token mean works
    # at any latent size (Anima SPEED's half-res stage).
    return windowed_override_patch(
        model,
        gate=gate,
        begin=begin,
        overrides=lambda _input_x: (override(index) for index in state.blocks),
        end=end,
        lock=_PATCH_LOCK,
        force_actual=force_actual,
        outermost=True,
    )
