"""Front-loaded cross-attention boost (``--xattn_boost`` of anima_lora).

While the plan of the image is written (sigma >= ``band``, where Anima's
cross-attention still carries text drive) every block's cross-attention
residual is scaled by ``gain`` on the conditional rows only, so the boost
lands in the CFG delta and not in the negative prompt::

    x_new = x + gain * gate * cross_attn(...)

Norm matching (the reference's shipped ``img`` mode) then rescales each
boosted image's hidden state toward the mean token norm it would have had
unboosted, ``x_new *= (mean|x + gate * attn| / mean|x_new|) ** frac``, so the
boost rotates the state toward the text direction instead of leaving the norm
shell the next blocks were trained on. ``frac`` 0 is the raw gain.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch

from modules import spectrum_force
from modules.anima_support import branch_rows, call_model_wrapper
from modules.forward_override import cached_overrides, forward_overrides

# anima_lora 703b34b defaults: --xattn_boost 2.0, band 0.85, renorm img, frac 0.5
DEFAULT_GAIN = 2.0
DEFAULT_BAND = 0.85
DEFAULT_RENORM = 0.5

_PATCH_LOCK = threading.RLock()
# Block.forward arguments after (x, emb, crossattn_emb), in order
_BLOCK_ARGS = ("rope_emb_L_1_1_D", "adaln_lora_B_T_3D", "extra_per_block_pos_emb", "transformer_options", "silu_emb_B_T_D")


def _sigma(value) -> float:
    return float(value.flatten()[0]) if torch.is_tensor(value) else float(value)


def row_gains(rows, batch: int, gain: float, device=None, dtype=None) -> torch.Tensor:
    """``(batch, 1, 1)`` factors: ``gain`` on ``rows``, 1 elsewhere."""

    gains = torch.ones(batch, 1, 1)
    gains[list(rows)] = float(gain)
    return gains.to(device=device, dtype=dtype)


def renorm_scale(plain: torch.Tensor, boosted: torch.Tensor, frac: float) -> torch.Tensor:
    """Per-image factor matching the mean token norm of ``boosted`` to that of
    ``plain`` (both ``(B, T, H, W, D)``), raised to ``frac``; float32
    ``(B, 1, 1, 1, 1)``. Rows where the two are equal get exactly 1."""

    dims = tuple(range(1, plain.ndim - 1))
    norm_plain = plain.float().norm(dim=-1, keepdim=True).mean(dim=dims, keepdim=True)
    norm_boosted = boosted.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).mean(dim=dims, keepdim=True)
    scale = norm_plain / norm_boosted
    return scale if frac == 1.0 else scale**frac


@dataclass
class BoostState:
    gain: float
    band: float
    renorm: float
    # (boosted rows, batch) of the current model call
    call: tuple | None = None
    # self/cross-attention outputs of the block being run (norm matching)
    captured: dict | None = None
    # (rows, batch, device, dtype) -> (B, 1, 1) row factors, built once per
    # pass on the device in the compute dtype (the rows repeat every call)
    cache: dict = field(default_factory=dict)

    def active(self, sigma) -> bool:
        if self.gain == 1.0:
            return False
        try:
            current = _sigma(sigma)
        except (TypeError, ValueError, IndexError):
            return False
        # sigma >= band, with the rounding slack of spectrum_force.in_sigma_window
        return current >= self.band - max(1e-8, self.band * 1e-7)

    def gains_like(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """The current call's ``(B, 1, 1)`` row factors on ``tensor``'s device
        in its dtype; None outside a call or for a call of another batch."""

        if self.call is None or self.call[1] != tensor.shape[0]:
            return None
        rows, batch = self.call
        key = (rows, batch, tensor.device, tensor.dtype)
        gains = self.cache.get(key)
        if gains is None:
            gains = self.cache[key] = row_gains(rows, batch, self.gain, tensor.device, tensor.dtype)
        return gains


def _make_cross_forward(original, state: BoostState):
    def forward(*args, **kwargs):
        out = original(*args, **kwargs)
        gains = state.gains_like(out)
        if gains is None:  # a call of another batch (e.g. a nested probe): untouched
            return out
        if state.captured is not None:
            state.captured["cross"] = out
        return out * gains

    return forward


def _make_self_capture(original, state: BoostState):
    def forward(*args, **kwargs):
        out = original(*args, **kwargs)
        if state.captured is not None:
            state.captured["self"] = out
        return out

    return forward


def _gate(mlp, silu_emb, adaln_lora):
    """The ``gate`` chunk of one AdaLN modulation, as ``Block.forward`` makes it."""

    modulation = mlp[2](mlp[1](silu_emb))
    if adaln_lora is not None:
        modulation = modulation + adaln_lora
    return modulation.chunk(3, dim=-1)[2][:, :, None, None, :]


def _make_block_forward(original, block, state: BoostState, attention_overrides):
    """The boost and norm matching around one block call.

    The attention overrides (``attention_overrides()`` pairs) go on for the
    block call only, installed at block time: the boost's cross-attention
    override then sits on top of every per-model-call one (Regional, Mixer,
    FreeFuse), whatever order their wrappers run in, and scales the
    cross-attention output the block actually adds.

    For norm matching the self/cross-attention outputs are captured to
    rebuild ``x`` (before cross-attention) and ``x_new``. The reference
    scales ``x_new`` by ``s`` before the MLP; the MLP reads it through a
    LayerNorm, which ``s`` does not change, so the result is
    ``out + (s - 1) * x_new``."""

    def forward(x, emb, crossattn_emb, *args, **kwargs):
        if state.call is None or state.call[1] != x.shape[0]:
            return original(x, emb, crossattn_emb, *args, **kwargs)
        if state.renorm <= 0.0:
            with forward_overrides(attention_overrides()):
                return original(x, emb, crossattn_emb, *args, **kwargs)
        state.captured = captured = {}
        try:
            with forward_overrides(attention_overrides()):
                out = original(x, emb, crossattn_emb, *args, **kwargs)
        finally:
            state.captured = None
        if "self" not in captured or "cross" not in captured:
            return out  # the block did not run its attentions through ours
        options = {**dict(zip(_BLOCK_ARGS, args)), **kwargs}
        silu_emb = options.get("silu_emb_B_T_D")
        if silu_emb is None:
            silu_emb = torch.nn.functional.silu(emb)
        adaln_lora = options.get("adaln_lora_B_T_3D")
        dtype = x.dtype
        if options.get("extra_per_block_pos_emb") is not None:
            x = x + options["extra_per_block_pos_emb"]
        shape = x.shape
        gate_self = _gate(block.adaln_modulation_self_attn, silu_emb, adaln_lora).to(dtype)
        gate_cross = _gate(block.adaln_modulation_cross_attn, silu_emb, adaln_lora).to(dtype)
        x_mid = torch.addcmul(x, gate_self, captured["self"].unflatten(1, shape[1:4]).to(dtype))
        cross = gate_cross * captured["cross"].unflatten(1, shape[1:4]).to(dtype)
        boosted = x_mid + cross * state.gains_like(cross).view(-1, 1, 1, 1, 1)
        scale = renorm_scale(x_mid + cross, boosted, state.renorm)
        return out + (scale - 1.0).to(out.dtype) * boosted.to(out.dtype)

    return forward


def apply_xattn_boost_patch(model, state: BoostState):
    patched = model.clone()
    dit = patched.model.diffusion_model
    previous = patched.model_options.get("model_function_wrapper")
    cross = cached_overrides(lambda index: dit.blocks[index].cross_attn, lambda original, _index: _make_cross_forward(original, state))
    capture = cached_overrides(lambda index: dit.blocks[index].self_attn, lambda original, _index: _make_self_capture(original, state))

    def attention_overrides(index):
        yield cross(index)
        if state.renorm > 0.0:
            yield capture(index)

    blocks = cached_overrides(
        lambda index: dit.blocks[index],
        lambda original, index: _make_block_forward(original, dit.blocks[index], state, lambda: attention_overrides(index)),
    )

    def wrapper(model_function, args):
        # Stateless and gated by its own sigma, so a PC3 endpoint probe sees
        # the same model as the step it predicts: no probe special case.
        if not state.active(args["timestep"]):
            return call_model_wrapper(previous, model_function, args)
        input_x = args["input"]
        markers = args.get("cond_or_uncond")
        if markers is None:
            markers = args.get("c", {}).get("transformer_options", {}).get("cond_or_uncond")
        rows = branch_rows(markers, input_x.shape[0])
        if not rows:  # no positive rows, or a batch the markers do not describe
            return call_model_wrapper(previous, model_function, args)
        with _PATCH_LOCK:
            state.call = (tuple(rows), input_x.shape[0])
            try:
                with forward_overrides(blocks(index) for index in range(len(dit.blocks))):
                    return call_model_wrapper(previous, model_function, args)
            finally:
                state.call = None
                state.captured = None

    # Outside its band the wrapper is a pass-through, and the request below
    # forces actual forwards inside it, so Spectrum may cache the other steps.
    wrapper.__spectrum_cache_safe__ = previous is None or getattr(previous, "__spectrum_cache_safe__", False)
    # A per-token gain and a per-image mean norm work at any latent size
    # (Anima SPEED's half-res stage).
    wrapper.__resolution_independent__ = previous is None or getattr(previous, "__resolution_independent__", False)
    patched.set_model_unet_function_wrapper(wrapper)
    spectrum_force.request(patched, state.active)
    return patched
