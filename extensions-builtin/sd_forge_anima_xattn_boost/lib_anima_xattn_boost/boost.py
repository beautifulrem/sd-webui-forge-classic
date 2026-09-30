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

from modules.anima_support import branch_rows, sigma_value, windowed_override_patch
from modules.forward_override import cached_overrides, forward_overrides

# anima_lora 703b34b defaults: --xattn_boost 2.0, band 0.85, renorm img, frac 0.5
DEFAULT_GAIN = 2.0
DEFAULT_BAND = 0.85
DEFAULT_RENORM = 0.5

_PATCH_LOCK = threading.RLock()
# Block.forward arguments after (x, emb, crossattn_emb), in order
_BLOCK_ARGS = ("rope_emb_L_1_1_D", "adaln_lora_B_T_3D", "extra_per_block_pos_emb", "transformer_options", "silu_emb_B_T_D")


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
    # Norms accumulate in float32 without a float32 copy of either tensor.
    norm_plain = torch.linalg.vector_norm(plain, dim=-1, keepdim=True, dtype=torch.float32).mean(dim=dims, keepdim=True)
    norm_boosted = torch.linalg.vector_norm(boosted, dim=-1, keepdim=True, dtype=torch.float32).clamp_min(1e-6).mean(dim=dims, keepdim=True)
    scale = norm_plain / norm_boosted
    return scale if frac == 1.0 else scale**frac


@dataclass
class BoostState:
    gain: float
    band: float
    renorm: float
    # (boosted rows, batch) of the current model call
    call: tuple | None = None
    # self/cross-attention outputs and self/cross AdaLN modulation Linear
    # outputs of the block being run (norm matching)
    captured: dict | None = None
    # (rows, batch, device, dtype) -> (B, 1, 1) row factors, built once per
    # pass on the device in the compute dtype (the rows repeat every call)
    cache: dict = field(default_factory=dict)

    def active(self, sigma) -> bool:
        if self.gain == 1.0:
            return False
        try:
            current = sigma_value(sigma)
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


def _make_capture(original, state: BoostState, key: str):
    """``original`` unchanged, its output recorded as ``state.captured[key]``
    while a norm-matched block call runs."""

    def forward(*args, **kwargs):
        out = original(*args, **kwargs)
        if state.captured is not None:
            state.captured[key] = out
        return out

    return forward


# captured key -> Block's AdaLN modulation MLP whose output Linear is captured
_MODULATIONS = (("modulation_self", "adaln_modulation_self_attn"), ("modulation_cross", "adaln_modulation_cross_attn"))


def _modulation_linear(block, name):
    """The Linear whose output ``Block.forward`` adds to ``adaln_lora`` and
    chunks (``mlp[2]``), or None for a block without that modulation MLP."""

    mlp = getattr(block, name, None)
    return mlp[2] if isinstance(mlp, torch.nn.Sequential) and len(mlp) == 3 else None


def _gate(modulation, adaln_lora):
    """The ``gate`` chunk of one AdaLN modulation Linear output, as
    ``Block.forward`` makes it."""

    if adaln_lora is not None:
        modulation = modulation + adaln_lora
    return modulation.chunk(3, dim=-1)[2][:, :, None, None, :]


def _modulation(captured, key, block, name, silu_emb):
    """The modulation Linear output this block call computed (captured), or,
    if the block did not run that Linear through the capture, a recompute."""

    out = captured.get(key)
    if out is not None:
        return out
    mlp = getattr(block, name)
    return mlp[2](mlp[1](silu_emb()))


def _make_block_forward(original, block, state: BoostState, attention_overrides):
    """The boost and norm matching around one block call.

    The attention overrides (``attention_overrides()`` pairs) go on for the
    block call only, installed at block time: the boost's cross-attention
    override then sits on top of every per-model-call one (Regional, Mixer,
    FreeFuse), whatever order their wrappers run in, and scales the
    cross-attention output the block actually adds.

    For norm matching the self/cross-attention outputs are captured to
    rebuild ``x`` (before cross-attention) and ``x_new``, and so are the
    self/cross AdaLN modulation Linear outputs for their gates, so those
    Linears (and an online LoRA's weight functions on them) run once per
    block call; a block that does not run them gets them recomputed. The
    reference scales ``x_new`` by ``s`` before the MLP; the MLP reads it through a
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

        def silu_emb():
            given = options.get("silu_emb_B_T_D")
            return torch.nn.functional.silu(emb) if given is None else given

        adaln_lora = options.get("adaln_lora_B_T_3D")
        dtype = x.dtype
        if options.get("extra_per_block_pos_emb") is not None:
            x = x + options["extra_per_block_pos_emb"]
        shape = x.shape
        (key_self, name_self), (key_cross, name_cross) = _MODULATIONS
        gate_self = _gate(_modulation(captured, key_self, block, name_self, silu_emb), adaln_lora).to(dtype)
        gate_cross = _gate(_modulation(captured, key_cross, block, name_cross, silu_emb), adaln_lora).to(dtype)
        x_mid = torch.addcmul(x, gate_self, captured["self"].unflatten(1, shape[1:4]).to(dtype))
        cross = gate_cross * captured["cross"].unflatten(1, shape[1:4]).to(dtype)
        plain = x_mid + cross
        # x_mid is this block's own temporary: boost into its buffer.
        boosted = x_mid.addcmul_(cross, state.gains_like(cross).view(-1, 1, 1, 1, 1))
        scale = renorm_scale(plain, boosted, state.renorm)
        return out + (scale - 1.0).to(out.dtype) * boosted.to(out.dtype)

    return forward


def apply_xattn_boost_patch(model, state: BoostState):
    dit = model.model.diffusion_model  # shared by the clone
    cross = cached_overrides(lambda index: dit.blocks[index].cross_attn, lambda original, _index: _make_cross_forward(original, state))
    capture = cached_overrides(lambda index: dit.blocks[index].self_attn, lambda original, _index: _make_capture(original, state, "self"))
    modulations = [
        (name, cached_overrides(lambda index, name=name: _modulation_linear(dit.blocks[index], name), lambda original, _index, key=key: _make_capture(original, state, key)))
        for key, name in _MODULATIONS
    ]

    def attention_overrides(index):
        # Installed for the block call only, like the self-attention capture.
        # Per-block compile never traces them: the block overrides, on for
        # the whole model call, already make it run the eager blocks.
        yield cross(index)
        if state.renorm > 0.0:
            yield capture(index)
            for name, modulation in modulations:
                if _modulation_linear(dit.blocks[index], name) is not None:
                    yield modulation(index)

    blocks = cached_overrides(
        lambda index: dit.blocks[index],
        lambda original, index: _make_block_forward(original, dit.blocks[index], state, lambda: attention_overrides(index)),
    )

    def gate(args):
        # Stateless and gated by its own sigma, so a PC3 endpoint probe sees
        # the same model as the step it predicts: no probe special case.
        if not state.active(args["timestep"]):
            return None
        input_x = args["input"]
        markers = args.get("cond_or_uncond")
        if markers is None:
            markers = args.get("c", {}).get("transformer_options", {}).get("cond_or_uncond")
        rows = branch_rows(markers, input_x.shape[0])
        # None: no positive rows, or a batch the markers do not describe
        return (tuple(rows), input_x.shape[0]) if rows else None

    def begin(call):
        state.call = call

    def end():
        state.call = None
        state.captured = None

    # The block overrides go on when this wrapper runs; DAVE's go on around
    # the model function (outermost), so DAVE attenuates the norm-matched
    # block output in either wrapper order. A per-token gain and a per-image
    # mean norm work at any latent size (Anima SPEED's half-res stage).
    return windowed_override_patch(
        model,
        gate=gate,
        begin=begin,
        overrides=lambda _call: (blocks(index) for index in range(len(dit.blocks))),
        end=end,
        lock=_PATCH_LOCK,
        force_actual=state.active,
    )
