"""Sampling runtime of the Anima Artist Scheduled Mixer: the artist
conditioning cache, the per-pass state, the model wrapper that installs the
artist cross-attention overrides for each model call, and the dispatch that
mixes the artist outputs into the base output."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lib_artist_mixer.contexts import (
    to_context as _to_context,
    to_context_like as _to_context_like,
    context_token_dim as _context_token_dim,
    concat_contexts as _concat_contexts,
    artist_mask_like as _artist_mask_like,
    joined_mask as _joined_mask,
    norm_limited_delta as _norm_limited_delta,
)
from lib_artist_mixer.schedule import curve_factor
from modules import spectrum_force
from modules.anima_lora_support import clamp as _clamp, sigma_progress
from modules.anima_support import NEGPIP_CONTEXT_MASK_KEY, NEGPIP_MASK_KEY, BranchRows, call_model_wrapper, conditioning_cache_key, encode_extra_prompts, negpip_mask_for, positive_rows, schedule_index
from modules.forward_override import cached_overrides, forward_overrides


logger = logging.getLogger("anima_artist_scheduled_mixer")

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"
FUSION_QUALITY_DELTA = "quality_delta"
# quality_delta is already anchored by comparing artist_out against base_out.
# Concatenating the base context again dilutes the artist token influence.
BASE_CONTEXT_FUSIONS = {FUSION_CONCAT_WITH_BASE}
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_CONCAT = "concat"


_COND_CACHE = OrderedDict()
_COND_CACHE_LIMIT = 64
_COND_CACHE_BYTES = 256 * 1024 * 1024  # CPU memory the cached artist conditionings may hold


def _cache_key(p, texts):
    return conditioning_cache_key(p), tuple(texts)


def _encode_text_batch(p, texts, use_cache=True):
    key = _cache_key(p, texts)
    if use_cache and key in _COND_CACHE:
        _COND_CACHE.move_to_end(key)
        return _COND_CACHE[key]
    # With NegPiP the mask gives the artist context the same negative-weight
    # semantics as the prompt.
    result = encode_extra_prompts(p, texts)
    if use_cache:
        _COND_CACHE[key] = result

        def cached_bytes():
            return sum(cond.nbytes + (0 if mask is None else mask.nbytes) for cond, mask in _COND_CACHE.values())

        while len(_COND_CACHE) > 1 and (len(_COND_CACHE) > _COND_CACHE_LIMIT or cached_bytes() > _COND_CACHE_BYTES):
            _COND_CACHE.popitem(last=False)
    return result


@dataclass
class ArtistRuntime:
    name: str
    weight: float
    blocks: set[int]
    start: float
    end: float
    peak: float
    curve: str  # a CURVE_* key (_build_artists resolves the panel's label)
    cond: torch.Tensor
    # NegPiP V-mask for this artist's tokens (None without negative weights).
    negpip_mask: torch.Tensor | None = None
    # Device copies of cond / mask for the attention context's device and
    # dtype (kept on the CPU otherwise): uploaded once, not per block
    on_device: dict = field(default_factory=dict, repr=False)
    # Prompt editing ([a:b:N], [a|b]) in the base prompts: the end steps of
    # the batch's schedule and ``(cond, negpip_mask)`` of each entry. Empty
    # when the prompts have one text: cond / negpip_mask then serve every step.
    ends: tuple[int, ...] = ()
    entries: tuple = field(default=(), repr=False)

    def at(self, step):
        """``(cond, negpip_mask)`` at sampler ``step``."""
        if not self.ends:
            return self.cond, self.negpip_mask
        return self.entries[schedule_index(self.ends, step)]

    def cond_like(self, context, step=0):
        return self._like("cond", self.at(step)[0], context)

    def mask_like(self, context, step=0):
        mask = self.at(step)[1]
        return None if mask is None else self._like("mask", mask, context)

    def _like(self, name, tensor, context):
        # By tensor: the entries of one text (every other step of [a|b]) share it
        key = (name, id(tensor), context.device, context.dtype)
        if key not in self.on_device:
            self.on_device[key] = _to_context(tensor, context)
        return self.on_device[key]


@dataclass
class MixerState:
    run_id: str
    enabled: bool
    global_strength: float
    combine_mode: str
    fusion_mode: str
    apply_uncond: bool
    artists: list[ArtistRuntime] = field(default_factory=list)
    progress_warning: bool = False
    # Output average runs the artists in one batched forward until it fails once
    batched_disabled: bool = False
    patched_blocks: list[int] = field(default_factory=list)
    dispatch_calls: int = 0
    active_calls: int = 0
    fallback_calls: int = 0
    wrapper_checks: int = 0
    diff_probe: tuple[float, float] | None = None
    mask_probe: tuple[int, int] | None = None
    # The generation's infotext, to report a runtime failure
    infotext: dict | None = None
    # Row masks already on the device, keyed by (mask, device)
    row_masks: dict = field(default_factory=dict)
    # Aligned artist contexts of the current model call (see _artist_kv)
    call_contexts: dict = field(default_factory=dict)
    # (sigmas, sampling_sigmas, progress) of the current model call: every
    # block of a call sees the same tensors, so progress is read once
    progress_cache: tuple | None = None
    # Sampling of its pass is over: its model wrapper passes through, so a
    # patcher clone that outlives the pass cannot mix stale artists
    finished: bool = False
    # The sampler step of the current model call (``anima_support.prompt_step``),
    # read only when the artists carry prompt-editing entries
    step_of: Callable[[], int] | None = None
    prompt_step: int = 0

    def target_blocks(self):
        blocks = set()
        for artist in self.artists:
            blocks.update(artist.blocks)
        return blocks

    def release_device_tensors(self):
        """Drop the device copies and per-call tensors; the run summary's
        counters stay."""
        self.call_contexts.clear()
        self.row_masks.clear()
        self.progress_cache = None
        for artist in self.artists:
            artist.on_device.clear()


# The passes of the current processing run, for postprocess's summary
_RUN_STATES = []


def _finish_run_states():
    """End sampling for the passes so far; their states stay for the summary."""
    for state in _RUN_STATES:
        state.finished = True


def _cleanup_runtime():
    states = list(_RUN_STATES)
    _RUN_STATES.clear()
    for state in states:  # device tensors of the finished run
        state.finished = True
        state.release_device_tensors()
    return states


def _validate_anima_unet(unet):
    try:
        dm = unet.model.diffusion_model
    except Exception:
        return None, "Cannot find diffusion_model"
    blocks = getattr(dm, "blocks", None)
    if not blocks:
        return None, "diffusion_model.blocks is empty"
    first = blocks[0]
    if not hasattr(first, "cross_attn"):
        return None, "blocks[0] has no cross_attn"
    if not hasattr(first.cross_attn, "context_dim"):
        return None, "cross_attn has no context_dim"
    return dm, "ok"


def _summarize_blocks(blocks):
    values = sorted(set(int(x) for x in blocks))
    if not values:
        return "none"
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _target_blocks(dm, state):
    """Indices of the blocks whose cross-attention some artist mixes into."""
    targets = state.target_blocks()
    return [idx for idx, block in enumerate(getattr(dm, "blocks", [])) if idx in targets and hasattr(block, "cross_attn")]


def _make_cross_attn_wrapper(original_forward, layer_idx, state):
    def wrapped(x, context=None, rope_emb=None, transformer_options={}):
        return _dispatch_cross_attn(original_forward, layer_idx, state, x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)

    return wrapped


def _install_model_wrapper(unet, dm, state):
    """Mix through ``unet``'s model wrapper: like Regional's, it installs the
    artist cross-attn overrides for each model call and restores them after
    it (also when it raises), so nothing outside sampling sees them."""
    options = getattr(unet, "model_options", None)
    if not isinstance(options, dict):
        return
    blocks = _target_blocks(dm, state)
    state.patched_blocks = blocks
    if not blocks:
        logger.warning("Anima artist mixer found no target cross-attn blocks to patch.")
        return
    logger.info("Anima artist mixer patches %d cross-attn blocks: %s", len(blocks), _summarize_blocks(blocks))
    previous = options.get("model_function_wrapper")
    override = cached_overrides(lambda index: dm.blocks[index].cross_attn, lambda original, index: _make_cross_attn_wrapper(original, index, state))

    def model_wrapper(apply_model, args):
        if state.finished or not state.enabled:
            return call_model_wrapper(previous, apply_model, args)
        state.wrapper_checks += 1
        if state.step_of is not None and any(artist.ends for artist in state.artists):
            step = state.step_of()
            if step != state.prompt_step:
                state.prompt_step = step
                state.call_contexts.clear()  # built from the previous entry's contexts
        with forward_overrides(override(index) for index in blocks):
            return call_model_wrapper(previous, apply_model, args)

    options["model_function_wrapper"] = model_wrapper
    # Artist contexts and per-step curves change the block features in a way
    # Spectrum cannot infer; it wraps this wrapper, so set the flag on the
    # patcher. Artist windows depend on sampler progress, not sigma alone,
    # so the whole pass is forced.
    spectrum_force.request(unet, "artist_mixer")


def _current_progress(transformer_options, state):
    if not isinstance(transformer_options, dict):
        return None
    cur = transformer_options.get("sigmas")
    all_sigmas = transformer_options.get("sampling_sigmas")
    if cur is None or all_sigmas is None:
        return None
    cached = state.progress_cache
    if cached is not None and cached[0] is cur and cached[1] is all_sigmas:
        return cached[2]
    progress = sigma_progress(cur, all_sigmas)
    if progress is None and not state.progress_warning:
        logger.warning("Anima artist mixer could not resolve sampling progress; time windows will act as full strength.")
        state.progress_warning = True
    state.progress_cache = (cur, all_sigmas, progress)
    return progress


def _active_artists(state, layer_idx, progress):
    active = []
    for artist in state.artists:
        if layer_idx not in artist.blocks:
            continue
        factor = curve_factor(progress, artist.start, artist.end, artist.peak, artist.curve)
        if factor <= 1e-6:
            continue
        active.append((artist, artist.weight * factor))
    return active


@dataclass
class _MixPlan:
    """What a dispatch mixes: the rows the artists apply to (``branch``) with
    their input, context and base NegPiP mask, the call's options, and the
    artist weights (relative to their total) with their total influence."""

    x: torch.Tensor
    context: torch.Tensor
    options: dict
    branch: BranchRows
    base_mask: torch.Tensor | None
    weights: list
    influence: float


def _mix_plan(state, x, context, transformer_options, active):
    """The :class:`_MixPlan` of a dispatch, or None when nothing is mixed
    (zero weights, or no row to mix into as in an uncond-only call)."""
    batch_size = context.shape[0]
    options = transformer_options if isinstance(transformer_options, dict) else {}
    cou = options.get("cond_or_uncond")
    if state.mask_probe is None:
        state.mask_probe = (len(cou) if cou is not None else -1, int(batch_size))
    # Artist forwards run on the rows they mix into only; the others keep the base output.
    branch = positive_rows(cou, batch_size, state.row_masks, context.device, every_row=state.apply_uncond)
    raw_weights = [float(weight) for _, weight in active]
    total_abs = sum(abs(w) for w in raw_weights)
    if total_abs <= 1e-8 or branch is None:
        return None
    weights = [w / total_abs for w in raw_weights]
    influence = _clamp(total_abs, 0.0, 1.0)
    # Every patched block of a model call gets the same context and base
    # mask, so what is built from them is kept for the call.
    base_source = options.get(NEGPIP_CONTEXT_MASK_KEY, options.get(NEGPIP_MASK_KEY))
    cache = state.call_contexts
    if cache.get("context") is not context or cache.get("base_source") is not base_source:
        cache.clear()
        cache["context"], cache["base_source"] = context, base_source
    base_mask = None
    if state.fusion_mode in BASE_CONTEXT_FUSIONS:
        if "base_mask" not in cache:
            cache["base_mask"] = negpip_mask_for(context, options)
        base_mask = cache["base_mask"]
    if branch.rows is None:
        return _MixPlan(x, context, transformer_options, branch, base_mask, weights, influence)
    key = ("rows", branch.rows)
    if key not in cache:
        cache[key] = (branch.select(context), None if base_mask is None else branch.select(base_mask))
    row_context, row_mask = cache[key]
    return _MixPlan(branch.select(x), row_context, transformer_options, branch, row_mask, weights, influence)


def _artist_kv(state, artist, plan, base_fusion):
    """``(kv, kv_mask)`` an artist attends with on the plan's rows: its
    context aligned to the base context (joined after it for base fusions)
    and the matching NegPiP mask, built once per model call."""
    cache = state.call_contexts
    key = ("artist", id(artist), plan.branch.rows, base_fusion)
    if key in cache:
        return cache[key]
    context = cache["context"]
    artist_context = _to_context_like(artist.cond_like(context, state.prompt_step), context)
    artist_mask = _artist_mask_like(artist.mask_like(artist_context, state.prompt_step), artist_context)
    artist_context = plan.branch.select(artist_context)
    artist_mask = None if artist_mask is None else plan.branch.select(artist_mask)
    if base_fusion:
        parts = [(plan.context, plan.base_mask), (artist_context, artist_mask)]
        artist_context, artist_mask = _concat_contexts(plan.context, artist_context), _joined_mask(parts, _context_token_dim(plan.context))
    cache[key] = (artist_context, artist_mask)
    return cache[key]


def _batched_kv(state, plan, artists, base_fusion):
    """``(context, mask)`` of the artists stacked along the batch, built once
    per model call for each set of active artists and rows."""
    cache = state.call_contexts
    key = ("batched", tuple(id(artist) for artist, _ in artists), plan.branch.rows, base_fusion)
    if key in cache:
        return cache[key]
    contexts = []
    masks = []
    for artist, _ in artists:
        kv, mask = _artist_kv(state, artist, plan, base_fusion)
        contexts.append(kv)
        masks.append((kv, mask))
    lengths = {item.shape[_context_token_dim(item)] for item in contexts}
    if len(lengths) > 1:
        raise RuntimeError(f"Cannot batch artist contexts with different token lengths: {lengths}")
    cache[key] = (torch.cat(contexts, dim=0), _joined_mask(masks, 0))
    return cache[key]


def _artist_forward_batched(state, original_forward, plan, rope_emb, artists, fusion_mode):
    batch_size = plan.context.shape[0]
    context_rep, mask_rep = _batched_kv(state, plan, artists, fusion_mode in BASE_CONTEXT_FUSIONS)
    count = len(artists)
    x_rep = plan.x.repeat(count, *([1] * (plan.x.dim() - 1)))
    opts = plan.branch.options(plan.options, "artist_context", mask_rep)
    cou = opts.get("cond_or_uncond")
    if cou is not None:
        opts["cond_or_uncond"] = list(cou) * count
    out = original_forward(x_rep, context=context_rep, rope_emb=rope_emb, transformer_options=opts)
    out = out.view(count, batch_size, *out.shape[1:])
    # Python-float weights: no host-to-device copy per block
    weights = plan.weights
    total = out[0] * weights[0]
    for index in range(1, count):
        total = total + out[index] * weights[index]
    return total


def _dispatch_cross_attn(original_forward, layer_idx, state, x, context=None, rope_emb=None, transformer_options={}):
    state.dispatch_calls += 1
    if not state.enabled or context is None or not state.artists:
        state.fallback_calls += 1
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    progress = _current_progress(transformer_options, state)
    active = _active_artists(state, layer_idx, progress)
    if not active:
        state.fallback_calls += 1
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    state.active_calls += 1
    try:
        if state.combine_mode == COMBINE_CONCAT:
            return _dispatch_concat(original_forward, state, x, context, rope_emb, transformer_options, active)
        return _dispatch_output_avg(original_forward, state, x, context, rope_emb, transformer_options, active)
    except Exception as exc:
        # Stop for the rest of the run instead of failing (and logging) in
        # every block of every step, and say so in the infotext.
        state.fallback_calls += 1
        state.enabled = False
        if state.infotext is not None:
            state.infotext["Anima Artist Mixer"] = f"disabled: {type(exc).__name__} at block {layer_idx}"
        logger.exception("Anima artist mixer failed at block %s and is off for this run: %s", layer_idx, exc)
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)


def _dispatch_output_avg(original_forward, state, x, context, rope_emb, transformer_options, active):
    plan = _mix_plan(state, x, context, transformer_options, active)
    if plan is None:
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    artist_total = None
    if len(active) > 1 and not state.batched_disabled:
        try:
            artist_total = _artist_forward_batched(state, original_forward, plan, rope_emb, active, state.fusion_mode)
        except Exception as exc:
            logger.warning("Anima artist mixer batched path failed, using serial artist forwards: %s", exc)
            state.batched_disabled = True
            artist_total = None
    if artist_total is None:
        base_fusion = state.fusion_mode in BASE_CONTEXT_FUSIONS
        for (artist, _), weight in zip(active, plan.weights):
            kv, kv_mask = _artist_kv(state, artist, plan, base_fusion)
            artist_options = plan.branch.options(plan.options, "artist_context", kv_mask)
            out_i = original_forward(plan.x, context=kv, rope_emb=rope_emb, transformer_options=artist_options)
            artist_total = out_i * weight if artist_total is None else artist_total + out_i * weight
    base_out = original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    return _blend_rows(state, base_out, artist_total, plan)


def _concat_kv(state, plan, active):
    """``(context, mask)`` of the weighted artist contexts joined along the
    tokens (after the base context for base fusions), built once per model
    call for each set of active artists, weights and rows."""
    base_fusion = state.fusion_mode in BASE_CONTEXT_FUSIONS
    cache = state.call_contexts
    key = ("concat", tuple(id(artist) for artist, _ in active), tuple(plan.weights), plan.branch.rows, base_fusion)
    if key in cache:
        return cache[key]
    token_dim = _context_token_dim(plan.context)
    parts = []
    for (artist, _), weight in zip(active, plan.weights):
        artist_context, artist_mask = _artist_kv(state, artist, plan, False)
        parts.append((artist_context * weight, artist_mask))
    combined = torch.cat([part for part, _ in parts], dim=token_dim)
    combined_mask = _joined_mask(parts, token_dim)
    if base_fusion:
        kv = _concat_contexts(plan.context, combined)
        kv_mask = _joined_mask([(plan.context, plan.base_mask), (combined, combined_mask)], token_dim)
    else:
        kv, kv_mask = combined, combined_mask
    cache[key] = (kv, kv_mask)
    return cache[key]


def _dispatch_concat(original_forward, state, x, context, rope_emb, transformer_options, active):
    plan = _mix_plan(state, x, context, transformer_options, active)
    if plan is None:
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    kv, kv_mask = _concat_kv(state, plan, active)
    artist_options = plan.branch.options(plan.options, "artist_context", kv_mask)
    artist_out = original_forward(plan.x, context=kv, rope_emb=rope_emb, transformer_options=artist_options)
    base_out = original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    return _blend_rows(state, base_out, artist_out, plan)


def _blend_rows(state, base_out, artist_out, plan):
    """Mix the artist output (of the plan's rows) into ``base_out``."""

    strength = _clamp(float(state.global_strength), 0.0, 1.0 if state.fusion_mode == FUSION_QUALITY_DELTA else 2.0) * plan.influence
    rows_out = plan.branch.select(base_out)
    if state.diff_probe is None and logger.isEnabledFor(logging.DEBUG):
        # Two syncs and full-size temporaries: only for debug logging
        with torch.no_grad():
            delta_norm = (artist_out - rows_out).detach().float().flatten(1).norm(dim=1).mean().item()
            base_norm = rows_out.detach().float().flatten(1).norm(dim=1).mean().clamp_min(1e-6).item()
            state.diff_probe = (float(delta_norm), float(delta_norm / base_norm))
    if state.fusion_mode == FUSION_QUALITY_DELTA:
        blended = _norm_limited_delta(rows_out, artist_out, strength)
    else:
        blended = rows_out * (1.0 - strength) + artist_out * strength
    return plan.branch.merge(base_out, blended)
