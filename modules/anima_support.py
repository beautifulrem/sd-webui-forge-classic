"""Shared helpers for Anima-only integrations: capability checks (engine,
auxiliary / Flow denoiser), prompt-batch and conditioning / negpip mask
plumbing, extra-prompt encoding and caching, block-list and branch-row
parsing, setting bounds (``clamp`` / ``to_float``), generated-frame (vs
edit reference frame) token spans, model-wrapper helpers (windowed
forward-override patches), the panels' script-argument check, infotext
paste keys and the block-patch panels' script mixin."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from contextlib import contextmanager
from types import SimpleNamespace


_AUXILIARY_DENOISER_ATTRIBUTE = "_anima_auxiliary_denoiser"


def is_anima_engine(model) -> bool:
    return bool(getattr(model, "is_anima_engine", False))


def is_anima_auxiliary_denoiser(process) -> bool:
    """Return whether the current denoiser call is a solver-only probe."""

    try:
        return bool(getattr(process, _AUXILIARY_DENOISER_ATTRIBUTE, False))
    except ReferenceError:  # a ``process_ref`` whose generation has ended
        return False


def process_ref(process):
    """A weak proxy of ``p`` for hooks stored on the model patcher.

    Those hooks outlive the generation (until the next one replaces the
    patcher); a strong reference would keep the finished ``p`` alive with its
    latents, conds and init images."""

    import weakref

    try:
        return weakref.proxy(process)
    except TypeError:  # None, or an object without weakref support
        return process


def effective_prompt_batch(process) -> tuple[list[str], list[str]]:
    """Return the positive and negative prompts used by the current pass."""

    if bool(getattr(process, "is_hr_pass", False)):
        positive = getattr(process, "hr_prompts", None) or getattr(
            process, "prompts", None
        )
        negative = getattr(process, "hr_negative_prompts", None) or getattr(
            process, "negative_prompts", None
        )
    else:
        positive = getattr(process, "prompts", None)
        negative = getattr(process, "negative_prompts", None)

    positive = list(positive or [getattr(process, "prompt", "")])
    negative = list(negative or [getattr(process, "negative_prompt", "")])
    if len(positive) == 1 and len(negative) > 1:
        positive *= len(negative)
    if len(negative) == 1 and len(positive) > 1:
        negative *= len(positive)
    if len(positive) != len(negative):
        raise RuntimeError(
            "Anima positive/negative prompt batch sizes differ for the current pass"
        )
    return [str(value) for value in positive], [str(value) for value in negative]


@contextmanager
def anima_auxiliary_denoiser(model):
    """Mark a solver-only model call without leaking process-global state."""

    process = getattr(model, "p", None)
    if process is None:
        yield
        return

    missing = object()
    previous = getattr(process, _AUXILIARY_DENOISER_ATTRIBUTE, missing)
    setattr(process, _AUXILIARY_DENOISER_ATTRIBUTE, True)
    try:
        yield
    finally:
        if previous is missing:
            delattr(process, _AUXILIARY_DENOISER_ATTRIBUTE)
        else:
            setattr(process, _AUXILIARY_DENOISER_ATTRIBUTE, previous)


def require_anima_flow_denoiser(model, feature: str) -> None:
    """Reject similarly shaped RF models before applying Anima-specific math."""

    process = getattr(model, "p", None)
    engine = getattr(process, "sd_model", None)
    if not is_anima_engine(engine):
        raise RuntimeError(f"{feature} requires an active Anima checkpoint")

    predictor = getattr(getattr(model, "inner_model", None), "predictor", None)
    from backend.modules.k_prediction import PredictionDiscreteFlow

    if not isinstance(predictor, PredictionDiscreteFlow):
        raise RuntimeError(
            f"{feature} requires Anima's discrete rectified-flow predictor"
        )


NEGPIP_MASK_KEY = "negpip_mask"  # base prompt mask, set by NegPiP's DiT hook
# Mask for a context a wrapper swapped in (Regional, Artist Mixer); None means
# "no negative weights here". Kept apart so the base mask cannot leak onto it.
NEGPIP_CONTEXT_MASK_KEY = "negpip_context_mask"
NEGPIP_EXTRA_PROMPTS = "negpip_extra_prompts"


def register_negpip_prompts(process, source: str, prompts) -> None:
    """Let NegPiP see negative weights in prompts encoded outside the main ones.

    ``source`` is the registering feature's name (see
    ``modules.anima_feature_conflicts``). Entries are replaced per ``source``
    so repeated ``process_images`` calls on one ``p`` (img2img batch, loopback,
    SD upscale) do not accumulate them.
    """

    extra = getattr(process, NEGPIP_EXTRA_PROMPTS, None)
    if not isinstance(extra, dict):
        extra = {}
        setattr(process, NEGPIP_EXTRA_PROMPTS, extra)
    extra[source] = [str(prompt) for prompt in prompts if isinstance(prompt, str) and prompt]


def negpip_extra_prompts(process) -> list[str]:
    """The registered extra prompts that will actually be encoded.

    The spatial features are one-of-three; one that yields to another active
    one (they mark themselves in ``process()``, before NegPiP's
    ``process_batch``) encodes nothing, whatever order the scripts ran in.
    """

    from modules.anima_feature_conflicts import ARTIST_MIXER, FREEFUSE, REGIONAL, is_feature_active

    spatial = (FREEFUSE, REGIONAL, ARTIST_MIXER)
    active = [feature for feature in spatial if is_feature_active(process, feature)]
    extra = getattr(process, NEGPIP_EXTRA_PROMPTS, None) or {}
    return [
        prompt
        for source, prompts in extra.items()
        if not active or source not in spatial or source in active
        for prompt in prompts
    ]


def split_conditioning(conds):
    """Split ``get_learned_conditioning`` output into ``(crossattn, mask)``.

    While NegPiP is active the output is ``{"crossattn", "c_negpip_mask"}``:
    negative-weight tokens are sign-restored in ``crossattn`` and NegPiP's
    attention hook negates only V with the mask. Callers that swap in their own
    contexts (Regional, Artist Mixer) keep the mask and pass it to the forward
    under ``NEGPIP_CONTEXT_MASK_KEY`` so those tokens get the same semantics as
    in the main prompt. Other outputs, and masks without negative entries, give
    ``mask=None``.
    """

    import torch

    if not isinstance(conds, dict):
        return conds, None
    context, mask = conditioning_context(conds), conds.get("c_negpip_mask")
    if isinstance(context, (list, tuple)):  # prompts padded to different lengths
        context = _stack_tiled(context)
        mask = _stack_tiled(mask, fill=1.0) if isinstance(mask, (list, tuple)) else None
    if not torch.is_tensor(mask) or not bool((mask < 0).any()):
        mask = None
    return context, mask


_MAX_TILE_FACTOR = 4  # ConditionCrossAttn.can_concat's limit


def _stack_tiled(values, fill: float = 0.0):
    """Stack per-prompt ``(tokens, ...)`` tensors, tiling each along the tokens
    to their least common multiple like Forge's ``ConditionCrossAttn``.
    Repeating a whole key/value sequence leaves attention unchanged.

    Prompts over 512 tokens have odd lengths (512, 600, 530 have an LCM of
    ~2M tokens), so past the factor ``ConditionCrossAttn.can_concat`` allows,
    every row is padded with ``fill`` to the longest instead: zeros for
    contexts, as Anima pads prompts under 512 tokens; 1 (neutral) for NegPiP
    masks, as NegPiP pads them (V of a zero context row is zero either way)."""

    import torch

    lengths = [value.shape[0] for value in values]
    length = math.lcm(*lengths)
    if length // min(lengths) <= _MAX_TILE_FACTOR:
        return torch.stack([value.repeat(length // value.shape[0], *([1] * (value.ndim - 1))) for value in values])
    longest = max(lengths)
    return torch.stack([torch.cat((value, value.new_full((longest - value.shape[0], *value.shape[1:]), fill))) for value in values])


@contextmanager
def preserved_reference_latents(sd_model):
    """Encode extra prompts (Regional, Artist Mixer) without touching the
    reference latents the main prompt set up: Anima's positive-prompt
    ``get_learned_conditioning`` resets them and consumes ``ini_latent``."""

    from backend.args import dynamic_args

    references = list(dynamic_args.ref_latents)
    missing = object()
    initial = getattr(sd_model, "ini_latent", missing)
    try:
        yield
    finally:
        dynamic_args.ref_latents = references
        if initial is not missing:
            sd_model.ini_latent = initial


def conditioning_cache_key(process) -> tuple:
    """What encoding an extra prompt depends on besides its text, for caches
    that outlive one pass (``forge_objects`` is a fresh copy every pass)."""

    from modules import sd_models, shared

    sd_model = getattr(process, "sd_model", None)
    clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
    return (
        str(getattr(getattr(sd_model, "sd_checkpoint_info", None), "filename", None)),
        # The checkpoint plus additional modules (text encoder).
        str(getattr(getattr(sd_models, "model_data", None), "forge_hash", "")),
        str(getattr(sd_model, "current_lora_hash", "")),
        # A new id whenever text-encoder LoRA patches (and so their Qwen layer
        # weights) are applied, which the LoRA hash alone does not reflect.
        str(getattr(getattr(clip, "patcher", None), "patches_uuid", None)),
        str(getattr(shared.opts, "emphasis", "")),
        # NegPiP replaces get_learned_conditioning while it is active.
        "get_learned_conditioning" in getattr(sd_model, "__dict__", {}),
    )


def encode_extra_prompts(process, texts):
    """Encode prompts besides the main ones (Regional regions, Artist Mixer
    artists) as ``(context, negpip_mask or None)`` on the CPU, one context row
    per text, leaving the main prompt's reference latents alone."""

    import torch

    from modules import prompt_parser

    conditioning = prompt_parser.SdConditioning(
        [str(text) for text in texts],
        width=getattr(process, "width", None),
        height=getattr(process, "height", None),
        distilled_cfg_scale=getattr(process, "distilled_cfg_scale", None),
    )
    with preserved_reference_latents(process.sd_model):
        conds = process.sd_model.get_learned_conditioning(conditioning)
    if isinstance(conds, (list, tuple)):  # one (1, tokens, dim) tensor per prompt
        conds = _stack_tiled([cond.reshape(-1, cond.shape[-1]) for cond in conds])
    context, mask = split_conditioning(conds)
    if not torch.is_tensor(context):
        raise RuntimeError("Could not extract the Anima conditioning tensor")
    if context.ndim == 2:
        context = context.unsqueeze(0)
    return context.detach().cpu(), None if mask is None else mask.detach().cpu()


def schedule_steps(process, *, hires: bool | None = None) -> tuple[int, int | None]:
    """``(steps, hires_steps)`` Forge resolves the current pass's prompt
    editing (``[a:b:N]``, ``[a|b]``) with, as ``processing.py`` passes them
    to ``get_learned_conditioning_prompt_schedules``: the base pass's sampler
    total steps (``firstpass_steps``) and, in a Hires. fix pass, the hires
    sampler's total steps as well (hires_steps None in the base pass).
    ``hires`` picks the pass explicitly (default: ``process.is_hr_pass``),
    for code that builds both passes' schedules before the Hires pass."""

    def total_steps(sampler_name, steps):
        from modules import sd_samplers

        config = sd_samplers.find_sampler_config(sampler_name)
        return config.total_steps(steps) if config else steps

    steps = int(getattr(process, "steps", 0) or 0)
    base = int(getattr(process, "firstpass_steps", 0) or 0) or total_steps(getattr(process, "sampler_name", None), steps)
    if not (bool(getattr(process, "is_hr_pass", False)) if hires is None else hires):
        return base, None
    hr_steps = int(getattr(process, "hr_second_pass_steps", 0) or 0) or steps
    sampler_name = getattr(process, "hr_sampler_name", None) or getattr(process, "sampler_name", None)
    return base, total_steps(sampler_name, hr_steps)


def prompt_schedule(prompts, steps: int, hires_steps: int | None = None) -> list[tuple[int, tuple[str, ...]]]:
    """Forge's prompt-editing schedule of a prompt batch: ``(end_step,
    texts)`` entries, ``texts`` holding each prompt's resolved text up to
    ``end_step`` (see :func:`schedule_steps` and :func:`schedule_index`).
    The prompts' own schedules are merged over their switch steps; entries
    with the same texts as the one before them are joined. A batch without
    prompt editing has one entry, with its prompts' text unchanged."""

    from modules import prompt_parser

    prompts = [str(prompt) for prompt in prompts]
    schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompts, max(int(steps), 1), hires_steps)
    ends = sorted({end for schedule in schedules for end, _text in schedule})
    merged = []
    for end in ends:
        texts = tuple(next((text for until, text in schedule if end <= until), schedule[-1][1]) for schedule in schedules)
        if merged and merged[-1][1] == texts:
            merged[-1] = (end, texts)
        else:
            merged.append((end, texts))
    return merged


def schedule_index(ends, step: int) -> int:
    """Entry of a schedule with end steps ``ends`` a model call at sampler
    ``step`` uses, exactly as ``prompt_parser.reconstruct_cond_batch`` picks
    the main prompt's: the first ending at or after it, else the first."""

    return next((index for index, end in enumerate(ends) if step <= end), 0)


def prompt_step(process) -> int:
    """The step Forge picks the main prompt's schedule entry by for the
    current model call: the CFG denoiser's call counter of the current pass's
    sampler (0 when there is none, or the generation has ended)."""

    try:
        return int(getattr(getattr(getattr(process, "sampler", None), "model_wrap_cfg", None), "step", 0) or 0)
    except ReferenceError:  # a ``process_ref`` whose generation has ended
        return 0


def conditioning_context(conds):
    """The cross-attention context of ``get_learned_conditioning`` output
    (the plain tensor, or ``crossattn`` of NegPiP's dict form)."""

    if isinstance(conds, dict):
        return conds.get("crossattn", conds.get("c_crossattn"))
    return conds


def negpip_mask_for(context, transformer_options):
    """NegPiP V-mask from ``transformer_options`` if it matches ``context``.

    A swapped-in context uses only its own ``NEGPIP_CONTEXT_MASK_KEY`` (which
    the swapping wrapper always sets, possibly to None). The mask must also
    fit the context (same length, tileable batch).
    """

    import torch

    options = transformer_options or {}
    if NEGPIP_CONTEXT_MASK_KEY in options:
        mask = options[NEGPIP_CONTEXT_MASK_KEY]
    else:
        mask = options.get(NEGPIP_MASK_KEY)
    if not torch.is_tensor(mask) or not torch.is_tensor(context):
        return None
    if mask.ndim != context.ndim or mask.shape[1] != context.shape[1]:
        return None
    if mask.shape[0] != context.shape[0]:
        if context.shape[0] % mask.shape[0]:
            return None
        mask = mask.repeat(context.shape[0] // mask.shape[0], *([1] * (mask.ndim - 1)))
    return mask.to(context)


def generated_token_frames(input_x, patch_temporal: int = 1) -> int | None:
    """Token frames of the generated latent ``input_x`` ``(B, C, T, H, W)``
    of a model call, None when it is not 5-D.

    An edit model's reference-latent frames are appended after these along T
    (``Anima.forward``), and the frame-major flattening of the blocks puts
    their tokens after the generated ones: a per-token edit covers the first
    ``frames * H * W`` tokens only (see :func:`pad_to_sequence`)."""

    if getattr(input_x, "ndim", None) != 5:
        return None
    return math.ceil(int(input_x.shape[2]) / max(1, int(patch_temporal or 1)))


def pad_to_sequence(values, length: int, fill: float = 0.0, *, dim: int = 1):
    """``values`` describing the generated tokens (or token frames), leading
    along ``dim``, extended to an Anima sequence of ``length`` with ``fill``
    for the reference-latent tokens an edit model appends after them (see
    :func:`generated_token_frames`). Raises when ``values`` is longer."""

    import torch

    size = int(values.shape[dim])
    if size > length:
        raise RuntimeError(f"Anima generated-token mask {size} does not match Anima sequence {length}")
    if size == length:
        return values
    shape = list(values.shape)
    shape[dim] = length - size
    return torch.cat((values, values.new_full(shape, float(fill))), dim=dim)


# How every weight-text entry point reads input typed with an IME or copied
# from prose: full-width ASCII forms (U+FF01-FF5E, "，；＝－＋．", "０-９",
# letters) by offset, the ideographic space, "：" as "=" (it separates a name
# from its value, "modules：mlp=0.5"), "。" as ".", and dash variants (hyphen,
# en/em dash, minus sign) as "-" ("0–9=0.5"). Only these: a superscript or
# circled digit ("0.5²", "①") stays as written and is rejected, not read as
# another number.
_WEIGHT_TEXT_FOLD = str.maketrans(
    {
        **{code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)},
        0x3000: " ",
        ord("："): "=",
        ord("。"): ".",
        **dict.fromkeys((0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xFE63), "-"),
    }
)


def fold_weight_text(text) -> str:
    """A weight text as every weight-text entry point reads it (see
    ``_WEIGHT_TEXT_FOLD``)."""

    return str(text or "").translate(_WEIGHT_TEXT_FOLD)


# The one number grammar of weight texts: decimal digits of any script (as
# float() reads them; not superscripts or circled digits), an optional
# exponent, no "_" separators ("0_5" would read as 5) and no inf / nan.
# Surrounding whitespace is stripped with str.strip first and float() only
# sees the stripped text, which is why plain_number never raises: str.strip
# removes every str.isspace() character, U+001C-U+001F included, which float()
# itself refuses ("0.5\x1c" reads as 0.5 here, float() raises on it)
_PLAIN_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def plain_number(text) -> float | None:
    """The finite value of a lone number such as "0.6", "-.5", "1e-1",
    "０．５" or "−0.5" (folded first), else None; it never raises."""

    folded = fold_weight_text(text).strip()
    if not _PLAIN_NUMBER_RE.fullmatch(folded):
        return None
    value = float(folded)
    return value if math.isfinite(value) else None


def is_plain_number(text) -> bool:
    return plain_number(text) is not None


def to_float(value, fallback=0.0):
    """``float(value)``, or ``fallback`` when it is not a finite number (None,
    "", text, and the NaN / Infinity JSON from the API or pasted infotext
    accepts: clamp(nan, low, high) would give ``high``)."""

    try:
        value = float(value)
    except Exception:
        return fallback
    return value if math.isfinite(value) else fallback


def clamp(value, low, high=math.inf):
    """``value`` bounded to ``[low, high]`` (no upper bound by default). The
    one bounding idiom of the Anima panels is ``clamp(to_float(value,
    default), low, high)``: ``max`` / ``min`` do not bound NaN, which
    ``to_float`` turns into ``default`` first."""

    return max(low, min(high, value))


def require_finite(owner: str, **values) -> None:
    """Refuse non-finite float script args of the panel ``owner``: raises
    ``ValueError`` naming it and each such argument, so a hook that calls it
    first changes nothing. Forge's script runner reports the exception and
    runs the other scripts; the panel is skipped for the pass.

    The API reads JSON ``NaN`` / ``Infinity`` as floats, and a run that
    recorded "nan" in its infotext replays it as one. Only float values are
    checked: a text field may hold "nan" (a prompt, a block list)."""

    bad = [f"{name}={value!r}" for name, value in values.items() if isinstance(value, numbers.Real) and not math.isfinite(value)]
    if bad:
        raise ValueError(f"{owner} refuses non-finite script arguments: {', '.join(bad)}")


# Past every block or layer index (the deepest Anima DiT has 52 blocks, the
# Qwen3 text encoder 28 layers): what a longer digit string reads as
BLOCK_INDEX_CAP = 1000


def block_index_token(text, *, signed: bool = False) -> int:
    """One block or layer index as every block field reads it: decimal digits
    with surrounding whitespace ignored and leading zeros allowed ("05"), and,
    when ``signed``, an optional "+" / "-" (what a negative index means is the
    caller's). No "_" separators (int() would read "0_5" as 5) and no inner
    whitespace ("1 0" is not 10). More than three significant digits read as
    ``BLOCK_INDEX_CAP`` (keeping the sign), past every index, without
    converting a prompt's 5000-digit string. Raises ValueError otherwise;
    whether a bad part is skipped or refused is the caller's."""

    value = str(text).strip()
    sign = 1
    if signed and value[:1] in ("+", "-"):
        sign = -1 if value[0] == "-" else 1
        value = value[1:]
    if not value.isdecimal():
        raise ValueError(f"not a block index: {text!r}")
    digits = value.lstrip("0") or "0"
    return sign * (int(digits) if len(digits) <= 3 else BLOCK_INDEX_CAP)


def parse_block_list(text: str, total: int, *, feature: str) -> set[int]:
    """Parse ``"0-3,7"`` into block indices below ``total``; out-of-range parts
    are clipped away and a reversed range (``"7-3"``) counts as ``"3-7"``.
    Raises when a part is not a number or range, or when nothing is left."""

    result = set()
    # IME input reads as in the weight fields: "0-9，19-27", "０－２７", "0–27"
    for part in fold_weight_text(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            sides = [block_index_token(side) for side in part.split("-", 1)]
            left, right = sides[0], sides[-1]
        except ValueError:
            raise ValueError(f"{feature} block selection has an invalid part {part!r} (use indices like 0-3,7)") from None
        low, high = sorted((left, right))
        result.update(range(max(0, low), min(total - 1, high) + 1))
    if not result:
        raise ValueError(f"{feature} block selection is empty")
    return result


def branch_rows(markers, batch: int, mark: int = 0) -> list[int] | None:
    """Rows of a CFG batch whose ``cond_or_uncond`` group is ``mark`` (0 is the
    positive prompt), or None when the markers are empty or do not divide it."""

    markers = [int(value) for value in markers or ()]
    if not markers or batch % len(markers):
        return None
    group = batch // len(markers)
    return [index * group + row for index, value in enumerate(markers) if value == mark for row in range(group)]


class BranchRows:
    """The rows of a cross-attention batch a wrapper runs its own contexts on
    (Regional regions, Artist Mixer artists); the other rows keep the base
    output. ``rows`` None means every row. See :func:`positive_rows`."""

    __slots__ = ("markers", "rows", "index")

    def __init__(self, markers, rows=None, index=None):
        self.markers = markers
        self.rows = rows
        self.index = index

    def select(self, tensor):
        """``tensor`` (batch first) on these rows."""
        return tensor if self.index is None else tensor.index_select(0, self.index)

    def merge(self, base, rows_out):
        """``base`` with these rows replaced by ``rows_out``."""
        return rows_out if self.index is None else base.index_copy(0, self.index, rows_out)

    def options(self, transformer_options, nag_skip: str, negpip_mask):
        """``transformer_options`` for a forward of these rows with a
        swapped-in context whose NegPiP mask is ``negpip_mask`` (None: no
        negative weights); the base prompt's mask describes other tokens."""
        options = dict(transformer_options) if isinstance(transformer_options, dict) else {}
        if self.rows is not None:
            options["cond_or_uncond"] = [0] * sum(int(marker) == 0 for marker in self.markers)
        # NAG's negative branch is meaningful for the base positive/negative
        # pair, not for swapped-in contexts.
        options["anima_nag_skip"] = nag_skip
        options[NEGPIP_CONTEXT_MASK_KEY] = negpip_mask
        return options


def positive_rows(markers, batch: int, cache: dict, device, every_row: bool = False) -> BranchRows | None:
    """The positive-prompt rows of a batch with ``cond_or_uncond`` ``markers``
    as :class:`BranchRows`: every row with ``every_row``, when they cover the
    batch or when the markers do not describe it; None when there is no
    positive row. The row index tensor is kept in ``cache`` per device."""

    import torch

    rows = branch_rows(markers, batch)
    if rows is None or every_row or len(rows) == batch:
        return BranchRows(markers)
    if not rows:
        return None
    rows = tuple(rows)
    key = ("rows", rows, device)
    if key not in cache:
        cache[key] = torch.tensor(rows, device=device)
    return BranchRows(markers, rows, cache[key])


# (weakref, version counter, value) of the last few tensors sigma_value read:
# every wrapper of a model call reads the same timestep (and the sampler's
# sigma), so the chain makes one host copy per tensor instead of one each.
_SIGMA_MEMO_SIZE = 4
_sigma_memo: tuple = ()


def sigma_value(value) -> float:
    """The first sigma of a model call's ``timestep``/``sigmas`` (a tensor,
    a sequence or a number) as a float.

    Tensors are remembered by identity and version counter, so an in-place
    update is read again.  Inference tensors (sampling runs under
    ``torch.inference_mode``) carry no version counter and are remembered by
    identity alone: the sampling loop builds its sigma tensors per call and
    never updates one in place."""

    global _sigma_memo
    import torch

    if torch.is_tensor(value):
        version = None if value.is_inference() else value._version
        memo = _sigma_memo
        for ref, seen, result in memo:
            if ref() is value and seen == version:
                return result
        result = float(value.detach().flatten()[0])
        import weakref

        _sigma_memo = ((weakref.ref(value), version, result), *(entry for entry in memo if entry[0]() is not value))[:_SIGMA_MEMO_SIZE]
        return result
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


WRAPPER_FLAGS = ("__spectrum_cache_safe__", "__resolution_independent__")


def inherit_wrapper_flags(wrapper, previous, *, spectrum_cache_safe=True, resolution_independent=True):
    """Mark a ``model_function_wrapper`` for Spectrum (may cache steps around
    it) and Anima SPEED (works at any latent size): each flag holds only if
    the wrapper itself qualifies and so does the wrapper it delegates to."""

    own = {"__spectrum_cache_safe__": spectrum_cache_safe, "__resolution_independent__": resolution_independent}
    for flag in WRAPPER_FLAGS:
        setattr(wrapper, flag, bool(own[flag]) and (previous is None or bool(getattr(previous, flag, False))))
    return wrapper


def call_model_wrapper(previous, model_function, args):
    """Run the ``model_function_wrapper`` a wrapper replaced, or the model
    itself when there was none."""

    if previous is not None:
        return previous(model_function, args)
    return model_function(args["input"], args["timestep"], **args["c"])


def windowed_override_patch(model, *, gate, overrides, lock, force_actual, begin=None, end=None, outermost=False, resolution_independent=True):
    """A clone of the UNet patcher ``model`` whose ``model_function_wrapper``
    runs the model calls ``gate`` admits under temporary forward overrides
    (``modules/forward_override.py``), and passes the others through.

    ``gate(args)`` returns a token for the call, or None to pass it through
    untouched; it runs outside ``lock``. Under ``lock``, ``begin(token)`` sets
    up the call, ``overrides(token)`` gives the ``(module, forward)`` pairs
    (read at install time, so each wraps the forward its module has then) and
    ``end()`` always runs after the call.

    An override installed later wraps the one installed before it. Overrides
    go on when this wrapper runs, before the wrappers it delegates to install
    theirs, so theirs wrap these. ``outermost`` installs them around the model
    function instead, once every wrapper beneath this one has installed its
    own: they then wrap every other override of the same modules whatever the
    wrapper order (an edit of a block's whole output, such as DAVE's).

    Outside the calls it admits the wrapper is a pass-through, and
    ``force_actual`` (a Spectrum request on the clone, see
    ``modules/spectrum_force.py``) makes Spectrum run actual forwards for
    them, so Spectrum may cache the other steps. ``resolution_independent``
    tells Anima SPEED whether the patch works at any latent size
    (:func:`inherit_wrapper_flags`)."""

    from modules import spectrum_force
    from modules.forward_override import forward_overrides

    patched = model.clone()
    previous = patched.model_options.get("model_function_wrapper")

    def wrapper(model_function, args):
        token = gate(args)
        if token is None:
            return call_model_wrapper(previous, model_function, args)
        with lock:
            try:
                if begin is not None:
                    begin(token)
                if not outermost:
                    with forward_overrides(overrides(token)):
                        return call_model_wrapper(previous, model_function, args)

                def overridden(*call_args, **call_kwargs):
                    with forward_overrides(overrides(token)):
                        return model_function(*call_args, **call_kwargs)

                return call_model_wrapper(previous, overridden, args)
            finally:
                if end is not None:
                    end()

    inherit_wrapper_flags(wrapper, previous, resolution_independent=resolution_independent)
    patched.set_model_unet_function_wrapper(wrapper)
    # Spectrum wraps this wrapper, so it must see the request on the patcher.
    spectrum_force.request(patched, force_actual)
    return patched


_MISSING_SHOWN = 5


def script_arguments(owner: str, names, values, *, defaults=None) -> SimpleNamespace:
    """A panel's positional script args ``values``, one attribute per name
    of ``names`` (the order ``ui`` returns its controls in). Call it before
    the hook changes anything: a wrong count raises ``TypeError`` naming
    ``owner`` and the counts, so nothing is half-applied.

    More values than names are refused: Forge and the API give a script
    exactly its own slots (the API fills every one from the UI defaults
    first), so an extra value means the list is misaligned. Fewer values are
    filled from ``defaults`` (a sequence as long as ``names``, or a mapping
    from name to default) when the panel gives them: an argument list stored
    before a control was added, e.g. a UI task Agent Scheduler queued earlier,
    is replayed as it was saved, with nothing filled in. A missing name
    without a default is refused.

    A float value that is not finite (NaN, Infinity) is refused with a
    ``ValueError`` naming ``owner`` and the argument (:func:`require_finite`):
    a range limit such as ``min(max(value, low), high)`` lets NaN through."""

    names = tuple(names)
    values = tuple(values)
    if len(values) > len(names):
        raise TypeError(f"{owner} takes {len(names)} script arguments, got {len(values)}")
    arguments = dict(zip(names, values))
    if len(values) < len(names):
        if isinstance(defaults, Mapping):
            fill = defaults
        elif defaults is not None:
            if len(defaults) != len(names):
                raise ValueError(f"{owner} has {len(defaults)} script argument defaults for {len(names)} names")
            fill = dict(zip(names, defaults))
        else:
            fill = {}
        missing = [name for name in names[len(values) :] if name not in fill]
        if missing:
            # The first few names locate the gap; a panel with hundreds of slots stays readable
            shown = ", ".join(missing[:_MISSING_SHOWN]) + (f" and {len(missing) - _MISSING_SHOWN} more" if len(missing) > _MISSING_SHOWN else "")
            raise TypeError(f"{owner} takes {len(names)} script arguments, got {len(values)}; missing: {shown}")
        arguments.update((name, fill[name]) for name in names[len(values) :])
    require_finite(owner, **arguments)
    return SimpleNamespace(**arguments)


class PasteKey(str):
    """An infotext key that is also the paste function of one control.

    Forge's ``PasteField(component, target)`` reads ``target`` as a label
    when it is a str and as a function when it is callable; a PasteKey is
    both, so each of Forge's three paste paths finds what it looks for:

    - UI paste (``infotext_utils._parse_info``) calls it, as any callable
      target: the function may fill many controls from one recorded value,
      or give a value the run left out of the infotext (an empty text).
    - "Send to" carries the fields whose target is in the panel's
      ``paste_field_names``: being its key, the control is carried like any
      plainly keyed field.
    - The API's infotext replay (``api.Api.apply_infotext``) takes
      ``field.function``'s value, running no ``on_infotext_pasted``
      callback: the function is the one place a panel converts a recorded
      value for both the UI paste and the API."""

    def __new__(cls, key, function):
        self = super().__new__(cls, key)
        self.function = function
        return self

    def __call__(self, params):
        return self.function(params)


def register_table_fields(script, rows, controls) -> list:
    """Infotext paste fields and Anima preset controls of a panel whose script
    arguments are the table ``rows``, each ``(name, infotext key or None,
    preset name or None, ...)``. ``controls``: the rows' controls by name, or
    in row order. Returns them in row order, which is what ``ui`` returns."""

    from modules.anima_presets import register_preset_control
    from modules.infotext_utils import PasteField

    rows = list(rows)
    in_order = controls if not isinstance(controls, dict) else [controls[name] for name, *_ in rows]
    fields = [(control, key, preset_name) for (_name, key, preset_name, *_), control in zip(rows, in_order, strict=True)]
    for control, _, preset_name in fields:
        if preset_name is not None:
            register_preset_control(script.tabname, preset_name, control)
    script.infotext_fields = [PasteField(control, key) for control, key, _ in fields if key is not None]
    script.paste_field_names = [key for _, key, _ in fields if key is not None]
    return [control for control, _, _ in fields]


class AnimaPatchPanel:
    """Mixin for an always-visible Anima panel that patches the UNet of each
    pass (DAVE, cross-attention boost). Its ``ui`` returns the enable switch
    and then the settings, in the order of ``INFOTEXT_KEYS``."""

    INFOTEXT_KEYS: tuple[str, ...] = ()
    # Anima preset control names, in the same order (see modules/anima_presets.py)
    PRESET_FIELDS: tuple[str, ...] = ()

    def show(self, is_img2img):
        from modules import scripts

        return scripts.AlwaysVisible

    def register_paste_fields(self, controls: list) -> list:
        """Infotext paste fields and Anima preset controls for ``ui``'s
        ``controls``; returns them."""

        return register_table_fields(self, zip(self.PRESET_FIELDS, self.INFOTEXT_KEYS, self.PRESET_FIELDS, strict=True), controls)

    @staticmethod
    def patch_pass(p, patch, infotext: dict) -> None:
        """Give this pass ``patch(unet)`` of its own UNet patcher and record
        ``infotext``. A fresh patch per pass on the pass's own patcher:
        nothing outlives it."""

        objects = p.sd_model.forge_objects
        objects.unet = patch(objects.unet)
        p.extra_generation_params.update(infotext)
