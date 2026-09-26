"""Shared capability checks for Anima-only integrations."""

from __future__ import annotations

from contextlib import contextmanager


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
        mask = _stack_tiled(mask) if isinstance(mask, (list, tuple)) else None
    if not torch.is_tensor(mask) or not bool((mask < 0).any()):
        mask = None
    return context, mask


def _stack_tiled(values):
    """Stack per-prompt ``(tokens, ...)`` tensors, tiling each along the tokens
    to their least common multiple like Forge's ``ConditionCrossAttn``.
    Repeating a whole key/value sequence leaves attention unchanged."""

    import math

    import torch

    length = math.lcm(*(value.shape[0] for value in values))
    return torch.stack([value.repeat(length // value.shape[0], *([1] * (value.ndim - 1))) for value in values])


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


def parse_block_list(text: str, total: int, *, feature: str) -> set[int]:
    """Parse ``"0-3,7"`` into block indices below ``total``; out-of-range parts
    are clipped away and a reversed range (``"7-3"``) counts as ``"3-7"``.
    Raises when a part is not a number or range, or when nothing is left."""

    result = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            left, right = (int(value) for value in part.split("-", 1)) if "-" in part else (int(part),) * 2
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


def call_model_wrapper(previous, model_function, args):
    """Run the ``model_function_wrapper`` a wrapper replaced, or the model
    itself when there was none."""

    if previous is not None:
        return previous(model_function, args)
    return model_function(args["input"], args["timestep"], **args["c"])
