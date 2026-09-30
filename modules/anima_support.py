"""Shared helpers for Anima-only integrations: capability checks (engine,
auxiliary / Flow denoiser), prompt-batch and conditioning / negpip mask
plumbing, extra-prompt encoding and caching, block-list and branch-row
parsing, generated-frame (vs edit reference frame) token spans,
model-wrapper helpers (windowed forward-override patches) and the block-patch
panels' script mixin."""

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

    import math

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

    import math

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

        from modules.anima_presets import register_preset_control
        from modules.infotext_utils import PasteField

        self.infotext_fields = [PasteField(control, key) for control, key in zip(controls, self.INFOTEXT_KEYS, strict=True)]
        self.paste_field_names = list(self.INFOTEXT_KEYS)
        for name, control in zip(self.PRESET_FIELDS, controls, strict=True):
            register_preset_control(self.tabname, name, control)
        return controls

    @staticmethod
    def patch_pass(p, patch, infotext: dict) -> None:
        """Give this pass ``patch(unet)`` of its own UNet patcher and record
        ``infotext``. A fresh patch per pass on the pass's own patcher:
        nothing outlives it."""

        objects = p.sd_model.forge_objects
        objects.unet = patch(objects.unet)
        p.extra_generation_params.update(infotext)
