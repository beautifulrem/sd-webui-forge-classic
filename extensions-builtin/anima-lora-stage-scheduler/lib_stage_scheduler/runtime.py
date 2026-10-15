"""A scheduled run: each pass's config, the LoRA strength patches and their
per-step factors, and the run's infotext."""

import logging
import weakref
import uuid
from dataclasses import dataclass, field

from modules.anima_lora_files import adapter_kind_for_file, is_anima_adapter_file, lora_coverage
from modules.anima_online_lora import acquire_online_lora, online_lora_leased_by_other_than, release_online_lora, user_online_lora
from modules.anima_lora_support import (
    LORA_TAG_RE,
    install_activate_hook,
    install_add_patches_hook,
    invalidate_cond_caches,
    lora_control_files,
    manual_window_bans,
    note_lora_control_yield,
    rewrite_batch_prompts,
    sigma_progress,
    window_curve_factor,
    lora_file_names as _names_for_lora_file,
    name_keys,
    network_misses,
    parse_targets as _parse_targets,
    resolve_network,
    active_template_parts,
    anima_lora_block_count,
    current_sampling_position,
    model_key as _model_key,
)
from modules.anima_support import clamp as _clamp, to_float as _to_float
from lib_stage_scheduler.options import (
    BUILTIN_TEMPLATES,
    CURVE_CODES,
    DISABLE_ALL,
    DISABLE_CODES,
    DISABLE_NONE,
    DISABLE_SELECTED,
    INFOTEXT_ENABLED,
    INFOTEXT_HIRES_DISABLE,
    INFOTEXT_HIRES_DISABLE_TAGS,
    INFOTEXT_HIRES_INDEPENDENT,
    MODE_PROMPT_ALL,
    MODE_TARGETS,
    PART_BAN,
    PanelWeightSpec,
    STAGE_CHARACTER,
    STAGE_CHOICES,
    TARGET_MODE_CODES,
    _format_float,
    _normalize_curve,
    _normalize_disable_mode,
    _normalize_target_mode,
    _panel_key,
    _parse_disable_tags,
    _parse_panel_weight,
)
from lib_stage_scheduler.templates import (
    _auto_stage_values,
    _parse_lora_template_overrides,
    _primary_template_part,
    _template_config,
    _template_parts,
)


logger = logging.getLogger("anima_lora_stage_scheduler")

_ACTIVE_STATE = None
# Name of this script's lease on Forge's online-LoRA switch
ONLINE_LORA_OWNER = "anima_lora_stage_scheduler"
# (current_lora_hash tag, load_networks' hash, _loaded_patch_config, strengths)
# of the patches the last run left loaded; see _prepare_for_extra_network_activation
_RELEASED_PATCHES = None
# PassConfig.text_encoder_signature of both passes of the last run
_TEXT_ENCODER_SIGNATURE = None


def _cleanup_runtime(p):
    """Undo ``p``'s own run. Never another's: a nested run (ADetailer) may
    carry a copied reference to the outer run's state, or none at all."""
    global _ACTIVE_STATE, _RELEASED_PATCHES

    release_online_lora(p, ONLINE_LORA_OWNER)
    state = getattr(p, "_anima_lora_stage_scheduler_state", None)
    if state is None or state.p is not p:
        return

    state.set_factor(1.0)
    sd_model = getattr(p, "sd_model", None)
    if state.loaded_pass is not None and sd_model is not None:
        # A nested run (ADetailer) loading a LoRA set of its own since left
        # its plain-strength patches: they are not this run's to hand on.
        if sd_model.current_lora_hash == state.loaded_hash:
            # The next run must not keep this run's scheduled / panel-weighted
            # patches, unless it schedules the very same ones.
            tag = f"anima-stage-scheduler:released:{uuid.uuid4().hex}"
            _RELEASED_PATCHES = (tag, sd_model.current_lora_hash, _loaded_patch_config(state, state.loaded_pass), list(state.wrappers))
            sd_model.current_lora_hash = tag
        state.loaded_pass = None
    p._anima_lora_stage_scheduler_state = None
    if _ACTIVE_STATE is state:
        _ACTIVE_STATE = state.outer_state


def _note_text_encoder_signature(p, signature):
    """Drop the cached prompt conditioning when the panels' text-encoder
    weights change: the cache key only sees the prompts."""
    global _TEXT_ENCODER_SIGNATURE

    if signature is not None and not any(signature):
        signature = None
    if signature != _TEXT_ENCODER_SIGNATURE:
        _TEXT_ENCODER_SIGNATURE = signature
        invalidate_cond_caches(p)


def _is_anima_lora_file(filename) -> bool:
    """An adapter (LoRA or LoKr) that trains Anima DiT blocks."""
    return bool(filename) and is_anima_adapter_file(filename, adapter_kind_for_file(filename) or "lora")


def _lora_block_count(filename) -> int | None:
    """The Anima depth (28/40/52) a LoRA file was trained for."""
    if not filename:
        return None
    return anima_lora_block_count(lora_coverage(filename, adapter_kind_for_file(filename) or "lora").blocks)


def _resolve_lora_filename(name: str, misses: set[str] | None = None):
    return getattr(resolve_network(name, misses), "filename", None)


def _is_unet_patch_key(key) -> bool:
    text = str(key or "").lower()
    if "qwen" in text or "text_encoder" in text or "text_encoders" in text:
        return False
    return text.startswith("diffusion_model.") or ".blocks." in text or "blocks_" in text


def _activation_pass_for_extra_network_data(state, extra_network_data) -> str | None:
    p = state.p
    if getattr(p, "enable_hr", False) and getattr(p, "hr_extra_network_data", None) is extra_network_data:
        return "hires"
    if getattr(p, "extra_network_data", None) is extra_network_data:
        return "base"
    return None


def _loaded_patch_config(state, patch_pass) -> tuple:
    """What the patches loaded for ``patch_pass`` depend on besides the LoRA
    set, which load_networks hashes itself."""
    return patch_pass, state.base.patch_signature, state.hires.patch_signature, state.original_online_lora, frozenset(lora_control_files())


def _prepare_for_extra_network_activation(p, extra_network_data):
    global _RELEASED_PATCHES

    state = getattr(p, "_anima_lora_stage_scheduler_state", None)
    if state is None or not state.enabled or getattr(p, "sd_model", None) is None:
        return None

    pass_name = _activation_pass_for_extra_network_data(state, extra_network_data)
    patch_pass = state.patch_pass(pass_name) if pass_name is not None else None
    if patch_pass is not None and patch_pass != state.loaded_pass:
        config = _loaded_patch_config(state, patch_pass)
        released, _RELEASED_PATCHES = _RELEASED_PATCHES, None
        if state.loaded_pass is None and released is not None and released[0] == p.sd_model.current_lora_hash and released[2] == config:
            # The last run left these very patches loaded (nothing loaded
            # others since): keep them, and schedule their strengths again.
            p.sd_model.current_lora_hash = released[1]
            state.wrappers = released[3]
        else:
            # The loaded patches were built for the other pass's panel, or by
            # another run: make load_networks rebuild them. A changed LoRA set
            # is reloaded by load_networks by itself.
            p.sd_model.current_lora_hash = f"anima-stage-scheduler:{patch_pass}:{hash(config):x}"
        state.loaded_pass = patch_pass
    return pass_name


def _sampling_progress(params) -> float:
    """Progress 0..1 of the current model call in the full sigma schedule (as
    the artist mixer reads it): Hires. fix and img2img start part-way, as
    their windows were denoised by the pass before. By steps without one."""
    p = getattr(getattr(params, "denoiser", None), "p", None)
    unet = getattr(getattr(getattr(p, "sd_model", None), "forge_objects", None), "unet", None)
    options = (getattr(unet, "model_options", None) or {}).get("transformer_options") or {}
    progress = sigma_progress(getattr(params, "sigma", None), options.get("sampling_sigmas"))
    if progress is not None:
        return progress
    step, total_steps = current_sampling_position(params)
    return _clamp(step / (total_steps - 1), 0.0, 1.0) if total_steps > 1 else 0.0


def _curve_factor_for_window(progress, start, end, peak, strength, curve) -> float:
    factor = window_curve_factor(progress, start, end, peak, CURVE_CODES.get(curve, "smooth"))
    return _clamp(factor * _clamp(strength, 0.0, 4.0), 0.0, 4.0)


def _stage_curve_factor(config, progress) -> float:
    stage_strength = _clamp(config.stage_strength, 0.0, 4.0)
    if config.parts:
        allow_factors = []
        ban_active = False

        # _template_parts normalized the parts: valid curve, clamped strength
        for part in config.parts:
            start, end, peak, _ = _auto_stage_values(part["stage"], config.shift)

            if part["mode"] == PART_BAN:
                window_start, window_end = (start, end) if start <= end else (end, start)
                if window_start <= progress <= window_end:
                    ban_active = True
                continue

            allow_factors.append(_curve_factor_for_window(progress, start, end, peak, part["strength"], part["curve"]))

        # The panel's stage strength scales the template's curves
        factor = max(allow_factors) * stage_strength if allow_factors else stage_strength
        return 0.0 if ban_active else _clamp(factor, 0.0, 4.0)

    if config.ban_window:
        start, end = sorted((config.start, config.end))
        return 0.0 if start <= progress <= end else stage_strength
    return _curve_factor_for_window(progress, config.start, config.end, config.peak, config.stage_strength, config.curve)


@dataclass
class PassConfig:
    target_mode: str
    target_loras: str
    panel_weight: PanelWeightSpec
    template_name: str
    lora_templates: dict[str, str]
    auto_timing: bool
    stage: str
    parts: list[dict]
    start: float
    end: float
    peak: float
    stage_strength: float
    curve: str
    shift: float
    prompt_names: set[str] = field(default_factory=set)
    # Manual timing: the start-end window blocks the LoRA instead of applying it
    ban_window: bool = False
    _lora_template_cache: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def panel_weight_nonzero(self) -> bool:
        return self.panel_weight.active

    @property
    def shapes_window(self) -> bool:
        """Whether this config's own curve shapes its window: no template
        parts run, and the window applies the LoRA rather than banning it."""
        return not self.parts and not self.ban_window

    @property
    def uses_panel_curve(self) -> bool:
        """Whether the panel's curve shapes a window of the run: its own, or a
        per-LoRA template's that has no curve of its own."""
        if self.shapes_window:
            return True
        names = set(self.lora_templates.values()) - {self.template_name}
        return any(self.config_for_template(name).shapes_window and not _template_config(name).get("curve") for name in names)

    @property
    def targets(self) -> set[str]:
        return _parse_targets(self.target_loras)

    @property
    def patch_signature(self) -> tuple:
        """What LoRA patches loaded under this pass depend on: which files are
        scheduled and their panel weights. Timing is looked up every step."""
        return self.target_mode, frozenset(self.targets), self.panel_weight.raw

    @property
    def text_encoder_signature(self) -> tuple | None:
        """The part of patch_signature the prompt conditioning depends on: a
        panel weight reaches the text encoder as a number or a qwen= section."""
        panel = self.panel_weight
        return self.patch_signature if panel.numeric is not None or panel.has_qwen else None

    def finalize_timing(self):
        """Resolve the template's parts, the stage (with auto timing that of
        its first applying part, else the template's, else the panel's, else
        with None the template's first applying part; character when
        invalid), the timing window, strength and curve (an invalid one reads
        as smooth)."""
        template = _template_config(self.template_name)
        template_parts = _template_parts(template)
        self.parts = active_template_parts(self.auto_timing, template_parts)
        self.ban_window = not self.auto_timing and manual_window_bans(template_parts)
        stage = _primary_template_part(template)["stage"] if self.parts else template.get("stage") or self.stage or _primary_template_part(template)["stage"]
        self.stage = stage if stage in STAGE_CHOICES else STAGE_CHARACTER

        if self.auto_timing:
            self.start, self.end, self.peak, _ = _auto_stage_values(stage, self.shift)
        else:
            self.start = _clamp(_to_float(self.start), 0.0, 1.0)
            self.end = _clamp(_to_float(self.end), 0.0, 1.0)
            if self.end < self.start:
                self.start, self.end = self.end, self.start
            self.peak = _clamp(_to_float(self.peak), self.start, self.end)
        self.stage_strength = _clamp(_to_float(self.stage_strength, 1.0), 0.0, 4.0)

        self.curve = _normalize_curve(self.curve)

    def config_for_file(self, filename, names=None):
        names = _names_for_lora_file(filename) if names is None else names
        template_name = None
        for selector_name, selector_template in self.lora_templates.items():
            if selector_name in names:
                template_name = selector_template
                break

        if not template_name or template_name == self.template_name:
            return self
        return self.config_for_template(template_name)

    def config_for_template(self, template_name):
        """This panel run on the per-LoRA template ``template_name`` (cached)."""
        # Template names are exact: "Late" and "late" are two templates
        cached = self._lora_template_cache.get(template_name)
        if cached is not None:
            return cached

        template = _template_config(template_name)
        auto_timing = bool(template.get("auto", self.auto_timing))
        start = template.get("start", self.start) if not auto_timing else self.start
        end = template.get("end", self.end) if not auto_timing else self.end
        peak = template.get("peak", self.peak) if not auto_timing else self.peak
        stage_strength = template.get("strength", self.stage_strength) if not auto_timing else self.stage_strength

        config = PassConfig(
            target_mode=self.target_mode,
            target_loras=self.target_loras,
            panel_weight=self.panel_weight,
            template_name=template_name,
            lora_templates=self.lora_templates,
            auto_timing=auto_timing,
            stage=self.stage,
            parts=[],
            start=start,
            end=end,
            peak=peak,
            stage_strength=stage_strength,
            curve=template.get("curve") or self.curve,
            shift=self.shift,
            prompt_names=self.prompt_names,
        )
        # Resolves the template's parts, stage and timing, as for the panel's
        config.finalize_timing()
        self._lora_template_cache[template_name] = config
        return config

    def prompt_name_matches(self, lora_name: str, misses: set[str] | None = None) -> bool:
        """``misses``: see network_misses."""
        if self.target_mode == MODE_PROMPT_ALL:
            return True
        if self.target_mode == MODE_TARGETS:
            return bool(name_keys(lora_name) & self.targets)

        filename = _resolve_lora_filename(lora_name, misses)
        return _is_anima_lora_file(filename)

    def file_matches(self, filename, names=None) -> bool:
        names = _names_for_lora_file(filename) if names is None else names
        if self.target_mode == MODE_PROMPT_ALL:
            return bool(names & self.prompt_names) if self.prompt_names else True
        if self.target_mode == MODE_TARGETS:
            return bool(names & self.targets)
        return _is_anima_lora_file(filename)


# Tag of patches that serve both passes (see RuntimeState.patch_pass)
BOTH_PASSES = "both"


class ScheduledStrength:
    """A LoRA patch strength that follows the schedule every step. It stands
    in for a float in the weight adapters' arithmetic (e.g. BOFT's
    ``1 - strength``); tensors defer to its reflected operators."""

    def __init__(self, base_strength, filename, pass_name: str):
        self.base_strength = _to_float(base_strength, 1.0)
        self.filename = filename
        # "base", "hires" or BOTH_PASSES: the passes these patches were built for
        self.pass_name = pass_name
        self.factor = 1.0

    def value(self) -> float:
        return float(self.base_strength) * float(self.factor)

    def __float__(self):
        return self.value()

    def __bool__(self):
        return self.value() != 0.0

    def __repr__(self):
        return _format_float(self.value())

    def __eq__(self, other):
        return self.value() == other

    def __ne__(self, other):
        return self.value() != other

    def __mul__(self, other):
        return self.value() * other

    def __rmul__(self, other):
        return other * self.value()

    def __truediv__(self, other):
        return self.value() / other

    def __rtruediv__(self, other):
        return other / self.value()

    def __add__(self, other):
        return self.value() + other

    def __radd__(self, other):
        return other + self.value()

    def __sub__(self, other):
        return self.value() - other

    def __rsub__(self, other):
        return other - self.value()

    def __neg__(self):
        return -self.value()

    def __abs__(self):
        return abs(self.value())

    def __lt__(self, other):
        return self.value() < other

    def __le__(self, other):
        return self.value() <= other

    def __gt__(self, other):
        return self.value() > other

    def __ge__(self, other):
        return self.value() >= other


@dataclass
class RuntimeState:
    p: object
    enabled: bool
    base: PassConfig
    hires: PassConfig
    hr_disable_mode: str
    hr_disable_tags: list[str]
    original_online_lora: bool
    current_activation_pass: str | None = None
    wrappers: list[weakref.ref] = field(default_factory=list)
    warned_offline: bool = False
    yielded_to_lora_control: set = field(default_factory=set)
    # The patch_pass() tag of the loaded LoRA patches
    loaded_pass: str | None = None
    # load_networks' current_lora_hash after this run's own last activation
    loaded_hash: str | None = None
    # The run this one interrupted (a nested process_images), active again after it
    outer_state: object = None
    _file_names: dict = field(default_factory=dict)
    _file_configs: dict = field(default_factory=dict)

    def current_config(self) -> PassConfig:
        pass_name = self.current_pass_name()
        if pass_name == "hires":
            return self.hires
        return self.base

    def current_pass_name(self) -> str:
        if self.current_activation_pass in {"base", "hires"}:
            return self.current_activation_pass
        if bool(getattr(self.p, "is_hr_pass", False)):
            return "hires"
        return "base"

    def patch_pass(self, pass_name: str) -> str:
        """Tag of the patches loaded for ``pass_name``: one set serves both
        passes unless their panels build different ones, so Hires. fix does
        not reload every LoRA from disk on each pass change."""
        if self.base.patch_signature == self.hires.patch_signature:
            return BOTH_PASSES
        return pass_name

    def file_names(self, filename) -> frozenset[str]:
        """lora_file_names, once per run: it scans every installed LoRA."""
        if filename not in self._file_names:
            self._file_names[filename] = _names_for_lora_file(filename)
        return self._file_names[filename]

    def file_config(self, config: PassConfig, filename) -> PassConfig:
        """``config`` with the per-LoRA template of ``filename`` applied."""
        key = (id(config), filename)
        if key not in self._file_configs:
            self._file_configs[key] = config.config_for_file(filename, self.file_names(filename))
        return self._file_configs[key]

    def register(self, wrapper: ScheduledStrength):
        # Weakly: a LoRA reload drops its patches (and their strengths), which
        # must then stop being updated every step.
        self.wrappers.append(weakref.ref(wrapper))

    def live_wrappers(self) -> list[ScheduledStrength]:
        alive = [ref() for ref in self.wrappers]
        if None in alive:
            self.wrappers = [ref for ref, wrapper in zip(self.wrappers, alive) if wrapper is not None]
        return [wrapper for wrapper in alive if wrapper is not None]

    def set_factor(self, factor: float):
        for wrapper in self.live_wrappers():
            wrapper.factor = factor


def _build_pass_config(target_mode, target_loras, panel_weight, template_name, lora_templates_text, auto_timing, stage, start, end, peak, stage_strength, curve, shift):
    target_mode = _normalize_target_mode(target_mode)
    config = PassConfig(
        target_mode=target_mode,
        target_loras=str(target_loras or ""),
        panel_weight=_parse_panel_weight(panel_weight),
        template_name=str(template_name or next(iter(BUILTIN_TEMPLATES))),
        lora_templates=_parse_lora_template_overrides(lora_templates_text),
        auto_timing=bool(auto_timing),
        stage=stage,  # validated by finalize_timing
        parts=[],
        start=_to_float(start, 0.2),
        end=_to_float(end, 0.8),
        peak=_to_float(peak, 0.5),
        stage_strength=_to_float(stage_strength, 1.0),
        curve=_normalize_curve(curve),
        shift=_to_float(shift, 3.0),
    )
    config.finalize_timing()
    return config


def _panel_infotext(pass_name: str, config: PassConfig, lora_templates_text) -> dict:
    """Infotext of one pass's panel, keyed by ``_panel_key``: what the run
    used (the timing resolved by finalize_timing, the Shift's under auto
    timing), with choices by code. Text a run ignored or left empty is left
    out; pasting such a panel empties that field (_translate_pasted_infotext)."""

    values = {
        "scope": TARGET_MODE_CODES[config.target_mode],
        "targets": config.target_loras.strip() if config.target_mode == MODE_TARGETS else "",
        "weight": config.panel_weight.raw,
        "template": config.template_name,
        # As typed: its lines are the per-LoRA template map
        "per-LoRA templates": str(lora_templates_text or "").strip(),
        "auto timing": config.auto_timing,
        "start": round(config.start, 4),
        "end": round(config.end, 4),
        "peak": round(config.peak, 4),
        "strength": round(config.stage_strength, 4),
        # Only when the run used it: a template's parts bring their own curves
        "curve": CURVE_CODES[config.curve] if config.uses_panel_curve else "",
    }
    return {_panel_key(pass_name, name): value for name, value in values.items() if value != ""}


def _run_infotext(p, state: RuntimeState, hr_independent, base_lora_templates, hr_lora_templates) -> dict:
    """The infotext of a scheduled run, keyed by INFOTEXT_KEYS. The Hires. fix
    section only when the run has a Hires. fix pass, its panel only when
    independent (else the base panel runs it at the Hires Shift)."""

    params = {INFOTEXT_ENABLED: True, **_panel_infotext("base", state.base, base_lora_templates)}
    if not getattr(p, "enable_hr", False):
        return params
    params[INFOTEXT_HIRES_INDEPENDENT] = bool(hr_independent)
    if hr_independent:
        params.update(_panel_infotext("hires", state.hires, hr_lora_templates))
    if state.hr_disable_mode != DISABLE_NONE:
        params[INFOTEXT_HIRES_DISABLE] = DISABLE_CODES[state.hr_disable_mode]
        if state.hr_disable_mode == DISABLE_SELECTED and state.hr_disable_tags:
            params[INFOTEXT_HIRES_DISABLE_TAGS] = "\n".join(state.hr_disable_tags)
    return params


def _remove_hires_lora_tags(p, state: RuntimeState):
    if not getattr(p, "enable_hr", False) or not hasattr(p, "all_hr_prompts") or not p.all_hr_prompts:
        return

    mode = state.hr_disable_mode
    if mode == DISABLE_NONE:
        return

    tags = state.hr_disable_tags

    def rewrite(prompt):
        text = str(prompt or "")
        if mode == DISABLE_ALL:
            return LORA_TAG_RE.sub("", text)
        if mode == DISABLE_SELECTED:
            for tag in tags:
                text = text.replace(tag, "")
        return text

    p.all_hr_prompts = [rewrite(prompt) for prompt in p.all_hr_prompts]


def _collect_prompt_names(prompts) -> set[str]:
    names = set()
    for prompt in prompts or []:
        for match in LORA_TAG_RE.finditer(str(prompt or "")):
            names |= name_keys(match.group(1))
    return names


def _rewrite_prompt_loras(prompts, config: PassConfig, misses: set[str] | None = None):
    """The prompts with the tags of the LoRAs this panel sets written at its
    multiplier. The panel's weight replaces all of the tag's: its text
    encoder / UNet arguments (``<lora:x:1:0.5>``, ``te=``, ``unet=``) are
    dropped, as add_patches sets both sides' strengths from the panel.

    Tags with an ``@`` stay as written: LoRA Control schedules
    ``<lora:x:[w@t,...]>`` (add_patches leaves those files to it), and Forge
    loads such a tag's text-encoder side at 0 only while the ``@`` is there."""
    rewritten = []

    def replace(match):
        name = match.group(1).strip()
        if "@" in (match.group(2) or ""):
            return match.group(0)
        if config.panel_weight_nonzero and config.prompt_name_matches(name, misses):
            return f"<lora:{name}:{_format_float(config.panel_weight.prompt_multiplier())}>"
        return match.group(0)

    for prompt in prompts or []:
        rewritten.append(LORA_TAG_RE.sub(replace, str(prompt or "")))
    return rewritten


def _rewrite_batch_prompts(p, state: RuntimeState):
    misses = network_misses(p)

    def rewrite(config):
        def apply(prompts):
            rewritten = _rewrite_prompt_loras(prompts, config, misses)
            config.prompt_names.update(_collect_prompt_names(rewritten))
            return rewritten

        return apply

    rewrite_batch_prompts(p, rewrite(state.base), rewrite(state.hires) if getattr(p, "enable_hr", False) else None)


def _patch_added_strengths(patch_destination, before_counts, state: RuntimeState, config: PassConfig, online_mode, lora_blocks=None, model_blocks=None, filename=None):
    pass_name = state.patch_pass(state.current_pass_name())

    def scheduled_patch(key, patch):
        if not isinstance(patch, (tuple, list)) or len(patch) < 5:
            return None

        strength_patch, patch_value, strength_model, offset, function = patch[:5]
        forge_fields = tuple(patch[5:])
        model_key = _model_key(key)
        base_strength = config.panel_weight.strength_for_key(model_key, strength_patch, lora_blocks, model_blocks) if config.panel_weight_nonzero else strength_patch

        if _is_unet_patch_key(model_key):
            if online_mode:
                strength_obj = ScheduledStrength(base_strength, filename, pass_name)
                state.register(strength_obj)
                strength_patch = strength_obj
            else:
                if not state.warned_offline:
                    logger.warning("Anima stage scheduler needs Forge online LoRA for per-step strength control.")
                    state.warned_offline = True
                strength_patch = _to_float(base_strength, 1.0)
        elif config.panel_weight_nonzero:
            strength_patch = _to_float(base_strength, 1.0)
        else:
            strength_patch = base_strength

        updated = (strength_patch, patch_value, strength_model, offset, function, *forge_fields)
        # Online patches are mutable lists that LoRA Control edits in place.
        return list(updated) if isinstance(patch, list) else updated

    for key, current_patches in patch_destination.items():
        start = before_counts.get(key, 0)
        if start >= len(current_patches):
            continue

        for index in range(start, len(current_patches)):
            entry = current_patches[index]
            # Online LoRAs are OnlineLoRAPatch weight wrappers holding a patch list.
            wrapped = getattr(entry, "patch", None)
            if isinstance(wrapped, list):
                for slot, patch in enumerate(wrapped):
                    updated = scheduled_patch(key, patch)
                    if updated is not None:
                        wrapped[slot] = updated
                continue

            updated = scheduled_patch(key, entry)
            if updated is not None:
                current_patches[index] = updated


def _add_patches(original, patcher, patches, strength_patch, strength_model, filename, online_mode):
    state = _ACTIVE_STATE
    # Only LoRAs the scheduled run itself loads: _activate marks the pass of
    # that run's activations alone. A nested run (ADetailer) loading its own
    # <lora:x:0.6> keeps its strength, not the outer run's panel.
    owned = state is not None and state.enabled and state.current_activation_pass is not None
    config = state.current_config() if owned else None
    should_control = config is not None and config.file_matches(filename, state.file_names(filename))
    if should_control and filename in lora_control_files():
        # Forge's LoRA Control (<lora:name:[w@t,...]>) overwrites the
        # strength slot every step; leave that LoRA to it rather than
        # having one schedule silently replace the other.
        should_control = False
        _note_lora_control(state, filename)
    if state is not None and not should_control and online_mode and not state.original_online_lora and not online_lora_leased_by_other_than(ONLINE_LORA_OWNER):
        # Only scheduled LoRAs need the online patches this run switched on;
        # the others keep Forge's faster merged weights, unless another script
        # (FreeFuse) holds online LoRA on for them too.
        online_mode = False

    # Forge Neo stores offline patches in ``patches`` and online LoRAs
    # as ``OnlineLoRAPatch`` entries in ``weight_wrapper_patches``.
    patch_destination = patcher.weight_wrapper_patches if online_mode else patcher.patches
    before_counts = {key: len(value) for key, value in patch_destination.items()} if should_control else {}

    loaded = original(patcher, patches, strength_patch=strength_patch, strength_model=strength_model, filename=filename, online_mode=online_mode)

    if should_control:
        lora_blocks = model_blocks = None
        if config.panel_weight.has_block:
            blocks = getattr(getattr(getattr(patcher, "model", None), "diffusion_model", None), "blocks", None)
            model_blocks = len(blocks) if blocks is not None else None
            lora_blocks = _lora_block_count(filename)
        _patch_added_strengths(patch_destination, before_counts, state, config, bool(online_mode), lora_blocks, model_blocks, filename)

    return loaded


def _activate(original, p, extra_network_data):
    state = getattr(p, "_anima_lora_stage_scheduler_state", None)
    previous_pass = getattr(state, "current_activation_pass", None) if state is not None else None
    pass_name = _prepare_for_extra_network_activation(p, extra_network_data)
    if state is not None and pass_name is not None:
        state.current_activation_pass = pass_name
    try:
        result = original(p, extra_network_data)
        if state is not None and pass_name is not None:
            state.loaded_hash = p.sd_model.current_lora_hash
        return result
    finally:
        if state is not None:
            state.current_activation_pass = previous_pass


def start_run(p, args):
    """Start a scheduled run of the panels' arguments ``args`` (enabled): both
    passes' configs, the online-LoRA lease and the run's infotext."""

    global _ACTIVE_STATE

    base_shift = _to_float(getattr(p, "distilled_cfg_scale", 3.0), 3.0)
    hires_shift = _to_float(getattr(p, "hr_distilled_cfg", base_shift), base_shift)

    # Each panel's _build_pass_config arguments except the Shift; with no
    # independent Hires panel the base one is reused at the Hires Shift.
    # The stage is None: finalize_timing resolves it from the template.
    base_panel = (args.base_target_mode, args.base_target_loras, args.base_panel_weight, args.base_template, args.base_lora_templates, args.base_auto, None, args.base_start, args.base_end, args.base_peak, args.base_stage_strength, args.base_curve)
    hires_panel = (args.hr_target_mode, args.hr_target_loras, args.hr_panel_weight, args.hr_template, args.hr_lora_templates, args.hr_auto, None, args.hr_start, args.hr_end, args.hr_peak, args.hr_stage_strength, args.hr_curve) if args.hr_independent else base_panel
    base_config = _build_pass_config(*base_panel, base_shift)
    hires_config = _build_pass_config(*hires_panel, hires_shift)

    state = RuntimeState(
        p=p,
        enabled=True,
        base=base_config,
        hires=hires_config,
        hr_disable_mode=_normalize_disable_mode(args.hr_disable_mode),
        hr_disable_tags=_parse_disable_tags(args.hr_disable_tags),
        original_online_lora=user_online_lora(),
        outer_state=_ACTIVE_STATE,
    )

    acquire_online_lora(p, ONLINE_LORA_OWNER)
    p._anima_lora_stage_scheduler_state = state
    _ACTIVE_STATE = state
    _note_text_encoder_signature(p, (base_config.text_encoder_signature, hires_config.text_encoder_signature))

    _remove_hires_lora_tags(p, state)

    # Keys match the Script's infotext_fields, so pasting restores the panels
    p.extra_generation_params.update(_run_infotext(p, state, args.hr_independent, args.base_lora_templates, args.hr_lora_templates))


def install_patch():
    # The wrappers are installed once per process; each call points them at
    # _add_patches / _activate, which hold the live run state
    if install_add_patches_hook("stage_scheduler.add_patches", _add_patches):
        logger.info("Installed Anima LoRA stage scheduler patch")
    if install_activate_hook("stage_scheduler.activate", _activate):
        logger.info("Installed Anima LoRA stage scheduler extra network patch")


def _note_lora_control(state, filename):
    note_lora_control_yield(
        state.yielded_to_lora_control,
        getattr(state.p, "extra_generation_params", None),
        filename,
        feature="Anima stage scheduler",
        infotext_key="Anima stage scheduler skipped",
    )


def _denoiser_callback(params):
    state = _ACTIVE_STATE
    if state is None or not state.enabled:
        return

    denoiser = getattr(params, "denoiser", None)
    p = getattr(denoiser, "p", None)
    if p is None:
        return
    if p is not state.p:
        # A nested run the scheduler is not part of (ADetailer) samples with
        # the loaded patches: at their plain strengths, not frozen at the
        # factor of the outer run's last step.
        state.set_factor(1.0)
        return

    pass_name = state.current_pass_name()
    pass_config = state.current_config()
    progress = _sampling_progress(params)
    # Hundreds of strengths share one or a few per-file configs: one curve each
    factors: dict[int, float] = {}
    for wrapper in state.live_wrappers():
        if wrapper.pass_name not in (pass_name, BOTH_PASSES):
            wrapper.factor = 0.0
            continue
        config = state.file_config(pass_config, wrapper.filename)
        if id(config) not in factors:
            factors[id(config)] = _stage_curve_factor(config, progress)
        wrapper.factor = factors[id(config)]
