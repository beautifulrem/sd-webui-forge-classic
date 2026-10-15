"""Shared, deterministic conflict rules for Anima generation controls.

The pure resolvers in this module are used by both Gradio callbacks and
processing-time validation.  GUI callbacks pass ``selected`` so the user's
last action wins.  API/CLI processing omits it, which keeps the primary
algorithm choice and safely drops incompatible auxiliary switches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any


STANDARD_GUIDANCE = "Standard / preserve existing"
SMC_GUIDANCE = "SMC-CFG"
FDG_GUIDANCE = "FDG (experimental)"
CFG_ZERO_STAR_GUIDANCE = "CFG-Zero*"
APG_GUIDANCE = "APG"
TCFG_GUIDANCE = "TCFG"
# One dropdown: the CFG combine modes are mutually exclusive by construction.
GUIDANCE_MODES = (STANDARD_GUIDANCE, SMC_GUIDANCE, CFG_ZERO_STAR_GUIDANCE, APG_GUIDANCE, TCFG_GUIDANCE, FDG_GUIDANCE)

# Runs part of the pass at half resolution; it falls back to Anima Flow Euler
# for img2img, masks and resolution-bound hooks (sd_samplers_anima_speed).
SPEED_SAMPLER = "Anima SPEED"
FLOW_SAMPLERS = frozenset(
    {"Anima Flow Euler", "Anima Flow UniPC2", "Anima Flow PC3", SPEED_SAMPLER}
)
MOMENTUM_SAMPLERS = frozenset(
    {"Euler", "Anima Flow Euler", "Anima FreeFuse Euler"}
)
CNS_SAMPLER = "Anima ER SDE CNS"
# DCW corrects the latent in place when the model input's sigma changes, so it
# needs exactly one main model evaluation per step.  Multi-evaluation samplers
# (Heun, DPM2, DPM++ SDE / 2S, Restart, ...) would correct an intermediate
# tensor instead.  PC3's endpoint probe is an auxiliary pass that DCW skips.
DCW_SAMPLERS = MOMENTUM_SAMPLERS | frozenset(
    {
        "Euler a",
        "DPM++ 2M",
        "DPM++ 2M SDE",
        "DPM++ 3M SDE",
        "ER SDE",
        CNS_SAMPLER,
        "LMS",
        "LCM",
        "Res Multistep",
        "Anima Flow UniPC2",
        "Anima Flow PC3",
        SPEED_SAMPLER,
    }
)
DCW_INPAINT_MESSAGE = "DCW disabled: incompatible with an inpaint mask"
CONFLICT_METADATA_KEY = "Anima conflict resolution"


@dataclass(frozen=True)
class ExclusiveFeatureState:
    enabled: tuple[bool, ...]
    interactive: tuple[bool, ...]
    winner: str | None


@dataclass(frozen=True)
class GuidanceConflictState:
    mode: str
    momentum: bool
    dcw: bool
    mode_choices: tuple[str, ...]
    momentum_interactive: bool
    dcw_interactive: bool
    messages: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleConflictState:
    scheduler: str
    dynamic_shift: bool
    scheduler_interactive: bool
    dynamic_shift_interactive: bool
    messages: tuple[str, ...]


@dataclass(frozen=True)
class FreeFuseRefinerConflictState:
    freefuse: bool
    refiner: bool
    checkpoint: str
    freefuse_interactive: bool
    refiner_interactive: bool
    checkpoint_interactive: bool
    messages: tuple[str, ...]


# Spatial-routing features that cannot share a pass. Headless runs keep the
# UI group's order (FreeFuse > Regional > Artist Mixer): the others yield to
# the first one that is on. Scripts mark what actually runs.
FREEFUSE = "Anima FreeFuse"
REGIONAL = "Anima Regional Conditioning"
ARTIST_MIXER = "Anima Artist Mixer"
_ACTIVE_FEATURES = "_anima_active_features"


def set_feature_active(process: Any, feature: str, active: bool = True) -> None:
    """Record whether ``feature`` runs in the current generation of ``process``."""

    features = set(getattr(process, _ACTIVE_FEATURES, None) or ())
    if active:
        features.add(feature)
    else:
        features.discard(feature)
    setattr(process, _ACTIVE_FEATURES, features)


def is_feature_active(process: Any, feature: str) -> bool:
    return feature in (getattr(process, _ACTIVE_FEATURES, None) or ())


def record_conflict_resolution(process: Any, *messages: str) -> None:
    """Append unique compatibility decisions to generation infotext."""

    params = process.extra_generation_params
    existing = [
        item.strip()
        for item in str(params.get(CONFLICT_METADATA_KEY, "")).split(";")
        if item.strip()
    ]
    for message in messages:
        message = str(message).strip()
        if message and message not in existing:
            existing.append(message)
    if existing:
        params[CONFLICT_METADATA_KEY] = "; ".join(existing)


def resolve_exclusive_features(
    names: tuple[str, ...],
    enabled: tuple[bool, ...],
    *,
    selected: str | None = None,
) -> ExclusiveFeatureState:
    """Resolve a one-of-N group without depending on Gradio.

    A newly enabled GUI item wins.  With no click identity (API/CLI) the first
    declared enabled item is the stable priority.  Only the winner remains
    editable so it can be turned off; turning it off unlocks the whole group.
    """

    if len(names) != len(enabled):
        raise ValueError("feature names and enabled values must have equal length")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")

    values = tuple(bool(value) for value in enabled)
    winner = None
    if selected in names and values[names.index(selected)]:
        winner = selected
    else:
        winner = next((name for name, value in zip(names, values) if value), None)

    if winner is None:
        return ExclusiveFeatureState(
            enabled=tuple(False for _ in names),
            interactive=tuple(True for _ in names),
            winner=None,
        )

    return ExclusiveFeatureState(
        enabled=tuple(name == winner for name in names),
        interactive=tuple(name == winner for name in names),
        winner=winner,
    )


def resolve_guidance_conflicts(
    *,
    sampler: str,
    mode: str,
    momentum: bool,
    dcw: bool,
    selected: str | None = None,
    inpaint_mask: bool = False,
) -> GuidanceConflictState:
    """Normalize Anima guidance controls for GUI and headless processing.

    ``inpaint_mask`` is known only at processing time (the GUI leaves it off).
    """

    mode = mode if mode in GUIDANCE_MODES else STANDARD_GUIDANCE
    momentum = bool(momentum)
    dcw = bool(dcw)
    messages: list[str] = []

    if sampler == CNS_SAMPLER and mode == FDG_GUIDANCE:
        mode = STANDARD_GUIDANCE
        messages.append("FDG disabled: incompatible with Anima ER SDE CNS")

    if momentum and sampler not in MOMENTUM_SAMPLERS:
        momentum = False
        messages.append(f"Momentum disabled: unsupported sampler {sampler}")

    if dcw and sampler not in DCW_SAMPLERS:
        dcw = False
        messages.append(f"DCW disabled: unsupported sampler {sampler}")

    # With an inpaint mask the CFG denoiser hands the pre-denoiser callback a
    # blended copy of the sampler latent, so DCW's in-place correction would
    # never reach the latent the sampler steps from.
    if dcw and inpaint_mask:
        dcw = False
        messages.append(DCW_INPAINT_MESSAGE)

    # In the GUI the last explicit switch wins.  Headless processing has no
    # click order, so the selected CFG controller remains authoritative.
    if selected == "momentum" and momentum and mode != STANDARD_GUIDANCE:
        mode = STANDARD_GUIDANCE
        messages.append("CFG guidance mode reset: Momentum requires Standard")
    elif mode != STANDARD_GUIDANCE and momentum:
        momentum = False
        messages.append(f"Momentum disabled: incompatible with {mode}")

    if selected == "dcw" and dcw and mode == FDG_GUIDANCE:
        mode = STANDARD_GUIDANCE
        messages.append("CFG guidance mode reset: DCW is incompatible with FDG")
    elif mode == FDG_GUIDANCE and dcw:
        dcw = False
        messages.append("DCW disabled: incompatible with FDG")

    choices = list(GUIDANCE_MODES)
    if sampler == CNS_SAMPLER or dcw:
        choices.remove(FDG_GUIDANCE)
    if momentum:
        choices = [STANDARD_GUIDANCE]

    return GuidanceConflictState(
        mode=mode,
        momentum=momentum,
        dcw=dcw,
        mode_choices=tuple(choices),
        momentum_interactive=(sampler in MOMENTUM_SAMPLERS and mode == STANDARD_GUIDANCE),
        dcw_interactive=(sampler in DCW_SAMPLERS and mode != FDG_GUIDANCE),
        messages=tuple(messages),
    )


# Label of the scheduler the Flow samplers lock to (their "scheduler" option in
# sd_samplers_kdiffusion; tests/test_anima_feature_conflicts.py keeps them equal)
FLOW_SCHEDULER = "Anima FlowMatch"


def _flow_scheduler_choice() -> str:
    """The Schedule type a Flow sampler shows: its locked scheduler, or
    "Automatic" (which ``get_sigmas`` resolves to it) when that scheduler is
    not among the dropdown's choices, which ``sd_schedulers.schedulers`` holds
    as built at startup (hiding one takes effect after a restart)."""

    from modules import sd_schedulers

    return FLOW_SCHEDULER if any(scheduler.label == FLOW_SCHEDULER for scheduler in sd_schedulers.schedulers) else "Automatic"


def resolve_schedule_conflicts(
    *,
    sampler: str,
    scheduler: str,
    dynamic_shift: bool,
    selected: str | None = None,
) -> ScheduleConflictState:
    """Normalize sampler/scheduler/Dynamic Shift combinations.

    The Flow-sampler lock is also enforced in ``KDiffusionSampler.get_sigmas``
    (``lock_scheduler_override``), so API/XYZ runs cannot bypass it."""

    dynamic_shift = bool(dynamic_shift)
    messages: list[str] = []

    if sampler in FLOW_SAMPLERS:
        if dynamic_shift:
            messages.append(f"Dynamic Shift disabled: {sampler} locks its scheduler")
        return ScheduleConflictState(
            scheduler=_flow_scheduler_choice(),
            dynamic_shift=False,
            scheduler_interactive=False,
            dynamic_shift_interactive=False,
            messages=tuple(messages),
        )

    if selected == "scheduler" and dynamic_shift and scheduler != "Automatic":
        dynamic_shift = False
        messages.append("Dynamic Shift disabled: Schedule Type was selected")
    elif dynamic_shift:
        if scheduler != "Automatic":
            messages.append(f"{scheduler} replaced: Dynamic Shift controls the schedule")
        scheduler = "Automatic"

    return ScheduleConflictState(
        scheduler=scheduler,
        dynamic_shift=dynamic_shift,
        scheduler_interactive=not dynamic_shift,
        dynamic_shift_interactive=True,
        messages=tuple(messages),
    )


def resolve_freefuse_refiner_conflicts(
    *,
    freefuse: bool,
    refiner: bool,
    checkpoint: str | None,
    selected: str | None = None,
) -> FreeFuseRefinerConflictState:
    """Resolve FreeFuse against an actually configured Refiner switch."""

    freefuse = bool(freefuse)
    refiner = bool(refiner)
    checkpoint = str(checkpoint or "None")
    has_checkpoint = checkpoint.strip().lower() not in {"", "none"}
    refiner_active = refiner and has_checkpoint
    messages: list[str] = []

    if selected == "freefuse" and freefuse:
        winner = "freefuse"
    elif selected in {"refiner", "checkpoint"} and refiner_active:
        winner = "refiner"
    elif freefuse:
        winner = "freefuse"
    elif refiner_active:
        winner = "refiner"
    else:
        winner = None

    if winner == "freefuse":
        if refiner or has_checkpoint:
            messages.append("Refiner cleared: FreeFuse requires a single checkpoint")
        return FreeFuseRefinerConflictState(
            freefuse=True,
            refiner=False,
            checkpoint="None",
            freefuse_interactive=True,
            refiner_interactive=False,
            checkpoint_interactive=False,
            messages=tuple(messages),
        )
    if winner == "refiner":
        if freefuse:
            messages.append("FreeFuse disabled: Refiner checkpoint switch is active")
        return FreeFuseRefinerConflictState(
            freefuse=False,
            refiner=True,
            checkpoint=checkpoint,
            freefuse_interactive=False,
            refiner_interactive=True,
            checkpoint_interactive=True,
            messages=tuple(messages),
        )
    return FreeFuseRefinerConflictState(
        freefuse=False,
        refiner=refiner,
        checkpoint=checkpoint,
        freefuse_interactive=True,
        refiner_interactive=True,
        checkpoint_interactive=True,
        messages=(),
    )


# region Declarative conflict locks (Remi parameter model)

# One-of-N groups declared with ``ui.Toggle(..., exclusive=(group, member))``.
# Headless runs keep the first enabled member in this order.
SPATIAL_GROUP = "anima_spatial_conditioning"
PID_HIRES_GROUP = "pid_hires"
EXCLUSIVE_GROUPS: dict[str, tuple[str, ...]] = {
    SPATIAL_GROUP: ("freefuse", "regional", "artist"),
    PID_HIRES_GROUP: ("pid", "hires"),
}

# Conflict roles: the params the pairwise rules read and lock, named by their
# Anima preset field (``ui.Input(preset=...)``); the Refiner checkpoint has
# no preset field and is bound by the core forms.
ROLE_SAMPLER = "native.sampler"
ROLE_SCHEDULER = "native.scheduler"
ROLE_REFINER = "native.refiner"
ROLE_REFINER_CHECKPOINT = "native.refiner_checkpoint"
ROLE_GUIDANCE_MODE = "guidance.mode"
ROLE_MOMENTUM = "guidance.momentum_enabled"
ROLE_DCW = "guidance.dcw_enabled"
ROLE_DYNAMIC_SHIFT = "dynamic_shift.enabled"
ROLE_FREEFUSE = "freefuse.enabled"
CONFLICT_ROLES = frozenset({ROLE_SAMPLER, ROLE_SCHEDULER, ROLE_REFINER, ROLE_REFINER_CHECKPOINT, ROLE_GUIDANCE_MODE, ROLE_MOMENTUM, ROLE_DCW, ROLE_DYNAMIC_SHIFT, ROLE_FREEFUSE})


@dataclass(frozen=True)
class ConflictLocks:
    """What the rules make of the current values, by role: ``values`` (the
    resolved value of every role a rule covered), ``enabled`` (whether the
    control stays interactive), ``choices`` (allowed choices, where a rule
    narrows them) and the ``messages`` explaining any change."""

    values: dict[str, Any] = field(default_factory=dict)
    enabled: dict[str, bool] = field(default_factory=dict)
    choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    messages: tuple[str, ...] = ()


def resolve_conflict_locks(values: Mapping[str, Any], *, selected: str | None = None) -> ConflictLocks:
    """Apply every pairwise rule whose roles are all present in ``values``
    (role -> current value), as the Gradio panels wired them: FreeFuse vs
    Refiner, guidance (mode / Momentum / DCW vs the sampler) and schedule
    (scheduler / Dynamic Shift vs the sampler). ``selected`` is the role the
    user just changed, so their last action wins. Pure; no Gradio."""

    resolved: dict[str, Any] = {}
    enabled: dict[str, bool] = {}
    choices: dict[str, tuple[str, ...]] = {}
    messages: list[str] = []

    if {ROLE_FREEFUSE, ROLE_REFINER, ROLE_REFINER_CHECKPOINT} <= values.keys():
        state = resolve_freefuse_refiner_conflicts(
            freefuse=bool(values[ROLE_FREEFUSE]),
            refiner=bool(values[ROLE_REFINER]),
            checkpoint=values[ROLE_REFINER_CHECKPOINT],
            selected={ROLE_FREEFUSE: "freefuse", ROLE_REFINER: "refiner", ROLE_REFINER_CHECKPOINT: "checkpoint"}.get(selected),
        )
        resolved.update({ROLE_FREEFUSE: state.freefuse, ROLE_REFINER: state.refiner, ROLE_REFINER_CHECKPOINT: state.checkpoint})
        enabled.update({ROLE_FREEFUSE: state.freefuse_interactive, ROLE_REFINER: state.refiner_interactive, ROLE_REFINER_CHECKPOINT: state.checkpoint_interactive})
        messages += state.messages

    sampler = values.get(ROLE_SAMPLER)
    if sampler is not None and {ROLE_GUIDANCE_MODE, ROLE_MOMENTUM, ROLE_DCW} <= values.keys():
        state = resolve_guidance_conflicts(
            sampler=str(sampler),
            mode=str(values[ROLE_GUIDANCE_MODE]),
            momentum=bool(values[ROLE_MOMENTUM]),
            dcw=bool(values[ROLE_DCW]),
            selected={ROLE_SAMPLER: "sampler", ROLE_GUIDANCE_MODE: "mode", ROLE_MOMENTUM: "momentum", ROLE_DCW: "dcw"}.get(selected),
        )
        resolved.update({ROLE_GUIDANCE_MODE: state.mode, ROLE_MOMENTUM: state.momentum, ROLE_DCW: state.dcw})
        enabled.update({ROLE_MOMENTUM: state.momentum_interactive, ROLE_DCW: state.dcw_interactive})
        choices[ROLE_GUIDANCE_MODE] = state.mode_choices
        messages += state.messages

    if sampler is not None and {ROLE_SCHEDULER, ROLE_DYNAMIC_SHIFT} <= values.keys():
        state = resolve_schedule_conflicts(
            sampler=str(sampler),
            scheduler=str(values[ROLE_SCHEDULER]),
            dynamic_shift=bool(values[ROLE_DYNAMIC_SHIFT]),
            selected={ROLE_SAMPLER: "sampler", ROLE_SCHEDULER: "scheduler", ROLE_DYNAMIC_SHIFT: "dynamic_shift"}.get(selected),
        )
        resolved.update({ROLE_SCHEDULER: state.scheduler, ROLE_DYNAMIC_SHIFT: state.dynamic_shift})
        enabled.update({ROLE_SCHEDULER: state.scheduler_interactive, ROLE_DYNAMIC_SHIFT: state.dynamic_shift_interactive})
        messages += state.messages

    return ConflictLocks(resolved, enabled, choices, tuple(dict.fromkeys(messages)))


def exclusive_group_order(group: str, members) -> tuple[str, ...]:
    """``members`` of ``group`` in its priority order (declared groups first
    by :data:`EXCLUSIVE_GROUPS`, then the others in the given order)."""

    members = list(dict.fromkeys(members))
    declared = [member for member in EXCLUSIVE_GROUPS.get(group, ()) if member in members]
    return tuple(declared + [member for member in members if member not in declared])


# endregion


_EXCLUSIVE_UI_COMPONENTS: dict[
    tuple[str, str], dict[str, Any]
] = {}
_EXCLUSIVE_UI_SIGNATURES: dict[tuple[str, str], tuple[int, ...]] = {}


def _exclusive_gradio_updates(
    names: tuple[str, ...], selected: str, *enabled: bool
):
    import gradio as gr

    state = resolve_exclusive_features(names, tuple(enabled), selected=selected)
    return [
        gr.update(value=value, interactive=interactive)
        for value, interactive in zip(state.enabled, state.interactive)
    ]


def wire_exclusive_components(
    names: tuple[str, ...], components: tuple[Any, ...]
) -> None:
    """Wire a complete one-of-N Gradio group using the pure resolver."""

    if len(names) != len(components) or any(component is None for component in components):
        return
    for selected, component in zip(names, components):
        component.change(
            fn=partial(_exclusive_gradio_updates, names, selected),
            inputs=list(components),
            outputs=list(components),
            queue=False,
            show_progress=False,
        )


def _tab_build(tab: str) -> Any:
    """The UI build of ``tab`` under way: its prompt, the first component
    each build of a tab creates (the marker the presets extension resets its
    registry on), in the Gradio Blocks being built. None outside a build."""

    from gradio.context import Context

    root = Context.root_block
    if root is None:
        return None
    marker = f"{tab}_prompt"
    return next((block for block in root.blocks.values() if getattr(block, "elem_id", None) == marker), root)


def _current_exclusive_updates(key, signature, names, selected, *enabled):
    """The group's updates from the handlers of its latest wiring; those of
    a wiring a later member superseded change nothing."""

    if _EXCLUSIVE_UI_SIGNATURES.get(key) != signature:
        import gradio as gr

        return [gr.update() for _ in names]
    return _exclusive_gradio_updates(names, selected, *enabled)


def register_exclusive_component(
    *,
    tab: str,
    group: str,
    names: tuple[str, ...],
    name: str,
    component: Any,
) -> None:
    """Collect cross-extension controls of a one-of-N group and wire the ones
    present: a member whose extension is disabled is left out.

    A new build of ``tab`` (Reload UI) starts the group afresh, so handles of
    the previous Gradio tree are not kept. Each member registered after the
    second re-wires the group with it; the handlers wired before then change
    nothing any more.
    """

    if name not in names:
        raise ValueError(f"{name!r} is not a member of conflict group {group!r}")
    key = (str(tab), str(group))
    build = _tab_build(tab)
    bucket = _EXCLUSIVE_UI_COMPONENTS.get(key)
    if bucket is None or bucket.get("build") is not build:
        bucket = _EXCLUSIVE_UI_COMPONENTS[key] = {"build": build, "members": {}}
        _EXCLUSIVE_UI_SIGNATURES.pop(key, None)
    members = bucket["members"]
    members[name] = component

    present = tuple(member for member in names if member in members)
    if len(present) < 2:
        return
    components = tuple(members[member] for member in present)
    signature = tuple(id(item) for item in components)
    if _EXCLUSIVE_UI_SIGNATURES.get(key) == signature:
        return
    _EXCLUSIVE_UI_SIGNATURES[key] = signature
    for selected, item in zip(present, components):
        item.change(
            fn=partial(_current_exclusive_updates, key, signature, present, selected),
            inputs=list(components),
            outputs=list(components),
            queue=False,
            show_progress=False,
        )
