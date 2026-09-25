"""Shared, deterministic conflict rules for Anima generation controls.

The pure resolvers in this module are used by both Gradio callbacks and
processing-time validation.  GUI callbacks pass ``selected`` so the user's
last action wins.  API/CLI processing omits it, which keeps the primary
algorithm choice and safely drops incompatible auxiliary switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any


STANDARD_GUIDANCE = "Standard / preserve existing"
SMC_GUIDANCE = "SMC-CFG"
FDG_GUIDANCE = "FDG (experimental)"
GUIDANCE_MODES = (STANDARD_GUIDANCE, SMC_GUIDANCE, FDG_GUIDANCE)

FLOW_SAMPLERS = frozenset(
    {"Anima Flow Euler", "Anima Flow UniPC2", "Anima Flow PC3"}
)
MOMENTUM_SAMPLERS = frozenset(
    {"Euler", "Anima Flow Euler", "Anima FreeFuse Euler"}
)
CNS_SAMPLER = "Anima ER SDE CNS"
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
) -> GuidanceConflictState:
    """Normalize Anima guidance controls for GUI and headless processing."""

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

    # In the GUI the last explicit switch wins.  Headless processing has no
    # click order, so the selected CFG controller remains authoritative.
    if selected == "momentum" and momentum and mode != STANDARD_GUIDANCE:
        mode = STANDARD_GUIDANCE
        messages.append("CFG guidance mode reset: Momentum requires Standard")
    elif mode in {SMC_GUIDANCE, FDG_GUIDANCE} and momentum:
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
        dcw_interactive=(mode != FDG_GUIDANCE),
        messages=tuple(messages),
    )


def resolve_schedule_conflicts(
    *,
    sampler: str,
    scheduler: str,
    dynamic_shift: bool,
    selected: str | None = None,
) -> ScheduleConflictState:
    """Normalize sampler/scheduler/Dynamic Shift combinations."""

    dynamic_shift = bool(dynamic_shift)
    messages: list[str] = []

    if sampler in FLOW_SAMPLERS:
        if dynamic_shift:
            messages.append(f"Dynamic Shift disabled: {sampler} locks its scheduler")
        return ScheduleConflictState(
            scheduler="Anima FlowMatch",
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


def register_exclusive_component(
    *,
    tab: str,
    group: str,
    names: tuple[str, ...],
    name: str,
    component: Any,
    reset: bool = False,
) -> None:
    """Collect cross-extension controls and wire them once all are present.

    ``reset`` is used by the first component built for a tab so Reload UI does
    not retain handles from the previous Gradio tree.
    """

    if name not in names:
        raise ValueError(f"{name!r} is not a member of conflict group {group!r}")
    key = (str(tab), str(group))
    if reset:
        _EXCLUSIVE_UI_COMPONENTS[key] = {}
        _EXCLUSIVE_UI_SIGNATURES.pop(key, None)
    bucket = _EXCLUSIVE_UI_COMPONENTS.setdefault(key, {})
    bucket[name] = component
    if not all(member in bucket for member in names):
        return

    components = tuple(bucket[member] for member in names)
    signature = tuple(id(item) for item in components)
    if _EXCLUSIVE_UI_SIGNATURES.get(key) == signature:
        return
    wire_exclusive_components(names, components)
    _EXCLUSIVE_UI_SIGNATURES[key] = signature
