"""The panel's ``remi.ui`` parameters: the layout, how a recorded infotext
pastes back into it, and the template actions' patches."""

import logging
from functools import partial

from lib_stage_scheduler.options import (
    BUILDER_PRESET_LABELS,
    BUILTIN_TEMPLATES,
    CURVE_CHOICES,
    CURVE_CODES,
    CURVE_LABELS_EN,
    DISABLE_ALIASES,
    DISABLE_CODES,
    DISABLE_NONE,
    DISABLE_SELECTED,
    INFOTEXT_ENABLED,
    INFOTEXT_HIRES_DISABLE,
    INFOTEXT_HIRES_DISABLE_TAGS,
    INFOTEXT_HIRES_INDEPENDENT,
    INTRO_EN,
    MODE_AUTO_ANIMA,
    MODE_TARGETS,
    PARAM_NAMES,
    PRESET_PARTS,
    STAGE_COMPOSITION,
    STAGE_LABELS_EN,
    STAGE_STYLE,
    TARGET_MODE_ALIASES,
    TARGET_MODE_CODES,
    TEMPLATE_LABELS_EN,
    TIMING_KEYS,
    _default_timing,
    _disable_choices,
    _lookup_choice,
    _panel_key,
    _target_mode_choices,
    _timing_value,
)
from lib_stage_scheduler.templates import _template_names
from lib_stage_scheduler.ui import apply_template, panel_template_after, template_choices, templates_payload
from modules.anima_support import PasteKey, table_params
from remi import ui

logger = logging.getLogger("anima_lora_stage_scheduler")

# Infotext fields of a panel's timing value, in TIMING_KEYS order
TIMING_FIELDS = ("auto timing", "start", "end", "peak", "strength", "curve")
# Arguments prefix -> infotext pass name
PASSES = {"base": "base", "hr": "hires"}
# Where a pass reads its Shift and step count: the first of these params
SHIFT_PARAMS = {"base": ["core.distilled_cfg_scale"], "hr": ["core.hr_distilled_cfg", "core.distilled_cfg_scale"]}
STEP_PARAMS = {"base": ["sampler.steps"], "hr": ["core.hr_second_pass_steps", "sampler.steps"]}


# region Paste


def _recorded(params, pass_name) -> bool:
    """Whether the infotext records the ``pass_name`` panel (its template)."""
    return _panel_key(pass_name, "template") in params


def _pasted_choice(key, codes, aliases, params):
    """A recorded choice (its code, the choice or its label) as the choice;
    None leaves the control alone (not recorded, or unknown)."""
    value = params.get(key)
    if value is None:
        return None
    choice = _lookup_choice(value, codes, aliases)
    if choice is None:
        logger.warning("Ignored unknown pasted %s: %s", key, value)
    return choice


def _pasted_text(pass_name, field, params):
    """A panel text: as recorded; empty when the recorded panel left it out
    (the run ignored it, or it was empty); else left alone."""
    value = params.get(_panel_key(pass_name, field))
    if value is not None:
        return str(value)
    return "" if _recorded(params, pass_name) else None


def _pasted_template(pass_name, params):
    """A recorded template that is installed here; else left alone."""
    name = params.get(_panel_key(pass_name, "template"))
    if name is None:
        return None
    if name not in _template_names():
        logger.warning("Pasted Anima LoRA stage template is not installed: %s", name)
        return None
    return name


def _pasted_timing(pass_name, params):
    """The timing value of a recorded panel: its auto switch, window,
    strength and curve (a curve the run did not use reads as smooth)."""
    if not _recorded(params, pass_name):
        return None
    recorded = {key: params.get(_panel_key(pass_name, field)) for key, field in zip(TIMING_KEYS, TIMING_FIELDS, strict=True)}
    return _timing_value({key: value for key, value in recorded.items() if value is not None})


def _pasted_hires(key, default, params):
    """A Hires. fix field, recorded with the Hires switch (else left alone);
    one the run left out reads as ``default``."""
    if INFOTEXT_HIRES_INDEPENDENT not in params:
        return None
    return params.get(key, default)


def _pasted_disable_mode(params):
    if INFOTEXT_HIRES_INDEPENDENT not in params:
        return None
    return _lookup_choice(params.get(INFOTEXT_HIRES_DISABLE, DISABLE_CODES[DISABLE_NONE]), DISABLE_CODES, DISABLE_ALIASES)


def _panel_rows(prefix):
    pass_name = PASSES[prefix]
    key = partial(_panel_key, pass_name)
    return (
        (f"{prefix}_target_mode", PasteKey(key("scope"), partial(_pasted_choice, key("scope"), TARGET_MODE_CODES, TARGET_MODE_ALIASES)), None),
        (f"{prefix}_target_loras", PasteKey(key("targets"), partial(_pasted_text, pass_name, "targets")), None),
        (f"{prefix}_panel_weight", key("weight"), None),
        (f"{prefix}_template", PasteKey(key("template"), partial(_pasted_template, pass_name)), None),
        (f"{prefix}_lora_templates", PasteKey(key("per-LoRA templates"), partial(_pasted_text, pass_name, "per-LoRA templates")), None),
        (f"{prefix}_timing", PasteKey(key("auto timing"), partial(_pasted_timing, pass_name)), None),
    )


# One row per params() input in argument order: (name, infotext key, preset)
PARAM_ROWS = (
    ("enabled", INFOTEXT_ENABLED, "lora_stage.enabled"),
    *_panel_rows("base"),
    ("hr_independent", INFOTEXT_HIRES_INDEPENDENT, None),
    *_panel_rows("hr"),
    ("hr_disable_mode", PasteKey(INFOTEXT_HIRES_DISABLE, _pasted_disable_mode), None),
    ("hr_disable_tags", PasteKey(INFOTEXT_HIRES_DISABLE_TAGS, partial(_pasted_hires, INFOTEXT_HIRES_DISABLE_TAGS, "")), None),
)
assert tuple(row[0] for row in PARAM_ROWS) == PARAM_NAMES


# endregion

# region Layout


def timeline_props(prefix):
    """Props of a panel's stage-timeline widget (templates read per schema)."""
    return {
        "panel": prefix,
        "template_param": f"{prefix}_template",
        "shift": SHIFT_PARAMS[prefix],
        "steps": STEP_PARAMS[prefix],
        "strength_max": 2.0,
        "curves": [{"value": curve, "label": CURVE_LABELS_EN[curve], "shape": CURVE_CODES[curve]} for curve in CURVE_CHOICES],
        "stages": [{"value": stage, "label": STAGE_LABELS_EN[stage], "kind": {STAGE_COMPOSITION: "composition", STAGE_STYLE: "style"}.get(stage, "character")} for stage in STAGE_LABELS_EN],
        "builder": [{"value": name, "label": TEMPLATE_LABELS_EN[name], "stage": part["stage"], "mode": part["mode"], "legacy_label": BUILDER_PRESET_LABELS[name]} for name, part in PRESET_PARTS.items()],
        "templates": templates_payload(),
    }


class _TemplateChoice(ui.Choice):
    """A panel's template dropdown. Templates are shared by both tabs, and a
    delete or rename refreshes the calling form only: a template that is
    gone runs as the default template, with a warning, rather than failing
    the job."""

    def coerce(self, value, ctx):
        if isinstance(value, str) and value not in self.values(ctx):
            logger.warning("Anima LoRA stage template %r no longer exists; using %r", value, self.default)
            value = self.default
        return super().coerce(value, ctx)


def _panel(prefix):
    default_template = next(iter(BUILTIN_TEMPLATES))
    return [
        ui.Row(
            ui.Choice(f"{prefix}_target_mode", _target_mode_choices(), MODE_AUTO_ANIMA, label="Target scope", style="radio"),
            ui.Text(
                f"{prefix}_panel_weight",
                "0",
                label="Panel weight, 0 = use prompt / layer-weight plugin",
                placeholder="0.8 or 0-18=1,19-27=0.2 or blocks=0-18=1,19-27=0.2;modules=mlp=0.8",
            ),
        ),
        ui.Text(
            f"{prefix}_target_loras",
            "",
            label="Target LoRA",
            placeholder="Separate multiple names with commas or new lines",
            visible_if=ui.param(f"{prefix}_target_mode", eq=MODE_TARGETS),
        ),
        _TemplateChoice(f"{prefix}_template", template_choices, default_template, label="Template", on_input=f"{prefix}_template_changed"),
        ui.Text(
            f"{prefix}_lora_templates",
            "",
            lines=3,
            label="Per-LoRA templates",
            placeholder="One per line: LoRA name=template name\nAlso supports: <lora:name:0.8>=仅人物特征阶段介入\nEmpty entries use the template above",
        ),
        ui.Custom(f"{prefix}_timing", "stage-timeline", _default_timing(default_template), partial(timeline_props, prefix), label="Stage timing", validate=_timing_value),
    ]


def params_root():
    """The panel; its inputs are PARAM_NAMES (bound by ``table_params``)."""
    hires = ui.param("hr_independent")
    return ui.Section(
        "Anima LoRA Stage Scheduler",
        ui.Markdown(INTRO_EN),
        ui.Group("Base pass", *_panel("base")),
        ui.Group(
            "Hires. fix stage",
            ui.Toggle("hr_independent", False, label="Use independent Hires. fix panel; disabled = inherit base settings with Hires Shift timing"),
            ui.Group("Hires. fix LoRA stage", *_panel("hr"), visible_if=hires),
            ui.Choice("hr_disable_mode", _disable_choices(), DISABLE_NONE, label="Disable LoRA during Hires. fix", style="radio"),
            ui.Text(
                "hr_disable_tags",
                "",
                lines=3,
                label="Full LoRA tags to disable",
                placeholder="<lora:name:0.8>\n<lora:name:blocks=0-9=0.8,10-18=1.0,19-27=0.7>",
                visible_if=ui.param("hr_disable_mode", eq=DISABLE_SELECTED),
            ),
            open=False,
        ),
        enable=ui.Toggle("enabled", False, label="Enable stage scheduling"),
    )


def bind(root):
    """Bind ``root``'s inputs to their infotext keys and preset field; the
    argument order."""
    return table_params(PARAM_ROWS, root)


# endregion

# region Template actions


def shift_of(ctx, prefix):
    """The Shift a panel's auto timing follows: the generation form's."""
    for ref in SHIFT_PARAMS[prefix]:
        value = ctx.core.get(ref.partition(".")[2])
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 3.0


def templates_patch(ctx, values, renamed=None) -> dict:
    """After the templates changed: both template dropdowns' choices, a
    panel whose template is gone (or renamed) on its new template with that
    template's timing, and the widgets' template data."""
    choices = template_choices()
    names = [name for _label, name in choices]
    patch = {}
    for prefix in PASSES:
        current = values.get(f"{prefix}_template")
        after = panel_template_after(current, names, renamed)
        entry = {"choices": choices}
        if after != current:
            entry["value"] = after
            patch[f"{prefix}_timing"] = ui.update(value=apply_template(after, shift_of(ctx, prefix)))
        patch[f"{prefix}_template"] = ui.update(**entry)
    patch["__outputs__"] = {"templates": templates_payload()}
    return patch


# endregion
