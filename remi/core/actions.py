"""Actions of the ``core`` owner (``POST /api/v2/actions/core/{action}``).

``ctx.tab`` names the form; ``ctx.scripts`` carries the scripts' values (the
request's ``scripts``) for the cross-owner actions; ``ctx.extra`` the
action's own arguments (``selected``, ``label``, ``preset``). Bad input
raises :class:`remi.ui.ValidationError`.
"""

from __future__ import annotations

from typing import Any

from remi import ui
from remi.ui.types import Patch


def _tab(ctx: ui.ActionContext) -> str:
    from remi.core.forms import TABS

    tab = ctx.tab or "txt2img"
    if tab not in TABS[:2]:
        raise ui.ValidationError([ui.ParamError(f"core actions run on txt2img or img2img, not {tab!r}", owner="core")])
    return tab


def _argument(ctx: ui.ActionContext, name: str) -> str:
    value = ctx.extra.get(name)
    if not isinstance(value, str) or not value:
        raise ui.ValidationError([ui.ParamError(f"the action needs a {name!r} string", owner="core", param=name)])
    return value


class CoreActions:
    """The core form's actions; :func:`remi.ui.call_action` runs them."""

    @ui.action("swap_dimensions")
    def swap_dimensions(self, ctx: ui.ActionContext, values: dict) -> Patch:
        from remi.core.forms import dimensions_layout

        current = dimensions_layout(_tab(ctx)).coerce(values, unknown="ignore")
        return {"width": {"value": current["height"]}, "height": {"value": current["width"]}}

    @ui.action("hr_resolution")
    def hr_resolution(self, ctx: ui.ActionContext, values: dict) -> Patch:
        """The "Upscaled resolution" line of Hires. fix (``modules.ui.calc_resolution_hires``)."""

        from remi.core.forms import hires_resolution_layout

        current = hires_resolution_layout().coerce(values, unknown="ignore")
        return {"__outputs__": {"hr_final_resolution": hires_resolution_html(current)}}

    @ui.action("__conflicts__")
    def conflicts(self, ctx: ui.ActionContext, values: dict) -> Patch:
        from remi.core.conflicts import conflict_patch

        selected = ctx.extra.get("selected")
        return conflict_patch(_tab(ctx), {**ctx.scripts, "core": values}, selected if isinstance(selected, str) else None)

    @ui.action("__preset__")
    def anima_preset(self, ctx: ui.ActionContext, values: dict) -> Patch:
        from remi.core.presets import anima_preset_patch

        return anima_preset_patch(_tab(ctx), _argument(ctx, "label"))

    @ui.action("__forge_preset__", gpu=True)
    def forge_preset(self, ctx: ui.ActionContext, values: dict) -> Patch:
        """Select a UI preset: the model switch happens here, on the server
        (behind generation); the patch carries the form defaults and the
        settings as stored now (``__settings__``)."""

        from remi.core.presets import apply_forge_preset, check_preset, forge_preset_patch

        preset = check_preset(_argument(ctx, "preset"))
        settings = apply_forge_preset(preset)
        return {**forge_preset_patch(_tab(ctx), preset), "__settings__": settings}


def hires_resolution_html(values: dict[str, Any]) -> str:
    """``from <w>x<h> to <W>x<H>`` for the Hires. fix values (empty when off)."""

    if not values.get("enable_hr"):
        return ""
    from modules import processing
    from remi.core.forms import snap_resolution

    p = processing.StableDiffusionProcessingTxt2Img(width=values["width"], height=values["height"], enable_hr=True, hr_scale=values["hr_scale"], hr_resize_x=values["hr_resize_x"], hr_resize_y=values["hr_resize_y"])
    p.calculate_target_resolution()
    width = snap_resolution(p.hr_resize_x or p.hr_upscale_to_x)
    height = snap_resolution(p.hr_resize_y or p.hr_upscale_to_y)
    return f"from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{width}x{height}</span>"


core_actions = CoreActions()
