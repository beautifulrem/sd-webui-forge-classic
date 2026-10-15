"""Render a ``params()`` layout with Gradio, for the legacy Gradio UI only.

Until the Gradio UI is deleted, a script ported to ``params()`` (without a
``ui()``) still has to occupy its argument slots there. This renders a plain
approximation of the tree: inputs in positional order, sections as
accordions; conditions, actions and custom widgets are not supported (custom
and JSON params keep their default in a ``gr.State``). Nothing outside the
legacy UI imports this module.
"""

from __future__ import annotations

from typing import Any

from remi.ui import nodes
from remi.ui.context import Context
from remi.ui.layout import ScriptLayout


class LazyGradio:
    """``gr`` for modules whose legacy Gradio handlers keep ``request:
    gr.Request`` annotations (Gradio injects the request by that
    annotation): Gradio is imported only when an attribute is read."""

    def __getattr__(self, name: str) -> Any:
        import gradio

        return getattr(gradio, name)


gr = LazyGradio()


def render(layout: ScriptLayout, ctx: Context | None = None) -> tuple[list[Any], list[Any]]:
    """Create the Gradio components of ``layout`` in the current Blocks
    context. Returns ``(controls in positional order, paste fields)``."""

    from modules.infotext_utils import PasteField

    ctx = ctx or Context()
    created: dict[str, Any] = {}
    if layout.root is not None:
        _render(layout.root, ctx, created)
    controls = [created[node.name] for node in layout.inputs]
    paste_fields = []
    for node in layout.inputs:
        target = node.paste or node.infotext
        if target is not None:
            paste_fields.append(PasteField(created[node.name], target))
    return controls, paste_fields


def _input(node: nodes.Input, ctx: Context) -> Any:
    import gradio as gr

    default = node.resolve_default(ctx)
    common: dict[str, Any] = {"label": node.label or node.name, "show_label": bool(node.label), "visible": not node.hidden}
    if node.info:
        common["info"] = node.info

    if isinstance(node, nodes.Toggle):
        return gr.Checkbox(value=default, **common)
    if isinstance(node, nodes.Slider):
        return gr.Slider(minimum=node.minimum, maximum=node.maximum, step=node.step, value=default, **common)
    if isinstance(node, nodes.Seed):
        return gr.Number(value=default, precision=0, **common)
    if isinstance(node, nodes.Number):
        return gr.Number(value=default, precision=node.precision, minimum=node.minimum, maximum=node.maximum, **common)
    if isinstance(node, nodes.Text):
        return gr.Textbox(value=default, lines=node.lines, max_lines=node.max_lines or max(node.lines, 20), placeholder=node.placeholder, **common)
    if isinstance(node, nodes.Choice):
        choices = [(label, value) for label, value in node.resolve_choices(ctx)]
        if node.style == "radio":
            return gr.Radio(choices=choices, value=default, type=node.returns, **common)
        return gr.Dropdown(choices=choices, value=default, type=node.returns, allow_custom_value=node.allow_custom, **common)
    if isinstance(node, nodes.MultiChoice):
        choices = [(label, value) for label, value in node.resolve_choices(ctx)]
        if node.style == "checkboxes":
            return gr.CheckboxGroup(choices=choices, value=default, type=node.returns, **common)
        return gr.Dropdown(choices=choices, value=default, multiselect=True, type=node.returns, allow_custom_value=node.allow_custom, **common)
    if isinstance(node, nodes.Color):
        return gr.ColorPicker(value=default, **common)
    if isinstance(node, nodes.Image):
        return gr.Image(type="pil", image_mode=node.mode or "RGB", sources=["upload"], **common)
    if isinstance(node, nodes.ImageList):
        return gr.Files(file_types=["image"], **common)
    if isinstance(node, nodes.File):
        return gr.File(type="filepath", **common)
    return gr.State(value=default)  # Json, Custom, State


def _render(node: nodes.Node, ctx: Context, created: dict[str, Any]) -> None:
    import gradio as gr

    from modules.ui_components import InputAccordion

    if isinstance(node, nodes.Input):
        created[node.name] = _input(node, ctx)
        return
    if isinstance(node, nodes.Section):
        if node.enable is not None:
            with InputAccordion(node.enable.resolve_default(ctx), label=node.title) as enable:
                created[node.enable.name] = enable
                _render_children(node, ctx, created)
        else:
            with gr.Accordion(node.title, open=node.open):
                _render_children(node, ctx, created)
        return
    if isinstance(node, nodes.Group):
        with gr.Accordion(node.title, open=node.open) if node.title else gr.Group():
            _render_children(node, ctx, created)
        return
    if isinstance(node, nodes.Tabs):
        with gr.Tabs():
            for tab in node.items:
                with gr.Tab(tab.title):
                    _render_children(tab, ctx, created)
        return
    if isinstance(node, (nodes.Row, nodes.Column)):
        with gr.Row() if isinstance(node, nodes.Row) else gr.Column():
            _render_children(node, ctx, created)
        return
    if isinstance(node, nodes.Markdown):
        gr.Markdown(node.text)
    elif isinstance(node, nodes.Html):
        gr.HTML(node.text)
    elif isinstance(node, nodes.Separator):
        gr.HTML("<hr>")
    # Buttons, outputs and slots have no legacy rendering


def _render_children(node: nodes.Container, ctx: Context, created: dict[str, Any]) -> None:
    for child in node.items:
        _render(child, ctx, created)
