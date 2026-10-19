"""``remi.ui``: the declarative parameter and layout model.

Scripts declare their parameters once, as a tree returned by ``params()``::

    from remi import ui

    def params(self, is_img2img):
        return ui.Section("DAVE", enable=ui.Toggle("enable", False, preset="dave.enabled"), children=[
            ui.Slider("strength", 0.0, 1.0, 0.3, 0.05, infotext="DAVE strength", preset="dave.strength"),
        ])

That tree drives the SPA (schema export), the HTTP API (named values to
positional script arguments), infotext paste, presets and conflict locks.
See docs/remi-webui/architecture.md and schema-format.md.
"""

from remi.ui import conditions
from remi.ui.actions import ActionContext, ActionSpec, action, actions_of, call_action, check_patch, merge_patches, message, toast, update
from remi.ui.conditions import all_of, any_of, evaluate, not_, param
from remi.ui.context import Context
from remi.ui.errors import ParamError, SchemaError, ValidationError
from remi.ui.files import ActionFiles
from remi.ui.layout import ScriptLayout
from remi.ui.nodes import (
    Button,
    Choice,
    Color,
    Column,
    Custom,
    File,
    Gallery,
    Group,
    Html,
    Image,
    ImageList,
    Input,
    Json,
    Markdown,
    MultiChoice,
    Node,
    Number,
    Output,
    RangeSlider,
    Row,
    Section,
    Seed,
    Separator,
    Slider,
    Slot,
    State,
    Tab,
    Tabs,
    Text,
    Toggle,
)
from remi.ui.schema import SCHEMA_VERSION, jsonable, to_schema

__all__ = [
    "SCHEMA_VERSION",
    "ActionContext",
    "ActionFiles",
    "ActionSpec",
    "Button",
    "Choice",
    "Color",
    "Column",
    "Context",
    "Custom",
    "File",
    "Gallery",
    "Group",
    "Html",
    "Image",
    "ImageList",
    "Input",
    "Json",
    "Markdown",
    "MultiChoice",
    "Node",
    "Number",
    "Output",
    "ParamError",
    "RangeSlider",
    "Row",
    "SchemaError",
    "ScriptLayout",
    "Section",
    "Seed",
    "Separator",
    "Slider",
    "Slot",
    "State",
    "Tab",
    "Tabs",
    "Text",
    "Toggle",
    "ValidationError",
    "action",
    "actions_of",
    "all_of",
    "any_of",
    "call_action",
    "check_patch",
    "conditions",
    "evaluate",
    "jsonable",
    "merge_patches",
    "message",
    "not_",
    "param",
    "to_schema",
    "toast",
    "update",
]
