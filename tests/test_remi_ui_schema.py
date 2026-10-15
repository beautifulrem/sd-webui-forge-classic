"""remi.ui schema export (docs/remi-webui/schema-format.md): a snapshot of
every node kind, strict JSON, and the settings schema."""

import json
from pathlib import Path

import pytest

from remi import ui

SNAPSHOT = Path(__file__).parent / "fixtures" / "remi_ui_schema_snapshot.json"


def demo_tree():
    """One node of every kind (keep in step with the snapshot)."""

    return ui.Section(
        "Demo",
        enable=ui.Toggle("enable", False, infotext="Demo", exclusive=("spatial", "demo")),
        open=True,
        children=[
            ui.Row(
                ui.Slider("strength", 0.0, 1.0, 0.3, 0.05, info="How strong", infotext="Demo strength", preset="dave.strength", width=6),
                ui.Number("seed_offset", 0, precision=0, minimum=0),
                ui.Seed("seed"),
            ),
            ui.Column(ui.Text("prompt", "", lines=3, prompt=True, placeholder="Prompt", label=""), width="fill"),
            ui.Group(
                "More",
                ui.Choice("mode", ["Fast", ("Slow and steady", "slow")], "Fast", style="radio", on_change="refresh"),
                ui.MultiChoice("tags", lambda: ["a", "b"], ["a"], style="checkboxes", allow_custom=True),
                ui.Color("tint", "#ff0000", visible_if=ui.param("mode", eq="slow")),
                open=False,
            ),
            ui.Tabs(
                ui.Tab("Upload", ui.Image("image", mode="RGBA"), ui.Image("mask", "mask", "image"), value="upload"),
                ui.Tab("Folder", ui.ImageList("batch"), ui.File("model", accept=".safetensors"), value="dir"),
                bind="source",
            ),
            ui.Choice("source", [("Upload", "upload"), ("Folder", "dir")], hidden=True),
            ui.Json("table", {"rows": []}),
            ui.Custom("timeline", "stage-timeline", [], props={"stages": 3}),
            ui.State("cache", None),
            ui.Markdown("**Note**"),
            ui.Html("<b>raw</b>"),
            ui.Separator(),
            ui.Button("Refresh", "refresh", "tool", enabled_if=ui.not_(ui.param("enable"))),
            ui.Gallery("previews", "Previews"),
            ui.Output("status", "html"),
            ui.Slot("seed"),
        ],
    )


def test_schema_matches_the_snapshot():
    schema = ui.to_schema(demo_tree())
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert schema == expected, json.dumps(schema, ensure_ascii=False, indent=1)


def test_schema_is_strict_json_with_stable_key_order():
    schema = ui.to_schema(demo_tree())
    text = json.dumps(schema, ensure_ascii=False, allow_nan=False)
    assert text == json.dumps(ui.to_schema(demo_tree()), ensure_ascii=False, allow_nan=False)
    assert list(schema["children"][0]["children"][0])[:5] == ["type", "name", "label", "info", "default"]
    assert ui.to_schema(None) is None


def test_every_node_kind_is_covered_by_the_snapshot():
    kinds = {node.type for node in demo_tree().walk()}
    exported = {cls.type for cls in vars(ui.nodes).values() if isinstance(cls, type) and issubclass(cls, ui.Node) and cls.type}
    assert exported <= kinds | {"section"}, exported - kinds


def test_dynamic_choices_and_defaults_are_resolved_once_per_context():
    calls = []

    def choices():
        calls.append(1)
        return ["a", "b"]

    node = ui.Choice("c", choices)
    ctx = ui.Context()
    first = ui.to_schema(ui.Group(node), ctx)
    node.coerce("b", ctx)
    assert first["children"][0]["choices"] == [{"value": "a", "label": "a"}, {"value": "b", "label": "b"}]
    assert first["children"][0]["default"] == "a"
    assert len(calls) == 1
    ui.to_schema(ui.Group(node))
    assert len(calls) == 2


def test_on_input_is_exported_next_to_on_change():
    data = ui.to_schema(ui.Group(ui.Choice("c", ["a"], on_change="changed", on_input="picked")))["children"][0]
    assert (data["on_change"], data["on_input"]) == ("changed", "picked")
    assert "on_input" not in ui.to_schema(ui.Group(ui.Choice("c", ["a"])))["children"][0]


def test_toasts_take_translatable_messages():
    toast = ui.toast(ui.message("Saved template: {{name}}", name="mine"), "success")
    assert toast == {"__toast__": {"level": "success", "message": {"text": "Saved template: {{name}}", "params": {"name": "mine"}}}}
    merged = ui.merge_patches(ui.toast("Done."), toast)
    assert merged["__toast__"] == {"level": "success", "message": "Done.; Saved template: mine"}
    with pytest.raises(ui.SchemaError):
        ui.toast({"text": 3})


def test_jsonable():
    assert ui.jsonable({"a": (1, float("nan")), 2: {3}}) == {"a": [1, None], "2": [3]}


class _Category:
    def __init__(self, id, label):
        self.id, self.label = id, label


class _Categories:
    mapping = {"sd": _Category("sd", "Stable Diffusion")}


def _options():
    from modules import options as options_module

    templates = {}
    templates.update(
        options_module.options_section(
            ("demo", "Demo", "sd"),
            {
                "steps": options_module.OptionInfo(20, "Steps", ui.Slider, {"minimum": 1, "maximum": 150, "step": 1}).info("per pass").needs_reload_ui(),
                "mode": options_module.OptionInfo("b", "Mode", ui.Choice, {"style": "radio", "choices": ("a", "b")}).link("wiki", "https://example.org"),
                "hint": options_module.OptionHTML("Read <b>this</b>"),
                "row_open": options_module.OptionRow(),
                "width": options_module.OptionInfo(1024, "Width"),
                "name": options_module.OptionInfo("x", "Name", component_args={"visible": False}),
                "row_close": options_module.OptionRow(),
                "div": options_module.OptionDiv(),
                "colors": options_module.OptionInfo(["red"], "Colors", ui.MultiChoice, lambda: {"choices": ["red", "blue"], "style": "checkboxes"}, refresh=lambda: None),
                "flag": options_module.OptionInfo(False, "Flag", onchange=lambda: None).info("see <b>X/Y/Z Plot</b>").info("plain").needs_restart(),
                "locked": options_module.OptionInfo("Automatic", "Locked", ui.Choice, {"choices": ("Automatic",), "interactive": False}, restrict_api=True),
            },
        )
    )
    templates.update(options_module.options_section((None, "Hidden"), {"secret": options_module.OptionInfo("s", "Secret")}))
    return options_module.Options(templates, set())


def test_settings_schema():
    from remi.ui.settings import settings_schema

    opts = _options()
    opts.data["steps"] = 30
    schema = settings_schema(opts, categories=_Categories(), quicksettings=["steps"])
    json.dumps(schema, allow_nan=False)
    assert [category["id"] for category in schema["categories"]] == ["sd"]
    (section,) = schema["categories"][0]["sections"]
    assert section["id"] == "demo" and section["label"] == "Demo"
    items = {item.get("name", item["type"]): item for item in section["items"]}
    assert list(items) == ["steps", "mode", "html", "row", "separator", "colors", "flag", "locked"]
    assert items["steps"] == {"type": "slider", "name": "steps", "label": "Steps", "default": 30, "minimum": 1, "maximum": 150, "step": 1, "integer": True, "notes": ["per pass"], "needs_reload_ui": True}
    assert items["mode"]["style"] == "radio" and items["mode"]["links"] == [{"label": "wiki", "url": "https://example.org"}]
    assert items["html"] == {"type": "html", "text": "Read <b>this</b>"}
    assert [child["name"] for child in items["row"]["children"]] == ["width", "name"]
    assert items["row"]["children"][0]["type"] == "number" and items["row"]["children"][0]["integer"]
    assert items["row"]["children"][1]["hidden"] is True
    assert items["colors"]["style"] == "checkboxes" and items["colors"]["refresh"] is True
    assert items["flag"]["onchange"] is True and items["flag"]["needs_restart"] is True
    # Notes are plain text; a note with markup moves to the trusted html help
    assert items["flag"]["notes"] == ["plain"] and items["flag"]["html"] == "<span class='info'>(see <b>X/Y/Z Plot</b>)</span>"
    assert items["locked"]["enabled_if"] is False and items["locked"]["restricted"] is True
    assert schema["quicksettings"] == ["steps"]
    assert schema["values"]["steps"] == 30 and schema["values"]["secret"] == "s" and "hint" not in schema["values"]


def test_settings_widgets():
    from modules import options as options_module
    from remi.ui.settings import setting_node

    node = setting_node("x", options_module.OptionInfo(["a"], "X", ui.MultiChoice, {"choices": ["a", "b"]}), ["a"])
    assert isinstance(node, ui.MultiChoice)
    node = setting_node("y", options_module.OptionInfo("a", "Y", ui.Choice, {"choices": ["a", "b"], "style": "radio"}), "a")
    assert isinstance(node, ui.Choice) and node.style == "radio"
    node = setting_node("z", options_module.OptionInfo(0.5, "Z", ui.Slider, {"minimum": 0, "maximum": 1, "step": 0.1}), 0.5)
    assert isinstance(node, ui.Slider) and not node.integer
    # A component that is not a remi.ui widget (an A1111 Gradio class) is picked from the default's type
    node = setting_node("w", options_module.OptionInfo(True, "W", object), True)
    assert isinstance(node, ui.Toggle)
    # An int setting stays an int slider though its stored value came back as a float
    node = setting_node("n", options_module.OptionInfo(4, "N", ui.Slider, {"minimum": 0, "maximum": 8, "step": 1}), 4.0)
    assert node.integer and node.resolve_default(ui.Context()) == 4
