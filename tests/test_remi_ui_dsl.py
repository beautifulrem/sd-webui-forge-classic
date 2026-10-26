"""remi.ui: nodes, conditions, coercion, ScriptLayout binding and actions."""

import math

import pytest

from remi import ui
from remi.ui.nodes import SEED_MAX


def _panel():
    return ui.Section(
        "Panel",
        enable=ui.Toggle("enable", False, infotext="Panel"),
        children=[
            ui.Row(
                ui.Slider("strength", 0.0, 2.0, 0.5, 0.05, infotext="Panel strength", preset="dave.strength"),
                ui.Slider("steps", 1, 50, 10, 1),
            ),
            ui.Group("Advanced", ui.Choice("mode", ["a", "b", ("Cee", "c")], "b", type="index"), ui.Text("blocks", "0-3"), open=False),
            ui.Tabs(ui.Tab("A", ui.Number("weight", 1.5, minimum=0, maximum=10), value="a"), ui.Tab("B", value="b"), bind="kind"),
            ui.Choice("kind", ["a", "b"], "a", hidden=True),
            ui.MultiChoice("tags", lambda: ["x", "y", "z"], ["x"]),
        ],
    )


class TestNodes:
    def test_declaration_order_is_depth_first_with_the_enable_toggle_first(self):
        layout = ui.ScriptLayout(_panel(), owner="panel")
        assert layout.names == ["enable", "strength", "steps", "mode", "blocks", "weight", "kind", "tags"]

    def test_explicit_order_must_list_every_param_once(self):
        root = ui.Row(ui.Toggle("a"), ui.Toggle("b"))
        assert ui.ScriptLayout(root, order=("b", "a")).names == ["b", "a"]
        with pytest.raises(ui.SchemaError):
            ui.ScriptLayout(root, order=("b",))
        with pytest.raises(ui.SchemaError):
            ui.ScriptLayout(root, order=("a", "a", "b"))

    def test_duplicate_names_and_bad_names_are_rejected(self):
        with pytest.raises(ui.SchemaError, match="duplicate"):
            ui.ScriptLayout(ui.Row(ui.Toggle("x"), ui.Group(ui.Slider("x", 0, 1, 0, 1))))
        with pytest.raises(ui.SchemaError):
            ui.Toggle("bad name")
        with pytest.raises(ui.SchemaError):
            ui.Toggle("a.b")

    def test_slider_integer_detection(self):
        assert ui.Slider("s", 0, 150, 20, 1).integer
        assert not ui.Slider("s", 0.0, 1.0, 0.5, 0.05).integer
        assert not ui.Slider("s", 0, 10, 2.5, 1).integer
        # Float literals make a float slider even when their values are integral
        assert not ui.Slider("s", 1.0, 24.0, 6.0, 1.0).integer
        assert not ui.Slider("s", 0.0, 2.0, 1.0, 1.0).integer
        assert not ui.Slider("s", 0, 10, 5, 1, integer=False).integer
        assert ui.Slider("s", 0.0, 10.0, 5.0, 1.0, integer=True).coerce(4.4, ui.Context()) == 4
        # A callable default does not decide
        assert ui.Slider("s", 1, 150, lambda: 20.0, 1).integer

    def test_bad_declarations(self):
        with pytest.raises(ui.SchemaError):
            ui.Slider("s", 2, 1, 1, 1)
        with pytest.raises(ui.SchemaError):
            ui.Slider("s", 0, math.inf, 1, 1)
        with pytest.raises(ui.SchemaError):
            ui.Choice("c", "abc")
        with pytest.raises(ui.SchemaError):
            ui.Choice("c", ["a"], style="checkboxes")
        with pytest.raises(ui.SchemaError):
            ui.Slider("s", 0, 1, 0, 1, exclusive=("group", "member"))
        with pytest.raises(ui.SchemaError):
            ui.Section("S", exclusive=("group", "member"))
        with pytest.raises(ui.SchemaError):
            ui.Tabs(ui.Toggle("t"))
        with pytest.raises(ui.SchemaError, match="unknown param"):
            ui.ScriptLayout(ui.Toggle("t", visible_if=ui.param("missing")))
        with pytest.raises(ui.SchemaError, match="not a Choice"):
            ui.ScriptLayout(ui.Group(ui.Toggle("t"), ui.Tabs(ui.Tab("A", value="a"), bind="t")))
        with pytest.raises(ui.SchemaError, match="not choices"):
            ui.ScriptLayout(ui.Group(ui.Choice("c", ["a"]), ui.Tabs(ui.Tab("B", value="b"), bind="c")))

    def test_section_exclusive_lands_on_the_enable_toggle(self):
        section = ui.Section("S", enable=ui.Toggle("on"), exclusive=("spatial", "regional"))
        layout = ui.ScriptLayout(section, owner="regional")
        assert section.enable.exclusive == ("spatial", "regional")
        assert layout.exclusive == {"spatial": {"regional": "on"}}

    def test_children_by_keyword_positional_and_none_entries(self):
        group = ui.Group("T", ui.Toggle("a"), None, [ui.Toggle("b"), None], children=[ui.Toggle("c")])
        assert [node.name for node in group.items] == ["a", "b", "c"]
        assert ui.Group(ui.Toggle("a")).title is None


class TestConditions:
    def test_grammar(self):
        assert ui.param("mode", in_=["a"]) == {"param": "mode", "in": ["a"]}
        assert ui.all_of(ui.param("a"), ui.not_(ui.param("b", eq=1))) == {"all": [{"param": "a"}, {"not": {"param": "b", "eq": 1}}]}
        with pytest.raises(ui.SchemaError):
            ui.conditions.check({"param": "a", "eq": 1, "ne": 2})
        with pytest.raises(ui.SchemaError):
            ui.conditions.check({"param": "a", "in": "abc"})
        with pytest.raises(ui.SchemaError):
            ui.conditions.check({"bogus": 1})
        # Ordering compares with finite numbers only (the client compares numbers)
        for operand in ("1", True, None, math.nan, [1]):
            with pytest.raises(ui.SchemaError):
                ui.param("cfg", gt=operand)
        assert ui.param("cfg", lte=1.5) == {"param": "cfg", "lte": 1.5}

    def test_evaluation_matches_the_client_rules(self):
        values = {"mode": "b", "cfg": 4.5, "tags": [], "on": True, "core.width": 1024}
        assert ui.evaluate(ui.param("mode", in_=["a", "b"]), values)
        assert ui.evaluate(ui.param("cfg", gt=1), values)
        assert not ui.evaluate(ui.param("tags"), values)
        assert ui.evaluate(ui.all_of(ui.param("on"), ui.param("core.width", gte=1024)), values)
        # A missing param is falsy, and "ne" is true for it
        assert not ui.evaluate(ui.param("missing"), values)
        assert not ui.evaluate(ui.param("missing", eq=None), values)
        assert ui.evaluate(ui.param("missing", ne=1), values)
        assert not ui.evaluate(ui.param("mode", gt=1), values)  # incomparable
        # eq/ne/in are Python == on JSON values, as the client's pyEqual: True == 1, "1" != 1
        assert ui.evaluate(ui.param("on", eq=1), values) and not ui.evaluate(ui.param("cfg", eq="4.5"), values)
        assert ui.evaluate(None, values) and not ui.evaluate(False, values)


class TestValues:
    def setup_method(self):
        self.layout = ui.ScriptLayout(_panel(), owner="panel")

    def test_defaults_resolve_callables_and_normalize(self):
        assert self.layout.defaults() == {"enable": False, "strength": 0.5, "steps": 10, "mode": "b", "blocks": "0-3", "weight": 1.5, "kind": "a", "tags": ["x"]}

    def test_values_to_args_fills_defaults_and_converts_index_choices(self):
        args = self.layout.values_to_args({"enable": "true", "steps": "12", "mode": "c"})
        assert args == [True, 0.5, 12, 2, "0-3", 1.5, "a", ["x"]]

    def test_coercion_clamps_rounds_and_matches_choices(self):
        values = self.layout.coerce({"strength": 9, "steps": 7.6, "weight": "2", "tags": ["z", "x"], "kind": "b"})
        assert values["strength"] == 2.0 and isinstance(values["strength"], float)
        assert values["steps"] == 8 and isinstance(values["steps"], int)
        assert values["weight"] == 2 and values["tags"] == ["z", "x"] and values["kind"] == "b"

    def test_every_bad_value_is_reported_with_its_param(self):
        with pytest.raises(ui.ValidationError) as caught:
            self.layout.coerce({"strength": float("nan"), "steps": True, "mode": "d", "weight": 11, "tags": "x", "bogus": 1, "enable": "maybe"})
        errors = {error["param"]: error["message"] for error in caught.value.to_json()["errors"]}
        assert set(errors) == {"strength", "steps", "mode", "weight", "tags", "bogus", "enable"}
        assert "finite" in errors["strength"] and "at most" in errors["weight"] and "unknown" in errors["bogus"]
        assert all(error["owner"] == "panel" for error in caught.value.to_json()["errors"])
        assert self.layout.coerce({"bogus": 1}, unknown="ignore")["steps"] == 10

    def test_infinity_strings_and_non_numbers_are_rejected(self):
        for bad in ("inf", "-Infinity", "nan", "abc", [], None):
            with pytest.raises(ui.ParamError):
                ui.Slider("s", 0, 1, 0, 1).coerce(bad, ui.Context())

    def test_choices_accept_numeric_strings_and_custom_values(self):
        ctx = ui.Context()
        assert ui.Choice("step", [8, 16, 64], 64).coerce("16", ctx) == 16
        assert ui.Choice("name", ["a"], allow_custom=True).coerce("zzz", ctx) == "zzz"
        with pytest.raises(ui.ParamError):
            ui.Choice("name", ["a"]).coerce("zzz", ctx)
        assert ui.Choice("empty", []).coerce(None, ctx) is None
        assert ui.Choice("stale", ["a", "b"], "gone").resolve_default(ctx) == "a"

    def test_same_key_maps_a_model_title_onto_its_current_choice(self):
        # A checkpoint title gains " [hash]" once the hash is known: a form
        # that still holds the title from before (or a pasted hashed title)
        # names the same model.
        ctx = ui.Context()
        hashed = ui.Choice("ckpt", ["anima [fba1195327]", "other [0123abcd]"], same=ui.without_model_hash)
        assert hashed.coerce("anima", ctx) == "anima [fba1195327]"
        assert ui.Choice("ckpt", ["anima", "other"], same=ui.without_model_hash).coerce("anima [fba1195327]", ctx) == "anima"
        assert ui.MultiChoice("ckpts", ["anima [fba1195327]"], same=ui.without_model_hash).coerce(["anima"], ctx) == ["anima [fba1195327]"]
        with pytest.raises(ui.ParamError):
            hashed.coerce("missing", ctx)
        # ambiguous: two choices with the same key
        with pytest.raises(ui.ParamError):
            ui.Choice("ckpt", ["m [0123abcd]", "m [89abcdef]"], same=ui.without_model_hash).coerce("m", ctx)
        # without the key the stale title is rejected, as before
        with pytest.raises(ui.ParamError):
            ui.Choice("ckpt", ["anima [fba1195327]"]).coerce("anima", ctx)
        assert ui.without_model_hash("a [b]") == "a [b]" and ui.without_model_hash(3) == 3
        with pytest.raises(ui.SchemaError):
            ui.Choice("ckpt", ["a"], same="key")

    def test_text_seed_color(self):
        ctx = ui.Context()
        assert ui.Text("t").coerce(None, ctx) == "" and ui.Text("t").coerce(12, ctx) == "12"
        assert ui.Seed("s").coerce("123", ctx) == 123 and ui.Seed("s").coerce(5.0, ctx) == 5 and ui.Seed("s").coerce(-1, ctx) == -1
        assert ui.Seed("s").coerce(str(SEED_MAX), ctx) == SEED_MAX
        for bad in (1.7, "2.5", -2, SEED_MAX + 1, True, None):
            with pytest.raises(ui.ParamError):
                ui.Seed("s").coerce(bad, ctx)
        assert ui.Color("c").coerce("#a0B", ctx) == "#a0B"
        with pytest.raises(ui.ParamError):
            ui.Color("c").coerce("red", ctx)

    def test_toggles_take_booleans_0_1_and_true_false_only(self):
        ctx = ui.Context()
        toggle = ui.Toggle("t")
        assert [toggle.coerce(value, ctx) for value in (True, 0, 1, "TRUE", " false ", None)] == [True, False, True, True, False, False]
        for bad in ("yes", "no", "on", "off", "", 2, 1.0, []):
            with pytest.raises(ui.ParamError):
                toggle.coerce(bad, ctx)
        # Infotext paste stays lenient, as Gradio's bool(text) was
        assert toggle.from_paste("yes", ctx) is True and toggle.from_paste("False", ctx) is False

    def test_json_values_reject_nan_and_infinity_anywhere(self):
        ctx = ui.Context()
        for node in (ui.Json("j"), ui.Custom("c", "widget"), ui.State("s")):
            assert node.coerce({"a": [1, 2.5, None]}, ctx) == {"a": [1, 2.5, None]}
            for bad in (math.nan, {"a": [math.inf]}, [{"b": -math.inf}]):
                with pytest.raises(ui.ParamError, match="finite"):
                    node.coerce(bad, ctx)

    def test_multichoice_index_type(self):
        ctx = ui.Context()
        layout = ui.ScriptLayout(ui.MultiChoice("parts", ["Prompt", "Seed", "Steps"], ["Seed"], type="index"))
        assert layout.values_to_args({}) == [[1]] and layout.values_to_args({"parts": ["Steps", "Prompt"]}) == [[2, 0]]
        assert layout.args_to_values([[0, 2]]) == {"parts": ["Prompt", "Steps"]}
        with pytest.raises(ui.SchemaError):
            ui.MultiChoice("parts", ["a"], type="index", allow_custom=True)
        assert ui.MultiChoice("parts", ["a"]).to_arg(("a",), ctx) == ["a"]

    def test_request_bodies_must_be_objects_of_named_values(self):
        for bad in ("abc", [1, 2], 5, {1: "x"}):
            with pytest.raises(ui.ValidationError) as caught:
                self.layout.values_to_args(bad)
            assert caught.value.to_json()["errors"][0]["owner"] == "panel"
        assert self.layout.coerce(None) == self.layout.defaults()

    def test_uploads_resolve_through_the_context(self):
        resolved = []

        def resolve(node, upload_id):
            resolved.append((node.name, upload_id))
            return f"<{upload_id}>"

        layout = ui.ScriptLayout(ui.Group(ui.Image("img"), ui.ImageList("batch"), ui.File("model")))
        args = layout.values_to_args({"img": "u1", "batch": ["u2", "u3"]}, ui.Context(resolve_upload=resolve))
        assert args == ["<u1>", ["<u2>", "<u3>"], None]
        assert resolved == [("img", "u1"), ("batch", "u2"), ("batch", "u3")]
        with pytest.raises(ui.ValidationError):
            layout.values_to_args({"img": 5})

    def test_image_mode_conversion(self):
        from PIL import Image

        image = ui.Image("img", mode="RGBA").to_arg(Image.new("RGB", (4, 4)), ui.Context())
        assert image.mode == "RGBA"

    def test_masks_are_alpha_coded_with_a_luminance_fallback(self):
        from PIL import Image

        from remi.ui.nodes import mask_channel

        painted = Image.new("RGBA", (4, 1), (0, 0, 0, 0))
        painted.putpixel((0, 0), (0, 0, 0, 255))  # the colour does not matter, the alpha does
        assert list(mask_channel(painted).getdata()) == [255, 0, 0, 0]
        plain = Image.new("L", (4, 1), 0)
        plain.putpixel((1, 0), 255)
        # An L mask, also once converted to opaque RGBA (Image(mode="RGBA")), reads by luminance
        assert list(mask_channel(plain).getdata()) == [0, 255, 0, 0]
        assert list(mask_channel(plain.convert("RGBA")).getdata()) == [0, 255, 0, 0]
        assert mask_channel(None) is None

    def test_args_round_trip(self):
        args = self.layout.values_to_args({"mode": "a", "tags": ["y"]})
        assert self.layout.args_to_values(args) == {**self.layout.defaults(), "mode": "a", "tags": ["y"]}
        assert self.layout.args_to_values(args[:2])["tags"] == ["x"]
        with pytest.raises(ValueError):
            self.layout.args_to_values([*args, 1])

    def test_paste_reads_keys_and_functions_and_skips_bad_values(self):
        layout = ui.ScriptLayout(
            ui.Group(
                ui.Toggle("on", paste=lambda d: "Panel" in d),
                ui.Slider("strength", 0.0, 2.0, 0.5, 0.05, infotext="Panel strength"),
                ui.Choice("mode", ["a", "b"], infotext="Panel mode"),
                ui.Toggle("flag", infotext="Panel flag"),
                ui.MultiChoice("tags", ["x", "y"], infotext="Panel tags"),
            ),
            owner="panel",
        )
        values, problems = layout.paste({"Panel": "1", "Panel strength": "0.75", "Panel mode": "zzz", "Panel flag": "False", "Panel tags": "['y']"})
        assert values == {"on": True, "strength": 0.75, "flag": False, "tags": ["y"]}
        assert [problem.param for problem in problems] == ["mode"]

    def test_preset_values(self):
        layout = ui.ScriptLayout(ui.Group(ui.Toggle("on", preset="dave.enabled"), ui.Choice("opt", ["平衡", "Quality"], preset=("artist.optimization", "平衡"))))
        assert layout.presets == {"dave.enabled": ("on", layout.presets["dave.enabled"][1]), "artist.optimization": ("opt", "平衡")}
        assert layout.preset_values({"dave.enabled": False, "artist.optimization": "Balance"}) == {"on": False, "opt": "平衡"}


class TestActions:
    def test_actions_are_collected_called_and_checked(self):
        class Owner:
            @ui.action("double")
            def double(self, ctx, values):
                return {"x": ui.update(value=values["x"] * 2, choices=["a", ("B", "b")]), **ui.toast("done")}

            @ui.action(gpu=True)
            def heavy(self, ctx, values):
                return {"x": {"bogus": 1}}

        specs = ui.actions_of(Owner())
        assert set(specs) == {"double", "heavy"} and specs["heavy"].gpu
        patch = ui.call_action(Owner(), "double", ui.ActionContext(tab="txt2img"), {"x": 2})
        assert patch == {"x": {"value": 4, "choices": [{"value": "a", "label": "a"}, {"value": "b", "label": "B"}]}, "__toast__": {"level": "info", "message": "done"}}
        with pytest.raises(ui.SchemaError):
            ui.call_action(Owner(), "heavy", ui.ActionContext(), {})
        with pytest.raises(KeyError):
            ui.call_action(Owner(), "missing", ui.ActionContext(), {})

    def test_merge_patches(self):
        merged = ui.merge_patches({"a": {"value": 1}, "__toast__": {"level": "info", "message": "x"}}, {"a": {"enabled": False}, "__toast__": {"level": "warning", "message": "y"}})
        assert merged == {"a": {"value": 1, "enabled": False}, "__toast__": {"level": "warning", "message": "x; y"}}


def test_refresh_patch_names_rescanned_lists():
    """``__refresh__`` lets an action announce rescanned server lists without
    reaching the server's services (the action route publishes them)."""

    from remi.ui.actions import check_patch
    from remi.ui.errors import SchemaError

    assert check_patch({"__refresh__": {"keys": ["upscalers"]}}) == {"__refresh__": {"keys": ["upscalers"]}}
    for bad in ({"keys": "upscalers"}, {"keys": [""]}, {"keys": [3]}, {}):
        with pytest.raises(SchemaError):
            check_patch({"__refresh__": bad})
