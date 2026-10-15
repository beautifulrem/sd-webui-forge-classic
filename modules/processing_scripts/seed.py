
from modules import scripts
from remi import ui


def _subseed_shown(params: dict) -> bool:
    return "Variation seed" in params or "Seed resize from-1" in params


class ScriptSeed(scripts.ScriptBuiltinUI):
    section = "seed"

    def title(self):
        return "Seed"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        shown = ui.param("subseed_show")
        return ui.Group(
            ui.Row(
                ui.Seed("seed", -1, label="Seed", infotext="Seed"),
                ui.Toggle("subseed_show", False, label="Extra", paste=_subseed_shown, width="auto"),
            ),
            ui.Group(
                ui.Row(
                    ui.Seed("subseed", -1, label="Variation seed", infotext="Variation seed"),
                    ui.Slider("subseed_strength", 0, 1, 0.0, 0.01, label="Variation strength", infotext="Variation seed strength"),
                ),
                ui.Row(
                    ui.Slider("seed_resize_from_w", 0, 2048, 0, 8, label="Resize seed from width", infotext="Seed resize from-1"),
                    ui.Slider("seed_resize_from_h", 0, 2048, 0, 8, label="Resize seed from height", infotext="Seed resize from-2"),
                ),
                visible_if=shown,
            ),
        )

    def setup(self, p, seed, seed_checkbox, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h):
        p.seed = seed

        if seed_checkbox and subseed_strength > 0:
            p.subseed = subseed
            p.subseed_strength = subseed_strength

        if seed_checkbox and seed_resize_from_w > 0 and seed_resize_from_h > 0:
            p.seed_resize_from_w = seed_resize_from_w
            p.seed_resize_from_h = seed_resize_from_h
