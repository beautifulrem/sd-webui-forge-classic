from modules import scripts, sd_samplers, sd_schedulers
from remi import ui
from remi.core import hints
from remi.core.presets import preset_default


def sampler_names() -> list[str]:
    return [x.name for x in sd_samplers.visible_samplers()]


def scheduler_names() -> list[str]:
    return [x.label for x in sd_schedulers.schedulers]


class ScriptSampler(scripts.ScriptBuiltinUI):
    section = "sampler"
    # Shown sampler, scheduler, steps; processed (and sent by the API) steps first
    param_order = ("steps", "sampler_name", "scheduler")

    def title(self):
        return "Sampler"

    def params(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        def from_preset(name, fallback):
            # Defaults follow the UI preset, as its page-load update did
            return lambda: preset_default(tab, f"sampler.{name}", fallback)

        return ui.Row(
            ui.Choice("sampler_name", sampler_names, from_preset("sampler_name", None), label="Sampling Method", hint=hints.SAMPLER, paste=sd_samplers.get_sampler_from_infotext, preset="native.sampler"),
            ui.Choice("scheduler", scheduler_names, from_preset("scheduler", None), label="Schedule Type", hint=hints.SCHEDULER, paste=sd_samplers.get_scheduler_from_infotext, preset="native.scheduler"),
            ui.Slider("steps", 1, 150, from_preset("steps", 20), 1, label="Sampling Steps", hint=hints.STEPS, infotext="Steps", preset="native.steps"),
        )

    def setup(self, p, steps, sampler_name, scheduler):
        p.steps = steps
        p.sampler_name = sampler_name
        p.scheduler = scheduler
