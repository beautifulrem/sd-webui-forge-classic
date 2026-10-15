from modules import scripts, sd_samplers, sd_schedulers
from remi import ui
from remi.core.presets import preset_default


def sampler_names() -> list[str]:
    return [x.name for x in sd_samplers.visible_samplers()]


def scheduler_names() -> list[str]:
    return [x.label for x in sd_schedulers.schedulers]


class ScriptSampler(scripts.ScriptBuiltinUI):
    section = "sampler"
    # Shown sampler, scheduler, steps; processed (and sent by the API) steps first
    param_order = ("steps", "sampler_name", "scheduler")

    def __init__(self):
        self.steps = None
        self.sampler_name = None
        self.scheduler = None

    def title(self):
        return "Sampler"

    def params(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        def from_preset(name, fallback):
            # Defaults follow the UI preset, as its page-load update did
            return lambda: preset_default(tab, f"sampler.{name}", fallback)

        return ui.Row(
            ui.Choice("sampler_name", sampler_names, from_preset("sampler_name", None), label="Sampling Method", paste=sd_samplers.get_sampler_from_infotext, preset="native.sampler"),
            ui.Choice("scheduler", scheduler_names, from_preset("scheduler", None), label="Schedule Type", paste=sd_samplers.get_scheduler_from_infotext, preset="native.scheduler"),
            ui.Slider("steps", 1, 150, from_preset("steps", 20), 1, label="Sampling Steps", infotext="Steps", preset="native.steps"),
        )

    def ui(self, is_img2img):
        import gradio as gr

        from modules.infotext_utils import PasteField
        from modules.ui import no_config
        from modules.ui_components import FormRow

        sampler_names = [x.name for x in sd_samplers.visible_samplers()]
        scheduler_names = [x.label for x in sd_schedulers.schedulers]

        with FormRow(elem_id=f"sampler_selection_{self.tabname}"):
            self.sampler_name = gr.Dropdown(label="Sampling Method", elem_id=f"{self.tabname}_sampling", choices=sampler_names, value=sampler_names[0])
            self.scheduler = gr.Dropdown(label="Schedule Type", elem_id=f"{self.tabname}_scheduler", choices=scheduler_names, value=scheduler_names[0])
            self.steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{self.tabname}_steps", label="Sampling Steps", value=20)

        self.infotext_fields = [
            PasteField(self.steps, "Steps", api="steps"),
            PasteField(self.sampler_name, sd_samplers.get_sampler_from_infotext, api="sampler_name"),
            PasteField(self.scheduler, sd_samplers.get_scheduler_from_infotext, api="scheduler"),
        ]

        no_config(self.sampler_name, self.scheduler, self.steps)
        return self.steps, self.sampler_name, self.scheduler

    def setup(self, p, steps, sampler_name, scheduler):
        p.steps = steps
        p.sampler_name = sampler_name
        p.scheduler = scheduler
