"""One-click, GUI-visible baseline for Remi's built-in Anima stack."""

from __future__ import annotations

import gradio as gr

from modules import scripts
from modules.anima_presets import (
    PRESET_CONFLICT_CONTROLS,
    PRESET_DESCRIPTIONS,
    PRESETS,
    SAFE_BASE_AESTHETIC_LABEL,
    preset_controls,
    preset_value,
    register_preset_control,
    reset_preset_controls,
)


class AnimaRemiPresetScript(scripts.Script):
    # Default section, after every other built-in extension has registered its
    # controls. Torch Compile currently uses 99999.
    sorting_priority = 100000
    create_group = False

    def title(self):
        return "Anima Remi Presets"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        elem_id = getattr(component, "elem_id", None) or kwargs.get("elem_id")
        tab = self.tabname

        # The prompt is one of the first per-tab components. Seeing it marks a
        # fresh UI build and prevents references from an earlier reload leaking
        # into the Apply event.
        if elem_id == f"{tab}_prompt":
            reset_preset_controls(tab)

        native_ids = {
            f"{tab}_sampling": "native.sampler",
            f"{tab}_scheduler": "native.scheduler",
            f"{tab}_steps": "native.steps",
            f"{tab}_cfg_scale": "native.cfg",
            f"{tab}_distilled_cfg_scale": "native.distilled_cfg",
            f"{tab}_enable-checkbox": "native.refiner",
        }
        if tab == "txt2img":
            native_ids["txt2img_hr-checkbox"] = "native.hires"

        name = native_ids.get(elem_id)
        if name is not None:
            register_preset_control(tab, name, component)

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        names, components, values = preset_controls(tab)

        with gr.Accordion(self.title(), open=False):
            preset = gr.Dropdown(
                choices=list(PRESETS),
                value=SAFE_BASE_AESTHETIC_LABEL,
                label="Preset",
            )
            description = gr.Markdown(PRESET_DESCRIPTIONS[SAFE_BASE_AESTHETIC_LABEL])
            gr.Markdown(
                "Optional guidance, cache, spatial, Hires, Refiner and "
                "post-processing stacks start disabled. Applying a preset "
                "preserves prompt text, model paths and LoRA/artist names."
            )
            apply = gr.Button("Apply preset", variant="primary")
            status = gr.Markdown("")

        preset.do_not_save_to_config = True
        description.do_not_save_to_config = True
        apply.do_not_save_to_config = True
        status.do_not_save_to_config = True

        preset.change(
            fn=lambda label: gr.update(value=PRESET_DESCRIPTIONS.get(label, "")),
            inputs=[preset],
            outputs=[description],
            queue=False,
            show_progress=False,
        )

        def apply_preset(label):
            label = label if label in PRESETS else SAFE_BASE_AESTHETIC_LABEL
            updates = []
            for name, value in zip(names, values):
                value = preset_value(label, name, value)
                if name in PRESET_CONFLICT_CONTROLS:
                    updates.append(gr.update(value=value, interactive=True))
                else:
                    updates.append(gr.update(value=value))
            return [
                *updates,
                gr.update(value=f"**Applied:** {label} ({len(names)} controls)"),
            ]

        apply.click(
            fn=apply_preset,
            inputs=[preset],
            outputs=[*components, status],
            queue=False,
            show_progress=False,
        )

        # This panel configures other scripts and does not add processing args.
        return []
