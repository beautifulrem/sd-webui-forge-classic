import gradio as gr

from modules import script_callbacks, scripts
from modules.anima_lora_support import restore_batch_prompts, restore_run_prompts
from modules.anima_lora_ui import intro_block, register_language_option
from modules.anima_presets import register_preset_control
from modules.infotext_utils import PasteField
from lib_stage_scheduler.options import (
    BUILDER_PRESET_CHOICES,
    BUILDER_PRESET_LABELS,
    BUILTIN_TEMPLATES,
    CURVE_CHOICES,
    CURVE_SMOOTH,
    DISABLE_NONE,
    INFOTEXT_KEYS,
    INTRO_EN,
    INTRO_ZH,
    LANGUAGE_OPTION,
    MODE_AUTO_ANIMA,
    PRESET_CHARACTER_ALLOW,
    SCRIPT_ARGS,
    STAGE_DEFAULT_TIMINGS,
    _arguments,
    _disable_choices,
    _display_disable_mode,
    _display_target_mode,
    _t,
    _target_mode_choices,
)
from lib_stage_scheduler.runtime import (
    _cleanup_runtime,
    _denoiser_callback,
    _note_text_encoder_signature,
    _rewrite_batch_prompts,
    _translate_pasted_infotext,
    install_patch,
    start_run,
)
from lib_stage_scheduler.templates import (
    _apply_template_ui,
    _auto_timing_ui,
    _close_template_modal_ui,
    _delete_template_ui,
    _open_template_builder_ui,
    _open_template_help_ui,
    _preset_combo_preview_text,
    _reapplying_changed_templates,
    _rename_template_ui,
    _save_preset_combo_template_ui,
    _save_template_ui,
    _template_names,
    _template_stage_from_name,
)


def _register_ui_settings():
    register_language_option(
        LANGUAGE_OPTION,
        "Anima LoRA Stage Scheduler language / Anima LoRA 阶段介入语言",
        ("anima-lora-stage-scheduler", "Anima LoRA Stage Scheduler"),
    )


script_callbacks.on_ui_settings(_register_ui_settings)


script_callbacks.on_infotext_pasted(_translate_pasted_infotext, name="anima_lora_stage_scheduler")


# Registered from this script: a Reload UI clears the callbacks and re-runs it
_CFG_CALLBACK_REGISTERED = False


def _register_cfg_callback_once():
    global _CFG_CALLBACK_REGISTERED
    if _CFG_CALLBACK_REGISTERED:
        return
    script_callbacks.on_cfg_denoiser(_denoiser_callback, name="anima_lora_stage_scheduler")
    _CFG_CALLBACK_REGISTERED = True


class Script(scripts.Script):
    section = None

    def __init__(self):
        _register_cfg_callback_once()
        self.base_shift_component = None
        self.hires_shift_component = None

    def title(self):
        return _t("title")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        elem_id = kwargs.get("elem_id")
        if self.is_txt2img:
            if elem_id == "txt2img_distilled_cfg_scale":
                self.base_shift_component = component
            elif elem_id == "txt2img_hr_distilled_cfg":
                self.hires_shift_component = component
        elif self.is_img2img and elem_id == "img2img_distilled_cfg_scale":
            self.base_shift_component = component

    def _stage_panel(self, prefix: str, label: str, default_open: bool, default_target_mode=MODE_AUTO_ANIMA):
        choices = _template_names()
        default_template = choices[0] if choices else "仅人物特征阶段介入"
        # The sliders start on the default template's window (what unticking auto timing keeps)
        default_start, default_end, default_peak = STAGE_DEFAULT_TIMINGS[_template_stage_from_name(default_template)]

        with gr.Accordion(label, open=default_open, elem_id=self.elem_id(f"{prefix}_panel")):
            with gr.Row():
                target_mode = gr.Radio(_target_mode_choices(), value=_display_target_mode(default_target_mode), label=_t("target_scope"), elem_id=self.elem_id(f"{prefix}_target_mode"))
                panel_weight = gr.Textbox(
                    label=_t("panel_weight"),
                    value="0",
                    lines=1,
                    placeholder=_t("panel_weight_placeholder"),
                    elem_id=self.elem_id(f"{prefix}_panel_weight"),
                )
            target_loras = gr.Textbox(label=_t("target_lora"), value="", lines=1, placeholder=_t("target_lora_placeholder"), elem_id=self.elem_id(f"{prefix}_target_loras"))

            with gr.Row():
                template = gr.Dropdown(label=_t("template"), choices=choices, value=default_template, elem_id=self.elem_id(f"{prefix}_template"))
                copy_template = gr.Button(_t("copy"), elem_id=self.elem_id(f"{prefix}_copy_template"))
                help_template = gr.Button("?", elem_id=self.elem_id(f"{prefix}_template_help"))
                open_builder = gr.Button(_t("builder"), elem_id=self.elem_id(f"{prefix}_open_template_builder"))
            auto_timing = gr.Checkbox(label=_t("auto_timing"), value=True, elem_id=self.elem_id(f"{prefix}_auto_timing"))
            lora_templates = gr.Textbox(
                label=_t("single_lora_template"),
                value="",
                lines=3,
                placeholder=_t("single_lora_template_placeholder"),
                elem_id=self.elem_id(f"{prefix}_lora_templates"),
            )

            with gr.Row():
                start = gr.Slider(0.0, 1.0, value=default_start, step=0.01, label=_t("start"), elem_id=self.elem_id(f"{prefix}_start"))
                end = gr.Slider(0.0, 1.0, value=default_end, step=0.01, label=_t("end"), elem_id=self.elem_id(f"{prefix}_end"))
                peak = gr.Slider(0.0, 1.0, value=default_peak, step=0.01, label=_t("peak"), elem_id=self.elem_id(f"{prefix}_peak"))

            with gr.Row():
                stage_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label=_t("strength"), elem_id=self.elem_id(f"{prefix}_stage_strength"))
                curve = gr.Dropdown(label=_t("curve"), choices=CURVE_CHOICES, value=CURVE_SMOOTH, elem_id=self.elem_id(f"{prefix}_curve"))

        return target_mode, target_loras, panel_weight, template, lora_templates, auto_timing, start, end, peak, stage_strength, curve, copy_template, help_template, open_builder

    def ui(self, is_img2img):
        template_css = f"""
        <style>
        #{self.elem_id("template_modal")} {{
            position: fixed;
            inset: 0;
            z-index: 5000;
            padding: 24px;
            overflow: auto;
            box-sizing: border-box;
            background: rgba(0, 0, 0, 0.92);
        }}
        #{self.elem_id("template_modal")}.hide {{
            display: none !important;
        }}
        #{self.elem_id("template_modal")}:not(.hide) {{
            display: flex !important;
            align-items: center;
            justify-content: center;
        }}
        #{self.elem_id("template_modal_box")} {{
            width: min(980px, calc(100vw - 48px));
            max-height: calc(100vh - 48px);
            margin: 0 auto;
            background: #111827 !important;
            border: 1px solid #475569;
            border-radius: 8px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.65);
            padding: 16px;
            overflow: auto;
            box-sizing: border-box;
        }}
        #{self.elem_id("template_modal_box")} > .form,
        #{self.elem_id("template_help_modal")},
        #{self.elem_id("template_builder_modal")} {{
            background: #111827 !important;
        }}
        </style>
        """
        with gr.Accordion(_t("title"), open=False, elem_id=self.elem_id("accordion")):
            intro_block(self.elem_id, INTRO_EN, INTRO_ZH, LANGUAGE_OPTION)
            enabled = gr.Checkbox(label=_t("enable"), value=False, elem_id=self.elem_id("enabled"))
            gr.HTML(template_css)

            base_controls = self._stage_panel("base", _t("base_panel"), True)

            with gr.Accordion(_t("hires_panel"), open=False, elem_id=self.elem_id("hires_panel")):
                hr_independent = gr.Checkbox(label=_t("hr_independent"), value=False, elem_id=self.elem_id("hr_independent"))
                hires_controls = self._stage_panel("hires", _t("hires_lora_panel"), True)
                with gr.Row():
                    hr_disable_mode = gr.Radio(_disable_choices(), value=_display_disable_mode(DISABLE_NONE), label=_t("hr_disable"), elem_id=self.elem_id("hr_disable_mode"))
                hr_disable_tags = gr.Textbox(
                    label=_t("hr_disable_tags"),
                    value="",
                    lines=3,
                    placeholder="<lora:name:0.8>\n<lora:name:blocks=0-9=0.8,10-18=1.0,19-27=0.7>",
                    elem_id=self.elem_id("hr_disable_tags"),
                )

            template_choices = _template_names()
            manage_default = template_choices[0] if template_choices else None
            with gr.Accordion(_t("template_manager"), open=False, elem_id=self.elem_id("template_manager")):
                with gr.Row():
                    manage_template = gr.Dropdown(label=_t("existing_template"), choices=template_choices, value=manage_default, elem_id=self.elem_id("manage_template"))
                    rename_template_name = gr.Textbox(label=_t("rename_to"), lines=1, elem_id=self.elem_id("rename_template_name"))
                with gr.Row():
                    rename_template = gr.Button(_t("rename_template"), elem_id=self.elem_id("rename_template"))
                    delete_template = gr.Button(_t("delete_template"), elem_id=self.elem_id("delete_template"))
                with gr.Row():
                    custom_name = gr.Textbox(label=_t("new_template_name"), lines=1, elem_id=self.elem_id("custom_name"))
                    custom_auto = gr.Checkbox(label=_t("save_auto_template"), value=True, elem_id=self.elem_id("custom_auto"))
                with gr.Row():
                    save_current_template = gr.Button(_t("save_current_base"), elem_id=self.elem_id("save_template"))
                template_status = gr.Markdown(value="", elem_id=self.elem_id("template_status"))

            with gr.Group(visible=False, elem_id=self.elem_id("template_modal")) as template_modal:
                with gr.Group(elem_id=self.elem_id("template_modal_box")):
                    with gr.Group(visible=False, elem_id=self.elem_id("template_help_modal")) as template_help_modal:
                        with gr.Row():
                            gr.Markdown(_t("template_help"))
                            close_help = gr.Button(_t("close"), elem_id=self.elem_id("close_template_help"))
                        template_help = gr.Markdown(value="", elem_id=self.elem_id("template_help_text"))
                    with gr.Group(visible=False, elem_id=self.elem_id("template_builder_modal")) as template_builder:
                        with gr.Row():
                            gr.Markdown(_t("combo_title"))
                            close_builder = gr.Button(_t("close"), elem_id=self.elem_id("close_template_builder"))
                        builder_name = gr.Textbox(label=_t("template_name"), lines=1, elem_id=self.elem_id("builder_name"))
                        builder_presets = gr.CheckboxGroup(
                            label=_t("base_preset_combo"),
                            choices=BUILDER_PRESET_CHOICES,
                            value=[BUILDER_PRESET_LABELS[PRESET_CHARACTER_ALLOW]],
                            elem_id=self.elem_id("builder_presets"),
                        )
                        builder_shift = gr.Number(label=_t("preview_shift"), value=3.0, precision=4, elem_id=self.elem_id("builder_shift"))
                        with gr.Row():
                            save_combo_template = gr.Button(_t("save_combo"), variant="primary", elem_id=self.elem_id("save_combo_template"))
                        builder_status = gr.Markdown(value="", elem_id=self.elem_id("builder_status"))

        (
            base_target_mode,
            base_target_loras,
            base_panel_weight,
            base_template,
            base_lora_templates,
            base_auto,
            base_start,
            base_end,
            base_peak,
            base_stage_strength,
            base_curve,
            base_copy_template,
            base_help_template,
            base_open_builder,
        ) = base_controls
        (
            hr_target_mode,
            hr_target_loras,
            hr_panel_weight,
            hr_template,
            hr_lora_templates,
            hr_auto,
            hr_start,
            hr_end,
            hr_peak,
            hr_stage_strength,
            hr_curve,
            hr_copy_template,
            hr_help_template,
            hr_open_builder,
        ) = hires_controls

        base_shift = self.base_shift_component or gr.Number(value=3.0, visible=False)
        hires_shift = self.hires_shift_component or self.base_shift_component or gr.Number(value=3.0, visible=False)

        for template_component, shift_component, auto_component, start_component, end_component, peak_component, strength_component, curve_component, help_button, copy_button in [
            (base_template, base_shift, base_auto, base_start, base_end, base_peak, base_stage_strength, base_curve, base_help_template, base_copy_template),
            (hr_template, hires_shift, hr_auto, hr_start, hr_end, hr_peak, hr_stage_strength, hr_curve, hr_help_template, hr_copy_template),
        ]:
            # User selections only (.input): a pasted template comes with its
            # own timing, which .change would overwrite with the template's.
            template_component.input(
                fn=_apply_template_ui,
                inputs=[template_component, shift_component],
                outputs=[auto_component, start_component, end_component, peak_component, strength_component, curve_component],
                show_progress=False,
            )
            auto_component.change(
                fn=_auto_timing_ui,
                inputs=[template_component, shift_component, auto_component, start_component, end_component, peak_component, strength_component],
                outputs=[start_component, end_component, peak_component, strength_component],
                show_progress=False,
            )
            help_button.click(
                fn=_open_template_help_ui,
                inputs=[template_component, shift_component],
                outputs=[template_modal, template_help_modal, template_builder, template_help],
                show_progress=False,
            )
            copy_button.click(
                fn=None,
                _js="(value) => { if (navigator.clipboard) { navigator.clipboard.writeText(value || ''); } return []; }",
                inputs=[template_component],
                outputs=[],
                show_progress=False,
            )

        base_open_builder.click(
            fn=_open_template_builder_ui,
            inputs=[base_template, base_shift],
            outputs=[template_modal, template_help_modal, template_builder, builder_name, builder_presets, builder_shift, builder_status],
            show_progress=False,
        )
        hr_open_builder.click(
            fn=_open_template_builder_ui,
            inputs=[hr_template, hires_shift],
            outputs=[template_modal, template_help_modal, template_builder, builder_name, builder_presets, builder_shift, builder_status],
            show_progress=False,
        )
        close_builder.click(
            fn=_close_template_modal_ui,
            inputs=[],
            outputs=[template_modal, template_help_modal, template_builder],
            show_progress=False,
        )
        close_help.click(
            fn=_close_template_modal_ui,
            inputs=[],
            outputs=[template_modal, template_help_modal, template_builder],
            show_progress=False,
        )
        builder_presets.change(
            fn=_preset_combo_preview_text,
            inputs=[builder_presets, builder_shift],
            outputs=[builder_status],
            show_progress=False,
        )
        builder_shift.change(
            fn=_preset_combo_preview_text,
            inputs=[builder_presets, builder_shift],
            outputs=[builder_status],
            show_progress=False,
        )
        manage_template.change(
            fn=lambda name: gr.update(value="" if str(name or "") in BUILTIN_TEMPLATES else str(name or "")),
            inputs=[manage_template],
            outputs=[rename_template_name],
            show_progress=False,
        )

        if self.base_shift_component is not None:
            self.base_shift_component.change(
                fn=_auto_timing_ui,
                inputs=[base_template, self.base_shift_component, base_auto, base_start, base_end, base_peak, base_stage_strength],
                outputs=[base_start, base_end, base_peak, base_stage_strength],
                show_progress=False,
            )
            if hasattr(self.base_shift_component, "release"):
                self.base_shift_component.release(
                    fn=_auto_timing_ui,
                    inputs=[base_template, self.base_shift_component, base_auto, base_start, base_end, base_peak, base_stage_strength],
                    outputs=[base_start, base_end, base_peak, base_stage_strength],
                    show_progress=False,
                )

        if self.hires_shift_component is not None:
            self.hires_shift_component.change(
                fn=_auto_timing_ui,
                inputs=[hr_template, self.hires_shift_component, hr_auto, hr_start, hr_end, hr_peak, hr_stage_strength],
                outputs=[hr_start, hr_end, hr_peak, hr_stage_strength],
                show_progress=False,
            )
            if hasattr(self.hires_shift_component, "release"):
                self.hires_shift_component.release(
                    fn=_auto_timing_ui,
                    inputs=[hr_template, self.hires_shift_component, hr_auto, hr_start, hr_end, hr_peak, hr_stage_strength],
                    outputs=[hr_start, hr_end, hr_peak, hr_stage_strength],
                    show_progress=False,
                )

        # See _reapplying_changed_templates
        panel_shifts = [base_shift, hires_shift]
        panel_timings = [base_auto, base_start, base_end, base_peak, base_stage_strength, base_curve, hr_auto, hr_start, hr_end, hr_peak, hr_stage_strength, hr_curve]
        save_current_template.click(
            fn=_reapplying_changed_templates(_save_template_ui),
            inputs=[custom_name, custom_auto, base_template, base_start, base_end, base_peak, base_stage_strength, base_curve, base_template, hr_template, *panel_shifts],
            outputs=[base_template, hr_template, manage_template, template_status, *panel_timings],
            show_progress=False,
        )
        save_combo_template.click(
            fn=_reapplying_changed_templates(_save_preset_combo_template_ui),
            inputs=[builder_name, builder_presets, builder_shift, base_template, hr_template, *panel_shifts],
            outputs=[base_template, hr_template, manage_template, builder_status, *panel_timings],
            show_progress=False,
        )
        rename_template.click(
            fn=_reapplying_changed_templates(_rename_template_ui),
            inputs=[manage_template, rename_template_name, base_template, hr_template, *panel_shifts],
            outputs=[base_template, hr_template, manage_template, template_status, *panel_timings],
            show_progress=False,
        )
        delete_template.click(
            fn=_reapplying_changed_templates(_delete_template_ui),
            inputs=[manage_template, base_template, hr_template, *panel_shifts],
            outputs=[base_template, hr_template, manage_template, template_status, *panel_timings],
            show_progress=False,
        )

        # Each control is the local variable named in SCRIPT_ARGS.
        named = locals()
        fields = [(named[name], key, preset_name) for name, key, preset_name in SCRIPT_ARGS]
        for control, _, preset_name in fields:
            if preset_name is not None:
                register_preset_control(self.tabname, preset_name, control)
        # _translate_pasted_infotext turns the recorded codes into these controls' values
        self.infotext_fields = [PasteField(control, key) for control, key, _ in fields]
        self.paste_field_names = list(INFOTEXT_KEYS)
        return [control for control, _, _ in fields]

    def process(self, p, *values):
        args = _arguments(values)
        if not args.enabled:
            _cleanup_runtime(p)
            _note_text_encoder_signature(p, None)
            return
        start_run(p, args)

    def before_process_batch(self, p, *args, **kwargs):
        state = getattr(p, "_anima_lora_stage_scheduler_state", None)
        if state is None or not state.enabled:
            return

        install_patch()
        _rewrite_batch_prompts(p, state)

    def process_before_every_sampling(self, p, *args, **kwargs):
        state = getattr(p, "_anima_lora_stage_scheduler_state", None)
        if state is None or not state.enabled:
            return

        state.set_factor(0.0)

    def postprocess_batch_list(self, p, pp, *args, **kwargs):
        # The batch's infotexts record the LoRA weights as written, not the
        # panel's multiplier the batch loaded them at
        restore_batch_prompts(p)

    def postprocess(self, p, processed, *args, **kwargs):
        _cleanup_runtime(p)
        restore_run_prompts(p)

    def cleanup(self, p, *args, **kwargs):
        _cleanup_runtime(p)
