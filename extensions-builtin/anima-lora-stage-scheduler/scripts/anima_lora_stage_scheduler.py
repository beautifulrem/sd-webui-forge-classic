from modules import script_callbacks, scripts
from modules.anima_lora_support import restore_batch_prompts, restore_run_prompts
from lib_stage_scheduler.options import _arguments, _t
from lib_stage_scheduler.params import bind, params_root, shift_of, templates_patch
from lib_stage_scheduler.runtime import (
    _cleanup_runtime,
    _denoiser_callback,
    _note_text_encoder_signature,
    _rewrite_batch_prompts,
    install_patch,
    start_run,
)
from lib_stage_scheduler.ui import apply_template, delete_template, rename_template, save_preset_combo_template, save_template
from remi import ui


# Registered from this script: a Reload UI clears the callbacks and re-runs it
_CFG_CALLBACK_REGISTERED = False


def _register_cfg_callback_once():
    global _CFG_CALLBACK_REGISTERED
    if _CFG_CALLBACK_REGISTERED:
        return
    script_callbacks.on_cfg_denoiser(_denoiser_callback, name="anima_lora_stage_scheduler")
    _CFG_CALLBACK_REGISTERED = True


def _text(ctx, name) -> str:
    """A string argument of an action request (``ctx.extra``)."""
    value = ctx.extra.get(name)
    return value if isinstance(value, str) else ""


def _panel(ctx) -> str:
    return "hr" if ctx.extra.get("panel") == "hr" else "base"


class Script(scripts.Script):
    section = None

    def __init__(self):
        _register_cfg_callback_once()

    def title(self):
        return _t("title")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        root = params_root()
        self.param_order = bind(root)
        return root

    # region Actions

    def _template_changed(self, ctx, values, prefix):
        """Selecting a template applies its timing to the panel."""
        return {f"{prefix}_timing": ui.update(value=apply_template(values.get(f"{prefix}_template"), shift_of(ctx, prefix)))}

    @ui.action("base_template_changed")
    def base_template_changed(self, ctx, values):
        return self._template_changed(ctx, values, "base")

    @ui.action("hr_template_changed")
    def hr_template_changed(self, ctx, values):
        return self._template_changed(ctx, values, "hr")

    def _after_edit(self, ctx, values, edit):
        """The edit's message, and when it changed the templates, the panels'
        template dropdowns and timings that follow from it."""
        patch = templates_patch(ctx, values, edit.renamed) if edit.ok else {}
        return {**patch, **ui.toast(edit.status, "success" if edit.ok else "warning")}

    @ui.action("template_save")
    def template_save(self, ctx, values):
        """Save a panel (``panel``: base / hr) as template ``name``; ``auto``
        saves it as an auto-timing template."""
        prefix = _panel(ctx)
        edit = save_template(_text(ctx, "name"), bool(ctx.extra.get("auto", True)), values.get(f"{prefix}_template"), values.get(f"{prefix}_timing"))
        return self._after_edit(ctx, values, edit)

    @ui.action("template_combo")
    def template_combo(self, ctx, values):
        """Save the base presets ``presets`` as the combo template ``name``."""
        presets = ctx.extra.get("presets")
        presets = [item for item in presets if isinstance(item, str)] if isinstance(presets, list) else []
        return self._after_edit(ctx, values, save_preset_combo_template(_text(ctx, "name"), presets))

    @ui.action("template_rename")
    def template_rename(self, ctx, values):
        return self._after_edit(ctx, values, rename_template(_text(ctx, "old"), _text(ctx, "new")))

    @ui.action("template_delete")
    def template_delete(self, ctx, values):
        return self._after_edit(ctx, values, delete_template(_text(ctx, "name")))

    # endregion

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
