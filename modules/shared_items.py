import sys

from modules import script_callbacks, scripts
from modules.options import OptionInfo, OptionMarkdown
from remi import ui


def postprocessing_scripts():
    import modules.scripts

    return modules.scripts.scripts_postproc.scripts


def refresh_vae_list():
    import modules.sd_vae

    modules.sd_vae.refresh_vae_list()


def sd_unet_items():
    import modules.sd_unet

    return ["Automatic", *(x.label for x in modules.sd_unet.unet_options), "None"]


def refresh_unet_list():
    import modules.sd_unet

    modules.sd_unet.list_unets()


def list_checkpoint_tiles(use_short=False):
    import modules.sd_models

    return modules.sd_models.checkpoint_tiles(use_short)


def refresh_checkpoints():
    import modules.sd_models

    return modules.sd_models.list_models()


def list_samplers():
    import modules.sd_samplers

    return modules.sd_samplers.all_samplers


def list_schedulers():
    import modules.sd_schedulers

    return [x.label for x in modules.sd_schedulers.all_schedulers]


def get_infotext_names():
    from modules import shared

    res = {}

    for info in shared.opts.data_labels.values():
        if info.infotext:
            res[info.infotext] = 1

    from remi.core.tabs import core_layout, runner_for

    for tab in ("txt2img", "img2img"):
        try:
            layouts = [core_layout(tab), *(script.layout for script in runner_for(tab).scripts if script.layout is not None)]
        except Exception:  # the runners are not built yet
            continue
        for layout in layouts:
            res.update(dict.fromkeys(layout.infotext, 1))

    res["Lora hashes"] = 1

    return list(res)


ui_reorder_categories_builtin_items = [
    "prompt",
    "image",
    "inpaint",
    "sampler",
    "accordions",
    "checkboxes",
    "dimensions",
    "cfg",
    "denoising",
    "seed",
    "batch",
    "override_settings",
]


def ui_reorder_categories():
    from modules import scripts

    yield from ui_reorder_categories_builtin_items

    sections = {}
    for script in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts:
        if isinstance(script.section, str) and script.section not in ui_reorder_categories_builtin_items:
            sections[script.section] = 1

    yield from sections

    yield "scripts"


def callbacks_order_settings():
    options = {
        # (A1111 reused "sd_vae_explanation" here, which replaced the VAE section's explanation)
        "callbacks_explanation": OptionMarkdown(
            """
    For categories below, callbacks added to dropdowns happen before others, in order listed.
    """
        ),
    }

    callback_options = {}

    for category, _ in script_callbacks.enumerate_callbacks():
        callback_options[category] = script_callbacks.ordered_callbacks(category, enable_user_sort=False)

    for method_name in scripts.scripts_txt2img.callback_names:
        callback_options["script_" + method_name] = scripts.scripts_txt2img.create_ordered_callbacks_list(method_name, enable_user_sort=False)

    for method_name in scripts.scripts_img2img.callback_names:
        callbacks = callback_options.get("script_" + method_name, [])

        for addition in scripts.scripts_img2img.create_ordered_callbacks_list(method_name, enable_user_sort=False):
            if any(x.name == addition.name for x in callbacks):
                continue

            callbacks.append(addition)

        callback_options["script_" + method_name] = callbacks

    for category, callbacks in callback_options.items():
        if not callbacks:
            continue

        option_info = OptionInfo([], f"{category} callback priority", ui.MultiChoice, {"choices": [x.name for x in callbacks]})
        option_info.needs_restart()
        option_info.default_order = [x.name for x in callbacks]
        options["prioritized_callbacks_" + category] = option_info

    return options


class Shared(sys.modules[__name__].__class__):
    """
    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    sd_model_val = None

    @property
    def sd_model(self):
        import modules.sd_models

        return modules.sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        import modules.sd_models

        modules.sd_models.model_data.set_sd_model(value)

    @property
    def batch_cond_uncond(self):
        from modules import shared

        # Before shared_init (tests, early imports) opts is None: batch as Forge did before the option
        return getattr(shared.opts, "batch_cond_uncond", True)


sys.modules["modules.shared"].__class__ = Shared
