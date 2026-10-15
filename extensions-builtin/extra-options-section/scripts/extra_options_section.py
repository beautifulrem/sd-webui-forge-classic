"""Extra options: chosen settings shown inside txt2img / img2img.

Each setting listed in ``extra_options_txt2img`` / ``extra_options_img2img``
becomes one input of this script, rendered with the settings page's widget
for it (``remi.ui.settings.setting_node``) and defaulting to the setting's
current value. As Gradio's page-load refresh did, every page load starts
from the current settings: the inputs are kept out of form drafts
(``persist=False``), so a setting changed on the Settings page is not
overridden by an old copy. ``before_process`` applies the values as
``p.override_settings`` unless the request already overrides that setting.
"""

from modules import errors, scripts, shared
from remi import ui
from remi.ui.settings import setting_node

# Settings that must not be overridden per generation (model selection)
NOT_ALLOWED = ("sd_model_checkpoint", "sd_vae", "CLIP_stop_at_last_layers", "forge_additional_modules")


def _current(key: str):
    return lambda: getattr(shared.opts, key)


def _infotext_keys() -> dict[str, str]:
    """setting -> infotext key, from the paste mapping the Gradio panel used."""

    from modules import infotext_utils

    return {setting: key for key, setting in infotext_utils.INFOTEXT_TO_SETTING}


class ExtraOptionsSection(scripts.Script):
    section = "extra_options"

    def __init__(self):
        self.setting_names = []

    def title(self):
        return "Extra options"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        opts = shared.opts
        extra_options = opts.extra_options_img2img if is_img2img else opts.extra_options_txt2img
        infotext_keys = _infotext_keys()

        self.setting_names = []
        inputs = []
        for key in extra_options or ():
            if key in NOT_ALLOWED or key in self.setting_names:
                continue
            info = opts.data_labels.get(key)
            try:
                if info is None:
                    raise KeyError(key)
                node = setting_node(key, info, _current(key))
                if not isinstance(node, ui.Input):
                    raise TypeError(f"{key} has no value")
            except Exception:
                errors.report(f"Can't add extra options for {key} in ui", exc_info=True)
                continue
            node.infotext = infotext_keys.get(key)
            node.persist = False
            inputs.append(node)
            self.setting_names.append(key)

        if not inputs:
            return None
        columns = max(1, int(opts.extra_options_cols))
        rows = [ui.Row(inputs[start : start + columns]) for start in range(0, len(inputs), columns)]
        if opts.extra_options_accordion:
            return ui.Section("Options", children=rows, open=False)
        return ui.Group(children=rows)

    def before_process(self, p, *args):
        for name, value in zip(self.setting_names, args):
            if name not in p.override_settings:
                p.override_settings[name] = value


shared.options_templates.update(
    shared.options_section(
        ("settings_in_ui", "Settings in UI", "ui"),
        {
            "settings_in_ui": shared.OptionHTML("This page allows you to add some settings to the main interface of txt2img and img2img tabs."),
            "extra_options_txt2img": shared.OptionInfo(
                [],
                "Settings for txt2img",
                ui.MultiChoice,
                lambda: {"choices": list(shared.opts.data_labels.keys())},
            )
            .js("info", "settingsHintsShowQuicksettings")
            .info("setting entries that also appear in txt2img interfaces")
            .needs_reload_ui(),
            "extra_options_img2img": shared.OptionInfo(
                [],
                "Settings for img2img",
                ui.MultiChoice,
                lambda: {"choices": list(shared.opts.data_labels.keys())},
            )
            .js("info", "settingsHintsShowQuicksettings")
            .info("setting entries that also appear in img2img interfaces")
            .needs_reload_ui(),
            "extra_options_cols": shared.OptionInfo(
                1,
                "Number of columns for added settings",
                ui.Slider,
                {"step": 1, "minimum": 1, "maximum": 20},
            )
            .info("displayed amount will depend on the actual browser window width")
            .needs_reload_ui(),
            "extra_options_accordion": shared.OptionInfo(False, "Place added settings into an accordion").needs_reload_ui(),
        },
    )
)
