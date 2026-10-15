import os

from lib_spectrum import logger
from lib_spectrum.forecaster import SpectrumNode
from lib_spectrum.pass_context import resolve_pass_context
from lib_spectrum.presets import CONTROLS, DEFAULT, ENABLE_KEY, INFOTEXT_KEYS, PresetManager

from modules import paths, scripts, shared
from modules.anima_support import script_arguments, table_params
from remi import ui

PresetManager.load_presets()

# The params() names of the settings, in CONTROLS order: their preset fields' last parts.
PARAM_NAMES = tuple(name.rsplit(".", 1)[-1] for _, name, _, _ in CONTROLS)
# The script arguments: the switch, the settings in CONTROLS order, then the
# name in the Presets box (UI only; ignored by processing).
SCRIPT_ARG_NAMES = ("enable", *INFOTEXT_KEYS, "preset_name")
_ARG_DEFAULTS = {**DEFAULT, "preset_name": ""}
# (param name, infotext key, preset field): infotext without ENABLE_KEY (older
# images, Spectrum off) leaves the switch alone.
ROWS = (("enable", ENABLE_KEY, "spectrum.enabled"), *((param, key, name) for param, (key, name, _, _) in zip(PARAM_NAMES, CONTROLS)), ("preset_name", None, None))
# Positions in the script arguments after ``enable`` (CONTROLS order).
_DEGREE = INFOTEXT_KEYS.index("spec_m")
_HISTORY = INFOTEXT_KEYS.index("spec_history_size")


class SpectrumForForge(scripts.Script):
    sorting_priority = 2026
    param_order = tuple(name for name, *_ in ROWS)

    def title(self):
        return "Spectrum Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        root = ui.Section(
            self.title(),
            enable=ui.Toggle("enable", False, label=self.title()),
            children=[
                ui.Row(
                    ui.Slider("weight", 0.0, 1.0, DEFAULT["spec_w"], 0.05, label="Prediction Weighting", info="higher = long-term trend ; lower = short-term changes"),
                    ui.Slider("degree", 1, 8, DEFAULT["spec_m"], 1, label="Polynomial Degree", info="higher = complex & subtle patterns ; lower = stable & faster"),
                ),
                ui.Row(
                    ui.Slider("regularization", 0.0, 2.0, DEFAULT["spec_lam"], 0.05, label="Regularization", info="higher = reduce overfitting ; lower = fit more data"),
                    ui.Slider("window", 1, 10, DEFAULT["spec_window_size"], 1, label="Cache Window", info="higher = skip more steps ; lower = slower but more accurate"),
                ),
                ui.Slider("window_growth", 0.0, 2.0, DEFAULT["spec_flex_window"], 0.05, label="Window Growth", info="higher = more speed & less accurate ; lower = more consistent accuracy but less speed gain"),
                ui.Row(
                    ui.Slider("warmup", 0, 20, DEFAULT["spec_warmup_steps"], 1, label="Warmup Steps", info="Run the full model before caching starts"),
                    ui.Slider("stop", 0.0, 1.0, DEFAULT["spec_stop_caching_step"], 0.05, label="Stop Caching Step", info="Run the full model for the last few steps"),
                ),
                ui.Row(
                    ui.Slider("tail_actual", 0, 12, DEFAULT["spec_tail_actual_steps"], 1, label="Minimum Tail Actual Steps", info="Always run at least this many final DiT steps; combined with Stop Caching Step using the safer value."),
                    ui.Slider("history", 3, 32, DEFAULT["spec_history_size"], 1, label="Feature History", info="Must be at least Polynomial Degree + 2. Larger values use more VRAM."),
                ),
                ui.Group(
                    "Schedule & compatibility",
                    ui.Row(
                        ui.Choice("schedule", ["Window", "SEA (auto-calibrated)"], DEFAULT["spec_schedule"], label="Refresh Schedule", info="SEA learns a content-aware threshold on the first matching run, then reuses it."),
                        ui.Slider("refresh_ratio", 0.0, 1.0, DEFAULT["spec_refresh_ratio"], 0.01, label="SEA Refresh Ratio", info="0 matches the Window schedule's compute budget automatically; otherwise sets the target actual-forward fraction."),
                        ui.Slider("sea_beta", 0.5, 4.0, DEFAULT["spec_sea_beta"], 0.1, label="SEA Spectral Beta", info="Natural-image spectral slope used only for SEA decisions."),
                    ),
                    ui.Choice(
                        "policy",
                        ["Conservative", "Strict", "Legacy / fastest"],
                        DEFAULT["spec_compat_policy"],
                        label="Compatibility Policy",
                        info="Conservative caches only simple Forge CFG batches and cache-safe wrapper chains. Strict runs every step actual (Spectrum off) because Forge has no stable branch UUIDs.",
                    ),
                    ui.Toggle("verbose", DEFAULT["spec_verbose"], label="Verbose Spectrum decisions"),
                    open=False,
                ),
                ui.Group(
                    "Presets",
                    ui.Choice("preset_name", PresetManager.list_preset, "", allow_custom=True, label="Preset Name"),
                    ui.Row(
                        ui.Button("Apply Preset", "apply_preset", "secondary"),
                        ui.Button("Save Preset", "save_preset", "primary"),
                        ui.Button("Delete Preset", "delete_preset", "danger"),
                    ),
                    open=False,
                ),
            ],
        )
        table_params(ROWS, root)  # infotext keys and preset fields
        return root

    def _settings(self, values: dict) -> dict:
        """The owner's current values, validated (an action's ``values`` are raw)."""

        layout = self.layout if self.layout is not None else ui.ScriptLayout(self.params(bool(getattr(self, "is_img2img", False))), owner="spectrum", order=self.param_order)
        return layout.coerce(values, unknown="ignore")

    @staticmethod
    def _preset_name(values: dict) -> str:
        name = values.get("preset_name")
        return name if isinstance(name, str) else ""

    @ui.action("apply_preset")
    def apply_preset(self, ctx: ui.ActionContext, values: dict) -> dict:
        name = self._preset_name(values)
        try:
            settings = PresetManager.get_preset(name)
            if settings is None:
                return ui.toast(ui.message('Preset "{{name}}" was not found', name=name), "error")
            # A hand-edited presets.json must not put values the form rejects
            coerced = self._settings(dict(zip(PARAM_NAMES, settings)))
        except ValueError as error:  # ValidationError included
            return ui.toast(ui.message('Preset "{{name}}" is invalid: {{error}}', name=name, error=str(error)), "error")
        return {param: {"value": coerced[param]} for param in PARAM_NAMES}

    @ui.action("save_preset")
    def save_preset(self, ctx: ui.ActionContext, values: dict) -> dict:
        current = self._settings(values)
        name = self._preset_name(current).strip()
        names = PresetManager.save_preset(name, *(current[param] for param in PARAM_NAMES))
        if names is None:
            return ui.toast("Enter a preset name", "error")
        return {"preset_name": {"choices": names, "value": name}, **ui.toast(ui.message('Preset "{{name}}" saved', name=name), "success")}

    @ui.action("delete_preset")
    def delete_preset(self, ctx: ui.ActionContext, values: dict) -> dict:
        name = self._preset_name(values)
        names = PresetManager.delete_preset(name)
        if names is None:
            return ui.toast(ui.message('Preset "{{name}}" was not found', name=name), "error")
        return {"preset_name": {"choices": names, "value": ""}, **ui.toast(ui.message('Preset "{{name}}" deleted', name=name), "success")}

    def cleanup(self, p, *args, **kwargs):
        # The wrapper (and its GPU feature history) stays on forge_objects
        # until the next generation; drop the history as soon as this one ends.
        for state in getattr(p, "_spectrum_states", None) or ():
            state.reset()
        p._spectrum_states = []

    def process_before_every_sampling(self, p, enable: bool, *args, **kwargs):
        # An argument list stored before the newer controls were added (e.g. a
        # UI task Agent Scheduler queued earlier) gets their defaults, as
        # saved presets do; one with extra values is refused.
        named = script_arguments("Spectrum", SCRIPT_ARG_NAMES, (enable, *args), defaults=_ARG_DEFAULTS)
        if not named.enable:
            return

        args = [getattr(named, key) for key in INFOTEXT_KEYS]
        degree = int(args[_DEGREE])
        minimum_history = degree + 2
        if int(args[_HISTORY]) < minimum_history:
            logger.warning("Feature History was raised to %d for Polynomial Degree %d.", minimum_history, degree)
            args[_HISTORY] = minimum_history
        args = tuple(args)

        if shared.opts.skip_early_cond > 0.0 or shared.opts.s_min_uncond > 0.0:
            logger.warning('Spectrum does not support "Ignore/Skip Negative Prompt" optimizations.')
            return

        x = kwargs.get("x")
        latent_shape = tuple(x.shape[-2:]) if x is not None else None
        pass_context = resolve_pass_context(p)
        sea_context = {
            "pass": "hires" if pass_context.is_hires else "base",
            "sampler": pass_context.sampler,
            "scheduler": pass_context.scheduler,
            "cfg": round(pass_context.cfg, 4),
            "shift": None if pass_context.shift is None else round(pass_context.shift, 4),
            "latent_batch": int(x.shape[0]) if x is not None else None,
            "latent_hw": latent_shape,
        }
        unet = p.sd_model.forge_objects.unet
        unet = SpectrumNode.patch(
            unet,
            pass_context.steps,
            *args,
            sea_cache_dir=os.path.join(paths.data_path, "cache", "spectrum-sea"),
            sea_cache_context=sea_context,
            process=p,
        )
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params[ENABLE_KEY] = True
        for k, v in zip(INFOTEXT_KEYS, args):
            p.extra_generation_params[k] = v
