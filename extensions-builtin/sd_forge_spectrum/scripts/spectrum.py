import os

import gradio as gr
from lib_spectrum import logger
from lib_spectrum.forecaster import SpectrumNode
from lib_spectrum.pass_context import resolve_pass_context
from lib_spectrum.presets import PresetManager

from modules import paths, scripts, shared
from modules.anima_support import is_anima_engine
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

PresetManager.load_presets()


class SpectrumForForge(scripts.Script):
    sorting_priority = 2026

    def title(self):
        return "Spectrum Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            with gr.Row():
                w = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Prediction Weighting",
                    info="higher = long-term trend ; lower = short-term changes",
                )
                m = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=4,
                    step=1,
                    label="Polynomial Degree",
                    info="higher = complex & subtle patterns ; lower = stable & faster",
                )
            with gr.Row():
                lam = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.1,
                    step=0.05,
                    label="Regularization",
                    info="higher = reduce overfitting ; lower = fit more data",
                )
                window_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Cache Window",
                    info="higher = skip more steps ; lower = slower but more accurate",
                )
            flex_window = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.0,
                step=0.05,
                label="Window Growth",
                info="higher = more speed & less accurate ; lower = more consistent accuracy but less speed gain",
            )
            with gr.Row():
                warmup_steps = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=6,
                    step=1,
                    label="Warmup Steps",
                    info="Run the full model before caching starts",
                )
                stop_caching_step = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Stop Caching Step",
                    info="Run the full model for the last few steps",
                )
            with gr.Row():
                tail_actual_steps = gr.Slider(
                    minimum=0,
                    maximum=12,
                    value=3,
                    step=1,
                    label="Minimum Tail Actual Steps",
                    info="Always run at least this many final DiT steps; combined with Stop Caching Step using the safer value.",
                )
                history_size = gr.Slider(
                    minimum=3,
                    maximum=32,
                    value=10,
                    step=1,
                    label="Feature History",
                    info="Must be at least Polynomial Degree + 2. Larger values use more VRAM.",
                )

            with gr.Accordion("Schedule & compatibility", open=False):
                with gr.Row():
                    schedule = gr.Dropdown(
                        ["Window", "SEA (auto-calibrated)"],
                        value="Window",
                        label="Refresh Schedule",
                        info="SEA learns a content-aware threshold on the first matching run, then reuses it.",
                    )
                    refresh_ratio = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.01,
                        label="SEA Refresh Ratio",
                        info="0 matches the Window schedule's compute budget automatically; otherwise sets the target actual-forward fraction.",
                    )
                    sea_beta = gr.Slider(
                        minimum=0.5,
                        maximum=4.0,
                        value=2.0,
                        step=0.1,
                        label="SEA Spectral Beta",
                        info="Natural-image spectral slope used only for SEA decisions.",
                    )
                compat_policy = gr.Dropdown(
                    ["Conservative", "Strict", "Legacy / fastest"],
                    value="Conservative",
                    label="Compatibility Policy",
                    info="Conservative caches only simple Forge CFG batches and cache-safe wrapper chains. Strict currently runs actual steps because Forge has no stable branch UUIDs.",
                )
                verbose = gr.Checkbox(False, label="Verbose Spectrum decisions")

            with gr.Accordion("Presets", open=False):
                _preset = gr.Dropdown(
                    value=None,
                    label="Preset Name",
                    choices=PresetManager.list_preset(),
                    allow_custom_value=True,
                )
                with gr.Row():
                    _load = gr.Button("Apply Preset", variant="secondary")
                    _save = gr.Button("Save Preset", variant="primary")
                    _del = gr.Button("Delete Preset", variant="stop")

                for comp in (_preset, _load, _save, _del):
                    comp.do_not_save_to_config = True

                args = (
                    w,
                    m,
                    lam,
                    window_size,
                    flex_window,
                    warmup_steps,
                    stop_caching_step,
                    tail_actual_steps,
                    history_size,
                    schedule,
                    refresh_ratio,
                    sea_beta,
                    compat_policy,
                    verbose,
                )

                _load.click(
                    fn=lambda name: PresetManager.get_preset(name),
                    inputs=[_preset],
                    outputs=[*args],
                    queue=False,
                )
                _save.click(
                    fn=lambda *args: PresetManager.save_preset(*args),
                    inputs=[_preset, *args],
                    outputs=[_preset],
                    queue=False,
                )
                _del.click(
                    fn=lambda name: PresetManager.delete_preset(name),
                    inputs=[_preset],
                    outputs=[_preset],
                    queue=False,
                )

        self.infotext_fields = [
            PasteField(w, "spec_w"),
            PasteField(m, "spec_m"),
            PasteField(lam, "spec_lam"),
            PasteField(window_size, "spec_window_size"),
            PasteField(flex_window, "spec_flex_window"),
            PasteField(warmup_steps, "spec_warmup_steps"),
            PasteField(stop_caching_step, "spec_stop_caching_step"),
            PasteField(tail_actual_steps, "spec_tail_actual_steps"),
            PasteField(history_size, "spec_history_size"),
            PasteField(schedule, "spec_schedule"),
            PasteField(refresh_ratio, "spec_refresh_ratio"),
            PasteField(sea_beta, "spec_sea_beta"),
            PasteField(compat_policy, "spec_compat_policy"),
            PasteField(verbose, "spec_verbose"),
        ]
        self.paste_field_names = [field.label for field in self.infotext_fields]

        return [
            enable,
            w,
            m,
            lam,
            window_size,
            flex_window,
            warmup_steps,
            stop_caching_step,
            tail_actual_steps,
            history_size,
            schedule,
            refresh_ratio,
            sea_beta,
            compat_policy,
            verbose,
        ]

    def process_before_every_sampling(self, p, enable: bool, *args, **kwargs):
        if not enable:
            return

        args = list(args)
        minimum_history = int(args[1]) + 2
        if int(args[8]) < minimum_history:
            logger.warning("Feature History was raised to %d for Polynomial Degree %d.", minimum_history, int(args[1]))
            args[8] = minimum_history
        args = tuple(args)

        if shared.opts.skip_early_cond > 0.0 or shared.opts.s_min_uncond > 0.0:
            logger.warning('Spectrum does not support "Ignore/Skip Negative Prompt" optimizations.')
            return

        if not is_anima_engine(getattr(p, "sd_model", None)):
            logger.warning("Spectrum vNext is limited to Anima because its cache seam is the Anima DiT final layer.")
            return

        x = kwargs.get("x")
        latent_shape = tuple(x.shape[-2:]) if x is not None else None
        pass_context = resolve_pass_context(p)
        sea_context = {
            "pass": "hires" if pass_context.is_hires else "base",
            "sampler": pass_context.sampler,
            "cfg": round(pass_context.cfg, 4),
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

        keys = [
            "spec_w",
            "spec_m",
            "spec_lam",
            "spec_window_size",
            "spec_flex_window",
            "spec_warmup_steps",
            "spec_stop_caching_step",
            "spec_tail_actual_steps",
            "spec_history_size",
            "spec_schedule",
            "spec_refresh_ratio",
            "spec_sea_beta",
            "spec_compat_policy",
            "spec_verbose",
        ]
        for k, v in zip(keys, args):
            p.extra_generation_params[k] = v
