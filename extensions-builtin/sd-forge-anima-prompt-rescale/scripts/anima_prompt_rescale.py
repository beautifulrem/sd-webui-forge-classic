"""Anima Prompt-Schedule Rescaler -- Forge Classic (neo) extension.

Companion to the Anima sampler. Fixes the common problem where a prompt-editing
schedule authored for a txt2img pass (e.g. ``[@guweiz:@wlop:15]`` at 35 steps)
no longer lands at the intended point once the prompt is sent to an img2img pass
at a different step count and a denoise < 1 -- because img2img only executes the
tail of the schedule.

Provides two things, both opt-in and non-destructive:

* A **manual tool**: paste a prompt + the step count it was authored for, and
  get back a schedule rewritten for the current target steps / denoise.
* An **auto-apply** checkbox (img2img only): rewrites the schedules of the
  generation's prompts for that single run, keyed to the live Steps slider and
  Denoising strength. Everything is wrapped so a failure can never break a
  generation -- on any error the original prompt is used unchanged.

The rescaling math lives in ``lib_anima_rescale`` and is independent of Forge.
"""

from __future__ import annotations

import os
import sys

import gradio as gr

from modules import scripts, shared

_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

from lib_anima_rescale.rescale import rescale_prompt_schedule


_INTRO = (
    "Rewrites prompt-editing schedules (`[from:to:when]`) so a switch authored "
    "for one pass lands at the **same relative point within the executed img2img "
    "window**. img2img with denoise `d` only runs the last ~`d` of the schedule, "
    "so a `when` set during a txt2img pass otherwise fires too early (or never). "
    "Alternation `[a|b]` and attention weights `(x:1.3)` are left untouched."
)

_AUTO_INFO = (
    "When on, the schedules in this img2img generation's prompt **and** negative "
    "prompt are rescaled for this run only, using the live **Steps** slider as the "
    "target and **Denoising strength** as the window. Set **Source steps** to the "
    "step count the schedule was originally written for (your txt2img steps). "
    "Safe: any parse problem falls back to the original prompt."
)


class AnimaPromptRescaleScript(scripts.Script):
    sorting_priority = 17

    def __init__(self):
        super().__init__()
        self.is_img2img = False

    def title(self):
        return "Anima Prompt-Schedule Rescaler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.is_img2img = bool(is_img2img)

        with gr.Accordion(label=self.title(), open=False):
            gr.Markdown(_INTRO)

            # --- Manual tool -------------------------------------------------
            with gr.Accordion(label="Manual rescale tool", open=True):
                src_in = gr.Textbox(
                    label="Prompt with schedule",
                    placeholder="[@guweiz:@wlop:15], masterpiece, (detailed:1.2)",
                    lines=3,
                )
                with gr.Row():
                    man_source = gr.Slider(minimum=1, maximum=150, step=1, value=35,
                                           label="Source steps (authored for)")
                    man_target = gr.Slider(minimum=1, maximum=150, step=1, value=35,
                                           label="Target steps (this pass)")
                with gr.Row():
                    man_denoise = gr.Slider(minimum=0.05, maximum=1.0, step=0.01, value=0.5,
                                            label="Denoise (this pass)")
                    man_fraction = gr.Checkbox(
                        value=False,
                        label="Output as fractions (portable across step counts)",
                    )
                man_btn = gr.Button("Rescale schedule", variant="primary")
                man_out = gr.Textbox(label="Rescaled prompt", lines=3,
                                     show_copy_button=True)

                def _do_manual(prompt, s, t, d, frac):
                    fixed_steps = bool(getattr(shared.opts, "img2img_fix_steps", False))
                    return rescale_prompt_schedule(
                        prompt or "",
                        int(s),
                        int(t),
                        float(d),
                        as_fraction=bool(frac),
                        executed_steps_override=int(t) if fixed_steps else None,
                    )

                man_btn.click(
                    fn=_do_manual,
                    inputs=[src_in, man_source, man_target, man_denoise, man_fraction],
                    outputs=[man_out],
                )

            # --- Auto-apply (img2img) ---------------------------------------
            auto_enable = gr.Checkbox(
                value=False,
                label="Auto-rescale this img2img generation's schedules",
                visible=self.is_img2img,
            )
            auto_source = gr.Slider(
                minimum=1, maximum=150, step=1, value=35,
                label="Source steps (txt2img steps the schedule was authored for)",
                visible=self.is_img2img,
            )
            auto_fraction = gr.Checkbox(
                value=False,
                label="Rewrite as fractions",
                visible=self.is_img2img,
            )
            if self.is_img2img:
                gr.Markdown(_AUTO_INFO)

        # Only these three feed process(); the manual tool is self-contained.
        return [auto_enable, auto_source, auto_fraction]

    def process(self, p, auto_enable=False, auto_source=35, auto_fraction=False):
        # Never touch txt2img, and never act unless explicitly enabled.
        if not self.is_img2img or not auto_enable:
            return
        if getattr(p, "init_images", None) in (None, []):
            # Not actually an img2img run (defensive).
            return
        try:
            target_steps = int(getattr(p, "steps", auto_source) or auto_source)
            denoise = float(getattr(p, "denoising_strength", 1.0) or 1.0)
            source_steps = int(auto_source)
            frac = bool(auto_fraction)
            fixed_steps = bool(getattr(shared.opts, "img2img_fix_steps", False))
            executed_override = target_steps if fixed_steps else None

            def _rw(text):
                if not isinstance(text, str) or not text:
                    return text
                return rescale_prompt_schedule(
                    text,
                    source_steps,
                    target_steps,
                    denoise,
                    as_fraction=frac,
                    executed_steps_override=executed_override,
                )

            # Rewrite the live prompt fields that conditioning is built from.
            # p.all_prompts / p.all_negative_prompts are assembled before
            # process() runs, so editing them in place feeds the rescaled
            # schedule straight into get_learned_conditioning for this run.
            changed = False
            for attr in ("all_prompts", "all_negative_prompts"):
                seq = getattr(p, attr, None)
                if isinstance(seq, list):
                    new = [_rw(x) for x in seq]
                    if new != seq:
                        changed = True
                    setattr(p, attr, new)
            if isinstance(getattr(p, "all_prompts", None), list) and p.all_prompts:
                p.main_prompt = p.all_prompts[0]
            if isinstance(getattr(p, "all_negative_prompts", None), list) and p.all_negative_prompts:
                p.main_negative_prompt = p.all_negative_prompts[0]
            for attr in ("prompt", "negative_prompt"):
                val = getattr(p, attr, None)
                if isinstance(val, str):
                    setattr(p, attr, _rw(val))

            if changed:
                p.extra_generation_params["Anima schedule rescale"] = (
                    f"src{source_steps}->tgt{target_steps}@"
                    f"{'fixed' if fixed_steps else f'd{denoise:g}'}"
                )
        except Exception as exc:  # never break a generation
            print(f"[anima-prompt-rescale] skipped (safe fallback): {exc}")
            return
