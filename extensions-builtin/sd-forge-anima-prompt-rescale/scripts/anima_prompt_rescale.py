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

from modules import prompt_rewrites, scripts, shared
from remi import ui

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


def manual_rescale(prompt, source_steps, target_steps, denoise, as_fraction) -> str:
    """The manual tool's result: ``prompt``'s schedules rescaled for a pass
    of ``target_steps`` at ``denoise`` (all of them with img2img fix steps)."""

    fixed_steps = bool(getattr(shared.opts, "img2img_fix_steps", False))
    return rescale_prompt_schedule(
        prompt or "",
        int(source_steps),
        int(target_steps),
        float(denoise),
        as_fraction=bool(as_fraction),
        executed_steps_override=int(target_steps) if fixed_steps else None,
    )


def manual_tool() -> ui.Group:
    """The manual tool's inputs, buttons and result line."""

    return ui.Group(
        "Manual rescale tool",
        ui.Text("manual_prompt", "", label="Prompt with schedule", placeholder="[@guweiz:@wlop:15], masterpiece, (detailed:1.2)", lines=3, arg=False),
        ui.Row(
            ui.Slider("manual_source", 1, 150, 35, 1, label="Source steps (authored for)", arg=False),
            ui.Slider("manual_target", 1, 150, 35, 1, label="Target steps (this pass)", arg=False),
        ),
        ui.Row(
            ui.Slider("manual_denoise", 0.05, 1.0, 0.5, 0.01, label="Denoise (this pass)", arg=False),
            ui.Toggle("manual_fraction", False, label="Output as fractions (portable across step counts)", arg=False),
        ),
        ui.Row(
            ui.Button("Rescale schedule", "rescale", "primary"),
            ui.Button("Use as prompt", "use_as_prompt"),
        ),
        ui.Output("rescaled", "text", label="Rescaled prompt"),
        open=True,
    )


# The auto-apply settings are the processing arguments (the legacy three); the
# manual tool's inputs are tool inputs (arg=False): its actions read them.
AUTO_PARAMS = ("auto_enable", "auto_source", "auto_fraction")
MANUAL_PARAMS = ("manual_prompt", "manual_source", "manual_target", "manual_denoise", "manual_fraction")


class AnimaPromptRescaleScript(scripts.Script):
    section = "accordions"
    sorting_priority = 17
    param_order = AUTO_PARAMS

    def __init__(self):
        super().__init__()
        self.is_img2img = False

    def title(self):
        return "Anima Prompt-Schedule Rescaler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        self.is_img2img = bool(is_img2img)
        # Auto-apply rewrites img2img runs only; txt2img keeps the (hidden) params
        txt2img = not self.is_img2img
        return ui.Section(
            self.title(),
            ui.Markdown(_INTRO),
            manual_tool(),
            ui.Toggle("auto_enable", False, label="Auto-rescale this img2img generation's schedules", preset="prompt_rescale.enabled", hidden=txt2img),
            ui.Slider(
                "auto_source",
                1,
                150,
                35,
                1,
                label="Source steps (txt2img steps the schedule was authored for)",
                preset="prompt_rescale.source_steps",
                hidden=txt2img,
            ),
            ui.Toggle("auto_fraction", False, label="Rewrite as fractions", preset="prompt_rescale.fractions", hidden=txt2img),
            None if txt2img else ui.Markdown(_AUTO_INFO),
        )

    @staticmethod
    def _manual_result(values: dict) -> str:
        current = ui.ScriptLayout(manual_tool(), owner="manual_tool").coerce(values, unknown="ignore")
        return manual_rescale(*(current[name] for name in MANUAL_PARAMS))

    @ui.action("rescale")
    def rescale(self, ctx: ui.ActionContext, values: dict) -> dict:
        return {"__outputs__": {"rescaled": self._manual_result(values)}}

    @ui.action("use_as_prompt")
    def use_as_prompt(self, ctx: ui.ActionContext, values: dict) -> dict:
        """The tool's result written into the tab's prompt box (the legacy
        result box's copy button)."""

        result = self._manual_result(values)
        return {"core.prompt": {"value": result}, "__outputs__": {"rescaled": result}}

    # The same p is run again by img2img batch, Loopback and SD upscale:
    # undo this script's rewrite before the next run rebuilds the prompts.
    # (Not after a run: Loopback / SD upscale report p.prompt afterwards.)
    def before_process(self, p, *args):
        prompt_rewrites.restore(p)

    def process(self, p, auto_enable=False, auto_source=35, auto_fraction=False, *_older_tool_args):
        # (``_older_tool_args``: jobs queued by Remi builds that passed the manual tool's values too)
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
                    prompt_rewrites.record(p, attr, val, getattr(p, attr))

            if changed:
                p.extra_generation_params["Anima schedule rescale"] = (
                    f"src{source_steps}->tgt{target_steps}@"
                    f"{'fixed' if fixed_steps else f'd{denoise:g}'}"
                )
        except Exception as exc:  # never break a generation
            print(f"[anima-prompt-rescale] skipped (safe fallback): {exc}")
            return
