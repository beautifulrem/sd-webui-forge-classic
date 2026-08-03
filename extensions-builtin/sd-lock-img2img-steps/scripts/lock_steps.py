"""
Lock img2img Steps
==================
Adds a "Lock Sampling Steps" checkbox to the img2img tab. When enabled,
the Sampling Steps slider will NOT be overwritten when an image / parameters
are sent to img2img from another tab, or when infotext is pasted.

Implementation: the actual locking is done in JavaScript
(javascript/lock_steps.js). This script only contributes the UI checkbox.

Why JS-only? Forge-Neo strips down the paste-field tuple shape to
exactly (component, key) and has other modules that iterate paste_fields
assuming that shape. Trying to inject extra metadata into those tuples
breaks unrelated UI (e.g. the "Ignore fields when reading infotext"
settings dropdown). The JS layer side-steps all of that by observing
the slider directly and restoring its value after any UI update.
"""

import gradio as gr

from modules import scripts


class LockImg2ImgSteps(scripts.Script):
    """Always-visible img2img-only UI: just the lock checkbox."""

    def title(self):
        return "Lock img2img Steps"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        if not is_img2img:
            return []

        with gr.Accordion("Lock img2img Steps", open=False):
            lock = gr.Checkbox(
                label="Lock Sampling Steps (ignore values from Send to img2img / paste)",
                value=False,
                elem_id="lock_img2img_steps_checkbox",
            )

        return [lock]
