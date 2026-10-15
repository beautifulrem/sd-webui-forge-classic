"""
Lock img2img Steps
==================
Adds a "Lock Sampling Steps" switch to the img2img sampler panel. While it
is on, pasted parameters, "Send to img2img", reused jobs, presets and other
programmatic form updates leave the Sampling Steps value alone; editing the
steps by hand still works and becomes the new locked value.

The lock is client-side form behaviour declared by the switch
(``Toggle(locks="sampler.steps")``): the form store drops every non-user
write to that param while the switch is on (webui/src/ui/locks.ts). The
script has no processing hooks; its value only travels with the form.
"""

from modules import scripts
from remi import ui

LABEL = "Lock Sampling Steps (ignore values from Send to img2img / paste)"
TARGET = "sampler.steps"


class LockImg2ImgSteps(scripts.Script):
    """Always-visible img2img-only panel: the lock switch."""

    # Render in Forge's native sampler section instead of the generic scripts area
    section = "sampler"
    sorting_priority = 1

    def title(self):
        return "Lock img2img Steps"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def params(self, is_img2img):
        if not is_img2img:
            return None
        return ui.Toggle("lock", False, label=LABEL, locks=TARGET)
