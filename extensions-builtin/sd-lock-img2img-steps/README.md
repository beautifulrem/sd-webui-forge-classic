# Lock img2img Steps

Adds a **"Lock Sampling Steps"** switch to the img2img sampler panel of the
Remi WebUI.

While the switch is on, the img2img Sampling Steps value is **not**
overwritten by "Send to img2img" (from txt2img, PNG Info, the gallery, …),
by pasted generation parameters / infotext, or by presets. Everything else
still updates.

Useful for workflows where you want to pull in an image + most of its
parameters for upscaling, but keep your own fixed step count.

## Usage

- Turn the switch on: the Sampling Steps value is locked.
- Send an image / paste parameters from anywhere: everything else updates,
  Steps stays put.
- You can still change Steps by hand while it is locked; your edit becomes
  the locked value.
- Turn the switch off to resume normal behaviour.

## How it works

The script declares one param, a switch that locks `sampler.steps`
(`ui.Toggle("lock", locks="sampler.steps")`). While it is on, the form store
drops every programmatic write to that param (pastes, Send to, reused jobs,
presets, action patches) and lets user edits through. The lock is part of
the form itself, so it also holds for values sent to img2img before the
img2img page was opened. The script has no processing hooks.
