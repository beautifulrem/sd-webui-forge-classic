# Lock img2img Steps

A small extension for **Stable Diffusion WebUI Forge (Classic / Neo)** and
compatible A1111-style forks that adds a **"Lock Sampling Steps"** checkbox
to the img2img tab.

When the checkbox is on, the Sampling Steps slider on img2img will **not**
be overwritten when you click "Send to img2img" from another tab (txt2img,
PNG Info, Extras), or when you paste generation parameters / infotext.

Useful for workflows where you want to pull in an image + most of its
parameters for upscaling, but keep your own fixed step count.

## Installation

1. Drop the `sd-lock-img2img-steps` folder into your WebUI's `extensions/`
   directory:

   ```
   sd-webui-forge-neo/
     extensions/
       sd-lock-img2img-steps/   <-- here
         scripts/
         javascript/
         README.md
   ```

2. Restart the WebUI (full restart, not just "Reload UI" the first time).

3. On the **img2img** tab, expand the **"Lock img2img Steps"** accordion
   near the bottom of the script section.

## Usage

- Tick the checkbox → the Sampling Steps slider is now locked.
- Send an image / paste parameters from anywhere → everything else updates,
  Steps stays put.
- Untick to resume normal behavior.
- You can still change Steps manually while it's locked; the lock only
  blocks *external* overwrites, not your own edits.

## How it works

The Python side just registers the checkbox UI on the img2img tab. The
actual locking happens in JavaScript: the slider's value is snapshotted
whenever you edit it, and after any UI update (which is what "Send to
img2img" and infotext paste trigger) the snapshotted value is restored
if the lock is checked.

This approach is robust to fork-specific changes in the paste-field
Python API — which Forge-Neo in particular has restructured in ways that
break the usual Python-side override hooks.

## Compatibility

Tested target: `Haoming02/sd-webui-forge-classic` (`neo` branch).
Should also work on `classic`, original `lllyasviel/stable-diffusion-webui-forge`,
and AUTOMATIC1111 with no changes — the elem IDs and extension API are
inherited from upstream.

If "Send to img2img" still overwrites Steps when the lock is on, open the
browser console and check for any errors from `lock_steps.js`, or verify
that the img2img Sampling Steps slider has `elem_id="img2img_steps"`
(it has for years on every fork in this lineage).

## License

Do whatever you want with this.
