# Post Processing Suite

A feature-packed, **reorderable** post-processing pipeline for
[sd-webui-forge-classic (neo)](https://github.com/Haoming02/sd-webui-forge-classic).
It adds a native generation section to **both txt2img and img2img** and runs a stack of image
effects in `postprocess_image_after_composite` — the final hook before the
image is saved — so every effect is baked into the saved file, the gallery and
the API result. Works alongside other extensions (it only touches the finished
pixels).

Pure `numpy` + `Pillow`, with the descreen using `numpy.fft`. **No extra
dependencies.**

## Install
Built into Remi (`extensions-builtin/`). Open the **Post Processing Suite**
section in either tab's generation controls.

## How it works
1. Tick the master **Post Processing Suite** toggle.
2. Stages are grouped into collapsible **categories** (Cleanup & Optics, Color &
   Tone, Detail, Film & Texture, Stylize, Overlays, Output). Open a category,
   then tick a stage to enable it. Parameters are laid out two per row.
3. Set each stage's **Order** number to control the pipeline order — lower runs
   first. Defaults are spaced (5, 10, 20, …) so you can slot stages between
   existing ones. Disabled stages are skipped.

Active stages are recorded into the PNG metadata under `PP Suite` (`PP Suite (input)`
for img2img input processing). Pasting the parameters restores them: the suite
switches on with exactly the recorded stages, orders and values (stages not in the
record are switched off; an image without the record switches the suite off). A
LUT file is not restored (the metadata only records where it was): pick it again.

## Presets
Open the **Presets** group at the top of the suite.

- **Save** — type a name into *Preset* and click *Save* to store the entire
  current configuration (master toggle + every stage's enabled state, order and
  parameters) as a JSON file under
  `extension-data/sd-postprocess-suite/presets/` in Forge's data directory. A
  LUT file is copied next to it (`presets/luts/`), so the preset keeps working
  after the upload is cleaned up.
- **Load** — pick a preset and click *Load* to apply it to all controls.
- **Delete** / **Refresh** — remove a preset / re-read the folder.

Presets are stored structured-by-stage, so they keep working even if future
versions add or remove stages (unknown keys are ignored, missing ones fall back
to defaults). The img2img "Apply to" selector is intentionally *not* part of a
preset, so looks stay portable between tabs.

## Where it runs (img2img: "Apply post-processing to")
On **img2img** a selector chooses where the pipeline is applied:

- **Final output** (default) — runs in `postprocess_image_after_composite`, the
  last step before the image is saved. Same as txt2img.
- **img2img input (pre-diffusion)** — runs on the **upscaled init image** right
  before it is encoded and diffused. The "Upscaler for img2img" enlarges the
  init image and those upscales often carry grid/halftone artifacts, ringing and
  color casts; this cleans/grades them *before* diffusion instead of after.
- **Both** — input cleanup **and** a final-output pass.

(Input mode applies to standard resize modes; it is skipped for inpaint
"only masked" full-res and latent-resize, to avoid disturbing those paths.)

## Send to img2img (txt2img only)
The **Apply PP → Send to img2img** button (the `pp-send` widget) runs the
current settings on the result selected on the stage (the newest one when none
is picked) on the server, then puts the processed copy into the img2img source
image and switches to img2img — your txt2img → img2img handoff with the
post-processing already applied.

## Stages
**Cleanup/optics:** Descreen/De-halftone (FFT notch), Lens Distortion,
Chromatic Aberration, Vignette, Bloom/Glow, Halation.
**Grading:** Tone (exposure/contrast/levels/gamma/highlights/shadows),
Color (saturation/vibrance/hue/temperature/tint), 3-way Color Balance,
Split Toning, **3D LUT (.cube)** loader.
**Detail:** Blur (gaussian/box), Unsharp-mask Sharpen.
**Film:** Film Grain (mono/colored, size, shadow-weighted), Noise, Scanlines.
**Stylize:** Sepia, Duotone, Posterize, Pixelate, Edge Detect, Pencil Sketch,
Comic/Cel.
**Overlays:** Light Leak, Lens Flare.
**Output:** JPEG Crush.

### About the descreen stage
Qwen-Image / Qwen-Image-Edit (and some other models) can emit a periodic
halftone/grid artifact. That pattern shows up as symmetric bright peaks in the
2D FFT away from the center. The descreen stage transforms to the frequency
domain, auto-detects those peaks (anything `threshold` std-devs above the local
background, outside a protected low-frequency core), notches them out, and
inverse-transforms. `luma` mode is color-safe and fast; `rgb` is stronger.
Raise *Strength*/lower *Peak threshold* for stubborn patterns; raise *Protect
low-freq radius* if real detail starts smearing.

## Extending
Add a dict to `STAGES` in `lib_postprocess/pipeline.py`, add a matching branch
in `apply_stage`, and (if it's a new operation) a function in
`lib_postprocess/effects.py`. The panel (`params()`, one collapsed section per
category and stage, params named `<stage>_<param>`) and the argument handling
are generated from the spec automatically. Add the new labels to
`localizations/zh_CN.json`; `tests/` checks the coverage.
