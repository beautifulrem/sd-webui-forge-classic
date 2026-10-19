# Anima Hires Guard

Forge Neo built-in, Anima-only generation guard for the base pass and Hires
fix target. It runs after the bundled Anima resolution randomizer and provides
GUI controls for:

- report-only or safe-clamp policy;
- Low VRAM, Balanced, Detail, and fully custom limits;
- base-pass and Hires megapixel ceilings;
- maximum per-axis upscale ratio;
- optional 16-pixel alignment.

The extension is disabled by default. It never changes non-Anima generations.
Forge rounds every Hires target to its Resolution Step (Settings, default
64 px) after the guard runs, so the Hires target is aligned to the least common
multiple of that step and the 16-pixel grid (just the step when alignment is
off). A target on that grid comes out of Forge's rounding unchanged, so the
checked size is the rendered size. When clamping is enabled and Hires is on,
the guarded target is always written as an explicit Hires width and height.
Every selected limit and applied adjustment is recorded in image metadata,
together with the request the guarded target replaced (`Hires upscale` for a
scale, `Anima Hires requested resize` for an explicit size). Pasting the
infotext restores that request and the original base size (from the
`base WxH -> ...` adjustment), so the pasted run is guarded the same way and
writes the same metadata.
Alignment is ceiling-aware: if nearest-grid rounding would cross an MP or
per-axis limit, the guard rounds downward. A target that would become smaller
than the base on either axis falls back to the already-safe base rectangle,
snapped onto the Hires grid when the base is off it.
txt2img's Upscale button starts from a finished image, so there the image
size is kept and only the Hires target relative to it is guarded.

The 16-pixel alignment follows Anima's 8x spatial VAE reduction followed by
the transformer's 2x spatial patch embedding. Existing 64-pixel Anima presets
remain valid because 64 is also divisible by 16.
