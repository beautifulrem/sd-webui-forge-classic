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
When clamping is enabled, the guarded target is written as an explicit Hires
width and height so both aligned axes survive Forge's target calculation.
Every selected limit and applied adjustment is recorded in image metadata.
Alignment is ceiling-aware: if nearest-grid rounding would cross an MP or
per-axis limit, the guard rounds downward. A target that would become smaller
than the base on either axis falls back to the already-safe base rectangle.

The 16-pixel alignment follows Anima's 8x spatial VAE reduction followed by
the transformer's 2x spatial patch embedding. Existing 64-pixel Anima presets
remain valid because 64 is also divisible by 16.
