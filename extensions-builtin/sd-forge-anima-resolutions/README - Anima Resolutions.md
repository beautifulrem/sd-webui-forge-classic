# sd-forge-anima-resolutions

Built-in Remi extension for [Anima](https://huggingface.co/circlestone-labs/Anima). Four features:

1. **Resolution presets** in the txt2img dimensions section: a standard list and a high-res list (searchable, with megapixels and aspect ratio). Picking one sets Width and Height at once; each list shows the preset matching the current size.
2. **img2img size fit**: when the img2img source image changes (*Send to img2img* / *Send to inpaint*, or an upload) and "Fit the size to each new source image" is on, Width and Height are set to multiples of 64 whose sum lands in **2560 – 3072** (neither side over 1856), the **highest total** whose aspect-ratio drift from the source stays within `MAX_UPSCALE_DRIFT` (3%). "Fit now" applies it on demand.
3. **Randomize toggles**: the standard toggle rolls each txt2img job's size from the standard list, the high-res toggle from the high-res list, both on from the combined pool. img2img is unaffected.
4. **Session prompt history**: every txt2img job's positive prompt is kept in a 100-entry session ring (most recent first, duplicates move to the top). Picking an entry puts it back into the prompt; **Refresh** loads the latest entries.

## txt2img picker constraints

**Standard preset list** (the standard Randomize pool):

- All dimensions multiples of 64.
- **W + H ≤ 2176** for every preset.
- Each dimension ≥ 640 and ≤ 1536.
- Comprehensive: every multiple-of-64 pair satisfying the above is included (~120 entries), grouped by orientation (Square / Landscape / Landscape wide / Ultra-wide / Portrait / Portrait tall / Ultra-tall) and sorted by aspect ratio within each group.
- Standard SDXL ~1MP buckets are included naturally (1024², 1152×896 / 896×1152, 1216×832 / 832×1216, 1344×768 / 768×1344, 1536×640 / 640×1536), as are all Anima-tuned higher-pixel-count pairs.

**High-res preset list** (also the high-res Randomize pool):

- All dimensions multiples of 64.
- **W + H ∈ [2560, 3072]**.
- Each dimension ≤ 1856 (the lower 2560 total with the 1856 cap forces each side ≥ 704).
- Comprehensive: every multiple-of-64 pair satisfying the above is included (135 entries), sorted by total pixel count, largest first.
- The [2560, 3072] window matches the img2img auto-resize target window exactly, so every entry is already a valid final-size target — the auto-resize never downscales it. An entry already at the top of the window (sum 3072) stays unchanged; one lower in the window is nudged up toward 3072 by the same max-total rule that applies to every source.

The standard picker and the standard Randomize roll use the `RESOLUTIONS` list at the top of `scripts/anima_resolutions.py`; the high-res picker and the high-res Randomize roll use `HIGH_RES_RESOLUTIONS`.

## img2img Resize-to target

- Post-upscale **W + H ∈ [2560, 3072]**, with **neither side exceeding 1856**. Going past 3072 total is quality-problematic and pointless given the VAE, so the window is deliberately tight.
- **Both target W and target H are multiples of 64**, so there is no latent-bucket distortion from non-divisible dimensions. (This is what required switching from "Resize by" to "Resize to": a scale multiplier rounds non-deterministically and can land on non-64 dims.)
- **Picks the highest-total pair (closest to 3072)** among the multiple-of-64 candidates whose aspect-ratio drift from the source stays within the drift cap (`MAX_UPSCALE_DRIFT`, default **3%** = 0.03). Tie on total → least drift wins. The goal is resolution: get the first pass as big as the window allows; the drift cap is just a guardrail so the size push never throws the ratio off by more than that. Raise the cap (e.g. 0.05) to chase the literal max total more aggressively; lower it (e.g. 0.015) to stay more ratio-faithful. If **no** candidate is within the cap (only very extreme source ratios), it falls back to the single least-drift pair so it still emits a sane target.
- Never downscales (target W ≥ source W and target H ≥ source H).
- **Pushes in-window sources up toward 3072 too.** A source already inside [2560, 3072] but below the within-cap maximum is upscaled to the largest within-cap pair; a source already *at* the within-cap maximum is left unchanged (its own dims are the highest-total, zero-drift option — a no-op, never a downscale). Standard-preset sources sit below the floor and always upscale.
- Applies uniformly to images generated from either preset list.
- **Source dimensions come from the source image's generation parameters** (`Size`, read with the WebUI's `parse_generation_parameters`, a plain `Size: WxH` regex as fallback), else from the image itself. With Randomize the size recorded is the rolled one; with Hires. fix it is the first pass.

**Use the "Resize to" tab in img2img** for the Width/Height to take effect; the fit only writes the values.

## How it is built

- `scripts/anima_resolutions.py` declares the panel with `params()` (`remi.ui`). The Randomize toggles are the script's processing arguments (`before_process`), first as before; the preset picker and the history pick follow them.
- The preset picker is the `resolution-presets` widget (`webui/src/widgets/resolution-presets`): it writes the form's width and height in the browser and runs their change effects (the Hires. fix size line).
- The size fit is the `source-size-fit` widget (`webui/src/widgets/source-size-fit`) on img2img: it calls the script action `fit_to_source` once per new source image (`fit_to_source_now` for the button), which reads the image's parameters from the file store and returns the Width/Height patch (`upscale_target`).
- The history is server-side and session-only (cleared when the WebUI exits); `before_process` records the prompt, the `use_history` / `refresh_history` actions read it.
- zh-CN strings: `localizations/zh_CN.json` (panel) and the widgets' `locales/zh-CN.json`.
- Tests: `tests/` (`fixtures/legacy_ui_*.json` is the Gradio contract snapshotted before the port, `legacy_upscale_targets.json` the Gradio version's targets).
