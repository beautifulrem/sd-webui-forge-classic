# sd-forge-anima-resolutions

Forge Neo extension for [Anima](https://huggingface.co/circlestone-labs/Anima). Four features:

1. **Resolution picker** inserted below the txt2img seed row — two preset dropdowns (standard and high-res), compact, no quick buttons.
2. **img2img "Resize to" auto-adjuster** — on *Send to img2img* / *Send to inpaint*, computes a target W and H that are both multiples of 64 and whose sum lands in **2560 – 3072** (neither side over 1856), choosing the pair with the **highest total** (as close to 3072 as possible) among those whose aspect-ratio drift from the source stays within the drift cap (`MAX_UPSCALE_DRIFT`, default 3%). Writes those targets directly into the img2img Width and Height fields (use the "Resize to" tab).
3. **Randomize toggles** — two independent checkboxes. The standard toggle rolls a random resolution from the standard preset list; the high-res toggle rolls from the high-res preset list; with both on, the roll draws from the combined pool of both lists. Applied per txt2img generation; img2img is unaffected.
4. **Session prompt history** — every Generate click silently logs the current positive prompt into a 100-entry session ring; the dropdown right below the preset dropdowns updates only when you click **↻ Refresh**.

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

The standard dropdown, the standard Randomize roll, and the send-to-img2img Resize-to all share the single `RESOLUTIONS` list at the top of `scripts/anima_resolutions.py`. The high-res dropdown and the high-res Randomize roll use `HIGH_RES_RESOLUTIONS`. Both lookups share `_LABEL_TO_DIMS` so the picker handler doesn't care which dropdown a label came from.

## img2img Resize-to target

- Post-upscale **W + H ∈ [2560, 3072]**, with **neither side exceeding 1856**. Going past 3072 total is quality-problematic and pointless given the VAE, so the window is deliberately tight.
- **Both target W and target H are multiples of 64**, so there is no latent-bucket distortion from non-divisible dimensions. (This is what required switching from "Resize by" to "Resize to": a scale multiplier rounds non-deterministically and can land on non-64 dims.)
- **Picks the highest-total pair (closest to 3072)** among the multiple-of-64 candidates whose aspect-ratio drift from the source stays within the drift cap (`MAX_UPSCALE_DRIFT`, default **3%** = 0.03). Tie on total → least drift wins. The goal is resolution: get the first pass as big as the window allows; the drift cap is just a guardrail so the size push never throws the ratio off by more than that. Raise the cap (e.g. 0.05) to chase the literal max total more aggressively; lower it (e.g. 0.015) to stay more ratio-faithful. If **no** candidate is within the cap (only very extreme source ratios), it falls back to the single least-drift pair so it still emits a sane target.
- Never downscales (target W ≥ source W and target H ≥ source H).
- **Pushes in-window sources up toward 3072 too.** A source already inside [2560, 3072] but below the within-cap maximum is upscaled to the largest within-cap pair; a source already *at* the within-cap maximum is left unchanged (its own dims are the highest-total, zero-drift option — a no-op, never a downscale). Standard-preset sources sit below the floor and always upscale.
- Applies uniformly to images generated from either preset list. The resize step reads source dims from the displayed image's PNG metadata, not from the picker, so it doesn't care which dropdown produced the source.
- **Source dimensions come from the displayed image's PNG metadata** (the hidden `generation_info_txt2img` textbox), not from the txt2img W/H sliders. This matters when Randomize is on, because in that case the sliders hold whatever the user picked but the image was actually generated at the rolled-random resolution. Reading from metadata always reflects what was actually generated.
- Parser uses the WebUI's `parse_generation_parameters` (extracts `Size-1` / `Size-2` keys), with a plain regex on `Size: WxH` as fallback.

**Important — you must be on the "Resize to" tab in img2img** for the auto-set Width/Height to take effect. The extension only writes the values; it doesn't switch tabs for you.

## Randomize feature

A `scripts.Script` subclass (`AlwaysVisible`) runs `before_process(p)` on every txt2img generation. When either inline randomize checkbox is on, it overwrites `p.width` / `p.height` with `random.choice(...)` from the matching pool before the noise is sampled: `RESOLUTIONS` for the standard toggle, `HIGH_RES_RESOLUTIONS` for the high-res toggle, or `RESOLUTIONS + HIGH_RES_RESOLUTIONS` when both are on. Console will log the pool used, e.g. `randomized (standard pool) 1024x1024 -> 768x1408` or `randomized (high-res pool) 1024x1024 -> 1536x1536`.

The roll pool is the **standard** list only — high-res presets are intentionally excluded, since they're meant for deliberate "generate big in the first pass" use rather than random sampling.

One Generate click → one random res (all iterations in `batch_count` share that res, all images in `batch_size` definitely share it since SD requires matching dims in a batch). Each *click* is a fresh roll.

The Script creates a small empty "Anima random resolution" section in the txt2img scripts accordion — a cosmetic side-effect of `AlwaysVisible`. The actual control is the inline checkbox; the accordion section can be ignored or collapsed.

## Prompt history

- Logs every Generate click's positive prompt to a session-only list (`_prompt_history` module variable).
- **LRU dedup** — if you regenerate the same prompt later, its existing entry is removed and re-added at the top, so the dropdown stays clean.
- **Cap 100** — once the list hits 100, oldest entries drop off when new ones come in.
- **Session only** — cleared when the webui process exits (no on-disk storage).
- **Preview** — dropdown labels show the first ~160 characters with whitespace collapsed; selecting fills the full prompt into the textbox.
- **Refresh-driven** — the dropdown only updates when you click the **↻ Refresh** button. Logging happens silently in the background on every Generate click; the UI just doesn't reflect it until you ask.

Negative prompts are not logged (per your spec).

## UI layout

```
Anima resolution helper   <small caption>
[ Preset dropdown                                                ▼ ]
[ High-res preset (txt2img total 2560–3072, max 1856 per side)   ▼ ]
[ Session prompt history (most recent first, max 100)            ▼ ]  [ ↻ Refresh ]
[ ] Randomize resolution each generation (standard preset list)   [ ] Randomize high-res resolution each generation (high-res preset list; both on = combined pool)
```

## Install

Extract into `extensions/`. Final layout:

```
sd-webui-forge-neo/
└── extensions/
    └── sd-forge-anima-resolutions/
        ├── README.md
        └── scripts/
            └── anima_resolutions.py
```

Then fully restart the webui.

## Console output to expect

```
[anima-resolution] script file is being imported
[anima-resolution] callback registered
[anima-resolution] picker injected after txt2img_seed_row
[anima-resolution] prompt-history logger wired on txt2img_generate (log-only; use Refresh to update dropdown)
[anima-resolution] resize-to auto-adjuster wired on txt2img_send_to_img2img (reads image PNG metadata, writes img2img_width / img2img_height)
[anima-resolution] resize-to auto-adjuster wired on txt2img_send_to_inpaint (reads image PNG metadata, writes img2img_width / img2img_height)
```

When you click a send button, you'll also see:

```
[anima-resolution] parsed Size from metadata: 832 x 1344
[anima-resolution] auto-upscale target: 832x1344 (ratio 0.619) -> 1152x1856 (sum 3008, ratio 0.621, drift 0.27%, scale ~1.385) [max-total within drift cap]
```

## If something doesn't work

- **No "prompt-history logger wired" line** → `txt2img_generate` has a different elem_id in your Forge build. Tell me the actual one.
- **Resize-to doesn't change the img2img Width/Height** → look for `could not parse Size from infotext; leaving Resize-to unchanged` in the console. That means the displayed image's metadata doesn't have a `Size:` field, which shouldn't normally happen. If it does, paste me a sample of what `txt2img`'s infotext textbox actually contains. Also check that both `resize-to auto-adjuster wired on ... (reads image PNG metadata, writes img2img_width / img2img_height)` startup lines appear — if the elem_id for the generation_info textbox or for the img2img W/H sliders is different in your Forge build, neither will appear and I need the actual elem_ids to re-target.
- **The Width/Height are set but the upscale still uses the source dimensions** → you're on the "Resize by" tab instead of "Resize to". Switch tabs.
- **History stays empty even after clicking Refresh** → the Generate click handler isn't logging. Check for `prompt-history logger wired on txt2img_generate (log-only; ...)` in the console. If absent, `txt2img_generate` has a different elem_id in your Forge build — tell me the actual one.
- **Randomize toggle does nothing** → confirm the `[anima-resolution] randomize_enabled = True` (or `randomize_highres_enabled = True`) line appears when you click the checkbox; if not, the checkbox isn't wired. If yes but resolution doesn't change, the Script subclass isn't being called — check the txt2img scripts accordion for an "Anima random resolution" section. If that section isn't there, scripts aren't being auto-registered for some reason.
