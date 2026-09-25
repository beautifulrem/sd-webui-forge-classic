# sd-webui-prompt-anchor

Adds an **anchor prompt** box above the main positive prompt on the
**txt2img** and **img2img** tabs. Whatever you put in the anchor gets
prepended to your positive prompt at generation time -- *without* ever
being written into the main prompt textarea.

![diagram](docs/diagram.png) <!-- placeholder; not required -->

## Why

You probably have a "stable foundation" of tags you reuse across nearly
every generation -- artist styles, quality tags, meta tags, base
LoRAs. Two choices today:

  1. Park them at the top of your main prompt and carefully edit
     *around* them every time. Annoying and error-prone.
  2. Use **Styles**. Better, but the UI is built around saving/loading
     discrete style presets -- it's not really designed for the "tweak
     this every 5th generation" workflow.

The anchor box is the third option. Type your foundation once, leave
it sitting at the top of the page, and use the main prompt box just for
the *subject* of each image. Both halves get joined at generation time
into a single positive prompt.

## Key behaviours

- **Foreign-extension proof.** Extensions that auto-fill the main
  `#txt2img_prompt` (booru-tag senders, captioners, etc.) will NOT
  touch the anchor box -- it has its own elem_id and is never wired
  into any "send to txt2img" flow. You can grow your main prompt
  with auto-appends as much as you want; the anchor stays put.

- **Wildcards work inside the anchor.** The prepend happens early
  enough in the generation pipeline that any extension expanding
  `__wildcard__` syntax (Dynamic Prompts, etc.) sees the joined
  text and processes wildcards in your anchor too.

- **Hires-fix aware.** If you've enabled hires fix without a separate
  hires prompt, the anchor flows through automatically. If you've
  written a *distinct* hires prompt, the anchor is also prepended to
  that one, so styles carry through both passes.

- **Saved with the image.** The anchor text goes into the PNG metadata
  as `Anchor prompt: "..."`, so it round-trips: "Send to txt2img"
  on any image generated with this extension restores both halves.

- **Survives reloads.** Anchor text is saved to your browser's
  localStorage on every keystroke, restored on page load. (No
  server-side state -- different browsers = different anchors.)

- **Collapsible.** Hit the `−` button to fold the box down to just its
  header row when you want the screen back.

- **Smart joining.** The default separator is `, `. Trailing commas
  on the anchor and leading commas on the main prompt are stripped
  before joining, so you can't accidentally produce `,,`. Override
  the separator per-tab under the **Advanced** accordion, or change
  the global default in **Settings → Prompt Anchor**.

## Installation

In Forge / Forge Neo / A1111:

  - **Extensions → Install from URL**, paste this repo's URL, click
    Install, then **Apply and restart UI**.

  - Or: `git clone` into `<webui>/extensions/`.

No additional Python packages required.

## Layout

After install you'll see a thin bar above your txt2img positive prompt:

```
+-------------------------------------------------+
| [x] Anchor prompt (prepended to positive...)  −|
+-------------------------------------------------+
|                                                 |
|  score_9, score_8_up, masterpiece, ...          |
|                                                 |
+-------------------------------------------------+
|  > Advanced (separator, etc.)                   |
+-------------------------------------------------+

  [main positive prompt textarea below...]
```

The same on img2img.

## Tips

- The two tabs (txt2img / img2img) have **independent** anchor
  contents. Setting one doesn't affect the other -- which is usually
  what you want, since image-to-image often calls for different
  foundation tags than text-to-image.

- The enable checkbox is the on/off switch. Uncheck it for a single
  generation if you want to test a prompt *without* the anchor; the
  anchor text itself is preserved.

- The "Prompt used" stored in image metadata is the JOINED text
  (anchor + main), not just the main prompt. So sharing PNG params
  with someone who doesn't have this extension installed still
  reproduces the same image -- the anchor just becomes part of the
  prompt they paste in.

## Notes for other extension authors

If you want your extension's output to land in the anchor box rather
than the main prompt, the anchor's elem_ids are:

  - `prompt_anchor_txt2img`
  - `prompt_anchor_img2img`

Standard Gradio textbox conventions apply. The infotext key for
round-tripping is `Anchor prompt` (value is double-quoted with
backslash-escapes for embedded quotes / newlines).

## License

AGPL-3.0, matching the host webui.
