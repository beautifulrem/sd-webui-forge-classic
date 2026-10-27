# Port: JSON Builder (`sd_forge_json_builder`)

Status: ported 2026-10-27 (upstream sync of Forge Neo `c6c2b43d`; the
extension came with `be6fa5d4` "Moss-Image"). Upstream registers an empty
Gradio tab, "JSON Builder", through `on_ui_tabs` and builds the whole editor in
JavaScript (`javascript/main.js`, `compile.js`, `decompile.js`, `style.css`).
It has no `Script.ui()`, so there is no legacy snapshot
([upstream-absorption.md §3](../upstream-absorption.md#3-procedure),
"Tab-only or JavaScript-only extensions").

## Where it went

| upstream | Remi |
| --- | --- |
| `scripts/builder.py` (`gr.Blocks()` tab) | `scripts/builder.py`: `pages.register(Page("json-builder", "JSON Builder", "braces", widget="json-builder"))`, no Gradio |
| `javascript/main.js` (editor, canvas, Send to) | `webui/src/pages/custom/json-builder/`: `index.tsx` (side panel, dialogs, send), `LayoutCanvas.tsx` (boxes, drag, resize, zoom, pan), `draft.ts` (editing model), `store.ts` (the draft, kept in the browser), `view.ts` (zoom / pan maths) |
| `javascript/compile.js` | `compile.ts` (`compilePrompt`, same rules and error messages) |
| `javascript/decompile.js` | `decompile.ts` (`decompilePrompt`, plus `splitPrompt`) |
| `style.css` | the SPA's design system (Tailwind classes); deleted |
| "Send to txt2img / img2img": writes `#txt2img_prompt`, `#txt2img_width`, `#txt2img_height` and switches tab | `deliver(tab, [{ owner: "core", values: { prompt, width, height } }])` (`features/send-to`): the form's own params, their `on_change` actions run, the generate view is shown |

The page is at `#/ext/json-builder`, in the navigation like the other
extension pages ("UI Tab Order", "Hide UI Tabs", gone when the extension is
disabled). Its title is translated by the extension's
`localizations/zh_CN.json`; the editor's strings are chrome strings
(`webui/src/locales/zh-CN.json`).

## What it does

A visual editor for MOSS-Image's structured ("Layout") prompts:

- **Resolution**: width and height (snapped to multiples of 16 when the
  field is left), or megapixels and an aspect ratio (1:1 … 21:9) with "Set
  resolution". A new size fits the canvas again and moves the boxes inside it.
- **Summary**: canvas type (single / multi-panel / collage), the caption,
  meta tags (comma separated) and "Extra", text appended after the compiled
  prompt (a LoRA tag).
- **Items**: object, character or text (with the literal text to render),
  a caption and a box `x1 y1 x2 y2` in pixels. On the canvas: drag a box,
  drag its corner to resize it (at least 16 px), arrow keys move the focused
  box (Shift: 16 px), a double click edits its caption or text in place. The
  wheel zooms at the cursor, the middle button (or the left one on the
  background while zoomed in) pans, a double click on the background or
  "Fit to screen" fits it again.
- **Send to txt2img / img2img** compiles the draft and writes the prompt
  (on one line), width and height into that form. Empty captions or texts
  are marked and nothing is sent; a draft the compiler refuses (a box outside
  the canvas, …) shows the compiler's message. "Copy prompt" copies it.
  The canvas has no upper size bound, but the forms do (txt2img 4096,
  img2img 2048, the maximum of their Width / Height sliders): a size past the
  target's is still sent and the toast warns that it must be changed before
  generating (upstream sends it silently; `size-limit.ts`).
- **Load prompt…** turns a compiled prompt (with the size it was made for)
  or a saved draft back into the editor; **Save JSON** / **Load JSON**
  download and open drafts (`layout.json`).

## Differences from upstream

- The compiled prompt is byte for byte upstream's (`compile.test.ts` holds a
  prompt recorded from upstream's `compile.js`; a differential run of 20,000
  random drafts, valid and invalid, gave the same prompts and error messages).
  One message differs on purpose: upstream's JSON scanner reads `true`,
  `false` and `null` as numbers and reports "Invalid JSON number"; the port
  parses them, as the reference compiler (OpenMOSS `compile_prompt.py`) does,
  and the field checks refuse them. No valid draft contains one.
- Loading a prompt shows its parentheses unescaped (`\(` → `(`): the
  compiler escapes them again, so the prompt does not change, but the fields
  read as typed. Upstream left the backslashes in the fields.
- "Load prompt…" also takes a prompt with the "Extra" text after it (as
  "Send to" writes it, e.g. copied back from PNG Info): the text after the
  JSON goes back into "Extra". Upstream refused it as invalid JSON.
- Errors say what is wrong (the dialog keeps the text and shows the message,
  file loads and sends show a toast); upstream only outlined the button red.
- The draft survives leaving the page and reloading (browser storage,
  `remi-json-builder`); upstream's lived in the tab until the page reloaded.
- "Save JSON" downloads `layout.json` (the browser may ask where); upstream
  asked for a file name with `prompt()`.
- No server action: the compiler runs in the browser only. API clients that
  want a Layout prompt compile it themselves (the format is OpenMOSS's).

## Tests

- vitest, `webui/src/pages/custom/json-builder/`: `compile.test.ts` (the
  recorded upstream prompt, every validation error, duplicate keys, number
  overflow), `decompile.test.ts` (rebuild for a size, round trip, errors,
  `splitPrompt`), `draft.test.ts` (resolution, boxes, draft key order,
  missing fields, the sent prompt, load with unescaping and a stable prompt),
  `view.test.ts`, `store.test.ts` (items, sizes, stored-draft validation),
  `index.test.tsx` (marking, send to txt2img, load prompt and file, drag,
  resize, zoom, in-place editing, keyboard).
- pytest, `extensions-builtin/sd_forge_json_builder/tests/`: the page
  registration (widget, no params, tab settings, unload), no Gradio /
  JavaScript / CSS in the extension, the SPA view and icon exist, zh_CN title.
