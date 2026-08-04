# Prompt Blacklist (Forge Neo / A1111-style WebUI)

Strips unwanted tags from your **positive prompt** with a single button press. Perfect for pasting in booru tag dumps and nuking the junk (`watermark`, `artist name`, usernames, etc.) without hand-editing.

- Works on **txt2img** and **img2img**, with a collapsible panel embedded next
  to each tab's native prompt controls
- **Manual only** — nothing scans or edits your prompt automatically
- **Never touches the negative prompt**
- Blacklist persists under `extension-data/sd-forge-prompt-blacklist/` in
  Forge's data directory

## Install

1. Copy the `sd-forge-prompt-blacklist` folder into your WebUI's `extensions/` directory
   (or use *Extensions → Install from URL* if you host it on git).
2. Restart the WebUI (a full restart, not just *Reload UI*, on first install).
3. Find the **🧹 Prompt Blacklist** accordion on the txt2img / img2img tabs.

## Usage

1. Type your blacklist entries, separated by commas or new lines.
2. Press **💾 Save Blacklist** to persist them (loaded automatically on startup).
3. Paste a prompt into the positive prompt box, press **🧹 Clean Prompt**.
4. The status line shows exactly which tags were removed.

### Matching rules

| Feature | Example |
|---|---|
| Case-insensitive | `Watermark` matches `watermark` |
| Underscore = space | `blue_hair` matches `blue hair` |
| Ignores emphasis/weights | `(watermark:1.4)`, `((watermark))` match `watermark` |
| Ignores escaped parens | `ganyu \(genshin impact\)` matches `ganyu (genshin impact)` |
| Wildcards | `*_username` removes `xyz_username`; `bad *` removes `bad anatomy`, `bad hands` |
| `BREAK` / `AND` are never removed | structural keywords are safe |

Optional checkbox: **Also remove duplicate tags** — deduplicates the prompt while cleaning (first occurrence wins).

### Notes

- The two tabs each show the blacklist loaded at startup. If you save on one tab, press **🔄 Reload** on the other to sync it.
- Only exact (or wildcard) tag matches are removed — `hair` will *not* remove `blue hair`. Use `*hair*` if you want substring behavior.

## License

MIT — do whatever you want with it.
