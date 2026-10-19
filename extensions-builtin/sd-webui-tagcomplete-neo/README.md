# Tag Autocomplete (Remi)

Tag autocompletion for every prompt editor of the Remi WebUI. This extension
contributes the data and the `tac_*` settings (Settings → Tag Autocomplete);
the completion runs in the SPA (`webui/src/features/prompt-tools`) against
`/api/v2/prompt-tools` (`remi/prompt_tools`). Disabling the extension turns
completion off.

- Booru tags from `tags/*.csv` (`name,category,count,"aliases"`): danbooru,
  e621, derpibooru, the merged list, an English dictionary. Prefix, word,
  alias, translation, `*substring` and abbreviation (`detco` →
  `detached_collar`) matches, ranked by match, post count and your own usage.
- Extra tags (`extra-quality-tags.csv`: `name,category,description,aliases,translation`),
  translations (`tag,translation` or the old `tag,category,translation`, or
  the quick tag groups' labels).
- `<` LoRA, `<e:` embeddings, `<c:` chants (`*.json`: `name`, `terms`,
  `content`), `__` wildcards (`scripts/wildcards`, every extension's
  `wildcards/`, Dynamic Prompts' folder; nested folders and YAML collections)
  and `__name__` values.
- Tab / Enter accept, ↑/↓ choose, Esc closes.
- Where it completes: txt2img, img2img, negative prompts and the prompt
  fields of extensions (ADetailer, MultiDiffusion, Prompt Workshop, …) can
  each be turned off, and a checkpoint list turns completion off for some
  checkpoints (or on for only those).

Usage counts of the old extension (`tag_frequency.db`) are imported once.

## Settings of the old extension

Every setting keeps its `tac_*` key, so saved values carry over; the dotted
keys, which settings names cannot have, are renamed with `_`
(`tac_activeIn.txt2img` → `tac_activeIn_txt2img`) and their saved values
moved. Changed: `tac_delayTime` defaults to 40 ms (the old 100 ms was capped
at 50 ms by the old code, `tagAutocomplete.js:1651`, and suggestions seen
before now show without waiting); `tac_modelKeywordCompletion` has no "Only
user list" (there is no separate keyword list: the LoRA's own activation
text is it, so a saved "Only user list" works like "Always");
`tac_animaArtistPrefix` gains "Always", and "Auto" detects Anima from the
loaded checkpoint without asking CivitAI.

### Not carried over

These settings of the old extension have no equivalent, for the reason given
(`tests/test_remi_prompt_tools.py` checks that every old setting is either a
setting here or a row of this table):

| Setting | Why |
|---|---|
| `tac_slidingPopup` | The list always opens at the word being completed and follows the caret (the old default); a list pinned under the field would cover the next lines of a multi-line editor. |
| `tac_showAllResults` | Tags are ranked on the server (match, post count, your usage) and "Suggestions shown" goes up to 50; a whole tag list (over 100,000 rows) in a popup is not readable and costs typing latency. LoRA, embedding, chant and wildcard lists are longer than that limit already. |
| `tac_resultStepLength` | Only paged the "show all results" list. |
| `tac_useIndexedSearch` | The search always runs on the server's index of the whole list (`remi/prompt_tools/tags.py`, under 1.5 ms per lookup); the old full-scan mode worked around results missing from the browser-side index, which this search does not have. |
| `tac_skipWildcardRefresh` | Wildcard folders are rescanned in the background (never while you type, `WildcardCache`), so a large collection cannot hang the page. |
| `tac_forceRefreshEmbeddings` | Embeddings come from the live network index (the Networks page's), so new ones appear after its refresh without a forced reload. |
| `tac_refreshTempFiles` | There are no temp files: the browser does not read tag lists through `file=` URLs any more. The refresh buttons next to the tag list settings reload the lists, wildcards and chants (`POST /api/v2/prompt-tools/refresh`). |
| `tac_useLycos` | LyCORIS / LoHa files are loaded by Forge's LoRA network and listed with the LoRAs: "Complete LoRA names" covers them. |
| `tac_useLoraPrefixForLycos` | Always `<lora:`, the only prefix Forge understands (it has no `<lyco:` network). |
| `tac_modelSortOrder` | Completions are ordered by how well the name matches what you type, then by your usage; browsing by name or date is the Networks page's, which has those sort orders. |
| `tac_useStyleVars` | `$style` names only did something with the separate style-vars extension, which expands them when generating; Remi has no such expansion, so an inserted `$name` would go into the prompt as text. Styles are applied from the styles menu of the prompt. |
| `tac_modelKeywordCivitai` | Completion sends nothing to third-party sites (this sent the hash of every LoRA you completed to CivitAI). Activation text comes from the LoRA's metadata, editable in Networks → Edit metadata. |
| `tac_civitaiApiKey` | Only used by the CivitAI lookups above. |
| `tac_translation.oldFormat` | Detected per file: `tag,translation` and the old `tag,category,translation` both load. |
| `tac_translation.liveTranslation` | The tag view of the prompt tools bar shows every tag with its translation (and translates unknown ones on request); it replaces this experimental line under the prompt. |
| `tac_keymap` | The list uses the prompt editor's keys, shared by every completion source: ↑/↓, Page Up/Down, Enter or Tab accept, Esc closes, Ctrl+Space opens; Home/End move the caret as in any text field. Other bindings would collide with the editor's own shortcuts (Ctrl+↑/↓ weight, Alt+←/→ move a tag, Ctrl+Enter generates). |
| `tac_colormap` | Category colours are theme colours worked out for contrast in both the light and the dark theme (`webui/src/features/prompt-tools/completion/colors.ts`); the old named colours (lightblue, gold, whitesmoke, …) fail contrast in one of the two. |

Based on [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)
by DominikDoom and its Forge Neo fork by eduardoabreu81 (MIT, see LICENSE);
the tag lists come from those projects.
