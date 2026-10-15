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

Usage counts of the old extension (`tag_frequency.db`) are imported once.

Based on [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)
by DominikDoom and its Forge Neo fork by eduardoabreu81 (MIT, see LICENSE);
the tag lists come from those projects.
