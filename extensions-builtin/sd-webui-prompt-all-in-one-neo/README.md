# Prompt Tools (Remi)

The prompt helpers of the Remi WebUI, successor of Prompt All-in-One: a bar
under every prompt editor with

- a tag view of the prompt (chips coloured by category with translations;
  drag or Alt+←/→ to reorder, click to edit text and weight, disable a tag
  as a prompt comment, delete);
- quick tag groups (`group_tags/<locale>.yaml`; a non-empty `custom.yaml`
  replaces them, `prepend.yaml` / `append.yaml` add categories);
- the prompt history (every queued generation, searchable, nameable);
- favorites in collections (export / import as JSON);
- translation through a provider you configure (Settings → Prompt Tools):
  an offline dictionary, LibreTranslate, an OpenAI-compatible API (a local
  model works) or DeepL. Nothing is sent anywhere until you choose one;
- with the OpenAI-compatible provider, **Write a prompt**: describe the image
  in any language and the model writes the prompt (or changes the current
  one); you edit the answer, then replace or extend the prompt. The
  instructions the model gets can be replaced in Settings → Prompt Tools;
- **Tidy up the separators**: one ", " between tags, line breaks kept,
  stray and repeated commas removed (weights and networks untouched).

This extension contributes the tag groups and the `prompt_tools_*` settings;
the tools live in `webui/src/features/prompt-tools` and `remi/prompt_tools`.
History and favorites of the old extension are imported once.

## What became of the old extension's features

Prompt All-in-One was a Vue app mounted over the Gradio prompt boxes with
its own HTTP API (`/physton_prompt/*`, 46 routes). The API is not kept (its
clients get `410 Gone`, see `docs/remi-webui/architecture.md`); the SPA talks
to `/api/v2/prompt-tools`. Feature by feature:

| Old feature | In Remi |
|---|---|
| Tag view of the prompt: drag to reorder, edit, weight, disable, delete, translation per tag | Kept: the tag view of the tools bar (`chips/`). |
| History, favorites (`history.vue`, `favorite.vue`) | Kept, with collections and JSON export / import; the old data is imported once. |
| Quick tag groups (`group_tags/*.yaml`) | Kept, same files (`custom.yaml`, `prepend.yaml`, `append.yaml`). |
| "ChatGPT" prompt writing (`chatgptPrompt.vue`, `gen_openai.py`) | Kept as **Write a prompt**, through the OpenAI-compatible translation provider (OpenAI, DeepSeek, Qwen and other compatible services, or a local Ollama / LM Studio / llama.cpp server); the old preset message is the editable `prompt_tools_generate_instructions`. |
| Translation with 21 backends (`translator/*.py`) | Merged into four providers: DeepL, an OpenAI-compatible API (which covers the LLM backends and most Chinese cloud models), LibreTranslate and the offline dictionary. Not carried over: Google, Bing, Baidu, Youdao, Alibaba, Tencent, Microsoft, Amazon, Caiyun, iFlytek, Niutrans, Volcengine, Yandex and MyMemory each needed their own request signing and keys (code that broke whenever a vendor changed its API), and the keyless "translators" backend scraped unofficial web endpoints, sending prompts to third parties without an agreement. |
| mbart50 offline translation (`mbart50.py`) | Not carried over: it downloaded a 2.4 GB model and ran it in the WebUI's process next to the image model. Offline translation is the dictionary (tag lists, tag groups) or a self-hosted LibreTranslate / local LLM. |
| Automatic translation while typing (`autoTranslate*` options) | Not carried over: it sent the prompt to a network service on every edit. Translation runs when you ask ("Write the prompt in English", "Show the tags in …"); known tags never leave the machine. |
| Translation blacklist ("forbidden words") | Not carried over: "Write the prompt in English" only sends tags that are not English, and "Show the tags in …" only displays translations, so there is no translated text in the prompt to protect. |
| Keyword blacklist (`blacklist.vue`: prompt, negative, LoRA, LyCORIS, embedding lists filtered automatically) | Merged into the **Prompt Blacklist** extension (`sd-forge-prompt-blacklist`), which strips listed tags from the positive prompt when you press its button. Silent automatic filtering, and filtering the negative prompt and networks, are not carried over: a prompt that changes by itself is hard to debug, and the negative prompt is where such tags usually belong. |
| Quality presets (`qualityPresets.vue`, `get_quality_presets.py`) | Merged into the core **Styles** (a style per model family, positive and negative quality tags, applied from the prompt's styles menu) and the **Prompt Anchor** extension (a block always prepended at generation). Not carried over: detecting the checkpoint's family by sending its SHA-256 to CivitAI (and the `paio_neo_civitai_api_key` setting it needed), and the built-in table of quality tags per base model, which goes stale as models change. |
| Prompt formatting options (`promptFormat.vue`) | Merged into **Tidy up the separators** and the tag view, which keeps your text as written (`serialize(parse(text)) === text`, an edit rewrites only its tag), so there is no output format to choose. Not carried over: commas without spaces (saves no tokens: CLIP's tokenizer splits them either way), NovelAI `{}` weights (Forge reads `{}` as text, not as emphasis), automatic line breaks around `BREAK` and comma removal around LoRAs (Forge ignores both). |
| Hotkeys for click / right-click / double-click / hover on a tag (`hotkey.vue`) | Not carried over: one interaction for everyone, usable from the keyboard: click (or Enter) opens the tag's editor (text, weight, disable, favorite, delete), Delete removes, Alt+←/→ moves, Ctrl+Z undoes. Remappable mouse actions would leave the remapped action without a keyboard path. |
| Keyword highlighting (`highlightPrompt.vue`) | Merged into the core prompt editor's syntax colouring (weights, networks, wildcards, comments, unbalanced brackets). |
| Token counter (`/token_counter`) | Merged into the core prompt field's token counter. |
| Extra networks popup (`extraNetworksPopup.vue`) | Merged into the core Networks page and `<` completion (Tag Autocomplete). |
| Tag CSV lists for its own suggestions (`csv.py`, `/get_csvs`) | Merged into Tag Autocomplete, which reads the same CSV files. |
| Themes and extension CSS (`extensionCss.vue`, `styles/`), language picker (`selectLanguage.vue`) | Merged into the WebUI's light / dark themes and UI language. |
| Installing Python packages at runtime (`packagesState.vue`, `/install_package`) | Not carried over: the WebUI never installs packages from a browser request; the providers above need none. |

## Settings of the old extension

### Not carried over

The old extension's only setting has no equivalent, for the reason given
(`tests/test_remi_extension_settings.py` checks that every old setting is either a
setting here or a row of this table):

| Setting | Why |
|---|---|
| `paio_neo_civitai_api_key` | Only used by the quality presets' checkpoint detection, which sent the checkpoint's SHA-256 to CivitAI (see "Quality presets" above). Quality tags per model family are core Styles and Prompt Anchor blocks now, chosen by you, so nothing is sent to a third-party site and no key is needed. |

Based on [sd-webui-prompt-all-in-one](https://github.com/Physton/sd-webui-prompt-all-in-one)
by Physton and its Forge Neo fork by eduardoabreu81 (MIT, see LICENSE); the
tag groups come from those projects.
