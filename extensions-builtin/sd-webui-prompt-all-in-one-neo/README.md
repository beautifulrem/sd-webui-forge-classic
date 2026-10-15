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
  model works) or DeepL. Nothing is sent anywhere until you choose one.

This extension contributes the tag groups and the `prompt_tools_*` settings;
the tools live in `webui/src/features/prompt-tools` and `remi/prompt_tools`.
History and favorites of the old extension are imported once.

Based on [sd-webui-prompt-all-in-one](https://github.com/Physton/sd-webui-prompt-all-in-one)
by Physton and its Forge Neo fork by eduardoabreu81 (MIT, see LICENSE); the
tag groups come from those projects.
