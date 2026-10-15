"""Settings of the prompt tools (``shared.opts``): tag autocompletion keeps
the ``tac_*`` keys of the tag autocomplete extension (saved values carry
over; its dotted keys, which settings names cannot have, are renamed with
``_`` and their values moved), the other tools use ``prompt_tools_*``. The extensions register them
(:func:`register_tag_options`, :func:`register_tool_options`); everything
else reads them through :func:`option` with these defaults."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

TAG_SECTION = ("tac", "Tag Autocomplete", "ui")
TOOLS_SECTION = ("prompt_tools", "Prompt Tools", "ui")

FREQUENCY_FUNCTIONS = ("Logarithmic (weak)", "Logarithmic (strong)", "Usage first")
WILDCARD_MODES = ("To next folder level", "To first difference", "Always fully")
KEYWORD_LOCATIONS = ("Start of prompt", "End of prompt", "Before LORA/LyCO", "After LORA/LyCO")
TRANSLATION_PROVIDERS = (("None", "none"), ("Offline dictionary", "dictionary"), ("LibreTranslate", "libretranslate"), ("OpenAI-compatible API", "openai"), ("DeepL", "deepl"))
TRANSLATION_PROVIDER_IDS = tuple(value for _, value in TRANSLATION_PROVIDERS)
TOOLBAR_SCOPES = ("All prompt fields", "Main prompts only")
"""``prompt_tools_toolbar``: where the prompt tools bar is shown"""
TRANSLATION_LANGUAGES = (("Simplified Chinese", "zh_CN"), ("Traditional Chinese", "zh_TW"), ("Japanese", "ja"), ("Korean", "ko"), ("Russian", "ru"), ("German", "de"), ("French", "fr"), ("Spanish", "es"))
TAG_GROUPS_TRANSLATION = "Tag groups"
"""a translation file choice: the labels of the quick tag groups"""
UNDERSCORE_EXCLUSIONS = "0_0,(o)_(o),+_+,+_-,._.,<o>_<o>,<|>_<|>,=_=,>_<,3_3,6_9,>_o,@_@,^_^,o_o,u_u,x_x,|_|,||_||"

DEFAULTS: dict[str, Any] = {
    "tac_active": True,
    "tac_activeIn_negativePrompts": True,
    "tac_tagFile": "danbooru.csv",
    "tac_extra_extraFile": "extra-quality-tags.csv",
    "tac_extra_addMode": "Insert before",
    "tac_translation_translationFile": "None",
    "tac_translation_searchByTranslation": True,
    "tac_alias_searchByAlias": True,
    "tac_maxResults": 12,
    "tac_useWildcards": True,
    "tac_sortWildcardResults": True,
    "tac_wildcardExclusionList": "",
    "tac_wildcardCompletionMode": "To next folder level",
    "tac_useEmbeddings": True,
    "tac_includeEmbeddingsInNormalResults": False,
    "tac_useLoras": True,
    "tac_chantFile": "demo-chants.json",
    "tac_showExtraNetworkPreviews": True,
    "tac_showWikiLinks": False,
    "tac_frequencySort": True,
    "tac_frequencyFunction": "Logarithmic (weak)",
    "tac_frequencyMinCount": 3,
    "tac_frequencyMaxAge": 30,
    "tac_frequencyRecommendCap": 10,
    "tac_replaceUnderscores": True,
    "tac_undersocreReplacementExclusionList": UNDERSCORE_EXCLUSIONS,
    "tac_escapeParentheses": True,
    "tac_appendComma": True,
    "tac_appendSpace": True,
    "tac_alwaysSpaceAtEnd": True,
    "tac_modelKeywordCompletion": "Never",
    "tac_modelKeywordLocation": "Start of prompt",
    "tac_animaArtistPrefix": "Off",
    "prompt_tools_history": True,
    "prompt_tools_history_limit": 1000,
    "prompt_tools_translation_provider": "none",
    "prompt_tools_toolbar": "All prompt fields",
    "prompt_tools_translation_url": "",
    "prompt_tools_translation_model": "",
    "prompt_tools_translation_language": "zh_CN",
}

TAG_OPTION_KEYS = tuple(key for key in DEFAULTS if key.startswith("tac_"))
TOOL_OPTION_KEYS = tuple(key for key in DEFAULTS if key.startswith("prompt_tools_"))


def option(name: str, default: Any = None) -> Any:
    """A setting's value (its default here when unset or unregistered)."""

    fallback = DEFAULTS.get(name, default)
    try:
        from modules import shared

        value = getattr(shared.opts, name, fallback)
    except Exception:
        return fallback
    return fallback if value is None else value


def _files(directory: Callable[[], str | None], suffix: str) -> list[str]:
    root = directory()
    if not root or not os.path.isdir(root):
        return []
    return sorted(name for name in os.listdir(root) if name.lower().endswith(suffix) and os.path.isfile(os.path.join(root, name)))


def tag_options(tag_dir: Callable[[], str | None], refresh: Callable[[], None]) -> dict[str, Any]:
    from modules.options import OptionInfo, options_section
    from remi import ui

    csvs = lambda: ["None", *_files(tag_dir, ".csv")]  # noqa: E731
    translation_files = lambda: ["None", TAG_GROUPS_TRANSLATION, *_files(tag_dir, ".csv")]  # noqa: E731
    jsons = lambda: ["None", *_files(tag_dir, ".json")]  # noqa: E731
    d = DEFAULTS
    return options_section(
        TAG_SECTION,
        {
            "tac_active": OptionInfo(d["tac_active"], "Enable tag autocompletion").info("in every prompt field; Tab or Enter accepts, ↑/↓ choose, Esc closes"),
            "tac_activeIn_negativePrompts": OptionInfo(d["tac_activeIn_negativePrompts"], "Complete in negative prompts"),
            "tac_tagFile": OptionInfo(d["tac_tagFile"], "Tag list", ui.Choice, lambda: {"choices": csvs()}, refresh=refresh).info("CSV files in the tags folder of the tag autocomplete extension"),
            "tac_extra_extraFile": OptionInfo(d["tac_extra_extraFile"], "Extra tags", ui.Choice, lambda: {"choices": csvs()}, refresh=refresh).info("a small list of custom tags"),
            "tac_extra_addMode": OptionInfo(d["tac_extra_addMode"], "Extra tags are listed", ui.Choice, {"choices": ("Insert before", "Insert after"), "style": "radio"}),
            "tac_translation_translationFile": OptionInfo(d["tac_translation_translationFile"], "Tag translations", ui.Choice, lambda: {"choices": translation_files()}, refresh=refresh).info("shown next to the tags; \"Tag groups\" uses the labels of the quick tag groups"),
            "tac_translation_searchByTranslation": OptionInfo(d["tac_translation_searchByTranslation"], "Search by translation"),
            "tac_alias_searchByAlias": OptionInfo(d["tac_alias_searchByAlias"], "Search by alias"),
            "tac_maxResults": OptionInfo(d["tac_maxResults"], "Suggestions shown", ui.Slider, {"minimum": 3, "maximum": 50, "step": 1}),
            "tac_useLoras": OptionInfo(d["tac_useLoras"], "Complete LoRA names after \"<\""),
            "tac_useEmbeddings": OptionInfo(d["tac_useEmbeddings"], "Complete embeddings after \"<e:\""),
            "tac_includeEmbeddingsInNormalResults": OptionInfo(d["tac_includeEmbeddingsInNormalResults"], "Also suggest embeddings among tags"),
            "tac_showExtraNetworkPreviews": OptionInfo(d["tac_showExtraNetworkPreviews"], "Show LoRA preview images"),
            "tac_chantFile": OptionInfo(d["tac_chantFile"], "Chants", ui.Choice, lambda: {"choices": jsons()}, refresh=refresh).info("prompt presets completed after \"<c:\""),
            "tac_useWildcards": OptionInfo(d["tac_useWildcards"], "Complete wildcards after \"__\""),
            "tac_sortWildcardResults": OptionInfo(d["tac_sortWildcardResults"], "Sort wildcard values alphabetically").info("off: keep the order of the file"),
            "tac_wildcardExclusionList": OptionInfo(d["tac_wildcardExclusionList"], "Skip wildcard folders").info("folder names, separated by commas"),
            "tac_wildcardCompletionMode": OptionInfo(d["tac_wildcardCompletionMode"], "Tab completes nested wildcards", ui.Choice, {"choices": WILDCARD_MODES}).info("Enter always inserts the whole name"),
            "tac_frequencySort": OptionInfo(d["tac_frequencySort"], "Prefer the tags you use often").info("usage is recorded on this machine"),
            "tac_frequencyFunction": OptionInfo(d["tac_frequencyFunction"], "Usage ranking", ui.Choice, {"choices": FREQUENCY_FUNCTIONS}),
            "tac_frequencyMinCount": OptionInfo(d["tac_frequencyMinCount"], "Uses before a tag counts as frequent", ui.Slider, {"minimum": 1, "maximum": 20, "step": 1}),
            "tac_frequencyMaxAge": OptionInfo(d["tac_frequencyMaxAge"], "Forget uses older than", ui.Slider, {"minimum": 0, "maximum": 365, "step": 1}).info("days; 0 keeps them forever"),
            "tac_frequencyRecommendCap": OptionInfo(d["tac_frequencyRecommendCap"], "Most frequent tags promoted", ui.Slider, {"minimum": 0, "maximum": 50, "step": 1}).info("0: no limit"),
            "tac_replaceUnderscores": OptionInfo(d["tac_replaceUnderscores"], "Insert spaces instead of underscores"),
            "tac_undersocreReplacementExclusionList": OptionInfo(d["tac_undersocreReplacementExclusionList"], "Keep underscores in").info("tags separated by commas (emoticons such as ^_^)"),
            "tac_escapeParentheses": OptionInfo(d["tac_escapeParentheses"], "Escape parentheses in inserted tags"),
            "tac_appendComma": OptionInfo(d["tac_appendComma"], "Append a comma after a tag"),
            "tac_appendSpace": OptionInfo(d["tac_appendSpace"], "Append a space after a tag"),
            "tac_alwaysSpaceAtEnd": OptionInfo(d["tac_alwaysSpaceAtEnd"], "Always append a space at the end of the prompt"),
            "tac_modelKeywordCompletion": OptionInfo(d["tac_modelKeywordCompletion"], "Insert LoRA activation text", ui.Choice, {"choices": ("Never", "Always"), "style": "radio"}).info("from the LoRA's user metadata"),
            "tac_modelKeywordLocation": OptionInfo(d["tac_modelKeywordLocation"], "Activation text goes to", ui.Choice, {"choices": KEYWORD_LOCATIONS}),
            "tac_animaArtistPrefix": OptionInfo(d["tac_animaArtistPrefix"], "Prefix artist tags with \"@\"", ui.Choice, {"choices": ("Off", "Auto", "Always"), "style": "radio"}).info("Auto: with Anima checkpoints (their artist tag convention)"),
            "tac_showWikiLinks": OptionInfo(d["tac_showWikiLinks"], "Link tags to their wiki page").info("opens danbooru or e621 in a new tab"),
        },
    )


def tool_options() -> dict[str, Any]:
    from modules.options import OptionInfo, options_section
    from remi import ui

    d = DEFAULTS
    return options_section(
        TOOLS_SECTION,
        {
            "prompt_tools_history": OptionInfo(d["prompt_tools_history"], "Keep a prompt history").info("the prompts of every queued generation"),
            "prompt_tools_history_limit": OptionInfo(d["prompt_tools_history_limit"], "Prompt history size", ui.Slider, {"minimum": 100, "maximum": 10000, "step": 100}).info("named entries are always kept"),
            "prompt_tools_toolbar": OptionInfo(d["prompt_tools_toolbar"], "Show the prompt tools bar under", ui.Choice, {"choices": TOOLBAR_SCOPES, "style": "radio"}).info(
                "main prompts: the prompt and negative prompt of txt2img and img2img"
            ),
            "prompt_tools_translation_provider": OptionInfo(d["prompt_tools_translation_provider"], "Translation", ui.Choice, {"choices": TRANSLATION_PROVIDERS}).info(
                "nothing is translated until a provider is chosen; the offline dictionary never leaves this machine; an API key is entered in the translate menu of the prompt tools bar (stored on the server, never shown again)"
            ),
            "prompt_tools_translation_language": OptionInfo(d["prompt_tools_translation_language"], "Show translations in", ui.Choice, {"choices": TRANSLATION_LANGUAGES}),
            "prompt_tools_translation_url": OptionInfo(d["prompt_tools_translation_url"], "Translation service URL").info(
                "LibreTranslate: the server, e.g. http://127.0.0.1:5000; OpenAI-compatible: the API base, e.g. http://127.0.0.1:11434/v1; DeepL: optional"
            ),
            "prompt_tools_translation_model": OptionInfo(d["prompt_tools_translation_model"], "Translation model").info("OpenAI-compatible APIs only"),
        },
    )


def _add(options: dict[str, Any]) -> None:
    from modules import shared

    for key, info in options.items():
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)


RENAMED = {old: old.replace(".", "_") for old in ("tac_activeIn.negativePrompts", "tac_extra.extraFile", "tac_extra.addMode", "tac_translation.translationFile", "tac_translation.searchByTranslation", "tac_alias.searchByAlias")}


def migrate_values(data: dict[str, Any]) -> list[str]:
    """Moves saved values of the old dotted keys to their new names (in place);
    returns the moved keys."""

    moved = []
    for old, new in RENAMED.items():
        if old in data:
            value = data.pop(old)
            if new not in data:
                data[new] = value
                moved.append(new)
    return moved


def register_tag_options(tag_dir: Callable[[], str | None], refresh: Callable[[], None]) -> None:
    from modules import shared

    migrate_values(shared.opts.data)
    _add(tag_options(tag_dir, refresh))


LEGACY_TRANSLATION_KEY = "prompt_tools_translation_key"
"""where an earlier version kept the translation API key (plain text in the settings)"""


def take_legacy_translation_key() -> str | None:
    """Removes the API key an earlier version saved in the settings and
    returns it (None when there is none); the settings file is rewritten
    without it."""

    try:
        from modules import shared
    except Exception:
        return None
    data = getattr(shared.opts, "data", None)
    if not isinstance(data, dict) or LEGACY_TRANSLATION_KEY not in data:
        return None
    value = data.pop(LEGACY_TRANSLATION_KEY)
    try:
        shared.opts.save(shared.config_filename)
    except Exception:
        # read-only settings: the key stays in the file but is no longer served
        pass
    return value.strip() if isinstance(value, str) and value.strip() else None


def register_tool_options() -> None:
    _add(tool_options())
