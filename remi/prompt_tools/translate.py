"""Prompt translation through a provider the user configured explicitly.

Nothing is sent anywhere unless ``prompt_tools_translation_provider`` names
a network provider and its URL/key are set. Providers:

* ``dictionary`` — offline: the tag translation file and the tag groups of
  the target language (both directions, exact tags only);
* ``libretranslate`` — a LibreTranslate server (self-hostable, so it can be
  offline too): ``POST {url}/translate``;
* ``openai`` — any OpenAI-compatible chat completions endpoint (a local
  Ollama / LM Studio / llama.cpp server works);
* ``deepl`` — the DeepL API.

Network results are cached in the store; the dictionary always answers
first, so known tags never leave the machine.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

PROVIDERS = ("none", "dictionary", "libretranslate", "openai", "deepl")
NETWORK_PROVIDERS = ("libretranslate", "openai", "deepl")
MAX_TEXTS = 200
MAX_CHARS = 20_000
TIMEOUT = (5, 60)
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


class TranslationError(RuntimeError):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "none"
    url: str = ""
    key: str = ""
    model: str = ""

    @property
    def enabled(self) -> bool:
        if self.provider == "dictionary":
            return True
        if self.provider == "libretranslate":
            return bool(self.url.strip())
        if self.provider == "openai":
            return bool(self.url.strip() and self.model.strip())
        if self.provider == "deepl":
            return bool(self.key.strip())
        return False

    @property
    def network(self) -> bool:
        return self.provider in NETWORK_PROVIDERS and self.enabled

    @property
    def cache_key(self) -> str:
        return f"{self.provider}:{self.url.strip()}:{self.model.strip()}"


def language_name(code: str) -> str:
    return {"en": "English", "zh": "Simplified Chinese", "zh_CN": "Simplified Chinese", "zh_TW": "Traditional Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian", "de": "German", "fr": "French", "es": "Spanish"}.get(code, code)


def has_cjk(text: str) -> bool:
    return CJK_RE.search(text) is not None


Post = Callable[[str, Mapping[str, str], Any], Any]
"""``(url, headers, json body) -> decoded JSON`` (tests replace it)."""


def _http_post(url: str, headers: Mapping[str, str], body: Any) -> Any:
    import requests

    try:
        response = requests.post(url, headers=dict(headers), json=body, timeout=TIMEOUT)
    except requests.RequestException as error:
        raise TranslationError(f"translation service unreachable: {error}") from error
    if response.status_code >= 400:
        raise TranslationError(f"translation service answered {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as error:
        raise TranslationError("translation service sent no JSON") from error


def _libretranslate(config: ProviderConfig, texts: Sequence[str], target: str, post: Post) -> list[str]:
    body: dict[str, Any] = {"q": list(texts), "source": "auto", "target": target.split("_")[0] if target != "zh_TW" else "zt", "format": "text"}
    if config.key.strip():
        body["api_key"] = config.key.strip()
    data = post(config.url.rstrip("/") + "/translate", {"Content-Type": "application/json"}, body)
    translated = data.get("translatedText") if isinstance(data, dict) else None
    if isinstance(translated, str):
        translated = [translated]
    if not isinstance(translated, list) or len(translated) != len(texts):
        raise TranslationError("unexpected LibreTranslate answer")
    return [str(item) for item in translated]


def _openai(config: ProviderConfig, texts: Sequence[str], target: str, post: Post) -> list[str]:
    instructions = (
        f"Translate each item of the JSON array into {language_name(target)}. The items are Stable Diffusion prompt tags or phrases: "
        "keep them short, keep booru tag conventions, keep any text that is already in the target language unchanged, "
        "and answer with only a JSON array of strings of the same length and order."
    )
    headers = {"Content-Type": "application/json"}
    if config.key.strip():
        headers["Authorization"] = f"Bearer {config.key.strip()}"
    body = {"model": config.model.strip(), "temperature": 0, "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": json.dumps(list(texts), ensure_ascii=False)}]}
    base = config.url.rstrip("/")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    data = post(url, headers, body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise TranslationError("unexpected chat completion answer") from error
    match = re.search(r"\[.*\]", str(content), re.DOTALL)
    try:
        translated = json.loads(match.group(0)) if match else None
    except ValueError:
        translated = None
    if not isinstance(translated, list) or len(translated) != len(texts):
        raise TranslationError("the model did not answer with a list of the same length")
    return [str(item).strip() for item in translated]


def _deepl(config: ProviderConfig, texts: Sequence[str], target: str, post: Post) -> list[str]:
    key = config.key.strip()
    base = config.url.strip().rstrip("/") or ("https://api-free.deepl.com" if key.endswith(":fx") else "https://api.deepl.com")
    target_lang = {"en": "EN-US", "zh_CN": "ZH-HANS", "zh": "ZH-HANS", "zh_TW": "ZH-HANT"}.get(target, target.split("_")[0].upper())
    data = post(f"{base}/v2/translate", {"Authorization": f"DeepL-Auth-Key {key}", "Content-Type": "application/json"}, {"text": list(texts), "target_lang": target_lang})
    translations = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(translations, list) or len(translations) != len(texts):
        raise TranslationError("unexpected DeepL answer")
    return [str(item.get("text", "")) for item in translations]


NETWORK = {"libretranslate": _libretranslate, "openai": _openai, "deepl": _deepl}

# region Prompt writing (Prompt All-in-One's "ChatGPT prompt")

MAX_DESCRIPTION = 4000

DEFAULT_INSTRUCTIONS = (
    "You write prompts for Stable Diffusion image models. Turn the user's description into one prompt: comma-separated "
    "English booru-style tags and short phrases, the most important first (subject, appearance, clothing, pose, setting, "
    "lighting, style), with (tag:1.2) weights only where emphasis matters. Answer with the prompt only, on one line, "
    "without explanations or quotes."
)
NEGATIVE_INSTRUCTIONS = (
    "You write negative prompts for Stable Diffusion image models: comma-separated English tags of what the image must "
    "not show. Answer with the negative prompt only, on one line, without explanations or quotes."
)


def _clean_prompt(text: str) -> str:
    """The model's answer as a prompt: code fences, a "Prompt:" label and
    enclosing quotes dropped, lines joined with commas."""

    text = re.sub(r"^```[\w-]*\s*|\s*```$", "", text.strip())
    text = re.sub(r"^(?:negative\s+)?prompt\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and (text[0], text[-1]) in (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」")):
        text = text[1:-1]
    lines = [line.strip().strip(",").strip() for line in text.splitlines()]
    return ", ".join(line for line in lines if line)


def generate_prompt(config: ProviderConfig, description: str, *, current: str = "", negative: bool = False, instructions: str = "", post: Post = _http_post) -> str:
    """A prompt written by the OpenAI-compatible model of ``config`` (the
    translation provider) from ``description``; ``current``: the prompt to
    build on. ``instructions`` replaces the built-in system message."""

    if config.provider != "openai" or not config.enabled:
        raise TranslationError("writing prompts needs the OpenAI-compatible translation provider with its URL and model", 409)
    description = description.strip()
    if not description:
        raise TranslationError("describe the image first", 422)
    if len(description) + len(current) > MAX_DESCRIPTION:
        raise TranslationError("the description is too long", 413)
    system = instructions.strip() or (NEGATIVE_INSTRUCTIONS if negative else DEFAULT_INSTRUCTIONS)
    request = description if not current.strip() else f"Current prompt: {current.strip()}\n\nChange it as follows: {description}"
    headers = {"Content-Type": "application/json"}
    if config.key.strip():
        headers["Authorization"] = f"Bearer {config.key.strip()}"
    body = {"model": config.model.strip(), "temperature": 0.7, "messages": [{"role": "system", "content": system}, {"role": "user", "content": request}]}
    base = config.url.rstrip("/")
    data = post(base if base.endswith("/chat/completions") else f"{base}/chat/completions", headers, body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise TranslationError("unexpected chat completion answer") from error
    prompt = _clean_prompt(str(content or ""))
    if not prompt:
        raise TranslationError("the model answered with an empty prompt")
    return prompt


# endregion


def dictionary_translate(texts: Sequence[str], forward: Mapping[str, str], target: str) -> dict[str, str]:
    """Exact-tag translations from ``forward`` (``tag → label``): English
    tags to the label, or (target ``en``) labels back to their tag."""

    if target == "en":
        backward: dict[str, str] = {}
        for tag, label in forward.items():
            backward.setdefault(label.strip().lower(), tag.replace("_", " "))
        return {text: backward[key] for text in texts if (key := text.strip().lower()) in backward}
    return {text: forward[key] for text in texts if (key := re.sub(r"\s+", "_", text.strip().lower())) in forward}


def translate(
    texts: Sequence[str],
    target: str,
    config: ProviderConfig,
    dictionary: Mapping[str, str],
    *,
    cached: Callable[[list[str]], Mapping[str, str]] = lambda texts: {},
    store: Callable[[Mapping[str, str]], None] = lambda pairs: None,
    post: Post = _http_post,
) -> list[str | None]:
    """One translation (or None) per text: the dictionary first, then the
    cache, then the configured network provider (in one batch)."""

    if not config.enabled:
        raise TranslationError("no translation provider is configured", 409)
    texts = [str(text) for text in texts][:MAX_TEXTS]
    if sum(len(text) for text in texts) > MAX_CHARS:
        raise TranslationError("too much text to translate at once", 413)
    result: dict[str, str] = dictionary_translate(texts, dictionary, target)
    if config.network:
        missing = [text for text in dict.fromkeys(texts) if text.strip() and text not in result]
        hits = cached(missing) if missing else {}
        result.update(hits)
        missing = [text for text in missing if text not in hits]
        if missing:
            translated = NETWORK[config.provider](config, missing, target, post)
            # unchanged texts are cached too (already in the target language)
            fresh = {source: value for source, value in zip(missing, translated, strict=False) if value.strip()}
            store(fresh)
            result.update(fresh)
    return [value if (value := result.get(text)) and value != text else None for text in texts]
