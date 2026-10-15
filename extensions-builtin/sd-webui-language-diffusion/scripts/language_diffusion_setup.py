"""
sd-webui-language-diffusion — the bundled UI translations.

The ten ``localizations/<locale>.json`` catalogs (English text → translation)
are picked up like every extension's: ``/api/v2/i18n/<locale>`` serves each
merged over the core catalog, and the SPA's language switcher (header →
Appearance) lists them by their own names. This script only

1. presents the ``localization`` setting (the server's default language, used
   by browsers that have not picked one) as "Language" with each choice in
   its own language ("English" for the source strings, persisted as "None");
2. re-scans the localization files once the extensions are loaded, so the
   catalogs are known whatever the start order;
3. logs the active language at start and when it changes.

The Gradio-era top-bar pin, flag icons (``javascript/``) and hint patches are
gone: the SPA has its own language switcher and translates everything
through the catalogs.
"""

from modules import shared

# Each language in its own language (autoglottonyms); "English" stands for
# the source strings and is itself translated by the catalogs.
LOCALE_DISPLAY_NAMES = {
    "None": "English",
    "it_IT": "Italiano",
    "es_ES": "Español",
    "fr_FR": "Français",
    "de_DE": "Deutsch",
    "zh_CN": "简体中文",
    "ja_JP": "日本語",
    "pt_BR": "Português (Brasil)",
    "ru_RU": "Русский",
    "ko_KR": "한국어",
    "pl_PL": "Polski",
}


def _autoglottonym_choices(choices):
    """``"it_IT"`` → ``("Italiano", "it_IT")``; (label, value) pairs and
    unknown codes are kept."""
    out = []
    for choice in choices:
        if isinstance(choice, tuple):
            out.append(choice)
        else:
            out.append((LOCALE_DISPLAY_NAMES.get(choice, choice), choice))
    return out


def present_localization_setting() -> None:
    """Label the ``localization`` setting "Language" and show its choices in
    their own languages. The stored values ("None", "it_IT", …) are
    unchanged. Idempotent; a no-op where the setting does not exist."""
    info = shared.opts.data_labels.get("localization")
    if info is None:
        return
    info.label = "Language"
    original = info.component_args
    if getattr(original, "_language_diffusion_patched", False):
        return

    if callable(original):

        def patched_args():
            args = dict(original())  # a copy: never mutate cached state
            args["choices"] = _autoglottonym_choices(args.get("choices", []))
            return args

    elif isinstance(original, dict):
        fixed = {**original, "choices": _autoglottonym_choices(original.get("choices", []))}

        def patched_args(_fixed=fixed):
            return _fixed

    else:
        return
    patched_args._language_diffusion_patched = True
    info.component_args = patched_args


def force_localization_rescan() -> None:
    """Re-scan the localization files (core + extensions) now that every
    extension is listed: the first scan may run before them."""
    try:
        from modules import localization
        from modules.shared_cmd_options import cmd_opts

        localization.list_localizations(cmd_opts.localizations_dir)
    except Exception:
        pass


# ANSI colour of the language in the console lines (bright green).
_BANNER_LANG_COLOR = "\033[92m"
_BANNER_COLOR_RESET = "\033[0m"

# Last-seen default language, so a change logs "old -> new".
_last_localization = None


def _lang_label(code: str) -> str:
    """'Italiano (it_IT)', or 'English (source strings)'."""
    if not code or code == "None":
        return "English (source strings)"
    return f"{LOCALE_DISPLAY_NAMES.get(code, code)} ({code})"


def _print(line: str, ascii_line: str) -> None:
    try:
        print(line)
    except UnicodeEncodeError:  # a legacy console (cp1252): native names dropped
        print(ascii_line)


def _on_localization_change() -> None:
    global _last_localization
    try:
        new_code = getattr(shared.opts, "localization", None) or "None"
        if new_code == _last_localization:
            return
        old = _last_localization or "None"
        _print(
            f"[Language Diffusion] Default UI language changed: {_lang_label(old)} → {_BANNER_LANG_COLOR}{_lang_label(new_code)}{_BANNER_COLOR_RESET}",
            f"[Language Diffusion] Default UI language changed: {old} -> {_BANNER_LANG_COLOR}{new_code}{_BANNER_COLOR_RESET}",
        )
        _last_localization = new_code
    except Exception:
        pass


def log_startup_state() -> None:
    """One console line: the extension is loaded, the default language and
    how many locales the catalogs cover. Never affects startup."""
    global _last_localization
    try:
        code = getattr(shared.opts, "localization", None) or "None"
        _last_localization = code
        try:
            from modules import localization

            count = len(localization.localizations)
        except Exception:
            count = 0
        tail = f"  ·  {count} locales available" if count else ""
        _print(
            f"[Language Diffusion] loaded — default UI language: {_BANNER_LANG_COLOR}{_lang_label(code)}{_BANNER_COLOR_RESET}{tail}",
            f"[Language Diffusion] loaded -- default UI language: {_BANNER_LANG_COLOR}{'English' if code == 'None' else code}{_BANNER_COLOR_RESET}" + (f" ({count} locales available)" if count else ""),
        )
    except Exception:
        pass


def setup() -> None:
    present_localization_setting()
    force_localization_rescan()
    try:
        shared.opts.onchange("localization", _on_localization_change, call=False)
    except Exception:
        pass
    log_startup_state()


setup()
