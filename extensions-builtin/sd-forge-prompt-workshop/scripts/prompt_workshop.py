"""
prompt_workshop.py
------------------
The "Prompt Workshop" page (``remi.pages``): structure prompts for the Anima
model (https://huggingface.co/circlestone-labs/Anima), which follows a
specific tag order:

    [quality/meta/year/safety] [1girl/1boy/1other] [character] [series] [artist] [general]

The page has:
  - One text box per tag-order category (plus eight manual extra categories).
  - "Build Prompt", which consolidates them in order into a copyable box,
    and buttons that append it to (or replace) the txt2img prompt.
  - A Booru random tag grabber (Danbooru, Aibooru, Gelbooru, Rule34) that
    fetches a random post (or one by id), normalises its tags into Anima
    format (lowercase, spaces, @ on artists, score_N kept with underscore) and
    either shows the result or pushes the tags into the category fields.
    Booru requests are only made when the user clicks a grab / download
    button.
  - A hand-off of the post image to the JoyCaption page.
"""

import io
import ipaddress
import json
import os
import re
import shutil
import socket
import threading
from typing import Tuple, List, Dict
from urllib.parse import urljoin, urlparse

import requests

from modules import paths_internal
from remi import pages, ui

EXT_TAG = "[prompt-workshop]"

# ----------------------------------------------------------------------------
# Tag normalisation
# ----------------------------------------------------------------------------

# Score tags are the one documented exception to "underscores -> spaces":
# the Anima model card explicitly says "Score tags are the only tags that use
# underscores." So score_1 .. score_9 must keep their underscore.
_SCORE_TAG_RE = re.compile(r"^score_\d+$", re.IGNORECASE)

# Subject-count tags that belong in the second tag-order section. This is the
# common booru vocabulary; the regex catches numeric forms (1girl, 2boys, ...)
# as well as the textual variants. "solo" and "no humans" are borderline but
# they're conventionally grouped with the character-count tags in prompts.
_SUBJECT_TAG_RE = re.compile(
    r"^("
    r"\d+(?:girl|boy|other)s?"
    r"|multiple (?:girls|boys|others)"
    r"|no humans"
    r"|solo"
    r"|solo focus"
    r")$"
)

USER_AGENT = "sd-forge-prompt-workshop/1.0 (https://github.com/)"
MAX_IMAGE_DOWNLOAD_BYTES = 64 * 1024 * 1024
IMAGE_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_IMAGE_REDIRECTS = 5


class _GrabCancelled(BaseException):
    """Raised by :func:`_http_get` once the grab task was cancelled. Not an
    ``Exception``: the grabbers' ``except Exception`` handlers let it through."""


# The grab task running on this thread (see PromptWorkshopPage._start_grab)
_grab_thread = threading.local()


def _http_get(*args, **kwargs):
    """``requests.get`` for the grabbers: stops before each request once the
    user cancelled the grab (hunt mode makes many)."""
    task = getattr(_grab_thread, "task", None)
    if task is not None and task.cancelled:
        raise _GrabCancelled()
    return requests.get(*args, **kwargs)

# Maps the UI checkbox labels for "block these categories from grabs" onto
# the internal bucket keys used inside the grab functions. Single source of
# truth for both the UI and the filter logic so they can't drift.
_CATEGORY_LABEL_TO_KEY = {
    "Quality/Meta/Safety": "quality",
    "Subject":             "subject",
    "Character":           "character",
    "Series":              "series",
    "Artist":              "artist",
    "General":             "general",
}
_CATEGORY_LABELS = list(_CATEGORY_LABEL_TO_KEY.keys())

# Comparators for the "post-count filter" on the random grabber. The user
# picks one of these and a low/high threshold; tags whose booru post_count
# fails the comparison are dropped before being assembled into the prompt.
#
# Each op is (low, high) -> bool. The single-bound ops (≥, ≤, =) ignore
# `high`; the range ops use both. Keeping a uniform 2-arg signature means
# every call site can pass `(low, high)` without branching.
#
# "not between" is the form the user typically wants for omission: e.g.
# "omit tags with count strictly between 50 and 100" -> not between 50 100,
# which keeps everything <=50 or >=100.
_COUNT_OP_CHOICES = ["≥", "≤", "=", "between", "not between"]
_COUNT_OPS = {
    "≥":           lambda c, lo, hi: c >= lo,
    "≤":           lambda c, lo, hi: c <= lo,
    "=":           lambda c, lo, hi: c == lo,
    "between":     lambda c, lo, hi: lo <= c <= hi,
    "not between": lambda c, lo, hi: not (lo <= c <= hi),
}


def _normalize_count_op_args(op_symbol: str,
                             threshold_low: int,
                             threshold_high: int) -> Tuple[int, int]:
    """
    Sort the two thresholds for range ops so the user can type them in either
    order and still get the expected result. No-op for single-bound ops.
    """
    if op_symbol in ("between", "not between"):
        lo, hi = sorted((int(threshold_low or 0), int(threshold_high or 0)))
        return lo, hi
    return int(threshold_low or 0), int(threshold_high or 0)


def _filter_tags_in_buckets(buckets: Dict[str, List[str]],
                            counts: Dict[str, int],
                            op_symbol: str,
                            threshold_low: int,
                            threshold_high: int,
                            target_categories: List[str]) -> Dict[str, List[str]]:
    """
    Apply the post-count filter to a category-keyed bucket dict.

    `target_categories` is a list of internal bucket keys ('artist',
    'general', ...). When non-empty, the filter applies ONLY to those
    buckets; everything else passes through untouched. When empty, the
    filter applies to ALL buckets (legacy behaviour, matches the previous
    single-list signature).

    `counts` is keyed by booru-wire-format tag name (the raw form, with
    underscores) since that's how the booru APIs return tag info. Tags that
    weren't returned by the lookup are treated as count=0 -- this handles
    deprecated/aliased tags gracefully and is the conservative choice (e.g.
    a >=50 filter drops unknown tags, which is what the user wants).

    The Anima-format strings have spaces and backslash-escaped parens; we
    strip the escapes and convert spaces back to underscores for the lookup.
    score_N tags keep their underscore in both forms.
    """
    op = _COUNT_OPS.get(op_symbol)
    if op is None:
        return buckets

    lo, hi = _normalize_count_op_args(op_symbol, threshold_low, threshold_high)
    targets = set(target_categories) if target_categories else None

    def keep(tag: str) -> bool:
        # Convert Anima-form 'mutsuki \(blue archive\)' back to wire-form
        # 'mutsuki_(blue_archive)' for the count dict. Strip paren-escape
        # backslashes first, then swap spaces for underscores. Score tags
        # already have underscores so this is a no-op for them.
        wire = tag.replace("\\(", "(").replace("\\)", ")").replace(" ", "_")
        count = counts.get(wire, 0)
        return op(count, lo, hi)

    out: Dict[str, List[str]] = {}
    for key, tags in buckets.items():
        if targets is None or key in targets:
            out[key] = [t for t in tags if keep(t)]
        else:
            out[key] = list(tags)
    return out

# Persisted booru API credentials live in Forge's user-data directory.
# Gelbooru has required api_key + user_id for ALL API access since mid-2022;
# Rule34 added the same requirement in August 2025. The credentials are free
# for both -- account holders can grab them from their site's account page:
#   Gelbooru: https://gelbooru.com/index.php?page=account&s=options
#   Rule34:   https://api.rule34.xxx/  (account options page)
# Each source's creds are stored under prefixed keys ({source}_api_key,
# {source}_user_id) so adding more credentialled sources later doesn't
# require a config migration.
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_DIR = os.path.join(
    paths_internal.data_path, "extension-data", "sd-forge-prompt-workshop"
)
_CONFIG_PATH = os.path.join(_STATE_DIR, "config.json")
_LEGACY_CONFIG_PATH = os.path.join(_EXT_DIR, "config.json")


def _migrate_legacy_config() -> None:
    if os.path.exists(_CONFIG_PATH) or not os.path.isfile(_LEGACY_CONFIG_PATH):
        return
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        shutil.copy2(_LEGACY_CONFIG_PATH, _CONFIG_PATH)
        os.chmod(_CONFIG_PATH, 0o600)
    except OSError as e:
        print(f"{EXT_TAG} could not migrate legacy config.json: {e}")


_migrate_legacy_config()


def _load_config() -> Dict[str, str]:
    path = _CONFIG_PATH if os.path.isfile(_CONFIG_PATH) else _LEGACY_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"{EXT_TAG} could not read config.json ({e}); using defaults")
    return {}


def _save_config(cfg: Dict[str, str]) -> None:
    try:
        # This file contains API credentials. Create it owner-only instead of
        # relying on the process umask, then also repair an existing file's
        # mode on POSIX systems.
        os.makedirs(_STATE_DIR, exist_ok=True)
        tmp_path = _CONFIG_PATH + ".tmp"
        opener = lambda path, flags: os.open(path, flags, 0o600)
        with open(tmp_path, "w", encoding="utf-8", opener=opener) as f:
            json.dump(cfg, f, indent=2)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, _CONFIG_PATH)
        try:
            os.chmod(_CONFIG_PATH, 0o600)
        except OSError:
            pass
    except Exception as e:
        print(f"{EXT_TAG} could not write config.json: {e}")


def _get_credentials(source_key: str) -> Tuple[str, str]:
    """Return (api_key, user_id) for the given source key (e.g. 'gelbooru',
    'rule34'); ('', '') if not set."""
    cfg = _load_config()
    return (cfg.get(f"{source_key}_api_key", "").strip(),
            cfg.get(f"{source_key}_user_id", "").strip())


# Back-compat shim: existing config.json files use the gelbooru_* keys
# directly, so the helper for Gelbooru is unchanged in behaviour.
def _get_gelbooru_credentials() -> Tuple[str, str]:
    return _get_credentials("gelbooru")


def save_credentials(source_key: str, api_key: str, user_id: str) -> str:
    """Persist credentials for a single source. Returns a status string for
    the UI Markdown box."""
    cfg = _load_config()
    cfg[f"{source_key}_api_key"] = api_key.strip()
    cfg[f"{source_key}_user_id"] = user_id.strip()
    _save_config(cfg)
    pretty = {"gelbooru": "Gelbooru", "rule34": "Rule34"}.get(
        source_key, source_key.capitalize())
    if cfg[f"{source_key}_api_key"] and cfg[f"{source_key}_user_id"]:
        return f"✓ {pretty} credentials saved."
    return (f"{pretty} credentials cleared. {pretty} calls will now fail "
            "with 401.")


def save_gelbooru_credentials(api_key: str, user_id: str) -> str:
    """Back-compat wrapper, kept so existing handlers don't need rewiring."""
    return save_credentials("gelbooru", api_key, user_id)


def save_rule34_credentials(api_key: str, user_id: str) -> str:
    return save_credentials("rule34", api_key, user_id)


# ----------------------------------------------------------------------------
# Booru source registry
# ----------------------------------------------------------------------------
#
# Each source belongs to one of two families:
#
#   danbooru: Danbooru-style JSON API. Posts have separate tag_string_*
#             fields per category, /posts/random.json gives a random post,
#             /tags.json supports rich search filters. Anonymous use is
#             capped at a small number of tags per query (Danbooru: 2;
#             Aibooru: same, since it's a Danbooru fork).
#
#   gelbooru: Gelbooru-style JSON API. Posts have a single flat space-
#             separated `tags` field, so categorising requires a separate
#             s=tag lookup. Random posts are obtained via a two-step
#             count-then-pid dance. Most newer Gelbooru-family sites now
#             require api_key + user_id (Gelbooru since 2022, Rule34 since
#             August 2025).
#
# Adding another booru in the future is mostly a matter of dropping an entry
# in here -- the grab functions take a source name and look everything up.
_BOORU_SOURCES = {
    "Danbooru": {
        "family":    "danbooru",
        "api_base":  "https://danbooru.donmai.us",
        "post_url":  lambda pid: f"https://danbooru.donmai.us/posts/{pid}",
        "image_hosts": ("donmai.us",),  # cdn.donmai.us
        "creds_key": None,            # no credentials needed
        "anon_tag_limit": 2,          # 2 tags including rating filter
    },
    "Aibooru": {
        "family":    "danbooru",
        "api_base":  "https://aibooru.online",
        "post_url":  lambda pid: f"https://aibooru.online/posts/{pid}",
        "image_hosts": ("aibooru.online", "aibooru.download"),  # cdn.aibooru.download
        "creds_key": None,
        "anon_tag_limit": 2,          # Aibooru is a Danbooru fork
    },
    "Gelbooru": {
        "family":    "gelbooru",
        "api_base":  "https://gelbooru.com/index.php",
        "post_url":  lambda pid: f"https://gelbooru.com/index.php?page=post&s=view&id={pid}",
        "image_hosts": ("gelbooru.com",),  # img*.gelbooru.com
        "creds_key": "gelbooru",
        "anon_tag_limit": None,       # no documented limit when authed
    },
    "Rule34": {
        # API host (api.rule34.xxx) differs from the display host (rule34.xxx).
        # API requires api_key + user_id since Aug 2025.
        "family":    "gelbooru",
        "api_base":  "https://api.rule34.xxx/index.php",
        "post_url":  lambda pid: f"https://rule34.xxx/index.php?page=post&s=view&id={pid}",
        "image_hosts": ("rule34.xxx",),  # api-cdn.rule34.xxx, wimg.rule34.xxx
        "creds_key": "rule34",
        "anon_tag_limit": None,
    },
}
_SOURCE_CHOICES = list(_BOORU_SOURCES.keys())


# Danbooru rating field: g = general/safe, s = sensitive, q = questionable, e = explicit
# (modern Danbooru; "s" used to mean "safe" but was repurposed).
_DANBOORU_RATING_MAP = {
    "g": "safe",
    "s": "sensitive",
    "q": "nsfw",
    "e": "explicit",
}

# Gelbooru ratings have both legacy and modern strings flying around.
_GELBOORU_RATING_MAP = {
    "general": "safe",
    "safe": "safe",
    "sensitive": "sensitive",
    "questionable": "nsfw",
    "explicit": "explicit",
}

# Maps the user-facing rating filter dropdown to the actual tag query each
# booru wants. None / "All" means no filter. Both sites have the same 4
# ratings (just different naming conventions), so there are no gaps.
_RATING_CHOICES = ["All ratings", "Safe / General", "Sensitive",
                   "Questionable", "Explicit"]
_DANBOORU_RATING_TAG = {
    "All ratings":     "",
    "Safe / General":  "rating:g",
    "Sensitive":       "rating:s",
    "Questionable":    "rating:q",
    "Explicit":        "rating:e",
}
_GELBOORU_RATING_TAG = {
    "All ratings":     "",
    "Safe / General":  "rating:general",
    "Sensitive":       "rating:sensitive",
    "Questionable":    "rating:questionable",
    "Explicit":        "rating:explicit",
}
# Rule34 uses the older 3-rating system; "Sensitive" and "General" both map
# to the closest Rule34 equivalent. Rule34 is mostly NSFW so the safe/sensitive
# buckets are rarely populated regardless.
_RULE34_RATING_TAG = {
    "All ratings":     "",
    "Safe / General":  "rating:safe",
    "Sensitive":       "rating:questionable",
    "Questionable":    "rating:questionable",
    "Explicit":        "rating:explicit",
}
_RULE34_RATING_MAP = {
    "safe":          "safe",
    "general":       "safe",
    "questionable":  "nsfw",
    "explicit":      "explicit",
}

# Rule34's HTML post page wraps each tag in a <li class="tag-type-X"> where
# X is the category. These map directly onto our internal category keys
# (with a fallback to 'general' for anything unrecognised). Used by the
# HTML-scraping path that recovers tag categories on Rule34, which lacks a
# working batch tag-info JSON endpoint.
_RULE34_HTML_TAG_TYPE = {
    "artist":    "artist",
    "character": "character",
    "copyright": "copyright",   # rule34 names this 'copyright', maps to series
    "metadata":  "metadata",
    "general":   "general",
}

# Pattern: matches `tag-type-{category}` followed (possibly across attribute
# boundaries) by a `tags={name}` URL parameter. DOTALL because the link
# attributes can straddle the class attribute on multi-line markup. Same
# strategy gallery-dl uses for the same problem on the same site.
_RULE34_TAG_TYPE_RE = re.compile(
    r'tag-type-([a-zA-Z]+)[^>]*>.*?[?;&]tags=([^"\'&;<>]+)',
    re.DOTALL,
)

# Gelbooru tag category codes from the s=tag API: 0=general, 1=artist,
# 3=copyright(series), 4=character, 5=meta. (2 is unused/deprecated.)
_GELBOORU_TAG_TYPE = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "metadata",
}


def _normalize_tag(tag: str) -> str:
    """Lowercase, strip, replace underscores with spaces. score_N stays as-is.

    Also escapes literal parentheses with backslashes so they aren't parsed
    as prompt-weighting syntax by Forge / A1111. Booru disambiguators like
    'mutsuki (blue archive)' or 'asura (asurauser)' become
    'mutsuki \\(blue archive\\)' / 'asura \\(asurauser\\)' -- otherwise the
    prompt parser reads the parens as a weight group and silently boosts
    that section, which is never what you want for a name.

    Already-escaped parens (preceded by a backslash) are left alone, so this
    is idempotent if called on a tag that's already been through it.
    """
    tag = tag.strip()
    if not tag:
        return ""
    if _SCORE_TAG_RE.match(tag):
        return tag.lower()
    out = tag.lower().replace("_", " ")
    # Escape any ( or ) that isn't already escaped. Negative-lookbehind keeps
    # this idempotent and safe when the user pastes pre-escaped input.
    out = re.sub(r"(?<!\\)([()])", r"\\\1", out)
    return out


def _clean_comma_list(s: str) -> str:
    """Parse a user-typed comma list, normalise each tag, rejoin with ', '."""
    if not s:
        return ""
    tags = [_normalize_tag(t) for t in s.split(",")]
    return ", ".join(t for t in tags if t)


def _format_artists(artist_str: str) -> str:
    """
    Anima requires artist tags to be prefixed with '@'. Ensure each comma-
    separated artist has exactly one '@' prefix, even if the user typed it
    without one (or typed two by accident).
    """
    if not artist_str.strip():
        return ""
    out = []
    for raw in artist_str.split(","):
        t = _normalize_tag(raw)
        if not t:
            continue
        t = "@" + t.lstrip("@").strip()
        out.append(t)
    return ", ".join(out)


def _split_subject_from_general(general_tags: List[str]) -> Tuple[List[str], List[str]]:
    """Pull subject-count tags (1girl, solo, ...) out of the general bucket."""
    subject, remaining = [], []
    for t in general_tags:
        (subject if _SUBJECT_TAG_RE.match(t) else remaining).append(t)
    return subject, remaining


def _booru_format_tag(tag: str) -> str:
    """
    Convert a tag from Anima format (spaces, lowercase) to the booru wire
    format (underscores, lowercase). Used when sending search tags into the
    API. score_N stays score_N (no spaces to convert anyway). A leading '@'
    is stripped since boorus don't use it.
    """
    t = tag.strip().lower().lstrip("@")
    return t.replace(" ", "_")


def _parse_search_tags(s: str) -> List[str]:
    """
    Split the user's search-tag input into a list of booru-wire-format tags.
    Always splits on commas (never whitespace), so multi-word tags like
    'long hair' stay as a single tag. Empty input -> empty list.
    """
    if not s or not s.strip():
        return []
    return [_booru_format_tag(p) for p in s.split(",") if p.strip()]


def _parse_blacklist_tags(s: str) -> set:
    """
    Parse the blacklist textbox into a set of normalised (Anima-format)
    tags. Always splits on commas so multi-word tags like 'long hair' or
    'looking at viewer' stay intact. Matching against grabbed tags is done
    after normalisation, so blacklisting 'long hair' catches both
    'long_hair' (booru-form) and 'long hair' (Anima-form).
    """
    if not s or not s.strip():
        return set()
    return {_normalize_tag(p) for p in s.split(",") if p.strip()}


def _dedupe_preserving_order(tags_str: str) -> str:
    """
    Given a 'tag1, tag2, tag2, tag3' string, keep only the first occurrence
    of each tag (case-insensitive). Used by build_prompt so duplicates across
    category boxes silently collapse — Anima uses tag dropout so duplicates
    don't strengthen anything, they just waste tokens.
    """
    seen = set()
    out: List[str] = []
    for raw in tags_str.split(","):
        t = raw.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return ", ".join(out)


def _apply_blacklists(
    buckets: Dict[str, List[str]],
    blacklist_tags: set,
    blacklist_categories: List[str],
) -> Dict[str, List[str]]:
    """
    Apply both tag-level and category-level blacklists to a dict of category
    buckets (already normalised). Returns a new dict with the same keys.

    - Tag-level: drops any tag in any bucket whose normalised form is in the
      `blacklist_tags` set.
    - Category-level: empties any bucket whose internal key is in
      `blacklist_categories` (list of values like 'quality', 'artist', etc).
    """
    blocked_keys = set(blacklist_categories)
    out: Dict[str, List[str]] = {}
    for key, tags in buckets.items():
        if key in blocked_keys:
            out[key] = []
            continue
        out[key] = [t for t in tags if t not in blacklist_tags]
    return out


# ----------------------------------------------------------------------------
# Build-prompt action
# ----------------------------------------------------------------------------

def build_prompt(quality, subject, character, series, artist, general,
                 composition="", pose="", clothing="", setting="",
                 lighting="", features="", style="", content=""):
    """Consolidate every category box into a single Anima-formatted prompt.

    The first six arguments are the *core* Anima-tag-order sections and must
    stay in this exact order. The eight that follow are the *manual extra*
    categories: they are appended verbatim AFTER the General section so the
    Anima priority ordering of the core sections is never disturbed. They are
    intentionally NOT wired into the grabber, blacklist, or count filter --
    they're plain comma-list buckets that exist only for manual organisation
    and Build Prompt. Each is run through the same cleaner as General (no @
    prefixing -- only the Artist box gets that). Order here matches the order
    of the accordions in the UI.
    """
    sections = [
        _clean_comma_list(quality),
        _clean_comma_list(subject),
        _clean_comma_list(character),
        _clean_comma_list(series),
        _format_artists(artist),
        _clean_comma_list(general),
        # ---- manual extra categories, appended after General ----
        _clean_comma_list(composition),
        _clean_comma_list(pose),
        _clean_comma_list(clothing),
        _clean_comma_list(setting),
        _clean_comma_list(lighting),
        _clean_comma_list(features),
        _clean_comma_list(style),
        _clean_comma_list(content),
    ]
    combined = ", ".join(s for s in sections if s)
    # Auto-dedupe: if the same tag shows up in two category boxes (or twice
    # in one), keep only the first occurrence. Anima uses random tag dropout
    # during training, so duplicates don't strengthen anything in the model.
    return _dedupe_preserving_order(combined)


# ----------------------------------------------------------------------------
# Danbooru random fetch
# ----------------------------------------------------------------------------

def _format_categorised(
    rating_text: str,
    meta_tags: List[str],
    general_tags: List[str],
    character_tags: List[str],
    series_tags: List[str],
    artist_tags: List[str],
    blacklist_tags: set,
    blacklist_categories: List[str],
    count_filter_args: Dict = None,
) -> Tuple[str, str, str, str, str, str, str]:
    """
    Given pre-categorised tag lists (already normalised), optionally apply
    the post-count filter, then the blacklists, then produce the seven
    return strings used by both grabbers: consolidated + six fields.

    The Quality bucket combines the rating tag + meta tags (highres,
    absurdres, ...), since Anima's tag order treats them as one section.

    `count_filter_args`, when not None, is a dict containing:
        counts:   Dict[str, int]   (wire-format tag name -> post_count)
        op:       str              (comparator symbol from _COUNT_OPS)
        low:      int              (low/single threshold)
        high:     int              (high threshold, only used for ranges)
        targets:  List[str]        (internal bucket keys, e.g. ['artist'];
                                    empty = apply to all)

    Rating text is exempt from the count filter (it isn't a real booru tag
    so its count would be 0 and the user would lose it spuriously). It IS
    affected by category blacklists -- blacklisting 'Quality/Meta/Safety'
    wipes both the rating string and the meta tags.
    """
    # Split the subject-count tags out of general before applying anything.
    subject_tags, remaining_general = _split_subject_from_general(general_tags)

    # Build the bucket dict for filter+blacklist. Rating text is held aside
    # so the count filter doesn't see it (otherwise a "Quality" target with
    # >=50 would drop the rating string for free, which is never wanted).
    buckets: Dict[str, List[str]] = {
        "quality":   list(meta_tags),
        "subject":   subject_tags,
        "character": character_tags,
        "series":    series_tags,
        "artist":    artist_tags,
        "general":   remaining_general,
    }

    # Apply per-category count filter (if enabled). When the targets list is
    # empty the filter is applied to every bucket, matching legacy behaviour.
    if count_filter_args:
        buckets = _filter_tags_in_buckets(
            buckets,
            count_filter_args.get("counts", {}),
            count_filter_args.get("op", ""),
            count_filter_args.get("low", 0),
            count_filter_args.get("high", 0),
            count_filter_args.get("targets", []),
        )

    # Now prepend rating text to the quality bucket. After filter, before
    # blacklist -- a Quality category blacklist will still wipe it.
    if rating_text:
        buckets["quality"] = [rating_text] + buckets["quality"]

    buckets = _apply_blacklists(buckets, blacklist_tags, blacklist_categories)

    # Artist tags get the @ prefix here (after blacklist filtering). This
    # also means that to blacklist an artist tag, the user types the bare
    # name 'foobar', not '@foobar' -- which matches how the artist box
    # accepts input anywhere else in the UI.
    artist_with_at = ["@" + t.lstrip("@") for t in buckets["artist"]]

    quality_str   = ", ".join(buckets["quality"])
    subject_str   = ", ".join(buckets["subject"])
    character_str = ", ".join(buckets["character"])
    series_str    = ", ".join(buckets["series"])
    artist_str    = ", ".join(artist_with_at)
    general_str   = ", ".join(buckets["general"])

    consolidated = ", ".join(
        s for s in (quality_str, subject_str, character_str,
                    series_str, artist_str, general_str) if s
    )
    return (consolidated, quality_str, subject_str, character_str,
            series_str, artist_str, general_str)


def _parse_danbooru_family_post(source: str,
                                post: dict,
                                blacklist_tags: set,
                                blacklist_categories: List[str],
                                count_filter_enabled: bool = False,
                                count_op: str = "≥",
                                count_threshold_low: int = 0,
                                count_threshold_high: int = 0,
                                count_target_categories: List[str] = None):
    """
    Convert a Danbooru-family (Danbooru, Aibooru) post JSON into the
    (consolidated, q, s, c, ser, a, g, info_line, post_url) tuple. Shared
    between random-grab and grab-by-id so they produce identical output.

    `source` is a key into _BOORU_SOURCES; used for the post URL and the
    /tags.json lookup base.

    If `count_filter_enabled`, also drops any tag whose post_count fails
    the comparator. Looks counts up via a single batch /tags.json call to
    the appropriate API base. The filter can be limited to specific
    `count_target_categories` (internal keys like 'artist'); empty list
    means apply to all.
    """
    cfg = _BOORU_SOURCES[source]
    api_base = cfg["api_base"]

    general_tags   = [_normalize_tag(t) for t in post.get("tag_string_general",   "").split() if t]
    character_tags = [_normalize_tag(t) for t in post.get("tag_string_character", "").split() if t]
    series_tags    = [_normalize_tag(t) for t in post.get("tag_string_copyright", "").split() if t]
    artist_tags    = [_normalize_tag(t) for t in post.get("tag_string_artist",    "").split() if t]
    meta_tags      = [_normalize_tag(t) for t in post.get("tag_string_meta",      "").split() if t]

    rating = post.get("rating", "")
    rating_text = _DANBOORU_RATING_MAP.get(rating, "")

    post_id  = post.get("id", "?")
    post_url = cfg["post_url"](post_id)
    image_url = (post.get("file_url") or post.get("large_file_url")
                 or post.get("preview_file_url") or "")
    info_line = f"{source} post #{post_id} → {post_url}"
    if image_url:
        info_line += f"\nImage: {image_url}"

    count_filter_args = None
    if count_filter_enabled:
        # Look up counts for every booru tag this post has, in one batch.
        wire_tags = []
        for field in ("tag_string_general", "tag_string_character",
                      "tag_string_copyright", "tag_string_artist",
                      "tag_string_meta"):
            wire_tags.extend(t for t in post.get(field, "").split() if t)
        counts = _danbooru_lookup_tag_counts(wire_tags, api_base)
        count_filter_args = {
            "counts":  counts,
            "op":      count_op,
            "low":     count_threshold_low,
            "high":    count_threshold_high,
            "targets": count_target_categories or [],
        }

    consolidated, q, s, c, ser, a, g = _format_categorised(
        rating_text, meta_tags, general_tags,
        character_tags, series_tags, artist_tags,
        blacklist_tags, blacklist_categories,
        count_filter_args,
    )
    return (consolidated, q, s, c, ser, a, g, info_line, post_url)


def _resolve_blacklists(blacklist_tag_str: str,
                        blacklist_cat_labels: List[str]
                        ) -> Tuple[set, List[str]]:
    """Translate the UI blacklist inputs into the internal forms used by
    the bucket logic. Shared by all grabbers."""
    blacklist_tags = _parse_blacklist_tags(blacklist_tag_str)
    blacklist_cats = [_CATEGORY_LABEL_TO_KEY[lbl]
                      for lbl in (blacklist_cat_labels or [])
                      if lbl in _CATEGORY_LABEL_TO_KEY]
    return blacklist_tags, blacklist_cats


def _resolve_count_targets(target_labels: List[str]) -> List[str]:
    """Translate UI target-category labels into internal bucket keys."""
    return [_CATEGORY_LABEL_TO_KEY[lbl]
            for lbl in (target_labels or [])
            if lbl in _CATEGORY_LABEL_TO_KEY]


def grab_danbooru_family(source: str,
                         rating_choice: str,
                         search_tag: str,
                         blacklist_tag_str: str,
                         blacklist_cat_labels: List[str],
                         count_filter_enabled: bool = False,
                         count_op: str = "≥",
                         count_threshold_low: int = 0,
                         count_threshold_high: int = 0,
                         count_target_labels: List[str] = None,
                         hunt_mode: bool = False):
    """
    Fetch one random post from a Danbooru-family source (Danbooru, Aibooru)
    optionally filtered by `search_tag`, apply blacklists, return the 9-tuple.

    If `hunt_mode` is on, instead of a plain random post, first search the
    booru's /tags.json for tags in the target categories whose post_count
    matches the comparator -- then pick one and search posts containing it.
    This guarantees the returned post will actually exercise the count
    filter on the target category (vs. rolling the dice on a random post
    which usually won't have any matching artist/character tag).
    """
    cfg = _BOORU_SOURCES[source]
    api_base = cfg["api_base"]
    anon_limit = cfg.get("anon_tag_limit")

    rating_tag = _DANBOORU_RATING_TAG.get(rating_choice, "")
    search_tags = _parse_search_tags(search_tag)
    count_targets = _resolve_count_targets(count_target_labels or [])

    # ---- Hunt-mode dispatch ------------------------------------------------
    if hunt_mode:
        if not count_filter_enabled:
            return ("[Hunt mode requires the post-count filter to be enabled "
                    "— it uses the filter to decide which posts qualify.]",
                    "", "", "", "", "", "", "", "")
        if not count_targets:
            return ("[Hunt mode needs at least one target category — tick a "
                    "box under 'Apply count filter to' so it knows what to "
                    "hunt for.]",
                    "", "", "", "", "", "", "", "")
        return _hunt_danbooru_family(
            source, rating_tag, search_tags,
            blacklist_tag_str, blacklist_cat_labels,
            count_op, count_threshold_low, count_threshold_high,
            count_target_labels,
        )

    # ---- Plain random grab -------------------------------------------------
    # Honour the anonymous tag-count cap (Danbooru/Aibooru: 2 tags total).
    if anon_limit is not None:
        tag_count = len(search_tags) + (1 if rating_tag else 0)
        if tag_count > anon_limit:
            msg = (f"[{source}'s anonymous search is limited to {anon_limit} "
                   f"tags total, but you have {len(search_tags)} search tag(s) "
                   f"+ {1 if rating_tag else 0} rating tag = {tag_count}. "
                   f"Try fewer search tags or set rating filter to "
                   f"'All ratings'.]")
            return (msg, "", "", "", "", "", "", "", "")

    url = f"{api_base.rstrip('/')}/posts/random.json"
    params = {}
    query_tags = list(search_tags)
    if rating_tag:
        query_tags.append(rating_tag)
    if query_tags:
        params["tags"] = " ".join(query_tags)

    try:
        r = _http_get(url, params=params,
                      headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        post = r.json()
    except Exception as e:
        msg = f"[error fetching from {source}: {e}]"
        print(f"{EXT_TAG} {msg}")
        return (msg, "", "", "", "", "", "", "", "")

    # Empty dict or no id field = no matching posts.
    if not post or not post.get("id"):
        msg = (f"[no {source} posts matched that search — try different tags "
               "or a less restrictive rating filter]")
        return (msg, "", "", "", "", "", "", "", "")

    blacklist_tags, blacklist_cats = _resolve_blacklists(
        blacklist_tag_str, blacklist_cat_labels)

    return _parse_danbooru_family_post(
        source, post, blacklist_tags, blacklist_cats,
        count_filter_enabled, count_op,
        count_threshold_low, count_threshold_high,
        count_targets,
    )


def grab_danbooru_family_by_id(source: str,
                               post_id: str,
                               blacklist_tag_str: str,
                               blacklist_cat_labels: List[str],
                               count_filter_enabled: bool = False,
                               count_op: str = "≥",
                               count_threshold_low: int = 0,
                               count_threshold_high: int = 0,
                               count_target_labels: List[str] = None):
    """Fetch a specific Danbooru-family post by ID. No rating filter, no
    search (you've already chosen the post). Hunt mode doesn't apply here."""
    pid = (post_id or "").strip()
    if not pid.isdigit():
        return (f"[invalid {source} post ID: {post_id!r}]",
                "", "", "", "", "", "", "", "")

    cfg = _BOORU_SOURCES[source]
    url = f"{cfg['api_base'].rstrip('/')}/posts/{pid}.json"
    try:
        r = _http_get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if r.status_code == 404:
            return (f"[{source} post #{pid} not found]",
                    "", "", "", "", "", "", "", "")
        r.raise_for_status()
        post = r.json()
    except Exception as e:
        msg = f"[error fetching {source} post #{pid}: {e}]"
        print(f"{EXT_TAG} {msg}")
        return (msg, "", "", "", "", "", "", "", "")

    blacklist_tags, blacklist_cats = _resolve_blacklists(
        blacklist_tag_str, blacklist_cat_labels)
    count_targets = _resolve_count_targets(count_target_labels or [])
    return _parse_danbooru_family_post(
        source, post, blacklist_tags, blacklist_cats,
        count_filter_enabled, count_op,
        count_threshold_low, count_threshold_high,
        count_targets,
    )


# Danbooru tag-category integers used by /tags.json search[category]. Same on
# Aibooru since it's a Danbooru fork.
#   0 = general, 1 = artist, 3 = copyright (series), 4 = character, 5 = meta
_DANBOORU_TAG_CATEGORY_INT = {
    "general":   0,
    "artist":    1,
    "series":    3,
    "character": 4,
    "quality":   5,   # meta = quality bucket for our purposes
}


def _count_filter_to_danbooru_query(op: str, lo: int, hi: int) -> List[str]:
    """
    Translate a comparator + thresholds into one or more Danbooru-style
    `search[post_count]` query strings. Most ops produce a single query;
    'not between' returns two (low half and high half) because the API has
    no direct exclusion syntax.

    Danbooru's range metatag syntax: `5..10` (inclusive), `>5`, `<=10`,
    `5` (exact). Source:
    https://github.com/danbooru/danbooru/blob/master/doc/api.txt
    """
    if op == "≥":           return [f">={lo}"]
    if op == "≤":           return [f"<={lo}"]
    if op == "=":           return [f"{lo}"]
    if op == "between":     return [f"{lo}..{hi}"]
    if op == "not between":
        # Keep things strictly outside [lo, hi]: count < lo OR count > hi.
        # Edge case: lo == 0 means the low half is empty -- skip it.
        queries = []
        if lo > 0:
            queries.append(f"<{lo}")
        queries.append(f">{hi}")
        return queries
    return []


def _hunt_danbooru_family(source: str,
                          rating_tag: str,
                          search_tags: List[str],
                          blacklist_tag_str: str,
                          blacklist_cat_labels: List[str],
                          count_op: str,
                          count_threshold_low: int,
                          count_threshold_high: int,
                          count_target_labels: List[str]):
    """
    Hunt mode for Danbooru-family sources. Strategy:

        1. For each target category, query /tags.json with
           search[category]=N + search[post_count]=<range> to get the list
           of tags in that category whose post_count satisfies the filter.
        2. Pool the candidates, pick one randomly.
        3. Search posts that have that tag (+ optional search/rating tags),
           using /posts/random.json.
        4. Parse the post normally so the filter is then re-applied to its
           tags (some unrelated tags may still get dropped from the output,
           but at least one matching target-category tag is guaranteed).

    This is much more reliable than rolling /posts/random.json and hoping
    a target-category tag in range happens to be present.
    """
    import random as _random

    cfg = _BOORU_SOURCES[source]
    api_base = cfg["api_base"]
    anon_limit = cfg.get("anon_tag_limit")
    count_targets = _resolve_count_targets(count_target_labels)

    # Translate categories + comparator into one or more tag-search queries.
    lo, hi = _normalize_count_op_args(
        count_op, count_threshold_low, count_threshold_high)
    count_queries = _count_filter_to_danbooru_query(count_op, lo, hi)
    if not count_queries:
        return (f"[invalid comparator for hunt: {count_op!r}]",
                "", "", "", "", "", "", "", "")

    candidate_tag_names: List[str] = []
    tag_url = f"{api_base.rstrip('/')}/tags.json"

    for cat_key in count_targets:
        cat_int = _DANBOORU_TAG_CATEGORY_INT.get(cat_key)
        if cat_int is None:
            # 'subject' has no Danbooru category integer -- subject-count
            # tags live inside the 'general' category. The user is unlikely
            # to want to hunt for these, but warn rather than silently skip.
            print(f"{EXT_TAG} hunt: skipping unsupported target category "
                  f"{cat_key!r} (no Danbooru category integer)")
            continue
        for cq in count_queries:
            params = {
                "search[category]":   str(cat_int),
                "search[post_count]": cq,
                "search[hide_empty]": "yes",
                "limit": "1000",
            }
            try:
                r = _http_get(tag_url, params=params,
                              headers={"User-Agent": USER_AGENT},
                              timeout=20)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"{EXT_TAG} hunt: {source} tag query failed: {e}")
                continue
            if not isinstance(data, list):
                continue
            for t in data:
                name = t.get("name", "")
                if name:
                    candidate_tag_names.append(name)

    if not candidate_tag_names:
        return (f"[hunt: no {source} tags in {count_target_labels} "
                f"satisfy {count_op} {lo}{'..' + str(hi) if count_op in ('between', 'not between') else ''} — "
                "try a wider range or different category]",
                "", "", "", "", "", "", "", "")

    # ---- Pick a candidate and search posts for it -----------------------
    # Shuffle so retries (if the first candidate has no posts matching the
    # search/rating filter) hit different tags.
    _random.shuffle(candidate_tag_names)

    # Cap retries so a misconfigured search doesn't loop forever.
    MAX_TRIES = 8

    for candidate in candidate_tag_names[:MAX_TRIES]:
        query_tags = list(search_tags) + [candidate]
        if rating_tag:
            query_tags.append(rating_tag)

        # Honour anon tag limit -- if we'd exceed it, drop the rating tag
        # first (the candidate is non-negotiable; the rating filter is a
        # nice-to-have when hunting).
        if anon_limit is not None and len(query_tags) > anon_limit:
            if rating_tag and len(query_tags) - 1 <= anon_limit:
                query_tags = list(search_tags) + [candidate]
            else:
                # Even without the rating tag we're over -- caller probably
                # had too many search tags. Short-circuit with a clear msg.
                return (f"[hunt: too many tags for {source}'s anonymous "
                        f"limit of {anon_limit}. Hunt needs one tag slot "
                        "for the candidate; reduce search tags.]",
                        "", "", "", "", "", "", "", "")

        url = f"{api_base.rstrip('/')}/posts/random.json"
        try:
            r = _http_get(url, params={"tags": " ".join(query_tags)},
                          headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            post = r.json()
        except Exception as e:
            print(f"{EXT_TAG} hunt: {source} post fetch failed for "
                  f"{candidate}: {e}")
            continue

        if not post or not post.get("id"):
            # No post for that candidate + search combo; try next candidate.
            continue

        blacklist_tags, blacklist_cats = _resolve_blacklists(
            blacklist_tag_str, blacklist_cat_labels)

        result = _parse_danbooru_family_post(
            source, post, blacklist_tags, blacklist_cats,
            count_filter_enabled=True,
            count_op=count_op,
            count_threshold_low=count_threshold_low,
            count_threshold_high=count_threshold_high,
            count_target_categories=count_targets,
        )

        # Re-decorate the info line so the user knows hunt mode produced it
        # and which candidate tag did so.
        consolidated, q, s, c, ser, a, g, info_line, post_url = result
        info_line = (f"🎯 Hunt hit on tag '{candidate}' → " + info_line)
        return (consolidated, q, s, c, ser, a, g, info_line, post_url)

    return (f"[hunt: found {len(candidate_tag_names)} matching tag(s) on "
            f"{source} but none had a post matching the other filters. "
            "Try widening rating filter or removing search tag(s).]",
            "", "", "", "", "", "", "", "")


# ----------------------------------------------------------------------------
# Gelbooru random fetch
# ----------------------------------------------------------------------------

def _gelbooru_lookup_tag_info(tag_names: List[str],
                              source: str = "Gelbooru"
                              ) -> Dict[str, Tuple[str, int]]:
    """
    Batch-lookup (category, post_count) for each tag via the booru's
    s=tag&q=index endpoint. Returns {tag_name: (category_string, count)}.

    Gelbooru-family per-post JSON only gives a flat space-separated tag
    list, so to sort tags into the Anima tag-order buckets we have to ask
    the booru what each one is. The s=tag endpoint accepts multiple names
    in a single call and also returns post-count for each, which the
    count filter uses (no extra API call needed when the filter's on).

    `source` is a key into _BOORU_SOURCES; both URL and credentials are
    looked up from there.

    **Rule34 caveat:** Rule34 doesn't expose a working batch tag-info
    endpoint -- calls to `s=tag&q=index` return an empty body, breaking
    JSON decode. We short-circuit Rule34 here and return empty; callers
    fall back to HTML scraping for tag categories (see
    `_rule34_scrape_tag_categories`). Counts aren't recoverable from HTML
    so the count filter degrades to "everything is 0" on Rule34.
    """
    out: Dict[str, Tuple[str, int]] = {}
    if not tag_names:
        return out
    if source == "Rule34":
        # See docstring; Rule34's `s=tag` endpoint is non-functional.
        return out

    cfg = _BOORU_SOURCES[source]
    api_key, user_id = _get_credentials(cfg["creds_key"]) if cfg["creds_key"] else ("", "")
    url = cfg["api_base"]

    # Reasonable chunk size to keep URLs short. Gelbooru handles much larger
    # but there's no point pushing it.
    chunk_size = 40
    for i in range(0, len(tag_names), chunk_size):
        chunk = tag_names[i:i + chunk_size]
        params = {
            "page":    "dapi",
            "s":       "tag",
            "q":       "index",
            "json":    "1",
            "names":   " ".join(chunk),
            "api_key": api_key,
            "user_id": user_id,
        }
        try:
            data = _gelbooru_request(url, params, source=source)
        except Exception as e:
            print(f"{EXT_TAG} {source} tag-info lookup failed: {e}")
            continue

        # Response is either {"tag": [{...}, ...]} or sometimes {"tag": {...}}
        # for a single result, depending on how the API feels that minute.
        tag_list = data.get("tag", [])
        if isinstance(tag_list, dict):
            tag_list = [tag_list]
        for t in tag_list:
            name = t.get("name", "")
            ttype = t.get("type", 0)
            if not name:
                continue
            try:
                count = int(t.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            try:
                cat = _GELBOORU_TAG_TYPE.get(int(ttype), "general")
            except (TypeError, ValueError):
                cat = "general"
            out[name] = (cat, count)
    return out


def _gelbooru_lookup_categories(tag_names: List[str]) -> Dict[str, str]:
    """
    Back-compat wrapper -- returns just the category map. Kept so external
    callers (if any) don't break. New code should use _gelbooru_lookup_tag_info.
    """
    return {n: info[0] for n, info in _gelbooru_lookup_tag_info(tag_names).items()}


def _rule34_scrape_tag_categories(post_id: str
                                  ) -> Dict[str, Tuple[str, int]]:
    """
    Recover tag categories on Rule34 by scraping the post's HTML page.

    Rule34 has no working batch tag-info JSON endpoint (calls to
    `s=tag&q=index` come back with an empty body), so without this
    fallback every tag would default to 'general' -- meaning artist
    tags wouldn't get the `@` prefix Anima needs, and character /
    series / meta sections would be empty. gallery-dl uses the same
    HTML-scraping approach on this site for the same reason.

    Returns `{wire_tag_name: (category, 0)}`. Post counts aren't
    available from HTML, so they're always 0 -- the per-tag count
    filter degrades to a no-op for Rule34 (every tag reads as count 0).
    """
    url = f"https://rule34.xxx/index.php?page=post&s=view&id={post_id}"
    try:
        r = _http_get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
    except Exception as e:
        # Quiet failure: caller will fall through to "all general" defaults.
        # Logged once per failed grab, not per tag, so it shouldn't spam.
        print(f"{EXT_TAG} Rule34 HTML category scrape failed for #{post_id}: {e}")
        return {}

    out: Dict[str, Tuple[str, int]] = {}
    # urllib.parse is in the stdlib but we haven't imported it yet -- do it
    # here so the import sits next to its only use site.
    from urllib.parse import unquote
    for tag_type, raw_tag in _RULE34_TAG_TYPE_RE.findall(r.text):
        cat = _RULE34_HTML_TAG_TYPE.get(tag_type.lower(), "general")
        # URL-decode the tag (e.g. mutsuki_%28blue_archive%29 -> the wire
        # form mutsuki_(blue_archive)). Skip empty matches defensively.
        name = unquote(raw_tag).strip()
        if name:
            out[name] = (cat, 0)
    return out


def _danbooru_lookup_tag_counts(tag_names: List[str],
                                api_base: str = "https://danbooru.donmai.us"
                                ) -> Dict[str, int]:
    """
    Batch-lookup post_count for each tag via the booru's /tags.json endpoint.
    Returns {tag_name: count}. Tags not returned by the API (deprecated,
    aliased, or otherwise missing) are simply absent -- callers should treat
    a missing key as count=0.

    `api_base` lets this work for both Danbooru and its forks (Aibooru,
    etc) without duplicating the lookup logic.

    Danbooru-family /tags.json supports search[name_comma]=a,b,c for batch
    lookup and is available anonymously, so no credentials are needed.
    """
    out: Dict[str, int] = {}
    if not tag_names:
        return out

    url = f"{api_base.rstrip('/')}/tags.json"

    # Danbooru caps /tags.json results at 1000 per page (limit param), which
    # is plenty for one post's tag list. Chunk anyway to keep URLs reasonable.
    chunk_size = 100
    for i in range(0, len(tag_names), chunk_size):
        chunk = tag_names[i:i + chunk_size]
        params = {
            "search[name_comma]": ",".join(chunk),
            "limit": str(len(chunk)),
        }
        try:
            r = _http_get(url, params=params,
                          headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"{EXT_TAG} {api_base} tag-count lookup failed: {e}")
            continue

        if not isinstance(data, list):
            continue
        for t in data:
            name = t.get("name", "")
            if not name:
                continue
            try:
                count = int(t.get("post_count", 0))
            except (TypeError, ValueError):
                count = 0
            out[name] = count
    return out


def _gelbooru_request(url: str, params: Dict[str, str], timeout: int = 20,
                      source: str = "Gelbooru"):
    """
    Wrapper around requests.get that detects 401 (missing/bad credentials)
    and raises a friendlier error message, since that's by far the most
    likely failure mode on the credentialled Gelbooru-family sources.

    `source` tailors the error message to the right site (Gelbooru since
    2022, Rule34 since Aug 2025 both require api_key + user_id).

    Also normalises the response shape: Gelbooru wraps its results as
    ``{"@attributes": {...}, "post": [...]}`` (or "tag": [...]), but Rule34
    returns the bare list directly (``[{...}, {...}]``). We re-wrap the list
    under the right key based on `params["s"]` so downstream call sites can
    use a single shape regardless of source. Bare empty lists become
    ``{"post": [], "@attributes": {"count": 0}}`` (or ``{"tag": []}``) so
    the "no results" check fires correctly.
    """
    try:
        r = _http_get(url, params=params,
                      headers={"User-Agent": USER_AGENT}, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"network error: {e}") from e

    if r.status_code == 401:
        creds_url = {
            "Gelbooru": "https://gelbooru.com/index.php?page=account&s=options",
            "Rule34":   "https://api.rule34.xxx/",
        }.get(source, "the site's account/options page")
        raise RuntimeError(
            f"401 Unauthorized — {source} requires API credentials for ALL "
            f"API access. Open the '{source} API credentials' section above "
            f"and enter your api_key and user_id from {creds_url}"
        )
    r.raise_for_status()
    data = r.json()

    # Re-wrap Rule34's bare-list response into Gelbooru's dict shape so
    # the rest of the code only has to handle one format. The Gelbooru API
    # uses `s` to identify the endpoint ('post' for image searches, 'tag'
    # for tag-info lookups); the same param drives the wrap key.
    if isinstance(data, list):
        key = params.get("s", "post")
        wrapped = {key: data}
        # Synthesize @attributes.count from the list length so downstream
        # "no results" checks work. Note: this is the length of THIS PAGE,
        # not the total match count -- the random-pid picker handles the
        # missing-total case separately by falling back to a blind pid.
        wrapped["@attributes"] = {"count": len(data)}
        data = wrapped
    return data


def _parse_gelbooru_family_post(source: str,
                                post: dict,
                                blacklist_tags: set,
                                blacklist_categories: List[str],
                                count_filter_enabled: bool = False,
                                count_op: str = "≥",
                                count_threshold_low: int = 0,
                                count_threshold_high: int = 0,
                                count_target_categories: List[str] = None,
                                tag_info_override: Dict[str, Tuple[str, int]] = None):
    """
    Shared parser for both random and by-id grabs on Gelbooru-family sources
    (Gelbooru, Rule34).

    `tag_info_override` lets the hunt-mode batch lookup reuse the tag info
    it already fetched instead of paying for it twice (the hunt code looks
    up info on many candidate posts at once).

    Gelbooru-family per-post JSON returns count and category in the same
    tag-info response, so the count filter doesn't cost an extra API call.
    """
    cfg = _BOORU_SOURCES[source]

    raw_tags = [t for t in post.get("tags", "").split() if t]
    rating = post.get("rating", "")
    rating_map = _RULE34_RATING_MAP if source == "Rule34" else _GELBOORU_RATING_MAP
    rating_text = rating_map.get(rating, "")

    post_id  = post.get("id", "?")
    post_url = cfg["post_url"](post_id)
    image_url = post.get("file_url", "")
    info_line = f"{source} post #{post_id} → {post_url}"
    if image_url:
        info_line += f"\nImage: {image_url}"

    if tag_info_override is not None:
        tag_info = tag_info_override
    elif source == "Rule34":
        # Rule34's `s=tag` JSON endpoint is non-functional; recover
        # categories from the post's HTML page instead. Counts come back
        # as 0 across the board -- the count filter is effectively a
        # no-op on Rule34 (documented in the README).
        tag_info = _rule34_scrape_tag_categories(str(post_id))
    else:
        tag_info = _gelbooru_lookup_tag_info(raw_tags, source=source)

    general_tags, character_tags, series_tags = [], [], []
    artist_tags, meta_tags = [], []
    for raw in raw_tags:
        cat, _ = tag_info.get(raw, ("general", 0))
        norm = _normalize_tag(raw)
        if   cat == "artist":    artist_tags.append(norm)
        elif cat == "character": character_tags.append(norm)
        elif cat == "copyright": series_tags.append(norm)
        elif cat == "metadata":  meta_tags.append(norm)
        else:                    general_tags.append(norm)

    count_filter_args = None
    if count_filter_enabled:
        counts = {n: info[1] for n, info in tag_info.items()}
        count_filter_args = {
            "counts":  counts,
            "op":      count_op,
            "low":     count_threshold_low,
            "high":    count_threshold_high,
            "targets": count_target_categories or [],
        }

    consolidated, q, s, c, ser, a, g = _format_categorised(
        rating_text, meta_tags, general_tags,
        character_tags, series_tags, artist_tags,
        blacklist_tags, blacklist_categories,
        count_filter_args,
    )
    return (consolidated, q, s, c, ser, a, g, info_line, post_url)


def _gelbooru_family_creds_check(source: str
                                 ) -> Tuple[bool, str, str, str]:
    """
    Returns (ok, api_key, user_id, error_msg). ok==False means error_msg is
    a user-facing string explaining what to do; api_key/user_id are empty
    in that case.

    Some Gelbooru-family sources don't need credentials (none currently in
    the registry, but kept for future-proofing). When creds_key is None,
    this always returns (True, "", "", "").
    """
    cfg = _BOORU_SOURCES[source]
    if not cfg["creds_key"]:
        return True, "", "", ""
    api_key, user_id = _get_credentials(cfg["creds_key"])
    if not api_key or not user_id:
        creds_link = {
            "gelbooru": "https://gelbooru.com/index.php?page=account&s=options",
            "rule34":   "https://api.rule34.xxx/",
        }.get(cfg["creds_key"], "the site's account/options page")
        msg = (f"[{source} API credentials not set — open the "
               f"'{source} API credentials' section above, enter your "
               f"api_key and user_id (free from {creds_link}), "
               "and click Save.]")
        return False, "", "", msg
    return True, api_key, user_id, ""


def grab_gelbooru_family(source: str,
                         rating_choice: str,
                         search_tag: str,
                         blacklist_tag_str: str,
                         blacklist_cat_labels: List[str],
                         count_filter_enabled: bool = False,
                         count_op: str = "≥",
                         count_threshold_low: int = 0,
                         count_threshold_high: int = 0,
                         count_target_labels: List[str] = None,
                         hunt_mode: bool = False):
    """
    Fetch one random post from a Gelbooru-family source (Gelbooru, Rule34).
    Two-step random workaround (count, then random pid) same as before.

    If `hunt_mode`, instead scan a batch of candidate posts and return the
    first one whose target-category tags satisfy the comparator. Gelbooru-
    family APIs don't support direct tag-by-count search, so we can't do
    the elegant Danbooru-style "find tag first, then post" -- the scan
    approach is best-effort but usually fast since one batch fetch + one
    batch tag-info call covers many candidates.
    """
    import random as _random  # local import keeps top-of-file imports lean

    count_targets = _resolve_count_targets(count_target_labels or [])
    rating_tag_map = _RULE34_RATING_TAG if source == "Rule34" else _GELBOORU_RATING_TAG
    rating_tag = rating_tag_map.get(rating_choice, "")
    search_tags = _parse_search_tags(search_tag)

    if hunt_mode:
        if not count_filter_enabled:
            return ("[Hunt mode requires the post-count filter to be enabled "
                    "— it uses the filter to decide which posts qualify.]",
                    "", "", "", "", "", "", "", "")
        if not count_targets:
            return ("[Hunt mode needs at least one target category — tick a "
                    "box under 'Apply count filter to' so it knows what to "
                    "hunt for.]",
                    "", "", "", "", "", "", "", "")
        return _hunt_gelbooru_family(
            source, rating_tag, search_tags,
            blacklist_tag_str, blacklist_cat_labels,
            count_op, count_threshold_low, count_threshold_high,
            count_target_labels,
        )

    ok, api_key, user_id, err = _gelbooru_family_creds_check(source)
    if not ok:
        return (err, "", "", "", "", "", "", "", "")

    cfg = _BOORU_SOURCES[source]
    url = cfg["api_base"]

    query_tags = list(search_tags)
    if rating_tag:
        query_tags.append(rating_tag)
    tag_query = " ".join(query_tags)

    base_params = {
        "page":     "dapi",
        "s":        "post",
        "q":        "index",
        "json":     "1",
        "tags":     tag_query,
        "api_key":  api_key,
        "user_id":  user_id,
    }

    # ---- Step 1: head probe to confirm the search has any matches at all,
    # and read @attributes.count on Gelbooru (Rule34 doesn't return a real
    # total -- _gelbooru_request synthesises count=len(page), which is only
    # the size of this page, not the total).
    try:
        head = _gelbooru_request(url, {**base_params, "limit": "1"},
                                 source=source)
    except Exception as e:
        msg = f"[error fetching from {source} (count): {e}]"
        print(f"{EXT_TAG} {msg}")
        return (msg, "", "", "", "", "", "", "", "")

    head_posts = head.get("post", [])
    if isinstance(head_posts, dict):
        head_posts = [head_posts]
    if not head_posts:
        msg = (f"[no {source} posts matched that search — try different tags "
               "or a less restrictive rating filter]")
        return (msg, "", "", "", "", "", "", "", "")

    # ---- Step 2: pick a random pid and fetch one post there ------------
    # Two strategies depending on whether the source exposes a true match
    # count:
    #   * Gelbooru: @attributes.count is the total -- pick a uniformly
    #     random pid in [0, count). PID_CAP guards against the anon
    #     pagination wall around 20k.
    #   * Rule34: no real total, so blindly pick a pid in [0, BLIND_CAP].
    #     If that page comes back empty (too far into the tail), fall back
    #     to pid=0 to guarantee we return something rather than a confusing
    #     "try again" error.
    PID_CAP = 20000           # anon pagination wall
    RULE34_BLIND_CAP = 200    # bounded random walk for Rule34 (≈first 200 posts)

    if source == "Rule34":
        pid = _random.randint(0, RULE34_BLIND_CAP)
        try:
            data = _gelbooru_request(
                url, {**base_params, "limit": "1", "pid": str(pid)},
                source=source)
        except Exception as e:
            msg = f"[error fetching from {source} (pid={pid}): {e}]"
            print(f"{EXT_TAG} {msg}")
            return (msg, "", "", "", "", "", "", "", "")
        # If the random pid landed past the end of the result set, fall
        # back to pid=0 (the head probe already confirmed at least 1 post).
        posts_check = data.get("post", [])
        if isinstance(posts_check, dict):
            posts_check = [posts_check]
        if not posts_check:
            data = head
    else:
        count = 0
        attrs = head.get("@attributes", {})
        if isinstance(attrs, dict):
            try:
                count = int(attrs.get("count", 0))
            except (TypeError, ValueError):
                count = 0
        if count <= 1:
            data = head
        else:
            max_pid = min(count, PID_CAP) - 1
            pid = _random.randint(0, max(max_pid, 0))
            try:
                data = _gelbooru_request(
                    url, {**base_params, "limit": "1", "pid": str(pid)},
                    source=source)
            except Exception as e:
                msg = f"[error fetching from {source} (pid={pid}): {e}]"
                print(f"{EXT_TAG} {msg}")
                return (msg, "", "", "", "", "", "", "", "")

    posts = data.get("post", [])
    if isinstance(posts, dict):
        posts = [posts]
    if not posts:
        msg = f"[no posts returned from {source} — try again]"
        return (msg, "", "", "", "", "", "", "", "")
    post = posts[0]

    blacklist_tags, blacklist_cats = _resolve_blacklists(
        blacklist_tag_str, blacklist_cat_labels)
    return _parse_gelbooru_family_post(
        source, post, blacklist_tags, blacklist_cats,
        count_filter_enabled, count_op,
        count_threshold_low, count_threshold_high,
        count_targets,
    )


def grab_gelbooru_family_by_id(source: str,
                               post_id: str,
                               blacklist_tag_str: str,
                               blacklist_cat_labels: List[str],
                               count_filter_enabled: bool = False,
                               count_op: str = "≥",
                               count_threshold_low: int = 0,
                               count_threshold_high: int = 0,
                               count_target_labels: List[str] = None):
    """Fetch a specific Gelbooru-family post by ID. Hunt mode N/A here."""
    pid = (post_id or "").strip()
    if not pid.isdigit():
        return (f"[invalid {source} post ID: {post_id!r}]",
                "", "", "", "", "", "", "", "")

    ok, api_key, user_id, err = _gelbooru_family_creds_check(source)
    if not ok:
        return (err, "", "", "", "", "", "", "", "")

    cfg = _BOORU_SOURCES[source]
    url = cfg["api_base"]
    params = {
        "page":     "dapi",
        "s":        "post",
        "q":        "index",
        "json":     "1",
        "id":       pid,
        "api_key":  api_key,
        "user_id":  user_id,
    }
    try:
        data = _gelbooru_request(url, params, source=source)
    except Exception as e:
        msg = f"[error fetching {source} post #{pid}: {e}]"
        print(f"{EXT_TAG} {msg}")
        return (msg, "", "", "", "", "", "", "", "")

    posts = data.get("post", [])
    if isinstance(posts, dict):
        posts = [posts]
    if not posts:
        return (f"[{source} post #{pid} not found]",
                "", "", "", "", "", "", "", "")

    blacklist_tags, blacklist_cats = _resolve_blacklists(
        blacklist_tag_str, blacklist_cat_labels)
    count_targets = _resolve_count_targets(count_target_labels or [])
    return _parse_gelbooru_family_post(
        source, posts[0], blacklist_tags, blacklist_cats,
        count_filter_enabled, count_op,
        count_threshold_low, count_threshold_high,
        count_targets,
    )


def _hunt_gelbooru_family(source: str,
                          rating_tag: str,
                          search_tags: List[str],
                          blacklist_tag_str: str,
                          blacklist_cat_labels: List[str],
                          count_op: str,
                          count_threshold_low: int,
                          count_threshold_high: int,
                          count_target_labels: List[str]):
    """
    Hunt mode for Gelbooru-family sources. Gelbooru/Rule34 don't support
    server-side filtering of tags by category + count, so we batch-scan
    candidate posts instead:

        1. Fetch a batch of posts at a random pid (or pid=0 if the search
           result is small) matching the rating/search filters.
        2. Look up tag info for the union of all tags across the batch in
           one call.
        3. Walk through posts and pick the first one whose target-category
           tags include at least one tag passing the count comparator.
        4. Parse that post normally so the count filter is applied to its
           output buckets.

    If no post in the batch qualifies, return a clear message rather than
    silently grabbing a non-matching post -- the user enabled hunt mode
    specifically to avoid that.
    """
    import random as _random

    ok, api_key, user_id, err = _gelbooru_family_creds_check(source)
    if not ok:
        return (err, "", "", "", "", "", "", "", "")

    count_targets = _resolve_count_targets(count_target_labels)
    target_cat_strings = set()
    # Map internal keys to the booru's category strings used in tag_info.
    cat_key_to_gel = {
        "artist":    "artist",
        "character": "character",
        "series":    "copyright",
        "quality":   "metadata",
        "general":   "general",
        "subject":   "general",  # subject tags live inside general
    }
    for k in count_targets:
        gel = cat_key_to_gel.get(k)
        if gel:
            target_cat_strings.add(gel)

    op_fn = _COUNT_OPS.get(count_op)
    if op_fn is None:
        return (f"[invalid comparator for hunt: {count_op!r}]",
                "", "", "", "", "", "", "", "")
    lo, hi = _normalize_count_op_args(
        count_op, count_threshold_low, count_threshold_high)

    cfg = _BOORU_SOURCES[source]
    url = cfg["api_base"]

    query_tags = list(search_tags)
    if rating_tag:
        query_tags.append(rating_tag)
    tag_query = " ".join(query_tags)

    base_params = {
        "page":     "dapi",
        "s":        "post",
        "q":        "index",
        "json":     "1",
        "tags":     tag_query,
        "api_key":  api_key,
        "user_id":  user_id,
    }

    # First peek at the result count so we can pick a sensible random pid.
    try:
        head = _gelbooru_request(url, {**base_params, "limit": "1"},
                                 source=source)
    except Exception as e:
        return (f"[hunt: {source} count query failed: {e}]",
                "", "", "", "", "", "", "", "")

    count = 0
    attrs = head.get("@attributes", {})
    if isinstance(attrs, dict):
        try:
            count = int(attrs.get("count", 0))
        except (TypeError, ValueError):
            count = 0
    if count == 0 and not head.get("post"):
        return (f"[hunt: no {source} posts matched the rating/search filter]",
                "", "", "", "", "", "", "", "")

    # Pick a random page within the available range and fetch a batch. A
    # bigger batch raises the chance of finding a matching post in one
    # round-trip; 50 posts is a reasonable balance between speed and yield.
    BATCH = 50
    PID_CAP = 20000
    max_first = max(count - BATCH, 0)
    max_first = min(max_first, PID_CAP)
    first = _random.randint(0, max_first) if max_first > 0 else 0

    try:
        data = _gelbooru_request(
            url, {**base_params, "limit": str(BATCH), "pid": str(first)},
            source=source)
    except Exception as e:
        return (f"[hunt: {source} batch fetch failed: {e}]",
                "", "", "", "", "", "", "", "")

    posts = data.get("post", [])
    if isinstance(posts, dict):
        posts = [posts]
    if not posts:
        return (f"[hunt: {source} returned no posts at pid={first}]",
                "", "", "", "", "", "", "", "")

    # Collect every unique tag across the batch and look them up at once.
    all_tags = set()
    for p in posts:
        for t in (p.get("tags", "") or "").split():
            if t:
                all_tags.add(t)
    tag_info = _gelbooru_lookup_tag_info(sorted(all_tags), source=source)

    # Find the first post whose target-category tags contain a match.
    matches = []
    for p in posts:
        ptags = [t for t in (p.get("tags", "") or "").split() if t]
        for t in ptags:
            cat, c = tag_info.get(t, ("general", 0))
            if cat in target_cat_strings and op_fn(c, lo, hi):
                matches.append((p, t))
                break

    if not matches:
        return (f"[hunt: scanned {len(posts)} {source} posts at pid={first}, "
                f"none had a tag in {count_target_labels} matching "
                f"{count_op} {lo}{'..' + str(hi) if count_op in ('between', 'not between') else ''}. "
                "Try widening the comparator range, removing the search "
                "tag, or retrying for a different batch.]",
                "", "", "", "", "", "", "", "")

    # Pick a random match so repeated hunts vary their output.
    _random.shuffle(matches)
    chosen_post, matched_tag = matches[0]

    blacklist_tags, blacklist_cats = _resolve_blacklists(
        blacklist_tag_str, blacklist_cat_labels)

    # Reuse the already-fetched tag_info so we don't pay for a second
    # /tag lookup. The override is a subset (posts in this batch), so
    # tags that happen not to be in any other batch post won't be in
    # tag_info -- _parse handles missing entries by defaulting to general,
    # which is acceptable for a hunt result.
    result = _parse_gelbooru_family_post(
        source, chosen_post, blacklist_tags, blacklist_cats,
        count_filter_enabled=True,
        count_op=count_op,
        count_threshold_low=count_threshold_low,
        count_threshold_high=count_threshold_high,
        count_target_categories=count_targets,
        tag_info_override=tag_info,
    )

    consolidated, q, s, c, ser, a, g, info_line, post_url = result
    info_line = (f"🎯 Hunt hit on tag '{matched_tag}' → " + info_line)
    return (consolidated, q, s, c, ser, a, g, info_line, post_url)


def grab_random(source: str,
                rating_choice: str,
                search_tag: str,
                blacklist_tag_str: str,
                blacklist_cat_labels: List[str],
                count_filter_enabled: bool = False,
                count_op: str = "≥",
                count_threshold_low: int = 0,
                count_threshold_high: int = 0,
                count_target_labels: List[str] = None,
                hunt_mode: bool = False):
    """Dispatch to the appropriate family grabber based on the source name.

    Hunt mode is silently disabled for Gelbooru-family sources -- those
    sites don't support server-side tag-by-count filtering, so the only
    feasible implementation is a slow batch-scan that often returns
    nothing. The UI restricts hunt mode to Danbooru-family; if a caller
    bypasses that (e.g. via a direct function call) we just ignore the
    flag here rather than ping-ponging with an error.
    """
    cfg = _BOORU_SOURCES.get(source)
    if not cfg:
        return (f"[unknown source: {source}]", "", "", "", "", "", "", "", "")
    if cfg["family"] == "danbooru":
        return grab_danbooru_family(source, rating_choice, search_tag,
                                    blacklist_tag_str, blacklist_cat_labels,
                                    count_filter_enabled, count_op,
                                    count_threshold_low, count_threshold_high,
                                    count_target_labels, hunt_mode)
    if cfg["family"] == "gelbooru":
        # Hunt mode is Danbooru-family only; always pass False here.
        return grab_gelbooru_family(source, rating_choice, search_tag,
                                    blacklist_tag_str, blacklist_cat_labels,
                                    count_filter_enabled, count_op,
                                    count_threshold_low, count_threshold_high,
                                    count_target_labels, False)
    return (f"[unsupported family for {source}]",
            "", "", "", "", "", "", "", "")


def grab_by_id(source: str,
               post_id: str,
               blacklist_tag_str: str,
               blacklist_cat_labels: List[str],
               count_filter_enabled: bool = False,
               count_op: str = "≥",
               count_threshold_low: int = 0,
               count_threshold_high: int = 0,
               count_target_labels: List[str] = None):
    """Dispatch by-id grab; hunt mode never applies here."""
    cfg = _BOORU_SOURCES.get(source)
    if not cfg:
        return (f"[unknown source: {source}]", "", "", "", "", "", "", "", "")
    if cfg["family"] == "danbooru":
        return grab_danbooru_family_by_id(source, post_id,
                                          blacklist_tag_str, blacklist_cat_labels,
                                          count_filter_enabled, count_op,
                                          count_threshold_low, count_threshold_high,
                                          count_target_labels)
    if cfg["family"] == "gelbooru":
        return grab_gelbooru_family_by_id(source, post_id,
                                          blacklist_tag_str, blacklist_cat_labels,
                                          count_filter_enabled, count_op,
                                          count_threshold_low, count_threshold_high,
                                          count_target_labels)
    return (f"[unsupported family for {source}]",
            "", "", "", "", "", "", "", "")


# ----------------------------------------------------------------------------
# Post image download (the preview, and the hand-off to JoyCaption)
# ----------------------------------------------------------------------------
# A grab names the post's image on the "Image: <url>" line of its info. The
# server remembers that URL per user (the grabbed info shown in the page is
# never trusted), and downloads only over https from the source's own image
# hosts, re-checking every redirect. Downloads are stored in the server's
# upload store (pruned like any upload), so the preview and the JoyCaption
# page share one file id.

# Captures the "Image: <url>" line of grabbed_info. Both grab paths
# (danbooru/aibooru and gelbooru/rule34) write this line in the same format.
_IMAGE_URL_RE = re.compile(r"^Image:\s*(\S+)", re.MULTILINE)

# user -> (source, image URL, post id hint) of the user's last grab
_last_grabs: Dict[object, Tuple[str, str, str]] = {}
_last_grabs_lock = threading.Lock()


def _remember_grab(user, source: str, info: str) -> None:
    """Keep the image URL of ``user``'s last grab (a grab without one, e.g.
    an error, forgets the previous)."""
    url, pid_hint = _parse_image_url_from_info(info)
    with _last_grabs_lock:
        if url:
            _last_grabs[user] = (source, url, pid_hint or "")
        else:
            _last_grabs.pop(user, None)


def _last_grab(user):
    with _last_grabs_lock:
        return _last_grabs.get(user)


def _safe_image_filename(url: str, post_id_hint: str = "") -> str:
    """Picks a sensible local filename for a downloaded booru image.
    Uses the URL's basename when it looks like a real image filename,
    otherwise falls back to post-<id>.<ext>. Strips anything weird."""
    from urllib.parse import urlparse
    try:
        base = os.path.basename(urlparse(url).path)
    except Exception:
        base = ""
    # Sanitise: keep only alnum, dash, dot, underscore.
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base or "")
    # Reject totally degenerate names.
    if not base or "." not in base:
        ext = ".png"
        # Try to sniff an extension from the URL even if path was weird.
        m = re.search(r"\.(png|jpg|jpeg|webp|gif|bmp|tiff|avif|jxl|heif)\b",
                      url, re.IGNORECASE)
        if m:
            ext = "." + m.group(1).lower()
        pid = re.sub(r"[^A-Za-z0-9_-]", "", str(post_id_hint)) or "img"
        base = f"post-{pid}{ext}"
    return base


def _resolve_host(host: str):
    """The addresses ``host`` resolves to (a seam for tests)."""
    return {
        ipaddress.ip_address(info[4][0].split("%", 1)[0])
        for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    }


def _image_url_problem(url: str, hosts) -> str:
    """Why ``url`` may not be fetched ("" when it may): https only, on one
    of ``hosts`` (or a subdomain), default port, public addresses only."""
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return "not a valid URL"
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not host:
        return "not an https URL"
    if parsed.username or parsed.password or port not in (None, 443):
        return "unexpected credentials or port in the URL"
    if not any(host == allowed or host.endswith("." + allowed) for allowed in hosts):
        return f"{host} is not an image host of this booru"
    try:
        addresses = _resolve_host(host)
    except (OSError, ValueError) as e:
        return f"cannot resolve {host}: {e}"
    for address in addresses:
        mapped = getattr(address, "ipv4_mapped", None)
        if not (mapped or address).is_global:
            return f"{host} resolves to a non-public address"
    return "" if addresses else f"cannot resolve {host}"


def _read_image_body(r, fname):
    """The response body, bounded by MAX_IMAGE_DOWNLOAD_BYTES: (bytes, error)."""
    declared = r.headers.get("Content-Length")
    if declared:
        try:
            if int(declared) > MAX_IMAGE_DOWNLOAD_BYTES:
                return None, (
                    "Download rejected: server reports an image larger than "
                    f"{MAX_IMAGE_DOWNLOAD_BYTES // (1024 * 1024)} MiB."
                )
        except ValueError:
            pass
    data = bytearray()
    for chunk in r.iter_content(chunk_size=IMAGE_DOWNLOAD_CHUNK_BYTES):
        if not chunk:
            continue
        if len(data) + len(chunk) > MAX_IMAGE_DOWNLOAD_BYTES:
            return None, (
                "Download rejected: image exceeded the "
                f"{MAX_IMAGE_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit."
            )
        data.extend(chunk)
    return bytes(data), None


def _download_image(url: str, pid_hint: str = "", hosts=()):
    """Fetch the post image ``url`` from one of ``hosts`` (see
    :func:`_image_url_problem`; redirects are followed by hand and checked
    hop by hop) and check that PIL can decode it. Returns
    (bytes | None, filename | None, error | None); errors never quote the
    response."""
    from PIL import Image as _PILImage, UnidentifiedImageError

    fname = _safe_image_filename(url, pid_hint)
    headers = {"User-Agent": USER_AGENT}
    # Some boorus (notably gelbooru) deny hot-linking unless the request
    # looks like it came from a page on their own domain. Pointing Referer
    # at the site root is enough; for everything else it is harmless.
    host = urlparse(url).hostname if url else None
    if host:
        headers["Referer"] = f"https://{host}/"

    current = url
    data = None
    try:
        for _hop in range(MAX_IMAGE_REDIRECTS + 1):
            problem = _image_url_problem(current, hosts)
            if problem:
                return None, None, f"Download refused: {problem}"
            with requests.get(current, headers=headers, timeout=30, stream=True, allow_redirects=False) as r:
                if r.is_redirect:
                    current = urljoin(current, r.headers.get("Location", ""))
                    continue
                r.raise_for_status()
                data, error = _read_image_body(r, fname)
                if error:
                    return None, fname, error
            break
        else:
            return None, None, "Download failed: too many redirects"
    except Exception as e:
        return None, None, f"Download failed: {e}"

    if not data:
        return None, fname, "Download failed: empty response"

    # Text (HTML/JSON) instead of an image is an error or anti-hotlink page;
    # PIL would only say "cannot identify image file".
    if data[:16].lstrip()[:1] in (b"<", b"{"):
        return None, fname, "Server returned non-image content (likely a block or error page)."

    try:
        with _PILImage.open(io.BytesIO(data)) as img:
            img.load()  # full decode: can PIL read this format at all?
    except UnidentifiedImageError:
        return None, fname, (
            "PIL can't identify this image. Server may have sent an "
            "unsupported format -- if the URL ends in .jxl/.avif/.heif "
            "your Pillow build may lack support."
        )
    except Exception as e:
        return None, fname, f"Decode failed: {e}"

    return data, fname, None


def _parse_image_url_from_info(grabbed_info_text: str):
    """Pulls the image URL and a post-ID hint out of a grabbed_info
    blob. Returns (url, pid_hint) or (None, None) if no URL line is
    present. The pid_hint is used only for cosmetic filename choice."""
    if not grabbed_info_text:
        return None, None
    m = _IMAGE_URL_RE.search(grabbed_info_text)
    if not m:
        return None, None
    url = m.group(1).strip()
    pid_match = re.search(r"#(\d+)", grabbed_info_text)
    return url, (pid_match.group(1) if pid_match else "")


# ----------------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------------

PAGE_ID = "prompt-workshop"

# The fourteen category boxes, in Build Prompt order: the six Anima
# tag-order sections, then the eight manual extras appended after General.
CORE_FIELDS = ("quality", "subject", "character", "series", "artist", "general")
EXTRA_FIELDS = ("composition", "pose", "clothing", "setting", "lighting", "features", "style", "content")
GRAB_STATES = tuple(f"grab_{name}" for name in CORE_FIELDS)

_EXTRA_BOXES = (
    ("composition", "7. Composition / Framing", "Composition / Framing", "cowboy shot, from above, close-up, upper body, from behind, pov"),
    ("pose", "8. Pose / Action / Expression", "Pose / Action / Expression", "sitting, arms up, hand on hip, looking at viewer, smile, open mouth"),
    ("clothing", "9. Clothing / Attire", "Clothing / Attire", "school uniform, dress, thighhighs, gloves, hat, glasses, jewelry"),
    ("setting", "10. Setting / Background", "Setting / Background", "outdoors, classroom, night, city, simple background, white background"),
    ("lighting", "11. Lighting / Atmosphere / Effects", "Lighting / Atmosphere / Effects", "cinematic lighting, backlighting, rim light, bloom, depth of field, petals"),
    ("features", "12. Physical Features", "Physical Features", "blue eyes, long hair, twintails, animal ears, freckles"),
    ("style", "13. Style / Aesthetic", "Style / Aesthetic", "retro artstyle, sketch, watercolor, flat color, 1990s (style)"),
    ("content", "14. Content / NSFW", "Content / NSFW", "explicit content tags kept separate from the rest"),
)

_SEND_CATEGORY = (
    ("quality", "→ 1. Quality"),
    ("subject", "→ 2. Subject"),
    ("character", "→ 3. Character"),
    ("series", "→ 4. Series"),
    ("artist", "→ 5. Artist"),
    ("general", "→ 6. General"),
)

# Values "Clear All Fields" resets (credentials are persistent setup, and
# the preview image stays, as before).
_CLEARED = {
    "consolidated": "",
    **{name: "" for name in CORE_FIELDS + EXTRA_FIELDS},
    "search_tag": "",
    "blacklist_tags": "",
    "blacklist_cats": [],
    "grabbed": "",
    "grabbed_info": "",
    "post_url": "",
    "source": "Danbooru",
    "rating": "All ratings",
    "post_id": "",
    "count_filter_enabled": False,
    "count_filter_op": "≥",
    "count_filter_threshold": 50,
    "count_filter_threshold_high": 100,
    "count_filter_targets": [],
    "hunt_mode": False,
    **{name: "" for name in GRAB_STATES},
}

_PRETTY = {"gelbooru": "Gelbooru", "rule34": "Rule34"}


def _creds_status(source_key: str):
    api_key, user_id = _get_credentials(source_key)
    if api_key and user_id:
        return ui.message("✓ {{source}} credentials set.", source=_PRETTY[source_key])
    return ui.message("⚠ Not set — {{source}} grabs return 401.", source=_PRETTY[source_key])


def _creds_set(source_key: str) -> bool:
    return all(_get_credentials(source_key))


def _send(name: str, label: str, props: dict, **kwargs):
    return ui.Custom(name, "send-button", None, props={"label": label, **props}, label="", persist=False, submit=False, **kwargs)


def _task_button(action: str, label: str):
    """A button running ``action`` as a page task (progress, Cancel)."""
    return ui.Custom(action, "task-runner", None, props={"action": action, "label": label}, label="", persist=False, submit=False)


def _credentials_section(source_key: str, title: str, help_text: str):
    prefix = "gel" if source_key == "gelbooru" else "r34"
    return ui.Section(
        title,
        open=not _creds_set(source_key),
        children=[
            ui.Markdown(help_text),
            ui.Row(
                ui.Custom(f"{prefix}_api_key", "secret-text", "", props={"placeholder": "Saved keys are not shown; leave empty to keep"}, label="api_key", persist=False, submit=False),
                ui.Text(f"{prefix}_user_id", lambda: _get_credentials(source_key)[1], label="user_id", persist=False, submit=False),
            ),
            ui.Row(
                ui.Button("Save", f"save_{source_key}", "primary"),
                ui.Button("Clear", f"clear_{source_key}", "danger"),
            ),
            ui.Output(f"{prefix}_status", kind="text"),
        ],
    )


class PromptWorkshopPage(pages.Page):
    def params(self):
        return ui.Column(
            ui.Markdown(
                "Build prompts in the Anima tag order: "
                "`[quality / meta / year / safety]  [1girl / 1boy / ...]  "
                "[character]  [series]  [artist]  [general]`. Fill the boxes "
                "below in any order then click **Build Prompt**, or use the "
                "**Booru Random Tag Grabber** at the bottom to pull tags from "
                "a Danbooru / Aibooru / Gelbooru / Rule34 post automatically."
            ),
            ui.Text(
                "consolidated",
                "",
                lines=4,
                prompt=True,
                label="Consolidated Prompt — copy this into txt2img",
                placeholder="Click 'Build Prompt' after filling in the category boxes below.",
            ),
            ui.Row(
                ui.Button("Build Prompt", "build", "primary"),
                _send("send_consolidated", "Send to txt2img Prompt", {"to": "prompt", "source": "consolidated", "mode": "append"}),
                _send("replace_consolidated", "Replace txt2img Prompt", {"to": "prompt", "source": "consolidated", "mode": "replace"}),
                ui.Button("Clear All Fields", "clear_all", "danger"),
            ),
            ui.Separator(),
            ui.Markdown("### Tag Categories (in Anima tag order)"),
            ui.Row(
                ui.Column(
                    ui.Text("quality", "", lines=2, prompt=True, label="1. Quality / Meta / Year / Safety", placeholder="masterpiece, best quality, score_7, highres, year 2024, newest, safe"),
                    ui.Text("subject", "", prompt=True, label="2. Subject Count   (1girl / 1boy / 1other / 2girls / solo / ...)", placeholder="1girl, solo"),
                    ui.Text("character", "", prompt=True, label="3. Character(s)", placeholder="oomuro sakurako"),
                ),
                ui.Column(
                    ui.Text("series", "", prompt=True, label="4. Series", placeholder="yuru yuri"),
                    ui.Text("artist", "", prompt=True, label="5. Artist   (@ will be auto-prepended)", placeholder="nnn yryr   →   will become   @nnn yryr"),
                    ui.Text("general", "", lines=4, prompt=True, label="6. General Tags", placeholder="smile, brown hair, hat, long hair, looking at viewer, ..."),
                ),
            ),
            # Per-category sends append one box to the txt2img prompt and stay
            # here (you might send several in a row).
            ui.Markdown("**Send a single category to txt2img:**"),
            ui.Row(*(_send(f"send_{name}", label, {"to": "prompt", "source": name, "mode": "append", "navigate": False}) for name, label in _SEND_CATEGORY)),
            ui.Separator(),
            ui.Markdown(
                "### Additional Categories (manual)\n"
                "Optional buckets for organising your own tags. They're appended "
                "**after General** when you click *Build Prompt*, and support tag "
                "autocomplete. They are not touched by the grabber, blacklist, or "
                "count filter. Collapsed by default — open only what you need."
            ),
            *(ui.Section(title, open=False, children=[ui.Text(name, "", lines=2, prompt=True, label=label, placeholder=placeholder)]) for name, title, label, placeholder in _EXTRA_BOXES),
            ui.Separator(),
            ui.Markdown(
                "### Booru Random Tag Grabber\n"
                "Pulls a random image from **Danbooru**, **Aibooru**, **Gelbooru**, "
                "or **Rule34** and rewrites its tags into Anima format. "
                "Gelbooru and Rule34 each require their own `api_key` + `user_id` "
                "(set them in the credentials sections below)."
            ),
            ui.Row(
                ui.Column(
                    _credentials_section(
                        "gelbooru",
                        "Gelbooru API credentials",
                        "Free — get them from "
                        "**[Account → Options](https://gelbooru.com/index.php?page=account&s=options)** "
                        "(*API Access Credentials* section). Stored locally; "
                        "sent only to Gelbooru. Does **not** authenticate Rule34.",
                    )
                ),
                ui.Column(
                    _credentials_section(
                        "rule34",
                        "Rule34 API credentials",
                        "Required since **August 2025**. Free — get them at "
                        "**[api.rule34.xxx](https://api.rule34.xxx/)**. Stored "
                        "locally under separate keys from Gelbooru.",
                    )
                ),
            ),
            ui.Row(
                ui.Choice("source", _SOURCE_CHOICES, "Danbooru", style="radio", label="Booru source"),
                ui.Choice("rating", _RATING_CHOICES, "All ratings", style="radio", label="Rating filter"),
            ),
            # Anima-format (spaces, lowercase) or booru format; several tags
            # are an AND filter (Danbooru/Aibooru anonymously: 2 incl. rating).
            ui.Text(
                "search_tag",
                "",
                label="Search tag(s) — find a random post with these tags (empty = any post)",
                placeholder="long hair · long hair, blue eyes · Danbooru/Aibooru anon: 2 tags total incl. rating",
            ),
            ui.Row(
                ui.Text("blacklist_tags", "", lines=2, label="Blacklist tags — won't appear in grabbed output", placeholder="e.g. lowres, jpeg artifacts, watermark, blurry", width=6),
                ui.MultiChoice("blacklist_cats", _CATEGORY_LABELS, (), style="checkboxes", label="Blacklist categories — whole sections dropped", width=6),
            ),
            ui.Row(
                ui.Toggle("count_filter_enabled", False, label="Filter tags by post count"),
                ui.Choice("count_filter_op", _COUNT_OP_CHOICES, "≥", label="Comparator"),
                ui.Number("count_filter_threshold", 50, minimum=0, step=1, precision=0, label="Min / value"),
                ui.Number("count_filter_threshold_high", 100, minimum=0, step=1, precision=0, label="Max (range only)"),
                ui.Toggle("hunt_mode", False, label="Hunt mode (Danbooru/Aibooru only)"),
            ),
            ui.MultiChoice("count_filter_targets", _CATEGORY_LABELS, (), style="checkboxes", label="Apply count filter only to these categories (empty = all)"),
            ui.Section(
                "ℹ How the count filter and hunt mode work",
                open=False,
                children=[
                    ui.Markdown(
                        "**Comparators.** `≥` / `≤` / `=` use only the Min/value box. "
                        "`between` keeps tags whose count is inside `[low, high]` "
                        "(inclusive). `not between` drops tags inside that range — "
                        "useful for *omit if greater than 50 but less than 100*-style "
                        "filters.\n\n"
                        "**Scope.** By default the filter applies to every tag in the "
                        "grabbed post. Tick boxes under *Apply count filter only to "
                        "these categories* to scope it (e.g. only Artist). Rating text "
                        "is always exempt; tags the booru doesn't recognise count as 0.\n\n"
                        "**Hunt mode** (Danbooru and Aibooru only) actively searches "
                        "for a post that has at least one tag in the target category "
                        "meeting the count criteria, instead of grabbing randomly and "
                        "hoping. It uses Danbooru's `/tags.json` endpoint, which is "
                        "fast and precise. Requires the count filter on and at least "
                        "one target category. Not available for Gelbooru / Rule34 "
                        "because those sites don't expose a comparable endpoint."
                    )
                ],
            ),
            _task_button("grab", "Grab Random Tags"),
            ui.Row(
                _send("open_post", "🔗 Open Post in New Tab", {"to": "url", "source": "post_url", "variant": "outline"}),
                _send("send_image", "📷 Send Image to JoyCaption", {"to": "page", "page": "joycaption", "param": "image", "source": "preview_image", "action": "download", "variant": "outline"}),
                ui.Button("🖼 Download for Preview", "download"),
                # Shown, never edited (as before): the server keeps what a grab found
                ui.Text("post_url", "", label="Post URL", width="fill", enabled_if=False),
            ),
            ui.Output("handoff_status", kind="text"),
            ui.Image("preview_image", label="Preview (downloaded post image — also used as source for 'Send to JoyCaption')"),
            _send("send_preview", "📤 Send Previewed Image to JoyCaption", {"to": "page", "page": "joycaption", "param": "image", "source": "preview_image", "variant": "outline"}),
            ui.Text("grabbed_info", "", lines=2, label="Grabbed from", enabled_if=False),
            ui.Text("grabbed", "", lines=4, prompt=True, label="Grabbed Tags (Anima-formatted, copyable)"),
            ui.Button("↑ Send Grabbed Tags into Category Fields Above", "send_grabbed"),
            ui.Section(
                "Grab from specific post ID",
                open=False,
                children=[
                    ui.Markdown(
                        "Paste a post ID (the number in the URL of a post page on "
                        "Danbooru, Aibooru, Gelbooru, or Rule34) to pull tags from "
                        "that exact post. Uses whichever **Source** is selected above. "
                        "Blacklists and the count filter apply; rating filter, search "
                        "tag, and hunt mode don't (you've already picked the post)."
                    ),
                    ui.Row(
                        ui.Text("post_id", "", label="Post ID", placeholder="e.g. 1234567", width="fill"),
                        _task_button("grab_by_id", "Grab from ID"),
                    ),
                ],
            ),
            # The last grab's per-category breakdown, for "Send Grabbed Tags
            # into Category Fields Above", and the URL the preview shows.
            *(ui.State(name, "", label="") for name in GRAB_STATES),
            ui.State("preview_url", "", label=""),
        )

    # ---- Prompt building ----------------------------------------------------

    @ui.action("refresh")
    def refresh(self, ctx, values):
        """On load: the credential status lines (local config only, no network)."""
        return {"__outputs__": {"gel_status": _creds_status("gelbooru"), "r34_status": _creds_status("rule34")}}

    @ui.action("build")
    def build(self, ctx, values):
        resolved = self.values(ctx, values, uploads=False)
        return {"consolidated": {"value": build_prompt(*(resolved[name] for name in CORE_FIELDS + EXTRA_FIELDS))}}

    @ui.action("clear_all")
    def clear_all(self, ctx, values):
        _remember_grab(ctx.user, "", "")  # the grabbed info is cleared: nothing left to download
        return {name: {"value": value} for name, value in _CLEARED.items()}

    # ---- Credentials ----------------------------------------------------------

    def _save(self, source_key: str, prefix: str, values) -> dict:
        api_key = str(values.get(f"{prefix}_api_key") or "").strip()
        user_id = str(values.get(f"{prefix}_user_id") or "").strip()
        saved_key, _ = _get_credentials(source_key)
        save_credentials(source_key, api_key or saved_key, user_id)
        status = _creds_status(source_key)
        return {f"{prefix}_api_key": {"value": ""}, "__outputs__": {f"{prefix}_status": status}}

    def _clear(self, source_key: str, prefix: str) -> dict:
        save_credentials(source_key, "", "")
        return {f"{prefix}_api_key": {"value": ""}, f"{prefix}_user_id": {"value": ""}, "__outputs__": {f"{prefix}_status": _creds_status(source_key)}}

    @ui.action("save_gelbooru")
    def save_gelbooru(self, ctx, values):
        return self._save("gelbooru", "gel", values)

    @ui.action("save_rule34")
    def save_rule34(self, ctx, values):
        return self._save("rule34", "r34", values)

    @ui.action("clear_gelbooru")
    def clear_gelbooru(self, ctx, values):
        return self._clear("gelbooru", "gel")

    @ui.action("clear_rule34")
    def clear_rule34(self, ctx, values):
        return self._clear("rule34", "r34")

    # ---- Grabbing (outbound HTTP, on the user's click only) --------------------

    @staticmethod
    def _grab_patch(result) -> dict:
        grabbed, *categories, info, post_url = result
        patch = {"grabbed": {"value": grabbed}, "grabbed_info": {"value": info}, "post_url": {"value": post_url}}
        patch.update({name: {"value": value} for name, value in zip(GRAB_STATES, categories)})
        return patch

    def _filters(self, resolved) -> tuple:
        return (
            resolved["blacklist_tags"],
            resolved["blacklist_cats"],
            resolved["count_filter_enabled"],
            resolved["count_filter_op"],
            int(resolved["count_filter_threshold"]),
            int(resolved["count_filter_threshold_high"]),
            resolved["count_filter_targets"],
        )

    def _start_grab(self, ctx, name: str, source: str, grab, progress) -> dict:
        """Run ``grab()`` as a page task (no GPU): requests can take long
        (hunt mode makes many), so the page shows progress and can cancel."""
        user = ctx.user

        def work(task):
            progress(task)
            _grab_thread.task = task
            try:
                result = grab()
            except _GrabCancelled:
                raise pages.TaskCancelled() from None
            finally:
                _grab_thread.task = None
            _remember_grab(user, source, result[-2])
            return self._grab_patch(result)

        return self.start_task(ctx, name, work)

    @ui.action("grab")
    def grab(self, ctx, values):
        resolved = self.values(ctx, values, uploads=False)
        blacklist_tags, blacklist_cats, *count = self._filters(resolved)
        source, hunt = resolved["source"], resolved["hunt_mode"]

        def progress(task):
            if hunt:
                task.progress(None, "Hunting for a post that passes the count filter…")
            else:
                task.progress(None, "Grabbing a random post…")

        return self._start_grab(
            ctx, "grab", source,
            lambda: grab_random(source, resolved["rating"], resolved["search_tag"], blacklist_tags, blacklist_cats, *count, hunt),
            progress,
        )

    @ui.action("grab_by_id")
    def grab_by_id(self, ctx, values):
        resolved = self.values(ctx, values, uploads=False)
        source, post_id = resolved["source"], resolved["post_id"]
        return self._start_grab(
            ctx, "grab_by_id", source,
            lambda: grab_by_id(source, post_id, *self._filters(resolved)),
            lambda task: task.progress(None, "Fetching post #{{id}}…", id=post_id.strip()),
        )

    @ui.action("send_grabbed")
    def send_grabbed(self, ctx, values):
        return {name: {"value": str(values.get(state) or "")} for name, state in zip(CORE_FIELDS, GRAB_STATES)}

    @ui.action("download")
    def download(self, ctx, values):
        """Download the image of this user's last grab (remembered by the
        server, never read from the page) into the preview (an upload); a
        preview of the same URL is reused, not fetched again. A failure
        clears the preview, so nothing stale is sent to JoyCaption."""
        grabbed = _last_grab(ctx.user)
        if grabbed is None:
            status = ui.message("❗ No image URL on the most recent grab — click 'Grab Random Tags' first.")
            return {"__outputs__": {"handoff_status": status}, **ui.toast(status, "warning")}
        source, url, pid_hint = grabbed
        if url == values.get("preview_url") and values.get("preview_image"):
            return {"__outputs__": {"handoff_status": ui.message("✅ Reusing already-downloaded preview (no re-fetch).")}}
        data, fname, err = _download_image(url, pid_hint, _BOORU_SOURCES[source]["image_hosts"])
        store = ctx.files
        if err or data is None or store is None:
            status = ui.message("❌ {{error}}", error=err or "the server's file store is unavailable")
            return {
                "preview_image": {"value": None},
                "preview_url": {"value": ""},
                "__outputs__": {"handoff_status": status},
                **ui.toast(status, "error"),
            }
        return {
            "preview_image": {"value": store.store_bytes(data, fname)},
            "preview_url": {"value": url},
            "__outputs__": {"handoff_status": ui.message("✅ Image downloaded ({{name}}).", name=fname)},
        }


pages.register(PromptWorkshopPage(PAGE_ID, "Prompt Workshop", "wand", on_load="refresh"))
