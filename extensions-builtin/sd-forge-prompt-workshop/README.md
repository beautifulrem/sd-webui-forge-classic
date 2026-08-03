# sd-forge-prompt-workshop

An **Anima Workshop** tab for [Haoming02/sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic) (the `neo` branch), designed for use with the [Anima](https://huggingface.co/circlestone-labs/Anima) image model.

Anima expects prompts written in a specific order:

```
[quality / meta / year / safety]  [1girl / 1boy / 1other / ...]  [character]  [series]  [artist]  [general]
```

…with a few format rules: tags are lowercase, words are separated by spaces (not underscores), every artist tag must begin with `@`, and the `score_N` quality tags keep their underscore. The extension provides one box per section, builds prompts in the correct order on demand, and can also pull a random post's tags from **Danbooru, Aibooru, Gelbooru, or Rule34** and rewrite them into Anima's format for you.

## What the tab contains

A new top-level tab called **Anima Workshop**, placed between **img2img** and **Extras**.

### Prompt builder

- **Six category boxes**, one per section of the Anima tag order: Quality/Meta/Year/Safety, Subject (1girl etc.), Character, Series, Artist, General. Fill in whichever ones you need, in any order.
- **Build Prompt** consolidates all six into a single comma-separated prompt in the correct order. Duplicate tags across boxes are automatically dropped (Anima uses tag dropout during training, so repeating a tag does not strengthen it).
- **Send to txt2img Prompt** appends the consolidated prompt to the txt2img positive prompt field and switches to that tab. It always appends, never replaces, so existing text in the txt2img prompt is preserved.
- **Per-category Send buttons** (one for each of the six category boxes) append just that one box's contents to txt2img. These do not switch tabs, so you can send several categories in a row.
- **Clear All Fields** wipes every input on the tab, including grabber state, but leaves your saved Gelbooru and Rule34 credentials alone.

### Booru random tag grabber

Pulls a random post from **Danbooru**, **Aibooru**, **Gelbooru**, or **Rule34**, reads its tags, and rewrites them into Anima format. Parentheses in tags (like the `(blue archive)` in `mutsuki (blue archive)`) are backslash-escaped automatically so Forge / A1111 doesn't read them as prompt-weighting syntax — you'll see `mutsuki \(blue archive\)` and `@asura \(asurauser\)` in the output.

- **Source** — Danbooru, Aibooru, Gelbooru, or Rule34. *Aibooru* is a Danbooru fork dedicated to AI-generated images; *Rule34* is a Gelbooru fork.
- **Rating filter** — `All ratings`, `Safe / General`, `Sensitive`, `Questionable`, or `Explicit`. On Rule34 there's no separate *Sensitive* rating (it uses the older three-rating system), so picking *Sensitive* there gracefully falls back to *Questionable*.
- **Search tag(s)** — restrict the random pick to posts that contain specific tags. You can type tags Anima-style (`long hair`) or booru-style (`long_hair`); both work. Multiple tags go in comma-separated. **Important:** anonymous searches on Danbooru and Aibooru are limited to 2 tags total (including the rating tag). The extension checks this before making any request and tells you if you've exceeded it. Gelbooru and Rule34 have no such limit.
- **Blacklist tags** — a comma-separated list of tags. These are filtered out of the grabbed output (the post still gets pulled, but those tags don't appear in your prompt).
- **Blacklist categories** — checkboxes for the six tag categories. Ticking one means that whole section is dropped from the grab. For example, ticking *Artist* lets you study a post's general tagging without copying its artist into your prompt.
- **Filter tags by post count** — when ticked, each tag on the grabbed post is checked against the booru's reported `post_count` using the comparator and threshold(s) you set, and tags that fail the check are dropped. Five comparators are available:
    - `≥`, `≤`, `=` — single-threshold (use the first threshold box only). Example: `≥ 50` drops obscure artists, one-off characters, and other long-tail tags so they don't bleed into your prompt.
    - `between` — keep tags whose count is **inside** `[low, high]` (inclusive). Uses both threshold boxes.
    - `not between` — drop tags whose count is **inside** `[low, high]`. Useful for *omit if greater than 50 but less than 100*-style filters where you want to skip the awkward middle band.

    By default the filter applies to **all** tag categories. Use **Apply count filter only to these categories** to scope it to a subset (e.g. only Artist), leaving every other category untouched. Rating text is exempt from the filter; tags the booru doesn't recognise are treated as count 0 (so a `≥` filter drops them). Off by default — no extra API calls are made when off.
- **Hunt mode** — by default, the count filter runs *after* a random post has been picked, which means you can get a post whose target tags all fail the filter (e.g. you wanted an obscure artist but the random post had a famous one). Hunt mode flips this: it **actively searches** for a post where at least one tag in the target category satisfies the count comparator.
    - Requires the count filter to be on and at least one target category selected.
    - On Danbooru and Aibooru this uses the site's `/tags.json` endpoint, which supports server-side filtering by category + post count — fast and precise.
    - On Gelbooru and Rule34 there's no such endpoint, so the extension scans a batch of candidate posts client-side and picks the first one that satisfies the criteria. This is slower and may give up if no post in the batch matches; in that case just click *Grab Random Tags* again or widen your criteria.
    - When hunt mode finds a hit, the grab info line is decorated with `🎯 Hunt hit on tag '…'` so you can see what tag landed it.
- **Grab Random Tags** — runs the grab with the above settings.
- **🔗 Open Post in New Tab** — opens the source post on the booru's website so you can see the image.
- **Grab from specific post ID** (collapsible section) — if you already know the post ID of a specific Danbooru, Aibooru, Gelbooru, or Rule34 post you want to learn from, paste it here. Uses the same Source radio as the random grab. The blacklists and count filter apply; rating filter, search tag, and hunt mode don't (you've already picked the exact post).

The grabbed tags are shown in a copyable box. A separate button distributes them back into the six category fields so you can edit them before building.

### Gelbooru and Rule34 API credentials

Both Gelbooru and Rule34 **require an `api_key` and `user_id` on every API call**:

- **Gelbooru** added this requirement in mid-2022.
- **Rule34** added the same requirement in August 2025.

Without credentials, every request to either site returns `401 Unauthorized`. The credentials are free in both cases, and they're stored separately so authenticating to one does not authenticate you to the other.

**Gelbooru:**

1. Log in to Gelbooru.
2. Go to **Account → Options** ([direct link](https://gelbooru.com/index.php?page=account&s=options)) and scroll to *API Access Credentials* near the bottom of the page. Copy the `api_key` and `user_id` values.
3. On the Anima Workshop tab, expand the **Gelbooru API credentials** section, paste both values, and click *Save Credentials*. The section is open by default until credentials are saved.

**Rule34:**

1. Log in to Rule34 (`rule34.xxx`).
2. Visit [api.rule34.xxx](https://api.rule34.xxx/) for the API credentials page and copy the `api_key` and `user_id`.
3. On the Anima Workshop tab, expand the **Rule34 API credentials** section, paste both values, and click *Save Credentials*. Like the Gelbooru section, it's open by default until credentials are saved.

Credentials are written to a file called `config.json` next to the extension, under prefixed keys so the two sites' credentials don't collide. Each set of credentials is only ever sent to the corresponding booru. **Danbooru and Aibooru** do not require credentials, but their anonymous searches are limited to 2 tags total (including the rating tag).

## Tag autocomplete compatibility

If you have the [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete) extension installed, Anima Workshop ships a small bridge file (`javascript/tagcomplete_compat.js`) that tries to enable tag autocomplete on the ten text-input fields in this tab: the six category boxes, the Consolidated Prompt and Grabbed Tags boxes, the booru search-tag input, and the blacklist-tags input.

**How it works.** Tagcomplete attaches autocomplete to a fixed list of textareas it knows about (txt2img prompt, img2img prompt, and a few specific third-party extensions it has built-in support for). It does not provide a public API for other extensions to register their own textareas. The bridge in this extension does a best-effort hook: after tagcomplete finishes setting up the standard textareas, it monkey-patches tagcomplete's internal identifier function so it recognises Anima Workshop's fields, then attaches the same listeners tagcomplete uses internally.

**When it will work.** Older versions of tagcomplete and most forks expose their helper functions as top-level globals on `window`. The bridge can hook into those and autocomplete should work on all ten Anima Workshop fields without any further setup.

**When it won't work.** Newer modular versions of tagcomplete may wrap their helpers inside a module so they aren't accessible from outside. In that case the bridge will log a clear message to the browser console explaining the situation. If this happens, the only way to get autocomplete in Anima Workshop is for tagcomplete itself to add the field IDs (`pw_quality`, `pw_subject`, `pw_character`, `pw_series`, `pw_artist`, `pw_general`, `pw_consolidated`, `pw_grabbed`, `pw_search_tag`, `pw_blacklist_tags`) to its supported textarea list. You can open a feature request at the [tagcomplete issues page](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/issues) to ask for this.

**Checking whether it worked.** Open the browser developer console after the UI loads. You'll see one of these messages:

- `[anima-workshop:tagcomplete] enabled tagcomplete on N Anima Workshop textareas` — working.
- `[anima-workshop:tagcomplete] tagcomplete is installed and active on txt2img/img2img, but its helper functions aren't exposed on window in your version.` — bridge can't hook in on your version of tagcomplete.
- `[anima-workshop:tagcomplete] tagcomplete not detected after 15s.` — tagcomplete isn't installed (this is the normal no-op message when you don't use tagcomplete).

The bridge is completely isolated from the main extension. If it can't hook in, every other Anima Workshop feature continues to work normally.

## Tag format rules applied automatically

Whether you type tags or grab them from a booru, the extension applies the rules from the Anima model card:

| Input                 | Output                |
|-----------------------|-----------------------|
| `Oomuro_Sakurako`     | `oomuro sakurako`     |
| `long_hair`           | `long hair`           |
| `score_7`             | `score_7`  *(underscore preserved — Anima's only exception)* |
| `nnn yryr` *(in the artist box)* | `@nnn yryr`           |
| `@@artist`            | `@artist`             |

Booru rating values are mapped onto Anima's vocabulary. Danbooru and Aibooru use `g/s/q/e` (general / sensitive / questionable / explicit); Gelbooru uses spelled-out `general/sensitive/questionable/explicit`; Rule34 uses an older three-rating system (`safe/questionable/explicit`, with `sensitive` mapped to `questionable`). All four become `safe`, `sensitive`, `nsfw`, or `explicit` in the prompt.

Subject-count tags (`1girl`, `2boys`, `multiple girls`, `no humans`, `solo`, etc.) are automatically split out of the general bucket into the Subject field, since the Anima tag order treats them as a separate section.

For Danbooru and Aibooru, the extension uses the API's per-category tag fields directly. For Gelbooru and Rule34, the API only returns a flat tag list per post, so the extension makes one additional call to look up each tag's category (general, artist, copyright/series, character, or meta).

## A few notes about Anima from its author

These are points raised in tdrussell's discussions on the Anima Hugging Face page that are worth knowing while building prompts:

- **Single artist tags work more reliably than multiple.** Multiple `@artist` tags in one prompt tend to degrade style consistency (discussion [#112](https://huggingface.co/circlestone-labs/Anima/discussions/112)).
- **Prompt weighting with `(tag:1.2)` syntax is not fully implemented yet** (discussion [#135](https://huggingface.co/circlestone-labs/Anima/discussions/135)). It will not behave the same way it does in other models.
- **Leading whitespace affects output.** `"best quality"` and `" best quality"` (with a leading space) produce slightly different stylistic tendencies (discussion [#57](https://huggingface.co/circlestone-labs/Anima/discussions/57)). The extension strips whitespace from every tag, so you will always be on the no-leading-space side.

## Install

1. Place the extension folder inside Forge's `extensions/` directory. Final layout:

    ```
    sd-webui-forge-neo/
    └── extensions/
        └── sd-forge-prompt-workshop/
            ├── README.md
            ├── config.json              (created automatically when you save credentials)
            ├── scripts/
            │   └── prompt_workshop.py
            └── javascript/
                ├── prompt_workshop.js
                └── tagcomplete_compat.js
    ```

    Both the `scripts/` and `javascript/` subfolders are required. Forge only discovers Python files inside `scripts/`, and only auto-loads JS from `javascript/`.

2. Fully close and restart `webui-user.bat`. The first startup is what registers the new tab; *Reload UI* is only enough for re-rendering.

3. The console should show, in order:

    ```
    [prompt-workshop] script file is being imported
    [prompt-workshop] on_ui_tabs callback registered
    [prompt-workshop] UI tab built
    ```

    And in the browser's developer console after the UI loads:

    ```
    [anima-workshop] tab reordered to before Extras
    ```

## Troubleshooting

**No `script file is being imported` line at startup** → Forge isn't finding the file. Either the folder structure is wrong or the extension is disabled in *Settings → Extensions*.

**`script file is being imported` shows but no `UI tab built`** → an exception fired while building the UI. There will be a traceback right after; it's almost always a Gradio version mismatch or a typo somewhere.

**The tab appears at the far right instead of between img2img and Extras** → the JavaScript that does the reordering didn't run, or ran before the UI had finished mounting. Check the browser developer console for `[anima-workshop]` messages. You can also reorder tabs manually under *Settings → User interface → Tab order*.

**Booru grab returns `[Gelbooru API credentials not set ...]` or `[Rule34 API credentials not set ...]`** → that booru requires credentials. See the *Gelbooru and Rule34 API credentials* section above.

**Booru grab returns `[error fetching from Gelbooru (count): 401 Unauthorized ...]` or the equivalent for Rule34** → the credentials are entered but the booru is rejecting them. Double-check that the `api_key` and `user_id` were copied exactly from the account / API credentials page (no extra spaces, no missing characters), then click *Save Credentials* again on the relevant section. Note that Gelbooru and Rule34 credentials are separate — entering Gelbooru credentials will not authenticate you to Rule34, and vice versa.

**Booru grab returns `[Danbooru's anonymous search is limited to 2 tags ...]` or the equivalent for Aibooru** → this is working as intended. Anonymous Danbooru and Aibooru cap queries at 2 tags total. Use fewer search tags, set Rating filter to *All ratings*, or switch the source to Gelbooru or Rule34.

**Booru grab returns `[no posts matched that search ...]`** → your search tags combined with the rating filter returned zero hits. Loosen the search or change the rating.

**Hunt mode returns `[Hunt mode requires the post-count filter to be enabled ...]` or `[Hunt mode needs at least one target category ...]`** → hunt mode needs both the count filter on and at least one target category ticked under *Apply count filter only to these categories*. It uses those settings to decide which posts qualify, so it can't run without them.

**Hunt mode on Gelbooru / Rule34 keeps coming back empty** → those sites don't support server-side filtering of tags by count, so the extension scans a batch of candidate posts and may not find one that matches. Click *Grab Random Tags* again or widen the count threshold / range. Danbooru and Aibooru don't have this issue because they support precise server-side tag-by-count searches.

**Booru grab returns `[error fetching from ...]` for any other reason** → the WebUI machine couldn't reach the booru. This could be a firewall, DNS, or captive portal issue. The post is fetched server-side from the Forge process, not the browser. The error text in the box is the exception message and usually identifies the problem (`ConnectionError`, `Timeout`, `429 Too Many Requests`, and so on).

**Grab returns nothing or empty tags from Gelbooru or Rule34** → the random page-id occasionally lands on a post with no indexed tags. Click *Grab Random Tags* again.

## Notes on rate limits

- **Danbooru** and **Aibooru** advertise a public rate limit of 10 requests per second; the extension makes at most one or two requests per button click (the post fetch plus, when the count filter is on, a tags-info lookup).
- **Gelbooru** and **Rule34** don't document explicit limits but rate-limit aggressive scrapers. Each random grab does one post fetch plus a small number of batched tag-category lookups (up to 40 tag names per call). Hunt mode on these sites adds one batch post-scan call. Do not loop the Grab button.
- All four APIs are called with a descriptive `User-Agent` header so the boorus can identify and contact the extension's source if there's ever a problem.

## Credits

- **Anima** by [tdrussell](https://huggingface.co/tdrussell) / [circlestone-labs](https://huggingface.co/circlestone-labs) — the model this extension is built around.
- **Forge classic / neo** by [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic) — the host environment.
