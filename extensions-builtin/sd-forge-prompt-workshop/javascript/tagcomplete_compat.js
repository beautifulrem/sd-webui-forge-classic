// tagcomplete_compat.js
// -----------------------------------------------------------------------------
// Best-effort bridge between Anima Workshop and the
// DominikDoom/a1111-sd-webui-tagcomplete extension.
//
// Tagcomplete attaches autocomplete to a HARDCODED list of textareas
// (txt2img_prompt, img2img_prompt, and their negatives, plus a few specific
// third-party extensions it has explicit support for). There is no public
// registration API: per tagcomplete issue #244, the maintainer says third-
// party integration has to be added inside tagcomplete itself.
//
// What this bridge does:
//   1. Waits for tagcomplete to finish setting up its built-in textareas
//      (detected by the `autocomplete` class it adds to them).
//   2. Looks for tagcomplete's internal helper functions on `window` --
//      `getTextAreaIdentifier`, `autocomplete`, `navigateInList`,
//      `hideResults`, `createResultsDiv`. Older + many forked versions
//      have these as top-level globals.
//   3. Monkey-patches `getTextAreaIdentifier` so it returns a unique
//      identifier for each Anima Workshop textarea instead of the empty
//      string it currently returns for "unknown" textareas. (The empty
//      string would make every Anima textarea share one popup div, which
//      is the actual bug that breaks multi-textarea integration.)
//   4. Replicates tagcomplete's per-textarea setup work on each Anima
//      Workshop textarea: create a results div, hide it, attach `input` +
//      `focusout` + `keydown` listeners that call tagcomplete's existing
//      handlers.
//
// What this bridge does NOT do:
//   - It does not modify tagcomplete's source code on disk.
//   - It does not change tagcomplete's behaviour on any of its built-in
//     textareas; the monkey-patch falls back to the original function for
//     everything that isn't one of ours.
//   - On modern tagcomplete versions where these functions are sealed
//     inside an IIFE / module and not on `window`, this bridge can't hook
//     in. It will log a clear message saying so and stop.
// -----------------------------------------------------------------------------

(function () {
    "use strict";

    const LOG = "[anima-workshop:tagcomplete]";

    // elem_ids of Anima Workshop's editable textareas. The two read-only
    // boxes (Post URL, Grabbed-from info) are deliberately excluded -- they
    // don't accept user input. Credentials are excluded for the same reason
    // they shouldn't have autocomplete (they're API keys, not tags).
    const PW_TEXTAREA_IDS = [
        "pw_quality",
        "pw_subject",
        "pw_character",
        "pw_series",
        "pw_artist",
        "pw_general",
        // manual extra categories (Build Prompt + autocomplete only;
        // not part of the grabber). Appended after General on build.
        "pw_composition",
        "pw_pose",
        "pw_clothing",
        "pw_setting",
        "pw_lighting",
        "pw_features",
        "pw_style",
        "pw_content",
        "pw_consolidated",
        "pw_grabbed",
        "pw_search_tag",
        "pw_blacklist_tags",
    ];

    function gApp() {
        return (typeof gradioApp === "function") ? gradioApp() : document;
    }

    function isTagcompleteSetUpOnBuiltIns() {
        // tagcomplete adds the class `autocomplete` to every textarea it has
        // bound. If it's bound to txt2img's positive prompt, tagcomplete has
        // finished its first-pass setup and we can safely run ours.
        const t2i = gApp().querySelector("#txt2img_prompt textarea");
        return !!(t2i && t2i.classList.contains("autocomplete"));
    }

    function getMyTextareas() {
        const out = [];
        for (const id of PW_TEXTAREA_IDS) {
            const wrap = gApp().querySelector("#" + id);
            if (!wrap) continue;
            const ta = wrap.querySelector("textarea");
            if (ta) out.push({ id, ta });
        }
        return out;
    }

    // Patch tagcomplete's identifier-resolution function so it returns a
    // unique tag for each Anima Workshop textarea (instead of an empty
    // string, which would collapse all of them onto the same popup div).
    // Returns true if the patch was applied; false if the function isn't
    // on `window` (modern modular tagcomplete -> bridge can't proceed).
    function patchIdentifier() {
        if (typeof window.getTextAreaIdentifier !== "function") {
            return false;
        }
        if (window.__pwIdentifierPatched) return true;  // idempotent
        const original = window.getTextAreaIdentifier;
        window.getTextAreaIdentifier = function (textArea) {
            const fromOriginal = original(textArea);
            if (fromOriginal) return fromOriginal;
            // Check each of our textareas
            for (const id of PW_TEXTAREA_IDS) {
                const wrap = gApp().querySelector("#" + id);
                if (wrap && wrap.querySelector("textarea") === textArea) {
                    // Use a class-safe identifier. The dot prefix is the
                    // format tagcomplete's `createResultsDiv` consumes when
                    // building its CSS class selector for the popup.
                    return ".pw_" + id;
                }
            }
            return "";
        };
        window.__pwIdentifierPatched = true;
        return true;
    }

    // A small debounce that mirrors what tagcomplete uses internally, just
    // in case tagcomplete's own `debounce` isn't on `window`.
    function debounce(fn, wait) {
        let timer = null;
        return function () {
            const ctx = this, args = arguments;
            if (timer) clearTimeout(timer);
            timer = setTimeout(function () {
                fn.apply(ctx, args);
            }, wait);
        };
    }

    // Replicate tagcomplete's per-textarea setup work for each Anima
    // Workshop textarea. Returns counts so we can log a useful summary.
    function setupAnimaTextareas() {
        const hasAutocomplete = typeof window.autocomplete === "function";
        const hasNavigate     = typeof window.navigateInList === "function";
        const hasHide         = typeof window.hideResults    === "function";
        const hasCreateDiv    = typeof window.createResultsDiv === "function";

        // We can technically work without createResultsDiv (skip popup div
        // creation -- tagcomplete's results would have nowhere to render),
        // but without `autocomplete` and `navigateInList` there's nothing
        // useful to wire up.
        if (!hasAutocomplete || !hasNavigate) {
            return { supported: false, attached: 0, alreadyDone: 0 };
        }

        const myTextareas = getMyTextareas();
        let attached = 0;
        let alreadyDone = 0;

        for (const { id, ta } of myTextareas) {
            if (ta.classList.contains("autocomplete")) {
                alreadyDone += 1;
                continue;
            }
            try {
                // Insert a results div for this textarea, if tagcomplete
                // exposes the helper. Without it, the popup will fall back
                // to whatever default container tagcomplete uses (which is
                // ok on most versions).
                if (hasCreateDiv) {
                    const resultsDiv = window.createResultsDiv(ta);
                    ta.parentNode.insertBefore(resultsDiv, ta.nextSibling);
                    if (hasHide) window.hideResults(ta);
                }

                ta.addEventListener("input",
                    debounce(function () { window.autocomplete(ta, ta.value); }, 100));
                ta.addEventListener("focusout",
                    debounce(function () { if (hasHide) window.hideResults(ta); }, 400));
                ta.addEventListener("keydown",
                    function (e) { window.navigateInList(ta, e); });

                ta.classList.add("autocomplete");
                attached += 1;
            } catch (e) {
                console.warn(LOG, "failed to attach autocomplete to #" + id + ":", e);
            }
        }

        return { supported: true, attached, alreadyDone };
    }

    function attempt() {
        if (!isTagcompleteSetUpOnBuiltIns()) {
            // Either tagcomplete isn't installed, or it hasn't finished
            // initialising yet. Either way, retry later.
            return false;
        }

        const patched = patchIdentifier();
        if (!patched) {
            console.log(LOG,
                "tagcomplete is installed and active on txt2img/img2img, but " +
                "its helper functions aren't exposed on `window` in your version. " +
                "This means the bridge can't hook in from outside. To get " +
                "autocomplete on Anima Workshop fields, open an issue at " +
                "https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/issues " +
                "requesting that the Anima Workshop textarea IDs " +
                "(pw_quality, pw_subject, pw_character, pw_series, pw_artist, " +
                "pw_general, pw_consolidated, pw_grabbed, pw_search_tag, " +
                "pw_blacklist_tags) be added to its supported textarea list.");
            return true;  // stop retrying — nothing more we can do here
        }

        const result = setupAnimaTextareas();
        if (!result.supported) {
            console.log(LOG,
                "tagcomplete is installed but its core functions (autocomplete, " +
                "navigateInList) aren't reachable as globals. Autocomplete on " +
                "Anima Workshop fields won't work on this version. See " +
                "https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/issues " +
                "to request third-party support.");
            return true;
        }

        if (result.attached > 0) {
            console.log(LOG,
                "enabled tagcomplete on " + result.attached + " Anima Workshop " +
                "textareas" + (result.alreadyDone ? " (" + result.alreadyDone +
                " were already done)" : ""));
        } else if (result.alreadyDone > 0) {
            // Bridge already ran on a previous UI refresh; nothing new to do.
        } else {
            console.log(LOG, "no Anima Workshop textareas found to attach to.");
        }
        return true;
    }

    function start() {
        // Tagcomplete sets up its built-in textareas asynchronously and the
        // timing varies (especially with many extensions installed). We
        // retry for up to ~15 seconds before giving up.
        let attempts = 0;
        const iv = setInterval(function () {
            attempts += 1;
            if (attempt() || attempts > 75) {
                clearInterval(iv);
                if (attempts > 75) {
                    console.log(LOG,
                        "tagcomplete not detected after 15s. If you have it " +
                        "installed, check the browser console for tagcomplete " +
                        "errors; otherwise this bridge is a no-op.");
                }
            }
        }, 200);
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(start);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
