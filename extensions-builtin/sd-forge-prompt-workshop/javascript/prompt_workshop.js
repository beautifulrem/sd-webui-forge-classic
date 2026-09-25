// prompt_workshop.js
// -----------------------------------------------------------------------------
// JS-side wiring for the Anima Workshop extension.
//
// Jobs (in roughly the order they fire after onUiLoaded):
//   1. Move the "Anima Workshop" tab button so it appears between img2img
//      and Extras. By default, extension tabs registered through on_ui_tabs
//      get appended at the end of the tab list.
//   2. Wire the main "Send to txt2img Prompt" button: APPEND the consolidated
//      prompt to the txt2img positive textarea (don't replace), then switch
//      to the txt2img tab. The sibling "Replace txt2img Prompt" button does
//      the same but OVERWRITES the textarea instead of appending.
//   3. Wire the six per-category "→ N. Whatever" buttons: append just that
//      one category box's content to the txt2img positive textarea. No tab
//      switch -- you might want to send several in a row.
//   4. Wire the "🔗 Open Post in New Tab" button: read the URL out of the
//      Post URL textbox and window.open() it.
//
// Everything below is purely additive -- no Forge code is patched, no Gradio
// internals are touched. Hooks: onUiLoaded(), defined globally by Forge.
// -----------------------------------------------------------------------------

(function () {
    "use strict";

    const LOG = "[anima-workshop]";

    // Per-category Send buttons: map button elem_id -> source textbox elem_id.
    const PER_CATEGORY_SENDS = {
        "pw_send_quality":   "pw_quality",
        "pw_send_subject":   "pw_subject",
        "pw_send_character": "pw_character",
        "pw_send_series":    "pw_series",
        "pw_send_artist":    "pw_artist",
        "pw_send_general":   "pw_general",
    };

    // ----- DOM helpers ----------------------------------------------------

    function root() {
        return (typeof gradioApp === "function") ? gradioApp() : document;
    }

    function findTabButton(label) {
        const buttons = root().querySelectorAll("#tabs > .tab-nav button");
        for (const b of buttons) {
            if (b.textContent.trim() === label) return b;
        }
        return null;
    }

    function findTabPanel(elemId) {
        return root().querySelector("#" + elemId);
    }

    function getTextareaValue(elemId) {
        const wrap = root().querySelector("#" + elemId);
        if (!wrap) return null;
        const ta = wrap.querySelector("textarea, input");
        return ta ? ta.value : null;
    }

    /**
     * Append text to a textarea, with smart comma separation:
     *   - empty existing  -> just set the text
     *   - existing ends with ','  -> append a space + the text
     *   - otherwise -> append ", " + the text
     * Then fire an 'input' event so Gradio's state catches up -- without
     * this, the next Generate would use the stale Python-side prompt.
     */
    function appendToTextarea(ta, text) {
        if (!ta || !text) return false;
        const trimmedNew = text.trim();
        if (!trimmedNew) return false;

        const current = ta.value.replace(/\s+$/, "");  // trim trailing whitespace only
        if (!current) {
            ta.value = trimmedNew;
        } else if (current.endsWith(",")) {
            ta.value = current + " " + trimmedNew;
        } else {
            ta.value = current + ", " + trimmedNew;
        }
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        return true;
    }

    /**
     * Replace the entire contents of a textarea with `text`, then fire an
     * 'input' event so Gradio's state catches up (same reasoning as
     * appendToTextarea -- without it the next Generate uses the stale
     * Python-side prompt).
     */
    function setTextarea(ta, text) {
        if (!ta) return false;
        const trimmedNew = (text || "").trim();
        if (!trimmedNew) return false;

        ta.value = trimmedNew;
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        return true;
    }

    function findTxt2imgPromptTextarea() {
        const wrap = root().querySelector("#txt2img_prompt")
                  || root().querySelector("#txt2img_toprow #txt2img_prompt");
        if (!wrap) return null;
        return wrap.querySelector("textarea");
    }

    function switchToTxt2imgTab() {
        const btn = findTabButton("txt2img");
        if (btn) btn.click();
    }

    // ----- Job 1: tab reordering -----------------------------------------

    function reorderTab() {
        const pwBtn      = findTabButton("Anima Workshop");
        const extrasBtn  = findTabButton("Extras");
        if (!pwBtn || !extrasBtn) return false;

        if (extrasBtn.previousElementSibling === pwBtn) return true;

        extrasBtn.parentNode.insertBefore(pwBtn, extrasBtn);

        const pwPanel     = findTabPanel("prompt_workshop_tab");
        const extrasPanel = findTabPanel("tab_extras");
        if (pwPanel && extrasPanel && pwPanel.parentNode === extrasPanel.parentNode) {
            extrasPanel.parentNode.insertBefore(pwPanel, extrasPanel);
        }

        console.log(LOG, "tab reordered to before Extras");
        return true;
    }

    // ----- Job 2: main Send to txt2img Prompt ----------------------------

    function wireSendToTxt2img() {
        const btn = root().querySelector("#pw_send_to_t2i");
        if (!btn || btn.dataset.pwBound === "1") return !!btn;
        btn.dataset.pwBound = "1";

        btn.addEventListener("click", function (e) {
            e.preventDefault();
            e.stopPropagation();
            const text = getTextareaValue("pw_consolidated");
            if (!text || !text.trim()) {
                console.log(LOG, "consolidated box is empty -- nothing to send");
                return;
            }
            const ta = findTxt2imgPromptTextarea();
            if (!ta) {
                console.warn(LOG, "could not find #txt2img_prompt textarea");
                return;
            }
            if (appendToTextarea(ta, text)) {
                switchToTxt2imgTab();
                console.log(LOG, "appended consolidated prompt to txt2img, switched tab");
            }
        });
        return true;
    }

    // ----- Job 2b: Replace txt2img Prompt --------------------------------

    function wireReplaceTxt2img() {
        const btn = root().querySelector("#pw_replace_t2i");
        if (!btn || btn.dataset.pwBound === "1") return !!btn;
        btn.dataset.pwBound = "1";

        btn.addEventListener("click", function (e) {
            e.preventDefault();
            e.stopPropagation();
            const text = getTextareaValue("pw_consolidated");
            if (!text || !text.trim()) {
                console.log(LOG, "consolidated box is empty -- nothing to send");
                return;
            }
            const ta = findTxt2imgPromptTextarea();
            if (!ta) {
                console.warn(LOG, "could not find #txt2img_prompt textarea");
                return;
            }
            if (setTextarea(ta, text)) {
                switchToTxt2imgTab();
                console.log(LOG, "replaced txt2img prompt with consolidated, switched tab");
            }
        });
        return true;
    }

    // ----- Job 3: per-category send buttons ------------------------------

    function wirePerCategorySends() {
        let allBound = true;
        for (const [btnId, srcId] of Object.entries(PER_CATEGORY_SENDS)) {
            const btn = root().querySelector("#" + btnId);
            if (!btn) { allBound = false; continue; }
            if (btn.dataset.pwBound === "1") continue;
            btn.dataset.pwBound = "1";

            // closure-capture srcId per iteration
            btn.addEventListener("click", (function (srcElemId, btnElemId) {
                return function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    const text = getTextareaValue(srcElemId);
                    if (!text || !text.trim()) {
                        console.log(LOG, btnElemId, "-> source box is empty");
                        return;
                    }
                    const ta = findTxt2imgPromptTextarea();
                    if (!ta) {
                        console.warn(LOG, "could not find #txt2img_prompt textarea");
                        return;
                    }
                    appendToTextarea(ta, text);
                    console.log(LOG, "appended", srcElemId, "to txt2img prompt");
                    // No tab switch on per-category sends -- user might want
                    // to dispatch multiple categories in a row.
                };
            })(srcId, btnId));
        }
        return allBound;
    }

    // ----- Job 4: Open Post in New Tab -----------------------------------

    function wireOpenPost() {
        const btn = root().querySelector("#pw_open_post");
        if (!btn || btn.dataset.pwBound === "1") return !!btn;
        btn.dataset.pwBound = "1";

        btn.addEventListener("click", function (e) {
            e.preventDefault();
            e.stopPropagation();
            const url = (getTextareaValue("pw_post_url") || "").trim();
            if (!url) {
                console.log(LOG, "post URL box is empty -- grab something first");
                return;
            }
            // noopener so the opened tab can't navigate our window via
            // window.opener, and noreferrer so we don't leak referrer info.
            window.open(url, "_blank", "noopener,noreferrer");
        });
        return true;
    }

    // ----- Job 5: "Send Image to JoyCaption" hand-off --------------------
    //
    // Three buttons funnel through these two JS helpers:
    //
    //   #pw_send_image_to_jc      ─► pw_trigger_jc_handoff
    //         (after Python download chain)
    //   #pw_download_preview      ─► no JS step (just downloads)
    //   #pw_send_preview_to_jc    ─► pw_trigger_jc_handoff_no_download
    //
    // Both trigger helpers end by clicking the hidden paste button --
    // registered server-side as a ParamBinding to tabname=joycaption --
    // and then switching the active tab to JoyCaption. They differ in
    // whether they expect a Python download to have just run (and
    // therefore whether they read the status markdown for errors).
    //
    // Why a hidden paste button instead of grabbing JoyCaption's image
    // component directly: Gradio components live in their own Blocks
    // contexts, so cross-tab events have to go through infotext_utils'
    // ParamBinding machinery. Trying to set #joycaption_single_image
    // from JS would not propagate the value to Python state.

    // Detects whether the preview image actually has content. The
    // gradio Image wrapper always exists in the DOM, but it only has
    // a child <img> after an image has been loaded into it. The
    // "Clear" button creates a similar void state.
    function previewImageHasContent() {
        const wrap = root().querySelector("#pw_preview_image");
        if (!wrap) return false;
        const img = wrap.querySelector("img");
        return !!(img && img.src && !img.src.endsWith("/empty.png"));
    }

    function fireJoycaptionHandoff() {
        const pasteBtn = root().querySelector("#pw_jc_handoff_paste");
        if (!pasteBtn) {
            console.warn(LOG, "hidden JoyCaption paste button not found");
            return false;
        }
        pasteBtn.click();
        const jcTab = findTabButton("JoyCaption");
        if (jcTab) jcTab.click();
        console.log(LOG, "image hand-off to JoyCaption fired");
        return true;
    }

    // Called after pw_send_image_to_jc's Python download chain. The
    // chain has already written either a fresh image or "gr.update()
    // (unchanged)" into the preview, plus a status line. We read the
    // status line to decide whether to proceed.
    window.pw_trigger_jc_handoff = function () {
        const statusEl = root().querySelector("#pw_jc_handoff_status");
        const statusText = (statusEl ? statusEl.textContent.trim() : "");
        // If the status starts with an error marker, the download
        // failed -- don't try to push an empty/stale preview.
        if (/^❌|^❗/.test(statusText)) {
            console.log(LOG, "JoyCaption hand-off skipped:", statusText);
            return;
        }
        fireJoycaptionHandoff();
    };

    // Called by #pw_send_preview_to_jc directly -- there's no Python
    // download to wait for. We just verify the preview has something
    // in it, then fire the same hand-off.
    window.pw_trigger_jc_handoff_no_download = function () {
        if (!previewImageHasContent()) {
            // Surface a status message so the user understands why
            // nothing happened. Gradio's Markdown components don't
            // accept JS-side text changes through their normal value
            // pipeline, so we poke the visible text directly. The
            // next real Python event will overwrite it normally.
            const statusEl = root().querySelector("#pw_jc_handoff_status");
            if (statusEl) {
                statusEl.textContent = "❗ Preview is empty — click 'Download for Preview' or 'Send Image to JoyCaption' first.";
            }
            console.log(LOG, "no preview to send to JoyCaption");
            return;
        }
        fireJoycaptionHandoff();
    };

    // ----- driver --------------------------------------------------------

    function tryAll() {
        const a = reorderTab();
        const b = wireSendToTxt2img();
        const b2 = wireReplaceTxt2img();
        const c = wirePerCategorySends();
        const d = wireOpenPost();
        return a && b && b2 && c && d;
    }

    function start() {
        if (tryAll()) return;
        let attempts = 0;
        const iv = setInterval(function () {
            attempts += 1;
            if (tryAll() || attempts > 40) {  // ~8 seconds at 200ms
                clearInterval(iv);
                if (attempts > 40) {
                    console.warn(LOG, "gave up waiting for UI elements");
                }
            }
        }, 200);
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(start);
    } else {
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", start);
        } else {
            start();
        }
    }
})();
