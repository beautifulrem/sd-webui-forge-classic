// joycaption.js
// -----------------------------------------------------------------------------
// JS-side wiring for the sd-webui-joycaption extension's bridge features.
//
// Jobs (in roughly the order they fire after onUiLoaded):
//   1. Wire the "→ Append to txt2img Prompt" button on the JoyCaption tab:
//      APPEND the contents of the combined-prompt textbox to the txt2img
//      positive textarea (do NOT replace), then switch to the txt2img tab.
//   2. Wire the "↻ Replace txt2img Prompt" button on the JoyCaption tab:
//      OVERWRITE the txt2img positive textarea with the contents of the
//      combined-prompt textbox, then switch to the txt2img tab. Useful
//      especially when the user has a sd-webui-prompt-anchor box up top
//      holding their style foundation -- the main prompt is then a
//      throwaway subject string and replacing it wholesale is fine.
//
// Everything below is purely additive -- no Forge code is patched, no Gradio
// internals are touched. Hooks: onUiLoaded(), defined globally by Forge.
//
// This file lives at <ext>/javascript/joycaption.js. Forge auto-loads
// every .js file in an extension's javascript/ folder, no registration
// needed.
// -----------------------------------------------------------------------------

(function () {
    "use strict";

    const LOG = "[joycaption-bridge]";

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

    // ----- Global helper: switch to the JoyCaption tab ------------------
    //
    // When the standard infotext_utils ParamBinding fires (e.g. the
    // "Send to JoyCaption" button on the PNG-info tab), the WebUI's
    // built-in flow ends the click chain by calling _js="switch_to_<tab>"
    // -- but those switch_to_* helpers are only auto-generated for the
    // four built-in destination tabs (txt2img/img2img/inpaint/extras).
    // For our custom tabname="joycaption" we have to provide our own.
    //
    // We expose this as a global function so it can be referenced from
    // Python via `gr.Button.click(..., _js="joycaption_switch_to_tab")`.
    // Returning the inputs unchanged (here: nothing) is the Gradio
    // convention for _js handlers that just have a side effect.
    window.joycaption_switch_to_tab = function () {
        const btn = findTabButton("JoyCaption");
        if (btn) {
            btn.click();
            console.log(LOG, "switched to JoyCaption tab");
        } else {
            console.warn(LOG, "JoyCaption tab button not found");
        }
    };

    function getTextareaValue(elemId) {
        const wrap = root().querySelector("#" + elemId);
        if (!wrap) return null;
        const ta = wrap.querySelector("textarea, input");
        return ta ? ta.value : null;
    }

    function findTxt2imgPromptTextarea() {
        const wrap = root().querySelector("#txt2img_prompt")
                  || root().querySelector("#txt2img_toprow #txt2img_prompt");
        if (!wrap) return null;
        return wrap.querySelector("textarea");
    }

    // Match the comma-joining semantics of prompt_workshop.js so the two
    // extensions behave the same way when they touch the txt2img prompt.
    function appendToTextarea(ta, text) {
        if (!ta || !text) return false;
        const trimmedNew = text.trim();
        if (!trimmedNew) return false;

        const current = ta.value.replace(/\s+$/, "");
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

    // Wholesale overwrite -- whatever was in the textarea is GONE.
    // Counterpart to appendToTextarea(). Used by the Replace button.
    // Empty `text` is rejected (returns false) so that an accidental
    // click while the combined box is empty doesn't silently wipe the
    // user's existing prompt; the caller has already verified non-empty
    // for the Append path and we mirror that here for consistency.
    function replaceTextarea(ta, text) {
        if (!ta || !text) return false;
        const trimmedNew = text.trim();
        if (!trimmedNew) return false;

        ta.value = trimmedNew;
        // Same input-event dispatch as the append path -- without it,
        // Gradio's internal value tracking diverges from the DOM and
        // the next generation submits the OLD value.
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        return true;
    }

    function switchToTxt2imgTab() {
        const btn = findTabButton("txt2img");
        if (btn) btn.click();
    }

    function wireSendToTxt2img() {
        const btn = root().querySelector("#joycaption_send_to_t2i");
        if (!btn || btn.dataset.jcBound === "1") return !!btn;
        btn.dataset.jcBound = "1";

        btn.addEventListener("click", function (e) {
            e.preventDefault();
            e.stopPropagation();
            const text = getTextareaValue("joycaption_txt2img_combined");
            if (!text || !text.trim()) {
                console.log(LOG, "combined box is empty -- nothing to send");
                return;
            }
            const ta = findTxt2imgPromptTextarea();
            if (!ta) {
                console.warn(LOG, "could not find #txt2img_prompt textarea");
                return;
            }
            if (appendToTextarea(ta, text)) {
                switchToTxt2imgTab();
                console.log(LOG, "appended combined prompt to txt2img, switched tab");
            }
        });
        return true;
    }

    // Counterpart to wireSendToTxt2img(): wires the "Replace txt2img
    // Prompt" button. Same flow, except the existing textarea contents
    // are discarded rather than preserved.
    //
    // We deliberately do NOT prompt for confirmation before overwriting.
    // The button label says "Replace ... (overwrites existing)" and the
    // user just clicked it; double-confirming every click would be
    // annoying in the common reuse flow (caption many images in a row,
    // each time replacing the prompt). If users want their prior
    // prompt back, the txt2img tab has its own undo via the prompt-
    // history dropdown built into the webui.
    function wireReplaceToTxt2img() {
        const btn = root().querySelector("#joycaption_replace_t2i");
        if (!btn || btn.dataset.jcBound === "1") return !!btn;
        btn.dataset.jcBound = "1";

        btn.addEventListener("click", function (e) {
            e.preventDefault();
            e.stopPropagation();
            const text = getTextareaValue("joycaption_txt2img_combined");
            if (!text || !text.trim()) {
                console.log(LOG, "combined box is empty -- refusing to wipe txt2img prompt with nothing");
                return;
            }
            const ta = findTxt2imgPromptTextarea();
            if (!ta) {
                console.warn(LOG, "could not find #txt2img_prompt textarea");
                return;
            }
            if (replaceTextarea(ta, text)) {
                switchToTxt2imgTab();
                console.log(LOG, "replaced txt2img prompt with combined prompt, switched tab");
            }
        });
        return true;
    }

    function tryAll() {
        // Both buttons live in the same UI block and appear together,
        // but we && them anyway -- if Gradio re-renders the row at
        // different times for some reason, we don't want the missing
        // button to permanently block the present one from being
        // wired on a future retry. Returning false just means we'll
        // re-enter on the next 200ms tick.
        const a = wireSendToTxt2img();
        const b = wireReplaceToTxt2img();
        return a && b;
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
