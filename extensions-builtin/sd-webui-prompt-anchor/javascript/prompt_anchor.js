// prompt_anchor.js
// -----------------------------------------------------------------------------
// JS-side wiring for the sd-webui-prompt-anchor extension.
//
// Two jobs, both done on onUiLoaded:
//
//   1. Reparent each anchor box (txt2img + img2img) to sit DIRECTLY ABOVE
//      the main positive prompt textarea on that tab. Gradio places our
//      AlwaysVisible script's UI in the standard "scripts accordion"
//      area below the prompt; we lift it to where the user expects it.
//
//   2. Persist the anchor text and enabled state in localStorage, keyed
//      by tab. Restores on page load. The "− / +" collapse button is
//      also wired purely on the JS side and its state persists too.
//
// Notes on robustness:
//   - We wait for both our wrapper AND the txt2img prompt textarea to
//     exist before reparenting -- both can be late in some load orders.
//   - We retry on a short interval for ~8 seconds, then give up
//     gracefully. The Python extension still functions if reparenting
//     fails; the box just sits below the prompt instead of above.
//   - We never patch any Forge code. We move a single DOM node, attach
//     an input listener, and read/write three localStorage keys. That's
//     the entire surface area.
//
// This file lives at <ext>/javascript/prompt_anchor.js. Forge auto-loads
// every .js file in an extension's javascript/ folder, no registration
// needed.
// -----------------------------------------------------------------------------

(function () {
    "use strict";

    const LOG = "[prompt-anchor]";

    // Tab definitions -- keep elem_ids in sync with prompt_anchor.py.
    const TABS = [
        {
            tab: "txt2img",
            groupId: "prompt_anchor_group_txt2img",
            boxId: "prompt_anchor_txt2img",
            enableId: "prompt_anchor_enable_txt2img",
            promptId: "txt2img_prompt",
            collapseId: "prompt_anchor_group_txt2img_collapse",
            storageKey: "promptAnchor.txt2img",
            collapsedKey: "promptAnchor.txt2img.collapsed",
            enabledKey: "promptAnchor.txt2img.enabled",
        },
        {
            tab: "img2img",
            groupId: "prompt_anchor_group_img2img",
            boxId: "prompt_anchor_img2img",
            enableId: "prompt_anchor_enable_img2img",
            promptId: "img2img_prompt",
            collapseId: "prompt_anchor_group_img2img_collapse",
            storageKey: "promptAnchor.img2img",
            collapsedKey: "promptAnchor.img2img.collapsed",
            enabledKey: "promptAnchor.img2img.enabled",
        },
    ];

    function root() {
        return (typeof gradioApp === "function") ? gradioApp() : document;
    }

    // Find the actual <textarea> or <input> element inside a Gradio
    // wrapper given the wrapper's elem_id. Gradio wraps its inputs in
    // a few layers of divs, so we drill down rather than try to apply
    // elem_id to the input itself.
    function findInputInside(elemId) {
        const wrap = root().querySelector("#" + elemId);
        if (!wrap) return null;
        return wrap.querySelector("textarea, input");
    }

    // Fire the synthetic 'input' event that Gradio listens for on form
    // fields. Without this dispatch, programmatic .value changes don't
    // make it back to the Python side -- so anything we restore from
    // localStorage on page load would NOT be visible to process().
    function notifyGradio(el) {
        if (!el) return;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        // gradio occasionally listens for 'change' too (checkboxes)
        el.dispatchEvent(new Event("change", { bubbles: true }));
    }

    // -------- 1. Reparent ----------------------------------------------------

    function reparent(def) {
        const group = root().querySelector("#" + def.groupId);
        if (!group) return false;

        const prompt = root().querySelector("#" + def.promptId);
        if (!prompt) return false;

        // Already lifted? Bail.
        if (group.dataset.paLifted === "1") return true;

        // Insert the whole group right BEFORE the prompt wrapper, in
        // the prompt wrapper's parent. parentNode is the toprow's main
        // flex column -- so we land in the same column as the prompt
        // textarea, sized the same way.
        const parent = prompt.parentNode;
        if (!parent) return false;
        parent.insertBefore(group, prompt);
        group.dataset.paLifted = "1";

        console.log(LOG, "reparented", def.groupId, "above", def.promptId);
        return true;
    }

    // -------- 2. localStorage persistence ------------------------------------

    function safeGet(key, fallback) {
        try {
            const v = localStorage.getItem(key);
            return (v === null) ? fallback : v;
        } catch (e) {
            return fallback;
        }
    }
    function safeSet(key, value) {
        try { localStorage.setItem(key, value); } catch (e) { /* quota? private mode? swallow */ }
    }

    function wireTextPersistence(def) {
        const ta = findInputInside(def.boxId);
        if (!ta) return false;
        if (ta.dataset.paBoundText === "1") return true;
        ta.dataset.paBoundText = "1";

        // Restore (only if the field is empty -- if Python has already
        // filled it via paste-from-PNG-info, that wins).
        const saved = safeGet(def.storageKey, "");
        if (saved && !ta.value) {
            ta.value = saved;
            notifyGradio(ta);
        }

        // Persist on every keystroke. 'input' fires for typing, paste,
        // and programmatic changes (the latter is what we just did
        // above, so the first save is a no-op write of the same value).
        ta.addEventListener("input", function () {
            safeSet(def.storageKey, ta.value);
        });
        return true;
    }

    function wireEnabledPersistence(def) {
        const cb = findInputInside(def.enableId);
        if (!cb) return false;
        if (cb.dataset.paBoundEnabled === "1") return true;
        cb.dataset.paBoundEnabled = "1";

        // Restore. Stored as the literal strings "1" / "0".
        const saved = safeGet(def.enabledKey, null);
        if (saved !== null) {
            const wanted = (saved === "1");
            if (cb.checked !== wanted) {
                cb.checked = wanted;
                notifyGradio(cb);
            }
        }
        cb.addEventListener("change", function () {
            safeSet(def.enabledKey, cb.checked ? "1" : "0");
        });
        return true;
    }

    // -------- 3. Collapse / expand toggle ------------------------------------
    //
    // Collapsing hides the textarea + advanced accordion but keeps the
    // header row (with the enabled checkbox) visible, so the user can
    // still flip it on/off without expanding the whole block. State
    // persists in localStorage.

    function applyCollapsedState(def, collapsed) {
        const group = root().querySelector("#" + def.groupId);
        if (!group) return;
        if (collapsed) {
            group.classList.add("pa-collapsed");
        } else {
            group.classList.remove("pa-collapsed");
        }
        const btn = root().querySelector("#" + def.collapseId);
        if (btn) {
            // Just the button label -- the actual hide/show is CSS.
            btn.textContent = collapsed ? "+" : "−";
            btn.title = collapsed ? "Expand anchor prompt" : "Collapse anchor prompt";
        }
    }

    function wireCollapseToggle(def) {
        const btn = root().querySelector("#" + def.collapseId);
        if (!btn) return false;
        if (btn.dataset.paBoundCollapse === "1") return true;
        btn.dataset.paBoundCollapse = "1";

        // Restore initial state. Default = NOT collapsed unless the
        // user previously collapsed it OR the global setting is on.
        // (The global setting is read from a data attr we'll set in
        // CSS via the body class -- but for now we just trust
        // localStorage; the global default is a Python-side concern.)
        const savedCollapsed = safeGet(def.collapsedKey, "0") === "1";
        applyCollapsedState(def, savedCollapsed);

        btn.addEventListener("click", function (e) {
            e.preventDefault();
            e.stopPropagation();
            const group = root().querySelector("#" + def.groupId);
            const nowCollapsed = !(group && group.classList.contains("pa-collapsed"));
            applyCollapsedState(def, nowCollapsed);
            safeSet(def.collapsedKey, nowCollapsed ? "1" : "0");
        });
        return true;
    }

    // -------- driver ---------------------------------------------------------

    function tryAll() {
        // Returns true once EVERY tab has been fully wired. We don't
        // require both tabs to succeed in the same tick -- the prompt
        // textarea on the inactive tab can still be late.
        let allDone = true;
        for (const def of TABS) {
            const r1 = reparent(def);
            const r2 = wireTextPersistence(def);
            const r3 = wireEnabledPersistence(def);
            const r4 = wireCollapseToggle(def);
            if (!(r1 && r2 && r3 && r4)) allDone = false;
        }
        return allDone;
    }

    function start() {
        if (tryAll()) return;
        let attempts = 0;
        const iv = setInterval(function () {
            attempts += 1;
            if (tryAll() || attempts > 40) {  // ~8 seconds at 200ms
                clearInterval(iv);
                if (attempts > 40) {
                    console.warn(LOG, "gave up waiting for UI elements; some pieces may not be wired");
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
