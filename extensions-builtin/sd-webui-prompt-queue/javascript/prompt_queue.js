// Prompt Queue — frontend
// - Adds an "Add to Queue" button under Generate on txt2img & img2img
// - Moves the "Queue" tab right next to img2img in the top tab bar
// - Renders the queue page and shows a live pending-count badge
// - Runner loop: when the webui is idle, pops the next item, fills the
//   prompt fields and clicks Generate; marks the item done when idle again.

(function () {
    "use strict";

    const API = "/prompt-queue";
    const TABS = ["txt2img", "img2img"];
    const POLL_MS = 1000;
    const START_TIMEOUT_MS = 30000; // generation must register as busy within this window

    let lastRenderKey = null;
    let dispatched = null; // { id, tab, clickedAt, sawBusy }
    let polling = false;

    // ------------------------------------------------ small helpers

    function esc(s) {
        return String(s == null ? "" : s)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    async function api(path, body) {
        const opts = body
            ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
            : { method: "GET" };
        const res = await fetch(API + path, opts);
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
    }

    function promptBox(tab, negative) {
        return gradioApp().querySelector("#" + tab + (negative ? "_neg_prompt" : "_prompt") + " textarea");
    }

    function setPromptFields(tab, prompt, negative) {
        const p = promptBox(tab, false);
        if (p) { p.value = prompt; updateInput(p); }
        const n = promptBox(tab, true);
        if (n) { n.value = negative; updateInput(n); }
    }

    function anchorControls(tab) {
        const group = gradioApp().querySelector("#prompt_anchor_group_" + tab);
        if (!group) return null;
        return {
            text: gradioApp().querySelector("#prompt_anchor_" + tab + " textarea"),
            enabled: gradioApp().querySelector("#prompt_anchor_enable_" + tab + " input[type=checkbox]"),
            separator: group.querySelector(".prompt-anchor-separator input, .prompt-anchor-separator textarea"),
        };
    }

    function snapshotAnchor(tab) {
        const controls = anchorControls(tab);
        if (!controls || !controls.enabled) return {};
        return {
            anchor: controls.text ? controls.text.value : "",
            anchor_enabled: !!controls.enabled.checked,
            anchor_separator: controls.separator ? controls.separator.value : ", ",
        };
    }

    function restoreAnchor(tab, item) {
        // Old queue entries intentionally have no anchor fields: leave the
        // current UI state untouched to retain their historical behaviour.
        if (!Object.prototype.hasOwnProperty.call(item, "anchor_enabled")) return;
        const controls = anchorControls(tab);
        if (!controls) return;
        if (controls.text) {
            controls.text.value = item.anchor || "";
            updateInput(controls.text);
        }
        if (controls.separator) {
            controls.separator.value = item.anchor_separator == null ? ", " : item.anchor_separator;
            updateInput(controls.separator);
        }
        if (controls.enabled) {
            controls.enabled.checked = !!item.anchor_enabled;
            updateInput(controls.enabled);
            controls.enabled.dispatchEvent(new Event("change", { bubbles: true }));
        }
    }

    function repeatRunnerActive() {
        return !!gradioApp().querySelector(".neo-repeat-generate-active");
    }

    function relTime(ts) {
        if (!ts) return "";
        const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
        if (s < 60) return s + "s ago";
        if (s < 3600) return Math.round(s / 60) + "m ago";
        return Math.round(s / 3600) + "h ago";
    }

    // ------------------------------------------------ "Add to Queue" buttons

    function injectQueueButton(tab) {
        const genBox = gradioApp().getElementById(tab + "_generate_box");
        const genBtn = gradioApp().getElementById(tab + "_generate");
        if (!genBox || !genBtn || gradioApp().getElementById(tab + "_pq_add")) return;

        const btn = document.createElement("button");
        btn.id = tab + "_pq_add";
        // Borrow the generate button's svelte classes so it looks native,
        // but render as a secondary button.
        btn.className = genBtn.className.replace(/\bprimary\b/, "secondary") + " pq-add-btn";
        btn.innerHTML = '<span class="pq-add-label">&#128203; Add to Queue</span><span class="pq-add-flash"></span>';
        btn.title = "Add the current prompt + negative prompt to the generation queue";

        btn.addEventListener("click", async function (ev) {
            ev.preventDefault();
            const prompt = (promptBox(tab, false) || {}).value || "";
            const negative = (promptBox(tab, true) || {}).value || "";
            try {
                const payload = Object.assign(
                    { tab: tab, prompt: prompt, negative: negative },
                    snapshotAnchor(tab)
                );
                const res = await api("/add", payload);
                if (res.ok) {
                    flash(btn, "\u2713 Queued  \u00b7  " + res.pending + " pending", false);
                } else {
                    flash(btn, res.error || "Could not queue", true);
                }
            } catch (e) {
                flash(btn, "Queue unavailable", true);
            }
        });

        genBox.insertAdjacentElement("afterend", btn);
    }

    function flash(btn, text, isError) {
        const label = btn.querySelector(".pq-add-label");
        const fx = btn.querySelector(".pq-add-flash");
        fx.textContent = text;
        btn.classList.add("pq-flashing");
        btn.classList.toggle("pq-flash-error", !!isError);
        label.style.visibility = "hidden";
        clearTimeout(btn._pqTimer);
        btn._pqTimer = setTimeout(function () {
            btn.classList.remove("pq-flashing", "pq-flash-error");
            label.style.visibility = "";
            fx.textContent = "";
        }, 1400);
    }

    // ------------------------------------------------ tab placement + badge

    function tabNavButtons() {
        return Array.from(gradioApp().querySelectorAll("#tabs > .tab-nav > button"));
    }

    function findTabButton(name) {
        return tabNavButtons().find(function (b) {
            return b.textContent.trim().toLowerCase().indexOf(name) === 0;
        });
    }

    function placeQueueTab() {
        const queueBtn = findTabButton("queue");
        const img2imgBtn = findTabButton("img2img");
        if (queueBtn && img2imgBtn && img2imgBtn.nextElementSibling !== queueBtn) {
            img2imgBtn.insertAdjacentElement("afterend", queueBtn);
        }
        if (queueBtn && !queueBtn.querySelector(".pq-badge")) {
            const badge = document.createElement("span");
            badge.className = "pq-badge";
            badge.style.display = "none";
            queueBtn.appendChild(badge);
        }
    }

    function updateBadge(state) {
        const queueBtn = findTabButton("queue");
        const badge = queueBtn && queueBtn.querySelector(".pq-badge");
        if (!badge) return;
        const running = state.items.some(function (i) { return i.status === "running"; });
        if (state.pending > 0 || running) {
            badge.textContent = running ? state.pending + " \u25b6" : String(state.pending);
            badge.style.display = "";
            badge.classList.toggle("pq-badge-live", running);
        } else {
            badge.style.display = "none";
        }
    }

    // ------------------------------------------------ queue page rendering

    function statusChip(item) {
        const map = { pending: "Pending", running: "Generating\u2026", done: "Done", failed: "Failed" };
        return '<span class="pq-chip pq-chip-' + item.status + '">' + map[item.status] + "</span>";
    }

    function itemCard(item, pendingIndex) {
        const isPending = item.status === "pending";
        const num = isPending ? '<span class="pq-num">' + (pendingIndex + 1) + "</span>" : '<span class="pq-num pq-num-dim">\u2013</span>';
        const neg = item.negative
            ? '<div class="pq-neg" title="Negative prompt">\u2296 ' + esc(item.negative) + "</div>"
            : "";
        const when = item.status === "pending" ? "added " + relTime(item.created)
            : item.status === "running" ? "started " + relTime(item.started)
            : relTime(item.finished);

        let actions = "";
        if (isPending) {
            actions =
                '<button class="pq-icon" data-action="up" data-id="' + item.id + '" title="Move up">\u25b2</button>' +
                '<button class="pq-icon" data-action="down" data-id="' + item.id + '" title="Move down">\u25bc</button>' +
                '<button class="pq-icon pq-danger" data-action="remove" data-id="' + item.id + '" title="Cancel">\u2715</button>';
        } else if (item.status === "running") {
            actions = '<button class="pq-icon pq-danger" data-action="interrupt" data-id="' + item.id + '" data-tab="' + item.tab + '" title="Interrupt this generation">\u25a0</button>';
        } else {
            actions =
                '<button class="pq-icon" data-action="requeue" data-id="' + item.id + '" title="Queue again">\u21bb</button>' +
                '<button class="pq-icon pq-danger" data-action="remove" data-id="' + item.id + '" title="Remove from history">\u2715</button>';
        }

        return (
            '<div class="pq-item pq-item-' + item.status + '">' +
                num +
                '<div class="pq-body">' +
                    '<div class="pq-meta"><span class="pq-tab pq-tab-' + item.tab + '">' + item.tab + "</span>" + statusChip(item) +
                        '<span class="pq-when">' + when + "</span></div>" +
                    '<div class="pq-prompt">' + esc(item.prompt) + "</div>" + neg +
                "</div>" +
                '<div class="pq-actions">' + actions + "</div>" +
            "</div>"
        );
    }

    function render(state) {
        const root = gradioApp().getElementById("prompt-queue-root");
        if (!root) return;

        const active = state.items.filter(function (i) { return i.status === "pending" || i.status === "running"; });
        const history = state.items.filter(function (i) { return i.status === "done" || i.status === "failed"; }).reverse();

        const statusText = state.busy
            ? '<span class="pq-live-dot"></span> Generating\u2026'
            : (state.enabled ? "Idle \u2014 waiting for prompts" : "Paused");

        let html =
            '<div class="pq-toolbar">' +
                '<button class="pq-run ' + (state.enabled ? "pq-run-on" : "pq-run-off") + '" data-action="toggle-run">' +
                    (state.enabled ? "\u23f8 Pause queue" : "\u25b6 Run queue") +
                "</button>" +
                '<span class="pq-status">' + statusText + "</span>" +
                '<span class="pq-count">' + state.pending + " / " + state.max + " queued</span>" +
                '<span class="pq-spacer"></span>' +
                '<button class="pq-ghost" data-action="clear-finished">Clear history</button>' +
                '<button class="pq-ghost pq-danger" data-action="clear-pending">Clear pending</button>' +
            "</div>";

        if (active.length === 0) {
            html +=
                '<div class="pq-empty">' +
                    "<div class=\"pq-empty-icon\">&#128203;</div>" +
                    "<div>The queue is empty.</div>" +
                    '<div class="pq-empty-hint">Write a prompt in <b>txt2img</b> or <b>img2img</b> and press <b>Add to Queue</b> (under Generate). ' +
                    "Items run one after another using whatever settings are currently active on that tab.</div>" +
                "</div>";
        } else {
            let pendingIdx = 0;
            html += '<div class="pq-list">' + active.map(function (i) {
                const card = itemCard(i, pendingIdx);
                if (i.status === "pending") pendingIdx++;
                return card;
            }).join("") + "</div>";
        }

        if (history.length) {
            html += '<div class="pq-history-label">History</div><div class="pq-list pq-list-history">' +
                history.map(function (i) { return itemCard(i, 0); }).join("") + "</div>";
        }

        root.innerHTML = html;
    }

    function bindRootEvents() {
        const root = gradioApp().getElementById("prompt-queue-root");
        if (!root || root._pqBound) return;
        root._pqBound = true;

        root.addEventListener("click", async function (ev) {
            const btn = ev.target.closest("[data-action]");
            if (!btn) return;
            const action = btn.getAttribute("data-action");
            const id = btn.getAttribute("data-id");
            try {
                if (action === "toggle-run") {
                    const st = await api("/state");
                    await api("/run", { enabled: !st.enabled });
                } else if (action === "remove") {
                    await api("/remove", { id: id });
                } else if (action === "up" || action === "down") {
                    await api("/move", { id: id, direction: action });
                } else if (action === "clear-pending") {
                    await api("/clear", { which: "pending" });
                } else if (action === "clear-finished") {
                    await api("/clear", { which: "finished" });
                } else if (action === "requeue") {
                    const st = await api("/state");
                    const item = st.items.find(function (i) { return i.id === id; });
                    if (item) {
                        const payload = {
                            tab: item.tab,
                            prompt: item.prompt,
                            negative: item.negative,
                        };
                        if (Object.prototype.hasOwnProperty.call(item, "anchor_enabled")) {
                            payload.anchor = item.anchor || "";
                            payload.anchor_enabled = !!item.anchor_enabled;
                            payload.anchor_separator = item.anchor_separator == null ? ", " : item.anchor_separator;
                        }
                        await api("/add", payload);
                    }
                } else if (action === "interrupt") {
                    const tab = btn.getAttribute("data-tab");
                    const ib = gradioApp().getElementById(tab + "_interrupt");
                    if (ib) ib.click();
                }
                poll(true); // refresh immediately
            } catch (e) {
                console.error("[Prompt Queue]", e);
            }
        });
    }

    // ------------------------------------------------ runner

    async function runner(state) {
        if (dispatched) {
            const mine = state.items.find(function (i) { return i.id === dispatched.id; });
            if (!mine || mine.status !== "running") {
                dispatched = null; // removed or finished elsewhere
                return;
            }
            if (state.busy) {
                dispatched.sawBusy = true;
            } else if (dispatched.sawBusy) {
                await api("/finish", { id: dispatched.id, status: "done" });
                dispatched = null;
            } else if (Date.now() - dispatched.clickedAt > START_TIMEOUT_MS) {
                // Generation never started (e.g. validation error on that tab).
                await api("/finish", { id: dispatched.id, status: "failed" });
                dispatched = null;
            }
            return;
        }

        if (!state.enabled || state.busy || state.pending === 0 || repeatRunnerActive()) return;

        const res = await api("/pop", {});
        const item = res && res.item;
        if (!item) return;

        restoreAnchor(item.tab, item);
        setPromptFields(item.tab, item.prompt, item.negative);
        const gen = gradioApp().getElementById(item.tab + "_generate");
        if (!gen) {
            await api("/finish", { id: item.id, status: "failed" });
            return;
        }
        dispatched = { id: item.id, tab: item.tab, clickedAt: Date.now(), sawBusy: false };
        gen.click();
    }

    // ------------------------------------------------ poll loop

    async function poll(force) {
        if (polling && !force) return;
        polling = true;
        try {
            const state = await api("/state");
            updateBadge(state);
            const key = state.version + ":" + state.busy + ":" + Math.floor(Date.now() / 30000);
            if (force || key !== lastRenderKey) {
                render(state);
                lastRenderKey = key;
            }
            await runner(state);
        } catch (e) {
            /* webui restarting or endpoint not up yet — try again next tick */
        } finally {
            polling = false;
        }
    }

    onUiLoaded(function () {
        TABS.forEach(injectQueueButton);
        placeQueueTab();
        bindRootEvents();
        poll(true);
        setInterval(poll, POLL_MS);
    });
})();
