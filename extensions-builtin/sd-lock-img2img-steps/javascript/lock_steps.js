// Lock img2img Steps - JS fallback
//
// Snapshots the img2img Sampling Steps value whenever the user edits it.
// After any UI update (which is what "Send to img2img" triggers), if the
// lock checkbox is checked, restores the snapshotted value.
//
// This is robust against fork-specific changes to the Python paste API.

(function () {
    "use strict";

    let savedSteps = null;
    let suppressNextChange = false;

    function getStepsInput() {
        // Forge / A1111 use elem_id "img2img_steps" for the Sampling Steps slider.
        // Gradio wraps it; the actual <input type=number> lives inside.
        const root = (typeof gradioApp === "function") ? gradioApp() : document;
        const container = root.querySelector("#img2img_steps");
        if (!container) return null;
        return container.querySelector('input[type="number"]');
    }

    function getLockCheckbox() {
        const root = (typeof gradioApp === "function") ? gradioApp() : document;
        const container = root.querySelector("#lock_img2img_steps_checkbox");
        if (!container) return null;
        return container.querySelector('input[type="checkbox"]');
    }

    function snapshot() {
        const steps = getStepsInput();
        if (steps && steps.value !== "") {
            savedSteps = steps.value;
        }
    }

    function restoreIfLocked() {
        const lock = getLockCheckbox();
        const steps = getStepsInput();
        if (!lock || !steps) return;

        if (lock.checked) {
            if (savedSteps !== null && steps.value !== savedSteps) {
                suppressNextChange = true;
                steps.value = savedSteps;
                steps.dispatchEvent(new Event("input", { bubbles: true }));
                steps.dispatchEvent(new Event("change", { bubbles: true }));
            }
        } else {
            // Lock is off; keep the snapshot fresh with whatever's there now.
            savedSteps = steps.value;
        }
    }

    function wireUp() {
        const steps = getStepsInput();
        if (!steps) return false;

        if (!steps.dataset.lockStepsWired) {
            steps.dataset.lockStepsWired = "1";

            const onUserChange = (event) => {
                if (suppressNextChange) {
                    suppressNextChange = false;
                    return;
                }
                const lock = getLockCheckbox();
                // Native user edits have isTrusted=true and should become the
                // new locked value. Gradio paste/send updates dispatch
                // synthetic events (isTrusted=false); ignore those while the
                // lock is active so restoreIfLocked can put the snapshot back.
                if (!lock || !lock.checked || event.isTrusted) {
                    savedSteps = steps.value;
                }
            };

            steps.addEventListener("change", onUserChange);
            steps.addEventListener("input", onUserChange);
            // Gradio may replace the entire input during Send-to-img2img.
            // Wiring that new node must not overwrite an existing lock
            // snapshot with the just-pasted value.
            const lock = getLockCheckbox();
            if (savedSteps === null || !lock || !lock.checked) {
                savedSteps = steps.value;
            }
        }
        return true;
    }

    // Hook into WebUI lifecycle. These globals are provided by the Forge/A1111
    // frontend; guard in case they aren't ready yet.
    function install() {
        if (typeof onUiLoaded === "function") {
            onUiLoaded(() => { wireUp(); snapshot(); });
        } else {
            document.addEventListener("DOMContentLoaded", () => {
                wireUp(); snapshot();
            });
        }

        if (typeof onAfterUiUpdate === "function") {
            onAfterUiUpdate(() => {
                // The Steps input may be re-rendered; re-wire if needed.
                wireUp();
                restoreIfLocked();
            });
        } else {
            // Fallback: poll lightly.
            setInterval(() => { wireUp(); restoreIfLocked(); }, 500);
        }
    }

    install();
})();
