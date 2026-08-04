(function () {
    const LOG_PREFIX = "[neo-repeat-generate]";
    const SESSION_HASH_KEY = "neo_repeat_generate_session_hash";
    const tabs = {
        txt2img: {
            toolRowId: "txt2img_tools",
            generateElemId: "txt2img_generate",
            interruptElemId: "txt2img_interrupt",
            submitFunctionName: "submit",
            localTaskKey: "txt2img_task_id",
            statusPrefix: "txt2img",
            galleryContainerId: "txt2img_gallery_container",
            galleryId: "txt2img_gallery",
        },
        img2img: {
            toolRowId: "img2img_tools",
            generateElemId: "img2img_generate",
            interruptElemId: "img2img_interrupt",
            submitFunctionName: "submit_img2img",
            localTaskKey: "img2img_task_id",
            statusPrefix: "img2img",
            galleryContainerId: "img2img_gallery_container",
            galleryId: "img2img_gallery",
        },
    };

    const states = {
        txt2img: createState(),
        img2img: createState(),
    };

    const taskToTab = new Map();
    let originalRequestProgress = null;
    let requestProgressWrapped = false;
    const wrappedSubmitFunctions = new Set();

    function createState() {
        return {
            running: false,
            busy: false,
            button: null,
            interruptButton: null,
            status: null,
            dependency: null,
            iteration: 0,
            lastTaskId: null,
            lastEventId: null,
            capturedArgs: null,
            seeded: false,
        };
    }

    function log(tabName, message, extra) {
        const timestamp = new Date().toISOString();
        if (extra === undefined) {
            console.log(`${LOG_PREFIX}[${tabName}] ${timestamp} ${message}`);
            return;
        }

        console.log(`${LOG_PREFIX}[${tabName}] ${timestamp} ${message}`, extra);
    }

    function grRoot() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function getComponentByElemId(elemId) {
        return window.gradio_config?.components?.find(function (component) {
            return component?.props?.elem_id === elemId;
        });
    }

    function getMainGenerateDependency(tabName) {
        const state = states[tabName];
        if (state.dependency) {
            return state.dependency;
        }

        const component = getComponentByElemId(tabs[tabName].generateElemId);
        if (!component) {
            throw new Error(`Component not found: ${tabs[tabName].generateElemId}`);
        }

        const dependency = (window.gradio_config?.dependencies || [])
            .filter(function (entry) {
                return Array.isArray(entry?.targets) && entry.targets.some(function (target) {
                    return Array.isArray(target) && target[0] === component.id && target[1] === "click";
                });
            })
            .sort(function (left, right) {
                return (right.inputs?.length || 0) - (left.inputs?.length || 0);
            })[0];

        if (!dependency) {
            throw new Error(`Generate dependency not found for ${tabName}`);
        }

        state.dependency = {
            fnIndex: dependency.id,
            inputCount: dependency.inputs?.length || 0,
            triggerId: component.id,
        };

        log(tabName, "generate dependency resolved", state.dependency);
        return state.dependency;
    }

    function getSessionHash() {
        const candidates = [
            window.__gradio_client__,
            window.gradio_client,
            window.client,
            window.app,
        ];

        for (const candidate of candidates) {
            if (candidate && typeof candidate.session_hash === "string" && candidate.session_hash) {
                return candidate.session_hash;
            }
        }

        for (const key of Object.getOwnPropertyNames(window)) {
            try {
                const value = window[key];
                if (value && typeof value === "object" && typeof value.session_hash === "string" && value.session_hash) {
                    return value.session_hash;
                }
            } catch (error) {
                // Ignore getter failures.
            }
        }

        let fallback = sessionStorage.getItem(SESSION_HASH_KEY);
        if (!fallback) {
            fallback = Math.random().toString(36).slice(2);
            sessionStorage.setItem(SESSION_HASH_KEY, fallback);
        }

        return fallback;
    }

    function cloneArgs(args) {
        return Array.isArray(args) ? args.slice() : null;
    }

    function createTaskId() {
        if (typeof window.randomId === "function") {
            return window.randomId();
        }

        return `task(${Math.random().toString(36).slice(2)})`;
    }

    function setStatus(tabName, text) {
        const state = states[tabName];
        if (state.status) {
            state.status.textContent = text;
        }
    }

    function updateButton(tabName) {
        const state = states[tabName];
        if (!state.button) {
            return;
        }

        if (state.running) {
            state.button.textContent = "■ Stop Repeat";
            state.button.classList.toggle("neo-repeat-generate-wait", false);
        } else {
            state.button.textContent = state.busy ? "⌛ Stopping" : "↻ Start Repeat";
            state.button.classList.toggle("neo-repeat-generate-wait", state.busy);
        }

        state.button.classList.toggle("neo-repeat-generate-active", state.running);

        if (state.interruptButton) {
            state.interruptButton.disabled = !state.busy;
            state.interruptButton.classList.toggle("neo-repeat-generate-interrupt-enabled", state.busy);
        }
    }

    function getGenerateButton(tabName) {
        return grRoot().getElementById(tabs[tabName].generateElemId);
    }

    function getInterruptButton(tabName) {
        return grRoot().getElementById(tabs[tabName].interruptElemId);
    }

    function runSubmitSideEffects(tabName, taskId) {
        const tab = tabs[tabName];

        if (typeof showSubmitButtons === "function") {
            showSubmitButtons(tab.statusPrefix, false);
        }

        if (typeof localSet === "function") {
            localSet(tab.localTaskKey, taskId);
        }

        if (typeof requestProgress === "function") {
            requestProgress(
                taskId,
                grRoot().getElementById(tab.galleryContainerId),
                grRoot().getElementById(tab.galleryId),
                function () {
                    if (typeof showSubmitButtons === "function") {
                        showSubmitButtons(tab.statusPrefix, true);
                    }

                    if (typeof localRemove === "function") {
                        localRemove(tab.localTaskKey);
                    }

                    if (typeof showRestoreProgressButton === "function") {
                        showRestoreProgressButton(tab.statusPrefix, false);
                    }
                },
            );
        }
    }

    async function queueJoin(tabName, args) {
        const dependency = getMainGenerateDependency(tabName);
        const sessionHash = getSessionHash();
        const payload = {
            data: args,
            event_data: null,
            fn_index: dependency.fnIndex,
            trigger_id: dependency.triggerId,
            session_hash: sessionHash,
        };

        log(tabName, "queue join request", {
            fn_index: payload.fn_index,
            trigger_id: payload.trigger_id,
            session_hash: payload.session_hash,
            args_count: args.length,
        });

        const response = await fetch("./queue/join", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        return await response.json();
    }

    async function startQueuedGeneration(tabName, reason) {
        const state = states[tabName];
        if (!state.running || state.busy) {
            return;
        }

        if (!state.capturedArgs) {
            throw new Error(`No captured submit args for ${tabName}`);
        }

        const args = cloneArgs(state.capturedArgs);
        const taskId = createTaskId();
        args[0] = taskId;

        state.busy = true;
        state.lastTaskId = taskId;
        taskToTab.set(taskId, tabName);
        updateButton(tabName);
        setStatus(tabName, `Preparing ${state.iteration + 1}...`);

        runSubmitSideEffects(tabName, taskId);

        log(tabName, "queue repeat start", {
            reason,
            taskId,
            argsCount: args.length,
        });

        try {
            const result = await queueJoin(tabName, args);
            state.lastEventId = result.event_id || null;
            state.iteration += 1;
            setStatus(tabName, `Queued ${state.iteration}`);
            log(tabName, "queue join accepted", {
                reason,
                taskId,
                event_id: state.lastEventId,
            });
        } catch (error) {
            taskToTab.delete(taskId);
            state.busy = false;
            state.running = false;
            updateButton(tabName);
            setStatus(tabName, `Error: ${error.message}`);
            if (typeof showSubmitButtons === "function") {
                showSubmitButtons(tabs[tabName].statusPrefix, true);
            }
            if (typeof localRemove === "function") {
                localRemove(tabs[tabName].localTaskKey);
            }
            log(tabName, "queue join failed", {
                reason,
                taskId,
                error: error.message,
            });
        }
    }

    function captureArgsFromSubmit(tabName, resultArgs) {
        const state = states[tabName];
        const cloned = cloneArgs(resultArgs);
        if (!cloned || !cloned.length) {
            return;
        }

        state.capturedArgs = cloned;
        state.lastTaskId = cloned[0] || null;
        state.seeded = true;
        taskToTab.set(state.lastTaskId, tabName);
        log(tabName, "submit args captured", {
            taskId: state.lastTaskId,
            argsCount: cloned.length,
            running: state.running,
        });
    }

    function installSubmitWrapper(tabName) {
        const functionName = tabs[tabName].submitFunctionName;
        if (wrappedSubmitFunctions.has(functionName) || typeof window[functionName] !== "function") {
            return;
        }

        const original = window[functionName];
        window[functionName] = function () {
            log(tabName, "submit called", {
                visibilityState: document.visibilityState,
                hidden: document.hidden,
            });

            const result = original.apply(this, arguments);
            captureArgsFromSubmit(tabName, result);

            log(tabName, "submit returned", {
                taskId: Array.isArray(result) ? result[0] : null,
                visibilityState: document.visibilityState,
                hidden: document.hidden,
            });

            return result;
        };

        wrappedSubmitFunctions.add(functionName);
        log(tabName, "submit wrapped", { functionName });
    }

    function handleTaskFinished(idTask) {
        const tabName = taskToTab.get(idTask);
        if (!tabName) {
            return;
        }

        const state = states[tabName];
        taskToTab.delete(idTask);
        state.busy = false;
        updateButton(tabName);

        log(tabName, "tracked task finished", {
            taskId: idTask,
            running: state.running,
            seeded: state.seeded,
        });

        if (!state.running) {
            setStatus(tabName, "Stopped");
            return;
        }

        setStatus(tabName, `Completed ${state.iteration || 1}`);
        window.setTimeout(function () {
            startQueuedGeneration(tabName, "requestProgress atEnd");
        }, 0);
    }

    function installRequestProgressWrapper() {
        if (requestProgressWrapped || typeof window.requestProgress !== "function") {
            return;
        }

        originalRequestProgress = window.requestProgress;
        window.requestProgress = function (idTask, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout) {
            const wrappedAtEnd = function () {
                try {
                    if (typeof atEnd === "function") {
                        atEnd();
                    }
                } finally {
                    handleTaskFinished(idTask);
                }
            };

            return originalRequestProgress.call(
                this,
                idTask,
                progressbarContainer,
                gallery,
                wrappedAtEnd,
                onProgress,
                inactivityTimeout,
            );
        };

        requestProgressWrapped = true;
        log("core", "requestProgress wrapped");
    }

    function clickGenerateToSeed(tabName) {
        const state = states[tabName];
        const button = getGenerateButton(tabName);
        if (!button) {
            throw new Error(`Generate button not found: ${tabs[tabName].generateElemId}`);
        }

        state.busy = true;
        updateButton(tabName);
        setStatus(tabName, "Seeding from Generate...");
        log(tabName, "generate click fired", { reason: "seed via click" });
        button.click();
    }

    function requestStop(tabName) {
        const state = states[tabName];
        state.running = false;
        updateButton(tabName);
        setStatus(tabName, "Stopping after current task...");
        log(tabName, "repeat stop requested", {
            busy: state.busy,
            taskId: state.lastTaskId,
        });
    }

    function requestInterruptStop(tabName) {
        const state = states[tabName];
        state.running = false;
        updateButton(tabName);
        setStatus(tabName, "Interrupting...");
        log(tabName, "repeat interrupt requested", {
            busy: state.busy,
            taskId: state.lastTaskId,
        });

        const interrupt = getInterruptButton(tabName);
        if (!interrupt) {
            setStatus(tabName, "Interrupt button not found");
            log(tabName, "interrupt button missing");
            return;
        }

        interrupt.click();
        log(tabName, "interrupt click fired");
    }

    function toggleRepeat(tabName) {
        const state = states[tabName];
        if (state.running) {
            requestStop(tabName);
            return;
        }

        state.running = true;
        state.busy = false;
        state.iteration = 0;
        state.lastEventId = null;
        state.capturedArgs = null;
        state.seeded = false;
        updateButton(tabName);
        log(tabName, "repeat started", {
            hasCapturedArgs: false,
        });

        clickGenerateToSeed(tabName);
    }

    function ensureUi(tabName) {
        const state = states[tabName];
        const toolRow = grRoot().getElementById(tabs[tabName].toolRowId);
        if (!toolRow) {
            return;
        }

        if (state.button && state.button.isConnected) {
            return;
        }

        const button = document.createElement("button");
        button.type = "button";
        button.className = "lg secondary gradio-button neo-repeat-generate-button";
        button.addEventListener("click", function () {
            toggleRepeat(tabName);
        });

        const interruptButton = document.createElement("button");
        interruptButton.type = "button";
        interruptButton.className = "lg secondary gradio-button neo-repeat-generate-interrupt-button";
        interruptButton.textContent = "Interrupt";
        interruptButton.addEventListener("click", function () {
            requestInterruptStop(tabName);
        });

        const status = document.createElement("span");
        status.className = "neo-repeat-generate-status";

        state.button = button;
        state.interruptButton = interruptButton;
        state.status = status;

        updateButton(tabName);
        setStatus(tabName, "Stopped");

        const buttons = document.createElement("div");
        buttons.className = "neo-repeat-generate-buttons";

        buttons.appendChild(button);
        buttons.appendChild(interruptButton);
        toolRow.appendChild(buttons);
        toolRow.appendChild(status);

        log(tabName, "ui mounted");
    }

    function mountUi() {
        installRequestProgressWrapper();
        installSubmitWrapper("txt2img");
        installSubmitWrapper("img2img");
        ensureUi("txt2img");
        ensureUi("img2img");
    }

    onUiLoaded(function () {
        mountUi();
    });

    onAfterUiUpdate(function () {
        mountUi();
    });
})();
