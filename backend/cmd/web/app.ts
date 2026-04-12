const sceneSelect = document.getElementById("scene-select");
const startSceneButton = document.getElementById("start-scene");
const runAllButton = document.getElementById("run-all");
const endSceneButton = document.getElementById("end-scene");
const endAllButton = document.getElementById("end-all");
const banner = document.getElementById("banner");
const fivemStatus = document.getElementById("fivem-status");
const runtimeStatus = document.getElementById("runtime-status");
const activeScene = document.getElementById("active-scene");
const lastCommand = document.getElementById("last-command");
const queueList = document.getElementById("queue");
const queueEmpty = document.getElementById("queue-empty");

let busy = false;
let selectedSceneName = "";

function setBusy(nextBusy) {
    busy = nextBusy;
    for (const button of [startSceneButton, runAllButton, endSceneButton, endAllButton]) {
        button.disabled = nextBusy;
    }
}

function setBanner(message, tone) {
    banner.textContent = message || "";
    banner.className = "banner";
    if (tone) {
        banner.classList.add(tone);
    }
}

function formatCommand(command) {
    if (!command) {
        return "Nothing queued yet";
    }
    const sceneSuffix = command.sceneName ? ` - ${command.sceneName}` : "";
    return `${command.type}${sceneSuffix}`;
}

function formatRuntimeStatus(status) {
    if (!status) {
        return "idle";
    }
    return status.replace(/[A-Z]/g, (match) => ` ${match.toLowerCase()}`);
}

function setConnectionPill(element, connected) {
    const tone = connected ? "ok" : "bad";
    const label = connected ? "Connected" : "Waiting for poll";
    element.innerHTML = `<span class="pill ${tone}">${label}</span>`;
}

function ensureSceneOptions(availableScenes) {
    if (!Array.isArray(availableScenes) || availableScenes.length === 0) {
        if (!sceneSelect.options.length) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "Waiting for FiveM scene list";
            sceneSelect.appendChild(option);
        }
        sceneSelect.dataset.signature = "";
        sceneSelect.disabled = true;
        return;
    }

    const nextSignature = JSON.stringify(availableScenes);
    if (sceneSelect.dataset.signature === nextSignature) {
        if (!selectedSceneName) {
            selectedSceneName = sceneSelect.value;
        }
        return;
    }

    sceneSelect.innerHTML = "";
    for (const scene of availableScenes) {
        const option = document.createElement("option");
        option.value = scene.name;
        option.textContent = scene.label;
        sceneSelect.appendChild(option);
    }
    sceneSelect.dataset.signature = nextSignature;
    sceneSelect.disabled = false;
    selectedSceneName = availableScenes[0].name;
    sceneSelect.value = selectedSceneName;
}

function renderQueue(pendingCommands) {
    queueList.innerHTML = "";

    if (!Array.isArray(pendingCommands) || pendingCommands.length === 0) {
        queueEmpty.hidden = false;
        return;
    }

    queueEmpty.hidden = true;
    for (const command of pendingCommands) {
        const item = document.createElement("li");
        item.innerHTML = `
            <strong>${command.type}</strong>
            <div class="muted">${command.sceneName || "No scene argument"}</div>
            <div class="muted">${command.createdAt || ""}</div>
        `;
        queueList.appendChild(item);
    }
}

async function fetchState() {
    const response = await fetch("/control/state", { cache: "no-store" });
    if (!response.ok) {
        throw new Error("failed to load control state");
    }
    return response.json();
}

async function sendCommand(type) {
    const payload = { type };
    if (type === "startScene") {
        payload.sceneName = sceneSelect.value;
    }

    const response = await fetch("/control/command", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || "command failed");
    }
    return result;
}

async function refresh() {
    const state = await fetchState();
    ensureSceneOptions(state.availableScenes);
    setConnectionPill(fivemStatus, Boolean(state.runtime && state.runtime.fivemConnected));
    runtimeStatus.textContent = formatRuntimeStatus(state.runtime && state.runtime.status);
    activeScene.textContent = (state.runtime && state.runtime.activeSceneName) || "None";
    lastCommand.textContent = formatCommand(state.lastCommand);
    renderQueue(state.pendingCommands);

    if (!Array.isArray(state.availableScenes) || state.availableScenes.length === 0) {
        setBanner("Waiting for scene list from FiveM.", "");
    } else if (state.runtime && state.runtime.lastError) {
        setBanner(state.runtime.lastError, "error");
    } else if (!busy && !banner.textContent) {
        setBanner("Control page ready.", "success");
    }
}

async function handleCommand(type, successMessage) {
    try {
        setBusy(true);
        setBanner("Queueing command...", "");
        await sendCommand(type);
        setBanner(successMessage, "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "command failed", "error");
    } finally {
        setBusy(false);
    }
}

sceneSelect.addEventListener("change", () => {
    selectedSceneName = sceneSelect.value;
});

startSceneButton.addEventListener("click", () => handleCommand("startScene", "Start scene queued."));
runAllButton.addEventListener("click", () => handleCommand("runAllScenes", "Run-all command queued."));
endSceneButton.addEventListener("click", () => handleCommand("endScene", "End-scene command queued."));
endAllButton.addEventListener("click", () => handleCommand("endAllScenes", "End-all command queued."));

setBusy(false);
refresh().catch((error) => {
    setBanner(error instanceof Error ? error.message : "failed to load state", "error");
});
setInterval(() => {
    refresh().catch((error) => {
        setBanner(error instanceof Error ? error.message : "failed to refresh state", "error");
    });
}, 1500);
