const sceneSelect = document.getElementById("scene-select");
const startSceneButton = document.getElementById("start-scene");
const runAllButton = document.getElementById("run-all");
const endSceneButton = document.getElementById("end-scene");
const endAllButton = document.getElementById("end-all");
const startInferenceButton = document.getElementById("start-inference");
const stopInferenceButton = document.getElementById("stop-inference");
const startEgoButton = document.getElementById("start-ego");
const stopEgoButton = document.getElementById("stop-ego");
const modelSelect = document.getElementById("model-select");
const loadModelButton = document.getElementById("load-model");
const banner = document.getElementById("banner");
const fivemStatus = document.getElementById("fivem-status");
const runtimeStatus = document.getElementById("runtime-status");
const activeScene = document.getElementById("active-scene");
const lastCommand = document.getElementById("last-command");
const queueList = document.getElementById("queue");
const queueEmpty = document.getElementById("queue-empty");
const inferenceState = document.getElementById("inference-state");
const inferenceSteering = document.getElementById("inference-steering");
const inferenceAcceleration = document.getElementById("inference-acceleration");
const inferenceSequence = document.getElementById("inference-sequence");
const inferencePredictedAt = document.getElementById("inference-predicted-at");
const inferenceError = document.getElementById("inference-error");
const actuatorReady = document.getElementById("actuator-ready");
const actuatorLastError = document.getElementById("actuator-last-error");
const actuatorApplied = document.getElementById("actuator-applied");
const actuatorConfigPath = document.getElementById("actuator-config-path");
const actuatorApplyButton = document.getElementById("actuator-apply");
const actuatorResetButton = document.getElementById("actuator-reset");
const actuatorSaveButton = document.getElementById("actuator-save");
const steerDeadzoneInput = document.getElementById("tuning-steer-deadzone");
const maxSteerScaleInput = document.getElementById("tuning-max-steer-scale");
const steerInputGainInput = document.getElementById("tuning-steer-input-gain");
const throttleInputGainInput = document.getElementById("tuning-throttle-input-gain");
const brakeInputGainInput = document.getElementById("tuning-brake-input-gain");
const steerRateInput = document.getElementById("tuning-steer-rate");
const throttleRateInput = document.getElementById("tuning-throttle-rate");
const brakeRateInput = document.getElementById("tuning-brake-rate");
const steerDeadzoneSaved = document.getElementById("saved-steer-deadzone");
const maxSteerScaleSaved = document.getElementById("saved-max-steer-scale");
const steerInputGainSaved = document.getElementById("saved-steer-input-gain");
const throttleInputGainSaved = document.getElementById("saved-throttle-input-gain");
const brakeInputGainSaved = document.getElementById("saved-brake-input-gain");
const steerRateSaved = document.getElementById("saved-steer-rate");
const throttleRateSaved = document.getElementById("saved-throttle-rate");
const brakeRateSaved = document.getElementById("saved-brake-rate");

let busy = false;
let selectedSceneName = "";
let selectedModelPath = "";
let lastModelsRefreshAt = 0;
let actuatorTuningDirty = false;

function setBusy(nextBusy) {
    busy = nextBusy;
    for (const button of [
        startSceneButton,
        runAllButton,
        endSceneButton,
        endAllButton,
        startInferenceButton,
        stopInferenceButton,
        startEgoButton,
        stopEgoButton,
        loadModelButton,
        actuatorApplyButton,
        actuatorResetButton,
        actuatorSaveButton,
    ]) {
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

function ensureModelOptions(models) {
    if (!Array.isArray(models) || models.length === 0) {
        modelSelect.innerHTML = "";
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No model checkpoints found";
        modelSelect.appendChild(option);
        modelSelect.disabled = true;
        selectedModelPath = "";
        return;
    }

    const nextSignature = JSON.stringify(models.map((model) => [model.path, model.label]));
    if (modelSelect.dataset.signature === nextSignature) {
        return;
    }

    modelSelect.innerHTML = "";
    for (const model of models) {
        const option = document.createElement("option");
        option.value = model.path;
        option.textContent = model.label;
        modelSelect.appendChild(option);
    }
    modelSelect.dataset.signature = nextSignature;
    modelSelect.disabled = false;
    selectedModelPath = modelSelect.value;
}

async function fetchInferenceModels() {
    const response = await fetch("/inference/models", { cache: "no-store" });
    if (!response.ok) {
        throw new Error("failed to load model list");
    }
    return response.json();
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

async function fetchInferenceStatus() {
    const response = await fetch("/inference/status", { cache: "no-store" });
    if (!response.ok) {
        throw new Error("failed to load inference status");
    }
    return response.json();
}

async function fetchActuatorState() {
    const response = await fetch("/actuator/state", { cache: "no-store" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok && response.status !== 501 && response.status !== 503) {
        throw new Error(result.error || "failed to load actuator state");
    }
    return result;
}

async function fetchActuatorTuning() {
    const response = await fetch("/actuator/tuning", { cache: "no-store" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || "failed to load actuator tuning");
    }
    return result;
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

async function postInference(endpoint, payload = {}) {
    return postJSON(endpoint, payload, "inference request failed");
}

async function postJSON(endpoint, payload = {}, fallbackMessage = "request failed") {
    const response = await fetch(endpoint, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || fallbackMessage);
    }
    return result;
}

function renderInferenceStatus(status) {
    const prediction = status && status.lastPrediction ? status.lastPrediction : null;
    inferenceState.textContent = status && status.state ? status.state : "idle";
    inferenceSteering.textContent = prediction ? Number(prediction.steering || 0).toFixed(4) : "0.0000";
    inferenceAcceleration.textContent = prediction ? Number(prediction.acceleration || 0).toFixed(4) : "0.0000";
    inferenceSequence.textContent = prediction && prediction.sequence !== undefined ? String(prediction.sequence) : "None";
    inferencePredictedAt.textContent = prediction && prediction.predictedAt ? prediction.predictedAt : "None";
    inferenceError.textContent = status && status.lastError ? status.lastError : "None";
}

function renderActuatorState(state) {
    const supported = state && state.supported !== false;
    const ready = Boolean(state && state.ready);
    actuatorReady.textContent = !supported ? "unsupported" : (ready ? "ready" : "not ready");
    actuatorLastError.textContent = state && state.lastError ? state.lastError : "None";

    const applied = state && state.applied ? state.applied : null;
    if (!applied) {
        actuatorApplied.textContent = "steer=0.000 throttle=0.000 brake=0.000";
        return;
    }

    actuatorApplied.textContent =
        `steer=${Number(applied.steer || 0).toFixed(3)} ` +
        `throttle=${Number(applied.throttle || 0).toFixed(3)} ` +
        `brake=${Number(applied.brake || 0).toFixed(3)}` +
        `${applied.handbrake ? " handbrake=on" : ""}` +
        `${applied.stale ? " stale" : ""}`;
}

function readActuatorTuningForm() {
    const tuning = {
        steerDeadzone: Number(steerDeadzoneInput.value),
        maxSteerScale: Number(maxSteerScaleInput.value),
        steerInputGain: Number(steerInputGainInput.value),
        throttleInputGain: Number(throttleInputGainInput.value),
        brakeInputGain: Number(brakeInputGainInput.value),
        steerRatePerSecond: Number(steerRateInput.value),
        throttleRatePerSecond: Number(throttleRateInput.value),
        brakeRatePerSecond: Number(brakeRateInput.value),
    };

    for (const [key, value] of Object.entries(tuning)) {
        if (!Number.isFinite(value)) {
            throw new Error(`invalid ${key}`);
        }
    }

    return tuning;
}

function formatTuningValue(value) {
    return Number(value || 0).toFixed(3);
}

function renderActuatorTuning(state, force = false) {
    const live = state && state.live ? state.live : null;
    const saved = state && state.saved ? state.saved : null;

    actuatorConfigPath.textContent = state && state.configPath
        ? `Config file: ${state.configPath}`
        : "Config file: unavailable";
    actuatorSaveButton.disabled = busy || !(state && state.saveSupported);

    if (saved) {
        steerDeadzoneSaved.textContent = `Saved: ${formatTuningValue(saved.steerDeadzone)}`;
        maxSteerScaleSaved.textContent = `Saved: ${formatTuningValue(saved.maxSteerScale)}`;
        steerInputGainSaved.textContent = `Saved: ${formatTuningValue(saved.steerInputGain)}`;
        throttleInputGainSaved.textContent = `Saved: ${formatTuningValue(saved.throttleInputGain)}`;
        brakeInputGainSaved.textContent = `Saved: ${formatTuningValue(saved.brakeInputGain)}`;
        steerRateSaved.textContent = `Saved: ${formatTuningValue(saved.steerRatePerSecond)}`;
        throttleRateSaved.textContent = `Saved: ${formatTuningValue(saved.throttleRatePerSecond)}`;
        brakeRateSaved.textContent = `Saved: ${formatTuningValue(saved.brakeRatePerSecond)}`;
    }

    if (!live || (actuatorTuningDirty && !force)) {
        return;
    }

    steerDeadzoneInput.value = formatTuningValue(live.steerDeadzone);
    maxSteerScaleInput.value = formatTuningValue(live.maxSteerScale);
    steerInputGainInput.value = formatTuningValue(live.steerInputGain);
    throttleInputGainInput.value = formatTuningValue(live.throttleInputGain);
    brakeInputGainInput.value = formatTuningValue(live.brakeInputGain);
    steerRateInput.value = formatTuningValue(live.steerRatePerSecond);
    throttleRateInput.value = formatTuningValue(live.throttleRatePerSecond);
    brakeRateInput.value = formatTuningValue(live.brakeRatePerSecond);
    actuatorTuningDirty = false;
}

async function refresh(forceModels = false) {
    const tasks = [fetchState(), fetchInferenceStatus(), fetchActuatorState(), fetchActuatorTuning()];
    const shouldRefreshModels = forceModels || (Date.now() - lastModelsRefreshAt > 10000);
    if (shouldRefreshModels) {
        tasks.push(fetchInferenceModels());
    }
    const [state, inference, actuator, tuning, modelResult] = await Promise.all(tasks);
    ensureSceneOptions(state.availableScenes);
    if (modelResult) {
        ensureModelOptions(modelResult.models);
        lastModelsRefreshAt = Date.now();
    }
    setConnectionPill(fivemStatus, Boolean(state.runtime && state.runtime.fivemConnected));
    runtimeStatus.textContent = formatRuntimeStatus(state.runtime && state.runtime.status);
    activeScene.textContent = (state.runtime && state.runtime.activeSceneName) || "None";
    lastCommand.textContent = formatCommand(state.lastCommand);
    renderQueue(state.pendingCommands);
    renderInferenceStatus(inference);
    renderActuatorState(actuator);
    renderActuatorTuning(tuning);

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

modelSelect.addEventListener("change", () => {
    selectedModelPath = modelSelect.value;
});

for (const input of [
    steerDeadzoneInput,
    maxSteerScaleInput,
    steerInputGainInput,
    throttleInputGainInput,
    brakeInputGainInput,
    steerRateInput,
    throttleRateInput,
    brakeRateInput,
]) {
    input.addEventListener("input", () => {
        actuatorTuningDirty = true;
    });
}

startSceneButton.addEventListener("click", () => handleCommand("startScene", "Start scene queued."));
loadModelButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        if (!modelSelect.value) {
            throw new Error("select a model checkpoint first");
        }
        setBanner("Loading selected model...", "");
        const result = await postInference("/inference/model/load", { checkpoint: modelSelect.value });
        const checkpoint = result && result.checkpoint ? String(result.checkpoint) : modelSelect.value;
        const checkpointLabel = checkpoint.replaceAll("\\", "/").split("/").pop() || checkpoint;
        setBanner(`Loaded model: ${checkpointLabel}`, "success");
        await refresh(true);
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to load model", "error");
    } finally {
        setBusy(false);
    }
});
runAllButton.addEventListener("click", () => handleCommand("runAllScenes", "Run-all command queued."));
endSceneButton.addEventListener("click", () => handleCommand("endScene", "End-scene command queued."));
endAllButton.addEventListener("click", () => handleCommand("endAllScenes", "End-all command queued."));
startEgoButton.addEventListener("click", () => handleCommand("startEgo", "Ego control queued."));
stopEgoButton.addEventListener("click", () => handleCommand("stopEgo", "Stop ego command queued."));
startInferenceButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setBanner("Starting backend inference...", "");
        await postInference("/inference/start");
        setBanner("Backend inference started.", "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to start inference", "error");
    } finally {
        setBusy(false);
    }
});
stopInferenceButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setBanner("Stopping backend inference...", "");
        await postInference("/inference/stop");
        setBanner("Backend inference stopped.", "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to stop inference", "error");
    } finally {
        setBusy(false);
    }
});
actuatorApplyButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setBanner("Applying actuator tuning...", "");
        const result = await postJSON("/actuator/tuning/apply", readActuatorTuningForm(), "failed to apply actuator tuning");
        renderActuatorTuning(result, true);
        setBanner("Actuator tuning applied live.", "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to apply actuator tuning", "error");
    } finally {
        setBusy(false);
    }
});
actuatorResetButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setBanner("Resetting live tuning to saved values...", "");
        const result = await postJSON("/actuator/tuning/reset", {}, "failed to reset actuator tuning");
        renderActuatorTuning(result, true);
        setBanner("Actuator tuning reset to saved values.", "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to reset actuator tuning", "error");
    } finally {
        setBusy(false);
    }
});
actuatorSaveButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setBanner("Saving live tuning to config...", "");
        const result = await postJSON("/actuator/tuning/save", {}, "failed to save actuator tuning");
        renderActuatorTuning(result, true);
        setBanner("Actuator tuning saved to config.", "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to save actuator tuning", "error");
    } finally {
        setBusy(false);
    }
});

setBusy(false);
refresh(true).catch((error) => {
    setBanner(error instanceof Error ? error.message : "failed to load state", "error");
});
setInterval(() => {
    refresh(true).catch((error) => {
        setBanner(error instanceof Error ? error.message : "failed to refresh state", "error");
    });
}, 1500);
