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
const inferenceFutureYawDelta = document.getElementById("inference-future-yaw-delta");
const inferenceFutureYaw = document.getElementById("inference-future-yaw");
const inferenceCurrentYaw = document.getElementById("inference-current-yaw");
const inferenceHeadingError = document.getElementById("inference-heading-error");
const inferenceFutureSpeed = document.getElementById("inference-future-speed");
const inferenceDeltaSpeed = document.getElementById("inference-delta-speed");
const inferenceMoveIntent = document.getElementById("inference-move-intent");
const inferenceSequence = document.getElementById("inference-sequence");
const inferenceFrameIndex = document.getElementById("inference-frame-index");
const inferencePredictedAt = document.getElementById("inference-predicted-at");
const inferenceControlSemantics = document.getElementById("inference-control-semantics");
const inferenceDebugFrames = document.getElementById("inference-debug-frames");
const inferenceDebugDir = document.getElementById("inference-debug-dir");
const inferenceError = document.getElementById("inference-error");
const inferenceWindowIndices = document.getElementById("inference-window-indices");
const inferenceFrameHashes = document.getElementById("inference-frame-hashes");
const actuatorReady = document.getElementById("actuator-ready");
const actuatorLastError = document.getElementById("actuator-last-error");
const actuatorLastCommand = document.getElementById("actuator-last-command");
const actuatorTarget = document.getElementById("actuator-target");
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
const modelSteerScaleInput = document.getElementById("tuning-model-steer-scale");
const modelAccelScaleInput = document.getElementById("tuning-model-accel-scale");
const steerRateInput = document.getElementById("tuning-steer-rate");
const throttleRateInput = document.getElementById("tuning-throttle-rate");
const brakeRateInput = document.getElementById("tuning-brake-rate");
const steerDeadzoneSaved = document.getElementById("saved-steer-deadzone");
const maxSteerScaleSaved = document.getElementById("saved-max-steer-scale");
const steerInputGainSaved = document.getElementById("saved-steer-input-gain");
const throttleInputGainSaved = document.getElementById("saved-throttle-input-gain");
const brakeInputGainSaved = document.getElementById("saved-brake-input-gain");
const modelSteerScaleSaved = document.getElementById("saved-model-steer-scale");
const modelAccelScaleSaved = document.getElementById("saved-model-accel-scale");
const steerRateSaved = document.getElementById("saved-steer-rate");
const throttleRateSaved = document.getElementById("saved-throttle-rate");
const brakeRateSaved = document.getElementById("saved-brake-rate");

const pageTabs = Array.from(document.querySelectorAll("[data-page-tab]"));
const pageControl = document.getElementById("page-control");
const pageInspector = document.getElementById("page-inspector");
const inspectorRunSelect = document.getElementById("data-run-select");
const inspectorSceneSelect = document.getElementById("data-scene-select");
const inspectorTripSelect = document.getElementById("data-trip-select");
const inspectorSourceSelect = document.getElementById("data-source-select");
const inspectorLayoutSelect = document.getElementById("data-layout-select");
const inspectorFieldSearch = document.getElementById("data-field-search");
const inspectorFieldList = document.getElementById("data-field-list");
const inspectorFieldEmpty = document.getElementById("data-field-empty");
const inspectorBanner = document.getElementById("data-banner");
const inspectorSourceMeta = document.getElementById("data-source-meta");
const inspectorRowCount = document.getElementById("data-row-count");
const inspectorLoadButton = document.getElementById("data-load-series");
const inspectorSelectVisibleButton = document.getElementById("data-select-visible");
const inspectorClearFieldsButton = document.getElementById("data-clear-fields");
const inspectorChart = document.getElementById("data-chart");
const inspectorChartLegend = document.getElementById("data-chart-legend");
const inspectorStats = document.getElementById("data-stats");
const inspectorStatsEmpty = document.getElementById("data-stats-empty");
const inspectorTableHead = document.getElementById("data-table-head");
const inspectorTableBody = document.getElementById("data-table-body");
const inspectorTableEmpty = document.getElementById("data-table-empty");

const INSPECTOR_COLORS = [
    "#38bdf8",
    "#fb7185",
    "#f59e0b",
    "#34d399",
    "#a78bfa",
    "#f97316",
    "#22c55e",
    "#e879f9",
];

let busy = false;
let selectedSceneName = "";
let selectedModelPath = "";
let lastModelsRefreshAt = 0;
let actuatorTuningDirty = false;
let actuatorRefreshInFlight = false;

let activePage = "control";
let inspectorRuns = [];
let inspectorAvailableFields = [];
let inspectorSelectedFields = new Set();
let inspectorSeries = null;
let inspectorFieldsLoading = false;
let inspectorSeriesLoading = false;

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

function setInspectorBusy(loadingFields, loadingSeries) {
    inspectorFieldsLoading = loadingFields;
    inspectorSeriesLoading = loadingSeries;
    const disabled = loadingFields || loadingSeries;
    inspectorRunSelect.disabled = disabled;
    inspectorSceneSelect.disabled = disabled;
    inspectorTripSelect.disabled = disabled;
    inspectorSourceSelect.disabled = disabled;
    inspectorLayoutSelect.disabled = loadingSeries;
    inspectorFieldSearch.disabled = loadingFields;
    inspectorSelectVisibleButton.disabled = loadingFields;
    inspectorClearFieldsButton.disabled = loadingFields;
    inspectorLoadButton.disabled = disabled;
}

function setBanner(message, tone) {
    banner.textContent = message || "";
    banner.className = "banner";
    if (tone) {
        banner.classList.add(tone);
    }
}

function setInspectorBanner(message, tone) {
    inspectorBanner.textContent = message || "";
    inspectorBanner.className = "banner";
    if (tone) {
        inspectorBanner.classList.add(tone);
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

async function fetchDataRuns() {
    const response = await fetch("/data/runs", { cache: "no-store" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || "failed to load runs");
    }
    return result;
}

async function fetchTripFields(runId, sceneKey, tripName, source) {
    const params = new URLSearchParams({
        runId,
        sceneKey,
        tripName,
        source,
    });
    const response = await fetch(`/data/trip/fields?${params.toString()}`, { cache: "no-store" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || "failed to load trip fields");
    }
    return result;
}

async function fetchTripSeries(runId, sceneKey, tripName, source, fields) {
    const params = new URLSearchParams({
        runId,
        sceneKey,
        tripName,
        source,
    });
    for (const field of fields) {
        params.append("field", field);
    }
    const response = await fetch(`/data/trip/series?${params.toString()}`, { cache: "no-store" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || "failed to load trip data");
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

async function postInference(endpoint, payload = {}) {
    return postJSON(endpoint, payload, "inference request failed");
}

function renderInferenceStatus(status) {
    const prediction = status && status.lastPrediction ? status.lastPrediction : null;
    inferenceState.textContent = status && status.state ? status.state : "idle";
    inferenceSteering.textContent = prediction ? Number(prediction.steering || 0).toFixed(4) : "0.0000";
    inferenceFutureYawDelta.textContent = prediction ? Number(prediction.futureYawDelta || 0).toFixed(4) : "0.0000";
    inferenceFutureYaw.textContent = prediction ? Number(prediction.futureYaw || 0).toFixed(4) : "0.0000";
    inferenceCurrentYaw.textContent = prediction ? Number(prediction.currentYaw || 0).toFixed(4) : "0.0000";
    inferenceHeadingError.textContent = prediction ? Number(prediction.headingErrorDeg || 0).toFixed(4) : "0.0000";
    inferenceFutureSpeed.textContent = prediction ? Number(prediction.futureSpeed || 0).toFixed(4) : "0.0000";
    inferenceDeltaSpeed.textContent = prediction ? Number(prediction.deltaSpeed || 0).toFixed(4) : "0.0000";
    inferenceMoveIntent.textContent = prediction && prediction.hasMoveIntent
        ? `${Number(prediction.moveIntentProb || 0).toFixed(3)} (${prediction.moveIntentActive ? "active" : "idle"})`
        : "None";
    inferenceSequence.textContent = prediction && prediction.sequence !== undefined ? String(prediction.sequence) : "None";
    inferenceFrameIndex.textContent = prediction && prediction.frameIndex !== undefined ? String(prediction.frameIndex) : "None";
    inferencePredictedAt.textContent = prediction && prediction.predictedAt ? prediction.predictedAt : "None";
    inferenceControlSemantics.textContent = prediction && prediction.controlSemantics
        ? String(prediction.controlSemantics)
        : "unknown";
    inferenceDebugFrames.textContent = status
        ? `${Number(status.debugFramesSaved || 0)} / ${Number(status.debugFramesLimit || 0)}`
        : "0 / 0";
    inferenceDebugDir.textContent = status && status.debugFramesDir ? String(status.debugFramesDir) : "None";
    inferenceWindowIndices.textContent = prediction && Array.isArray(prediction.windowFrameIndices) && prediction.windowFrameIndices.length
        ? prediction.windowFrameIndices.join(", ")
        : "None";
    inferenceFrameHashes.textContent = prediction && Array.isArray(prediction.windowFrameHashes) && prediction.windowFrameHashes.length
        ? prediction.windowFrameHashes.map((value) => String(value).slice(0, 8)).join(" ")
        : "None";
    inferenceError.textContent = status && status.lastError ? status.lastError : "None";
}

function formatTimestamp(value) {
    if (!value) {
        return "never";
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return String(value);
    }
    return parsed.toLocaleTimeString();
}

function formatControlLine(control, options = {}) {
    if (!control) {
        return "None";
    }
    const parts = [
        `steer=${Number(control.steer || 0).toFixed(3)}`,
        `throttle=${Number(control.throttle || 0).toFixed(3)}`,
        `brake=${Number(control.brake || 0).toFixed(3)}`,
    ];
    if (control.handbrake) {
        parts.push("handbrake=on");
    }
    if (options.includeEnabled) {
        parts.push(control.enabled === false ? "disabled" : "enabled");
    }
    if (options.includeMode && control.inputMode) {
        parts.push(control.inputMode);
    }
    if (options.includeSequence && control.sequence !== undefined) {
        parts.push(`seq=${control.sequence}`);
    }
    if (options.includeTimestamp && control.updatedAt) {
        parts.push(`at=${formatTimestamp(control.updatedAt)}`);
    }
    if (options.includeTimestamp && control.receivedAt) {
        parts.push(`at=${formatTimestamp(control.receivedAt)}`);
    }
    return parts.join(" ");
}

function renderActuatorState(state) {
    const supported = state && state.supported !== false;
    const ready = Boolean(state && state.ready);
    actuatorReady.textContent = !supported ? "unsupported" : (ready ? "ready" : "not ready");
    actuatorLastCommand.textContent = formatControlLine(state && state.lastCommand ? state.lastCommand : null, {
        includeEnabled: true,
        includeMode: true,
        includeSequence: true,
        includeTimestamp: true,
    });
    actuatorTarget.textContent = formatControlLine(state && state.target ? state.target : null, {
        includeEnabled: true,
        includeTimestamp: true,
    });
    actuatorApplied.textContent = formatControlLine(state && state.applied ? state.applied : null, {
        includeEnabled: true,
        includeTimestamp: true,
    });

    const controllerStatus = [];
    if (state && state.target && state.target.timedOut) {
        controllerStatus.push("timeout neutralizing");
    } else if (state && state.applied && state.applied.holding) {
        controllerStatus.push("holding latched input");
    } else if (state && state.target && state.target.enabled) {
        controllerStatus.push("tracking latest input");
    }
    if (state && state.lastApplyError) {
        controllerStatus.push(`error ${state.lastApplyError}`);
    } else if (state && state.lastApplySucceededAt) {
        controllerStatus.push(`ok at ${formatTimestamp(state.lastApplySucceededAt)}`);
    } else if (state && state.lastApplyAttemptedAt) {
        controllerStatus.push(`attempted at ${formatTimestamp(state.lastApplyAttemptedAt)}`);
    } else {
        controllerStatus.push("idle");
    }
    if (state && state.lastError) {
        controllerStatus.push(`service ${state.lastError}`);
    }
    actuatorLastError.textContent = controllerStatus.join(" · ");
}

function readActuatorTuningForm() {
    const tuning = {
        steerDeadzone: Number(steerDeadzoneInput.value),
        maxSteerScale: Number(maxSteerScaleInput.value),
        steerInputGain: Number(steerInputGainInput.value),
        throttleInputGain: Number(throttleInputGainInput.value),
        brakeInputGain: Number(brakeInputGainInput.value),
        modelSteerScale: Number(modelSteerScaleInput.value),
        modelAccelScale: Number(modelAccelScaleInput.value),
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
        modelSteerScaleSaved.textContent = `Saved: ${formatTuningValue(saved.modelSteerScale)}`;
        modelAccelScaleSaved.textContent = `Saved: ${formatTuningValue(saved.modelAccelScale)}`;
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
    modelSteerScaleInput.value = formatTuningValue(live.modelSteerScale);
    modelAccelScaleInput.value = formatTuningValue(live.modelAccelScale);
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

async function refreshActuator() {
    if (actuatorRefreshInFlight || document.hidden) {
        return;
    }
    actuatorRefreshInFlight = true;
    try {
        const state = await fetchActuatorState();
        renderActuatorState(state);
    } catch (error) {
        actuatorLastError.textContent = error instanceof Error ? error.message : "failed to refresh actuator";
    } finally {
        actuatorRefreshInFlight = false;
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

function setActivePage(pageName) {
    activePage = pageName;
    for (const tab of pageTabs) {
        const isActive = tab.dataset.pageTab === pageName;
        tab.classList.toggle("active", isActive);
        tab.setAttribute("aria-pressed", isActive ? "true" : "false");
    }
    pageControl.hidden = pageName !== "control";
    pageInspector.hidden = pageName !== "inspector";
    if (pageName === "inspector" && inspectorRuns.length === 0) {
        refreshInspectorRuns().catch((error) => {
            setInspectorBanner(error instanceof Error ? error.message : "failed to load runs", "error");
        });
    }
}

function renderInspectorRunOptions() {
    inspectorRunSelect.innerHTML = "";
    if (!inspectorRuns.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No runs found";
        inspectorRunSelect.appendChild(option);
        inspectorRunSelect.disabled = true;
        renderInspectorSceneOptions();
        return;
    }
    for (const run of inspectorRuns) {
        const option = document.createElement("option");
        option.value = run.runId;
        option.textContent = `${run.runId} · ${run.tripCount} trips`;
        inspectorRunSelect.appendChild(option);
    }
    inspectorRunSelect.disabled = false;
}

function getSelectedInspectorRun() {
    return inspectorRuns.find((run) => run.runId === inspectorRunSelect.value) || null;
}

function getSelectedInspectorScene() {
    const run = getSelectedInspectorRun();
    if (!run || !Array.isArray(run.scenes)) {
        return null;
    }
    return run.scenes.find((scene) => scene.sceneKey === inspectorSceneSelect.value) || null;
}

function getSelectedInspectorTrip() {
    const scene = getSelectedInspectorScene();
    if (!scene || !Array.isArray(scene.trips)) {
        return null;
    }
    return scene.trips.find((trip) => trip.tripName === inspectorTripSelect.value) || null;
}

function renderInspectorSceneOptions() {
    inspectorSceneSelect.innerHTML = "";
    const run = getSelectedInspectorRun();
    if (!run || !Array.isArray(run.scenes) || !run.scenes.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No scenes found";
        inspectorSceneSelect.appendChild(option);
        inspectorSceneSelect.disabled = true;
        renderInspectorTripOptions();
        return;
    }
    for (const scene of run.scenes) {
        const option = document.createElement("option");
        option.value = scene.sceneKey;
        option.textContent = `${scene.sceneKey} · ${scene.tripCount} trips`;
        inspectorSceneSelect.appendChild(option);
    }
    inspectorSceneSelect.disabled = false;
}

function renderInspectorTripOptions() {
    inspectorTripSelect.innerHTML = "";
    const scene = getSelectedInspectorScene();
    if (!scene || !Array.isArray(scene.trips) || !scene.trips.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No trips found";
        inspectorTripSelect.appendChild(option);
        inspectorTripSelect.disabled = true;
        updateInspectorSourceAvailability();
        return;
    }
    for (const trip of scene.trips) {
        const option = document.createElement("option");
        option.value = trip.tripName;
        option.textContent = `${trip.tripName} · raw ${trip.rawAvailable ? "yes" : "no"} · processed ${trip.processedAvailable ? "yes" : "no"}`;
        inspectorTripSelect.appendChild(option);
    }
    inspectorTripSelect.disabled = false;
    updateInspectorSourceAvailability();
}

function updateInspectorSourceAvailability() {
    const trip = getSelectedInspectorTrip();
    const rawOption = inspectorSourceSelect.querySelector('option[value="raw"]');
    const processedOption = inspectorSourceSelect.querySelector('option[value="processed"]');
    if (!trip) {
        rawOption.disabled = false;
        processedOption.disabled = false;
        return;
    }
    rawOption.disabled = !trip.rawAvailable;
    processedOption.disabled = !trip.processedAvailable;
    if (inspectorSourceSelect.value === "raw" && rawOption.disabled) {
        inspectorSourceSelect.value = processedOption.disabled ? "raw" : "processed";
    }
    if (inspectorSourceSelect.value === "processed" && processedOption.disabled) {
        inspectorSourceSelect.value = rawOption.disabled ? "processed" : "raw";
    }
}

function resetInspectorSeriesState() {
    inspectorSeries = null;
    inspectorSourceMeta.textContent = "No source loaded.";
    inspectorRowCount.textContent = "0 rows";
    renderInspectorChart();
    renderInspectorStats();
    renderInspectorTable();
}

function renderInspectorFieldList() {
    const query = inspectorFieldSearch.value.trim().toLowerCase();
    inspectorFieldList.innerHTML = "";
    const visibleFields = inspectorAvailableFields.filter((field) => field.name.toLowerCase().includes(query));
    inspectorFieldEmpty.hidden = visibleFields.length > 0;
    if (!visibleFields.length) {
        inspectorFieldEmpty.textContent = inspectorAvailableFields.length
            ? "No fields match the current filter."
            : "Select a run, trip, and source to load available fields.";
        return;
    }

    for (const field of visibleFields) {
        const label = document.createElement("label");
        label.className = "field-option";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = inspectorSelectedFields.has(field.name);
        input.addEventListener("change", () => {
            if (input.checked) {
                inspectorSelectedFields.add(field.name);
            } else {
                inspectorSelectedFields.delete(field.name);
            }
            renderInspectorFieldList();
        });

        const name = document.createElement("span");
        name.className = "field-name";
        name.textContent = field.name;

        const kind = document.createElement("span");
        kind.className = "field-kind";
        kind.textContent = field.kind || "unknown";

        label.appendChild(input);
        label.appendChild(name);
        label.appendChild(kind);
        inspectorFieldList.appendChild(label);
    }
}

function chooseDefaultInspectorFields(fields) {
    const preferred = [
        "yaw",
        "currentSpeed",
        "routeForwardDelta",
        "label.future_yaw_delta",
        "label.currentSpeed",
    ];
    const chosen = [];
    for (const name of preferred) {
        const match = fields.find((field) => field.name === name);
        if (match && chosen.length < 4) {
            chosen.push(match.name);
        }
    }
    if (!chosen.length) {
        for (const field of fields) {
            if (field.kind === "number") {
                chosen.push(field.name);
            }
            if (chosen.length >= 4) {
                break;
            }
        }
    }
    return chosen;
}

async function refreshInspectorRuns() {
    setInspectorBusy(true, false);
    try {
        const result = await fetchDataRuns();
        inspectorRuns = Array.isArray(result.runs) ? result.runs : [];
        renderInspectorRunOptions();
        if (inspectorRuns.length) {
            inspectorRunSelect.value = inspectorRunSelect.value || inspectorRuns[0].runId;
        }
        renderInspectorSceneOptions();
        const run = getSelectedInspectorRun();
        if (run && run.scenes.length) {
            inspectorSceneSelect.value = inspectorSceneSelect.value || run.scenes[0].sceneKey;
        }
        renderInspectorTripOptions();
        const scene = getSelectedInspectorScene();
        if (scene && scene.trips.length) {
            inspectorTripSelect.value = inspectorTripSelect.value || scene.trips[0].tripName;
        }
        updateInspectorSourceAvailability();
        await refreshInspectorFields();
        setInspectorBanner("Inspector ready.", "success");
    } finally {
        setInspectorBusy(false, false);
    }
}

async function refreshInspectorFields() {
    const run = getSelectedInspectorRun();
    const scene = getSelectedInspectorScene();
    const trip = getSelectedInspectorTrip();
    resetInspectorSeriesState();
    inspectorAvailableFields = [];
    inspectorSelectedFields = new Set();
    renderInspectorFieldList();
    if (!run || !scene || !trip) {
        return;
    }

    setInspectorBusy(true, false);
    try {
        const result = await fetchTripFields(run.runId, scene.sceneKey, trip.tripName, inspectorSourceSelect.value);
        inspectorAvailableFields = Array.isArray(result.fields) ? result.fields : [];
        inspectorSelectedFields = new Set(chooseDefaultInspectorFields(inspectorAvailableFields));
        inspectorSourceMeta.textContent = `${run.runId} · ${scene.sceneKey} · ${trip.tripName} · ${inspectorSourceSelect.value}`;
        renderInspectorFieldList();
        if (!inspectorAvailableFields.length) {
            setInspectorBanner("No scalar fields were found for this source.", "");
        } else {
            setInspectorBanner(`Loaded ${inspectorAvailableFields.length} fields.`, "success");
        }
    } catch (error) {
        setInspectorBanner(error instanceof Error ? error.message : "failed to load fields", "error");
        throw error;
    } finally {
        setInspectorBusy(false, false);
    }
}

async function loadInspectorSeries() {
    const run = getSelectedInspectorRun();
    const scene = getSelectedInspectorScene();
    const trip = getSelectedInspectorTrip();
    const selectedFields = Array.from(inspectorSelectedFields);
    if (!run || !scene || !trip) {
        setInspectorBanner("Select a run, scene, and trip first.", "error");
        return;
    }
    if (!selectedFields.length) {
        setInspectorBanner("Select at least one field to load.", "error");
        return;
    }

    setInspectorBusy(false, true);
    try {
        const result = await fetchTripSeries(run.runId, scene.sceneKey, trip.tripName, inspectorSourceSelect.value, selectedFields);
        inspectorSeries = result;
        inspectorSourceMeta.textContent = `${run.runId} · ${scene.sceneKey} · ${trip.tripName} · ${inspectorSourceSelect.value}`;
        inspectorRowCount.textContent = `${Number(result.rowCount || 0)} rows · ${selectedFields.length} fields`;
        renderInspectorChart();
        renderInspectorStats();
        renderInspectorTable();
        setInspectorBanner("Trip data loaded.", "success");
    } catch (error) {
        resetInspectorSeriesState();
        setInspectorBanner(error instanceof Error ? error.message : "failed to load trip data", "error");
    } finally {
        setInspectorBusy(false, false);
    }
}

function renderInspectorChart() {
    inspectorChart.innerHTML = "";
    inspectorChartLegend.innerHTML = "";
    if (!inspectorSeries || !Array.isArray(inspectorSeries.rows) || !inspectorSeries.rows.length) {
        inspectorChart.innerHTML = `<div class="empty-state">Load numeric fields to render the normalized trend preview.</div>`;
        return;
    }

    const numericFields = inspectorSeries.fields.filter((field) => field.kind === "number");
    if (!numericFields.length) {
        inspectorChart.innerHTML = `<div class="empty-state">The current selection has no numeric fields to chart.</div>`;
        return;
    }

    const width = 980;
    const height = 240;
    const padding = 18;
    const svgParts = [
        `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Normalized trend preview">`,
        `<rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>`,
    ];

    for (let lineIndex = 0; lineIndex < 4; lineIndex += 1) {
        const y = padding + ((height - (padding * 2)) / 3) * lineIndex;
        svgParts.push(`<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(148,163,184,0.12)" stroke-width="1"></line>`);
    }

    numericFields.forEach((field, index) => {
        const color = INSPECTOR_COLORS[index % INSPECTOR_COLORS.length];
        const values = inspectorSeries.rows
            .map((row) => row[field.name])
            .filter((value) => typeof value === "number" && Number.isFinite(value));
        if (!values.length) {
            return;
        }
        const min = Math.min(...values);
        const max = Math.max(...values);
        const span = Math.max(max - min, Number.EPSILON);
        const points = inspectorSeries.rows.map((row, rowIndex) => {
            const value = row[field.name];
            if (typeof value !== "number" || !Number.isFinite(value)) {
                return null;
            }
            const normalized = span <= Number.EPSILON ? 0.5 : (value - min) / span;
            const x = padding + ((width - (padding * 2)) * rowIndex / Math.max(inspectorSeries.rows.length - 1, 1));
            const y = height - padding - normalized * (height - (padding * 2));
            return `${x.toFixed(2)},${y.toFixed(2)}`;
        }).filter(Boolean);
        if (!points.length) {
            return;
        }
        svgParts.push(`<polyline fill="none" stroke="${color}" stroke-width="2.2" points="${points.join(" ")}"></polyline>`);

        const chip = document.createElement("span");
        chip.className = "legend-chip";
        chip.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>${field.name}`;
        inspectorChartLegend.appendChild(chip);
    });

    svgParts.push("</svg>");
    inspectorChart.innerHTML = svgParts.join("");
}

function renderInspectorStats() {
    inspectorStats.innerHTML = "";
    const rows = inspectorSeries && Array.isArray(inspectorSeries.rows) ? inspectorSeries.rows : [];
    if (!rows.length) {
        inspectorStatsEmpty.hidden = false;
        return;
    }

    inspectorStatsEmpty.hidden = true;
    for (const field of inspectorSeries.fields) {
        const values = rows.map((row) => row[field.name]).filter((value) => value !== undefined && value !== null);
        const card = document.createElement("article");
        card.className = "stat-card";
        const title = document.createElement("strong");
        title.textContent = field.name;
        const meta = document.createElement("div");
        meta.className = "stat-meta";

        if (field.kind === "number") {
            const numeric = values.filter((value) => typeof value === "number" && Number.isFinite(value));
            if (numeric.length) {
                const min = Math.min(...numeric);
                const max = Math.max(...numeric);
                const mean = numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
                meta.textContent = `count ${numeric.length} · min ${formatCellValue(min)} · max ${formatCellValue(max)} · mean ${formatCellValue(mean)}`;
            } else {
                meta.textContent = "No numeric values in current rows.";
            }
        } else if (field.kind === "boolean") {
            const trueCount = values.filter((value) => value === true).length;
            meta.textContent = `count ${values.length} · true ${trueCount} · false ${values.length - trueCount}`;
        } else {
            const unique = new Set(values.map((value) => String(value)));
            meta.textContent = `count ${values.length} · unique ${unique.size}`;
        }

        card.appendChild(title);
        card.appendChild(meta);
        inspectorStats.appendChild(card);
    }
}

function formatCellValue(value) {
    if (value === null || value === undefined) {
        return "";
    }
    if (typeof value === "number") {
        if (!Number.isFinite(value)) {
            return String(value);
        }
        if (Math.abs(value) >= 1000) {
            return value.toFixed(1);
        }
        if (Math.abs(value) >= 10) {
            return value.toFixed(3);
        }
        return value.toFixed(4);
    }
    if (typeof value === "boolean") {
        return value ? "true" : "false";
    }
    return String(value);
}

function formatTimelineLabel(row, index) {
    if (row.time !== undefined) {
        return `#${index} · t=${formatCellValue(row.time)}`;
    }
    if (row.anchor_game_time !== undefined) {
        return `#${index} · gt=${formatCellValue(row.anchor_game_time)}`;
    }
    return `#${index}`;
}

function renderInspectorTable() {
    inspectorTableHead.innerHTML = "";
    inspectorTableBody.innerHTML = "";

    const rows = inspectorSeries && Array.isArray(inspectorSeries.rows) ? inspectorSeries.rows : [];
    const fields = inspectorSeries && Array.isArray(inspectorSeries.fields) ? inspectorSeries.fields : [];
    if (!rows.length || !fields.length) {
        inspectorTableEmpty.hidden = false;
        return;
    }
    inspectorTableEmpty.hidden = true;

    const layout = inspectorLayoutSelect.value;
    if (layout === "rows") {
        renderInspectorTableRows(rows, fields);
        return;
    }
    renderInspectorTableColumns(rows, fields, inspectorSeries.timelineFields || []);
}

function renderInspectorTableColumns(rows, fields, timelineFields) {
    const headRow = document.createElement("tr");
    for (const label of [...timelineFields, ...fields.map((field) => field.name)]) {
        const th = document.createElement("th");
        th.textContent = label;
        headRow.appendChild(th);
    }
    inspectorTableHead.appendChild(headRow);

    for (const row of rows) {
        const tr = document.createElement("tr");
        for (const key of timelineFields) {
            const td = document.createElement("td");
            td.textContent = formatCellValue(row[key]);
            tr.appendChild(td);
        }
        for (const field of fields) {
            const td = document.createElement("td");
            td.textContent = formatCellValue(row[field.name]);
            tr.appendChild(td);
        }
        inspectorTableBody.appendChild(tr);
    }
}

function renderInspectorTableRows(rows, fields) {
    const headRow = document.createElement("tr");
    const headLabel = document.createElement("th");
    headLabel.textContent = "Field";
    headRow.appendChild(headLabel);
    rows.forEach((row, index) => {
        const th = document.createElement("th");
        th.textContent = formatTimelineLabel(row, index);
        headRow.appendChild(th);
    });
    inspectorTableHead.appendChild(headRow);

    for (const field of fields) {
        const tr = document.createElement("tr");
        const labelCell = document.createElement("td");
        labelCell.textContent = field.name;
        tr.appendChild(labelCell);
        rows.forEach((row) => {
            const td = document.createElement("td");
            td.textContent = formatCellValue(row[field.name]);
            tr.appendChild(td);
        });
        inspectorTableBody.appendChild(tr);
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
    modelSteerScaleInput,
    modelAccelScaleInput,
    steerRateInput,
    throttleRateInput,
    brakeRateInput,
]) {
    input.addEventListener("input", () => {
        actuatorTuningDirty = true;
    });
}

for (const tab of pageTabs) {
    tab.addEventListener("click", () => {
        setActivePage(tab.dataset.pageTab);
    });
}

inspectorRunSelect.addEventListener("change", async () => {
    renderInspectorSceneOptions();
    const run = getSelectedInspectorRun();
    if (run && run.scenes.length) {
        inspectorSceneSelect.value = run.scenes[0].sceneKey;
    }
    renderInspectorTripOptions();
    const scene = getSelectedInspectorScene();
    if (scene && scene.trips.length) {
        inspectorTripSelect.value = scene.trips[0].tripName;
    }
    updateInspectorSourceAvailability();
    await refreshInspectorFields();
});

inspectorSceneSelect.addEventListener("change", async () => {
    renderInspectorTripOptions();
    const scene = getSelectedInspectorScene();
    if (scene && scene.trips.length) {
        inspectorTripSelect.value = scene.trips[0].tripName;
    }
    updateInspectorSourceAvailability();
    await refreshInspectorFields();
});

inspectorTripSelect.addEventListener("change", async () => {
    updateInspectorSourceAvailability();
    await refreshInspectorFields();
});

inspectorSourceSelect.addEventListener("change", async () => {
    await refreshInspectorFields();
});

inspectorFieldSearch.addEventListener("input", () => {
    renderInspectorFieldList();
});

inspectorSelectVisibleButton.addEventListener("click", () => {
    const query = inspectorFieldSearch.value.trim().toLowerCase();
    for (const field of inspectorAvailableFields) {
        if (!query || field.name.toLowerCase().includes(query)) {
            inspectorSelectedFields.add(field.name);
        }
    }
    renderInspectorFieldList();
});

inspectorClearFieldsButton.addEventListener("click", () => {
    inspectorSelectedFields.clear();
    renderInspectorFieldList();
});

inspectorLoadButton.addEventListener("click", async () => {
    await loadInspectorSeries();
});

inspectorLayoutSelect.addEventListener("change", () => {
    renderInspectorTable();
});

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
setInspectorBusy(false, false);
setActivePage("control");

Promise.all([
    refresh(true),
    refreshInspectorRuns(),
]).catch((error) => {
    const message = error instanceof Error ? error.message : "failed to initialize page";
    setBanner(message, "error");
    setInspectorBanner(message, "error");
});

setInterval(() => {
    refreshActuator();
}, 100);

setInterval(() => {
    refresh(true).catch((error) => {
        setBanner(error instanceof Error ? error.message : "failed to refresh state", "error");
    });
}, 1500);
