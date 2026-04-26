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
const translationMode = document.getElementById("translation-mode");
const translationCommand = document.getElementById("translation-command");
const translationSubmitStatus = document.getElementById("translation-submit-status");
const translationHeading = document.getElementById("translation-heading");
const translationLongitudinal = document.getElementById("translation-longitudinal");
const translationDriveState = document.getElementById("translation-drive-state");
const translationReasons = document.getElementById("translation-reasons");
const uiErrorRuntime = document.getElementById("ui-error-runtime");
const uiErrorInference = document.getElementById("ui-error-inference");
const uiErrorTranslation = document.getElementById("ui-error-translation");
const uiErrorActuator = document.getElementById("ui-error-actuator");
const uiTranslationConfig = document.getElementById("ui-translation-config");
const actuatorReady = document.getElementById("actuator-ready");
const actuatorLastError = document.getElementById("actuator-last-error");
const actuatorLastCommand = document.getElementById("actuator-last-command");
const actuatorTarget = document.getElementById("actuator-target");
const actuatorApplied = document.getElementById("actuator-applied");
const translationTuningApplyButton = document.getElementById("translation-tuning-apply");
const translationTuningResetButton = document.getElementById("translation-tuning-reset");
const translationTuningSaveButton = document.getElementById("translation-tuning-save");
const translationTuningStatus = document.getElementById("translation-tuning-status");
const translationTuningConfig = document.getElementById("translation-tuning-config");
const translationThrottleGainInput = document.getElementById("translation-tuning-throttle-gain");
const translationBrakeGainInput = document.getElementById("translation-tuning-brake-gain");
const translationBrakeThresholdInput = document.getElementById("translation-tuning-brake-threshold");
const translationOverspeedMarginInput = document.getElementById("translation-tuning-overspeed-margin");
const translationBrakeReleaseMarginInput = document.getElementById("translation-tuning-brake-release-margin");
const translationBrakeEnterHoldInput = document.getElementById("translation-tuning-brake-enter-hold");
const translationThrottleHoldMinInput = document.getElementById("translation-tuning-throttle-hold-min");
const translationThrottleRampUpInput = document.getElementById("translation-tuning-throttle-ramp-up");
const translationThrottleDecayInput = document.getElementById("translation-tuning-throttle-decay");
const translationTargetSpeedGainInput = document.getElementById("translation-tuning-target-speed-gain");
const translationTargetAccelGainInput = document.getElementById("translation-tuning-target-accel-gain");
const translationLaunchThrottleMinInput = document.getElementById("translation-tuning-launch-throttle-min");
const translationSavedThrottleGain = document.getElementById("translation-saved-throttle-gain");
const translationSavedBrakeGain = document.getElementById("translation-saved-brake-gain");
const translationSavedBrakeThreshold = document.getElementById("translation-saved-brake-threshold");
const translationSavedOverspeedMargin = document.getElementById("translation-saved-overspeed-margin");
const translationSavedBrakeReleaseMargin = document.getElementById("translation-saved-brake-release-margin");
const translationSavedBrakeEnterHold = document.getElementById("translation-saved-brake-enter-hold");
const translationSavedThrottleHoldMin = document.getElementById("translation-saved-throttle-hold-min");
const translationSavedThrottleRampUp = document.getElementById("translation-saved-throttle-ramp-up");
const translationSavedThrottleDecay = document.getElementById("translation-saved-throttle-decay");
const translationSavedTargetSpeedGain = document.getElementById("translation-saved-target-speed-gain");
const translationSavedTargetAccelGain = document.getElementById("translation-saved-target-accel-gain");
const translationSavedLaunchThrottleMin = document.getElementById("translation-saved-launch-throttle-min");

const pageTabs = Array.from(document.querySelectorAll("[data-page-tab]"));
const pageControl = document.getElementById("page-control");
const pageInspector = document.getElementById("page-inspector");
const pageTraining = document.getElementById("page-training");
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
const trainingRuntimeMeta = document.getElementById("training-runtime-meta");
const trainingNameInput = document.getElementById("training-name");
const trainingNotesInput = document.getElementById("training-notes");
const trainingEpochsInput = document.getElementById("training-epochs");
const trainingLearningRateInput = document.getElementById("training-learning-rate");
const trainingLossWeights = document.getElementById("training-loss-weights");
const trainingConsistency = document.getElementById("training-consistency");
const trainingYawLossWeightingEnabled = document.getElementById("training-yaw-loss-weighting-enabled");
const trainingYawLossWeighting = document.getElementById("training-yaw-loss-weighting");
const trainingTurnOversamplingEnabled = document.getElementById("training-turn-oversampling-enabled");
const trainingTurnOversampling = document.getElementById("training-turn-oversampling");
const trainingWidthMultiplierInput = document.getElementById("training-width-multiplier");
const trainingTrainRunList = document.getElementById("training-train-run-list");
const trainingValRunList = document.getElementById("training-val-run-list");
const trainingStateInputsContainer = document.getElementById("training-state-inputs");
const trainingQueueJobButton = document.getElementById("training-queue-job");
const trainingResetDefaultsButton = document.getElementById("training-reset-defaults");
const trainingUseLastButton = document.getElementById("training-use-last");
const trainingBanner = document.getElementById("training-banner");
const trainingBatchJson = document.getElementById("training-batch-json");
const trainingLoadYawBatchButton = document.getElementById("training-load-yaw-batch");
const trainingQueueBatchButton = document.getElementById("training-queue-batch");
const trainingQueueList = document.getElementById("training-queue-list");
const trainingQueueEmpty = document.getElementById("training-queue-empty");
const trainingActiveSummary = document.getElementById("training-active-summary");
const trainingActiveEmpty = document.getElementById("training-active-empty");
const trainingActiveLog = document.getElementById("training-active-log");
const trainingStopActiveButton = document.getElementById("training-stop-active");
const trainingRecentList = document.getElementById("training-recent-list");
const trainingRecentEmpty = document.getElementById("training-recent-empty");
const trainingRecentSearch = document.getElementById("training-recent-search");
const trainingRecentStatus = document.getElementById("training-recent-status");
const trainingClearHistoryButton = document.getElementById("training-clear-history");
const trainingSelectedSummary = document.getElementById("training-selected-summary");
const trainingSelectedEmpty = document.getElementById("training-selected-empty");
const trainingSelectedLog = document.getElementById("training-selected-log");
const trainingRequeueSelectedButton = document.getElementById("training-requeue-selected");
const trainingDeleteSelectedButton = document.getElementById("training-delete-selected");

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

const TRANSLATION_TUNING_FIELDS = [
    { key: "steeringGain", input: translationTargetSpeedGainInput, saved: translationSavedTargetSpeedGain },
    { key: "throttleGain", input: translationThrottleGainInput, saved: translationSavedThrottleGain },
    { key: "throttleFloor", input: translationThrottleHoldMinInput, saved: translationSavedThrottleHoldMin },
];

let busy = false;
let selectedSceneName = "";
let selectedModelPath = "";
let lastModelsRefreshAt = 0;
let actuatorRefreshInFlight = false;
const liveValueTimers = new WeakMap();
let translationTuningState = null;
let translationTuningDraft = null;
let translationTuningDirty = false;

let activePage = "control";
let inspectorRuns = [];
let inspectorAvailableFields = [];
let inspectorSelectedFields = new Set();
let inspectorSeries = null;
let inspectorFieldsLoading = false;
let inspectorSeriesLoading = false;
let trainingConfig = null;
let trainingState = null;
let trainingReady = false;
let trainingRefreshInFlight = false;
let trainingSelectedJobId = "";
let trainingSelectedJob = null;
const TRAINING_STATE_INPUT_ORDER = [
    "currentSpeed",
    "routeForwardDelta",
    "routeHeadingError",
    "routeDistance",
    "leadVehicleDistance",
    "hasLeadVehicle",
];

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
        translationTuningApplyButton,
        translationTuningResetButton,
        translationTuningSaveButton,
        trainingQueueJobButton,
        trainingResetDefaultsButton,
        trainingUseLastButton,
        trainingLoadYawBatchButton,
        trainingQueueBatchButton,
        trainingStopActiveButton,
        trainingRequeueSelectedButton,
        trainingClearHistoryButton,
        trainingDeleteSelectedButton,
    ]) {
        if (!button) {
            continue;
        }
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

function setTrainingBanner(message, tone) {
    trainingBanner.textContent = message || "";
    trainingBanner.className = "banner";
    if (tone) {
        trainingBanner.classList.add(tone);
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

async function waitForTelemetry(timeoutMs = 8000) {
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
        const state = await fetchState();
        const telemetry = state && state.telemetry ? state.telemetry : null;
        const runtime = state && state.runtime ? state.runtime : null;
        if (telemetry && runtime && runtime.activeSceneName === "ego-control") {
            return state;
        }
        await new Promise((resolve) => setTimeout(resolve, 250));
    }
    throw new Error("ego control did not start producing telemetry");
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

async function fetchTranslationState() {
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

async function fetchTrainingConfig() {
    const response = await fetch("/training/config", { cache: "no-store" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || "failed to load training config");
    }
    return result;
}

async function fetchTrainingState() {
    const response = await fetch("/training/state", { cache: "no-store" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || "failed to load training state");
    }
    return result;
}

async function fetchTrainingJob(id) {
    const response = await fetch(`/training/jobs/${encodeURIComponent(id)}`, { cache: "no-store" });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || "failed to load training job");
    }
    return result;
}

async function fetchTrainingJobLog(id, tailLines = 0) {
    const params = new URLSearchParams();
    if (tailLines > 0) {
        params.set("tailLines", String(tailLines));
    }
    const suffix = params.toString() ? `?${params.toString()}` : "";
    const response = await fetch(`/training/jobs/${encodeURIComponent(id)}/log${suffix}`, { cache: "no-store" });
    const text = await response.text();
    if (!response.ok) {
        try {
            const parsed = JSON.parse(text);
            throw new Error(parsed.error || "failed to load training log");
        } catch {
            throw new Error(text || "failed to load training log");
        }
    }
    return text;
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

function renderInferenceStatus(controlState, status) {
    const telemetry = controlState && controlState.telemetry ? controlState.telemetry : null;
    const runtime = controlState && controlState.runtime ? controlState.runtime : null;
    const prediction = status && status.lastPrediction ? status.lastPrediction : null;
    const telemetryTimestampMs = telemetry ? (telemetry.receivedAtMs || telemetry.timestampMs) : 0;
    const telemetryAgeMs = telemetryTimestampMs > 0 ? Math.max(0, Date.now() - Number(telemetryTimestampMs)) : NaN;

    setLiveValue(inferenceState, status && status.state ? status.state : "idle");
    setLiveValue(inferenceSteering, formatMetric(telemetry && telemetry.currentSpeed, 3));
    setLiveValue(inferenceFutureYawDelta, formatMetric(telemetry && telemetry.currentYaw, 3));
    setLiveValue(inferenceFutureYaw, formatMetric(telemetry && telemetry.yawRate, 3));
    setLiveValue(inferenceCurrentYaw, formatMetric(telemetry && telemetry.steering, 3));
    setLiveValue(inferenceHeadingError, formatMetric(telemetry && telemetry.acceleration, 3));
    setLiveValue(inferenceFutureSpeed, formatAgeMs(telemetryTimestampMs), {
        stale: !Number.isFinite(telemetryAgeMs) || telemetryAgeMs > 500,
    });
    setLiveValue(inferenceDeltaSpeed, telemetry && telemetry.gameTimeMs ? String(telemetry.gameTimeMs) : "None");
    setLiveValue(inferenceMoveIntent, runtime && runtime.fivemConnected ? "live" : "waiting", {
        stale: !(runtime && runtime.fivemConnected),
    });

    setLiveValue(inferenceSequence, prediction && prediction.sequence !== undefined ? String(prediction.sequence) : "None");
    setLiveValue(inferenceFrameIndex, prediction && prediction.frameIndex !== undefined ? String(prediction.frameIndex) : "None");
    setLiveValue(inferencePredictedAt, prediction && prediction.predictedAt ? formatTimestamp(prediction.predictedAt) : "None");
    setLiveValue(inferenceControlSemantics, prediction ? formatPlannerCommand(prediction.collapsedCommand) : "None");
    setLiveValue(inferenceDebugFrames, prediction ? formatPlannerCommand(prediction.postProcessedCommand) : "None");
    setLiveValue(inferenceWindowIndices, prediction && Array.isArray(prediction.windowFrameIndices) && prediction.windowFrameIndices.length
        ? prediction.windowFrameIndices.join(", ")
        : "None");
    setLiveValue(inferenceFrameHashes, prediction && Array.isArray(prediction.selectedTelemetryTimestampsMs) && prediction.selectedTelemetryTimestampsMs.length
        ? prediction.selectedTelemetryTimestampsMs.join(", ")
        : "None");
    setLiveValue(inferenceDebugDir, prediction ? formatPlannerHorizon(prediction.rawPredControls) : "None");
    setLiveValue(inferenceError, status && status.lastError ? status.lastError : "None", {
        stale: Boolean(status && status.lastError),
    });
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

function pulseLiveValue(element) {
    if (!element) {
        return;
    }
    const existing = liveValueTimers.get(element);
    if (existing) {
        clearTimeout(existing);
    }
    element.classList.add("live-change");
    const timer = setTimeout(() => {
        element.classList.remove("live-change");
        liveValueTimers.delete(element);
    }, 700);
    liveValueTimers.set(element, timer);
}

function setLiveValue(element, value, options = {}) {
    if (!element) {
        return;
    }
    const next = value === null || value === undefined || value === "" ? "None" : String(value);
    const previous = element.dataset.liveValue;
    if (previous !== undefined && previous !== next) {
        pulseLiveValue(element);
    }
    element.dataset.liveValue = next;
    element.textContent = next;
    element.classList.toggle("stale", Boolean(options.stale));
}

function formatMetric(value, digits = 3, suffix = "") {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return "None";
    }
    return `${number.toFixed(digits)}${suffix}`;
}

function formatAgeMs(timestampMs) {
    const numeric = Number(timestampMs);
    if (!Number.isFinite(numeric) || numeric <= 0) {
        return "None";
    }
    return `${Math.max(0, Date.now() - numeric)}ms`;
}

function formatPlannerCommand(control) {
    if (!control || typeof control !== "object") {
        return "None";
    }
    const steering = Number(control.steering ?? control.steer);
    const throttle = Number(control.throttle ?? control.acceleration);
    if (!Number.isFinite(steering) || !Number.isFinite(throttle)) {
        return "None";
    }
    return `steer=${steering.toFixed(3)} throttle=${throttle.toFixed(3)}`;
}

function formatPlannerHorizon(controls) {
    if (!Array.isArray(controls) || controls.length === 0) {
        return "None";
    }
    return controls.map((item, index) => {
        const steering = Number(Array.isArray(item) ? item[0] : NaN);
        const throttle = Number(Array.isArray(item) ? item[1] : NaN);
        if (!Number.isFinite(steering) || !Number.isFinite(throttle)) {
            return null;
        }
        return `t+${index + 1}(${steering.toFixed(2)},${throttle.toFixed(2)})`;
    }).filter(Boolean).join(" | ") || "None";
}

function yesNo(value) {
    return value ? "yes" : "no";
}

function formatProcessorRange(state) {
    const range = state && state.recentRange ? state.recentRange : null;
    if (!range) {
        return "None";
    }
    const minSteering = Number(range.minSteering);
    const maxSteering = Number(range.maxSteering);
    const minThrottle = Number(range.minThrottle);
    const maxThrottle = Number(range.maxThrottle);
    if (![minSteering, maxSteering, minThrottle, maxThrottle].every(Number.isFinite)) {
        return "None";
    }
    return `steer ${minSteering.toFixed(2)}..${maxSteering.toFixed(2)} | throttle ${minThrottle.toFixed(2)}..${maxThrottle.toFixed(2)}`;
}

function formatProcessorCounters(state) {
    const stats = state && state.stats ? state.stats : null;
    if (!stats) {
        return "None";
    }
    return [
        `clamp=${Number(stats.clampedCommands || 0)}`,
        `deadzone=${Number(stats.deadzoneCommands || 0)}`,
        `rate=${Number(stats.rateLimitedCommands || 0)}`,
        `smooth=${Number(stats.smoothedCommands || 0)}`,
        `fallback=${Number(stats.fallbackCommands || 0)}`,
    ].join(" ");
}

function formatControlLine(control, options = {}) {
    if (!control) {
        return "None";
    }
    const parts = [
        `steer=${Number(control.steer || 0).toFixed(3)}`,
        `throttle=${Number(control.throttle || 0).toFixed(3)}`,
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
    setLiveValue(actuatorReady, !supported ? "unsupported" : (ready ? "ready" : "not ready"), {
        stale: !ready,
    });
    setLiveValue(actuatorLastCommand, formatControlLine(state && state.lastCommand ? state.lastCommand : null, {
        includeEnabled: true,
        includeMode: true,
        includeSequence: true,
        includeTimestamp: true,
    }));
    setLiveValue(actuatorTarget, formatControlLine(state && state.target ? state.target : null, {
        includeEnabled: true,
        includeTimestamp: true,
    }));
    setLiveValue(actuatorApplied, formatControlLine(state && state.applied ? state.applied : null, {
        includeEnabled: true,
        includeTimestamp: true,
    }));

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
    setLiveValue(actuatorLastError, controllerStatus.join(" · "), {
        stale: Boolean(state && (state.lastError || state.lastApplyError)),
    });
}

function renderProcessorState(status) {
    const prediction = status && status.lastPrediction ? status.lastPrediction : null;
    const processorDebug = prediction && prediction.processorDebug ? prediction.processorDebug : null;
    const processorState = prediction && prediction.processorState ? prediction.processorState : null;

    setLiveValue(translationMode, prediction ? (prediction.fallbackApplied ? "fallback" : "planner") : "idle");
    setLiveValue(translationCommand, prediction ? formatPlannerCommand(prediction.collapsedCommand) : "None");
    setLiveValue(translationSubmitStatus, prediction ? formatPlannerCommand(prediction.postProcessedCommand) : "None");
    setLiveValue(translationHeading, processorDebug
        ? `clamp=${yesNo(processorDebug.clampApplied)} deadzone=${yesNo(processorDebug.deadzoneApplied)}`
        : "clamp=no deadzone=no");
    setLiveValue(translationLongitudinal, processorDebug
        ? `rate=${yesNo(processorDebug.rateLimitApplied)} smooth=${yesNo(processorDebug.smoothingApplied)}`
        : "rate=no smooth=no");
    setLiveValue(translationDriveState, formatProcessorRange(processorState));
    setLiveValue(translationReasons, formatProcessorCounters(processorState));
}

function formatTuningValue(value) {
    return Number(value || 0).toFixed(3);
}

function parseTuningInputValue(input, fallbackValue) {
    const nextValue = Number(input && input.value);
    if (!Number.isFinite(nextValue)) {
        return Number(fallbackValue || 0);
    }
    return nextValue;
}

function cloneTranslationTuning(tuning) {
    if (!tuning || typeof tuning !== "object") {
        return null;
    }
    return {
        ...tuning,
    };
}

function updateTranslationTuningActions() {
    const hasState = Boolean(translationTuningState);
    const saveSupported = Boolean(translationTuningState && translationTuningState.saveSupported);
    if (translationTuningApplyButton) {
        translationTuningApplyButton.disabled = busy || !translationTuningDirty || !hasState;
    }
    if (translationTuningResetButton) {
        translationTuningResetButton.disabled = busy || !hasState;
    }
    if (translationTuningSaveButton) {
        translationTuningSaveButton.disabled = busy || !hasState || !saveSupported;
    }
}

function renderTranslationTuningState(tuningState) {
    translationTuningState = tuningState || null;
    const live = tuningState && tuningState.live ? tuningState.live : null;
    const saved = tuningState && tuningState.saved ? tuningState.saved : null;

    if (!translationTuningDirty || !translationTuningDraft) {
        translationTuningDraft = cloneTranslationTuning(live);
    }

    const draft = translationTuningDraft || live || {};
    for (const field of TRANSLATION_TUNING_FIELDS) {
        if (field.input && (!translationTuningDirty || document.activeElement !== field.input)) {
            field.input.value = formatTuningValue(draft[field.key]);
        }
        if (field.saved) {
            field.saved.textContent = formatTuningValue(saved ? saved[field.key] : 0);
        }
    }

    if (translationTuningStatus) {
        if (!tuningState) {
            translationTuningStatus.textContent = "Live actuator tuning unavailable.";
        } else if (translationTuningDirty) {
            translationTuningStatus.textContent = "Live edits pending apply.";
        } else if (tuningState.saveSupported) {
            translationTuningStatus.textContent = "Live actuator tuning synced. Save persists config.";
        } else {
            translationTuningStatus.textContent = "Live actuator tuning synced. Config save unavailable.";
        }
    }
    if (translationTuningConfig) {
        translationTuningConfig.textContent = tuningState && tuningState.configPath
            ? `Config file: ${tuningState.configPath}`
            : "Config file: unavailable";
    }
    updateTranslationTuningActions();
}

function buildTranslationTuningPayload() {
    const base = cloneTranslationTuning(translationTuningState && translationTuningState.live);
    if (!base) {
        throw new Error("actuator tuning is unavailable");
    }
    const next = {
        ...base,
    };
    for (const field of TRANSLATION_TUNING_FIELDS) {
        next[field.key] = parseTuningInputValue(field.input, base[field.key]);
    }
    return next;
}


function renderErrorSurface(state, runtime, inference, actuator) {
    uiErrorRuntime.textContent = runtime && runtime.lastError ? String(runtime.lastError) : "None";
    uiErrorInference.textContent = inference && inference.lastError ? String(inference.lastError) : "None";
    uiErrorTranslation.textContent = "Bypassed in planner runtime";
    uiErrorActuator.textContent = actuator && actuator.lastError
        ? String(actuator.lastError)
        : (actuator && actuator.lastApplyError ? String(actuator.lastApplyError) : "None");
    uiTranslationConfig.textContent = state && state.configPath
        ? `Config file: ${state.configPath}`
        : "Config file: unavailable";
}

async function refresh(forceModels = false) {
    const tasks = [fetchState(), fetchInferenceStatus(), fetchActuatorState(), fetchTranslationState()];
    const shouldRefreshModels = forceModels || (Date.now() - lastModelsRefreshAt > 10000);
    if (shouldRefreshModels) {
        tasks.push(fetchInferenceModels());
    }
    const [state, inference, actuator, translationState, modelResult] = await Promise.all(tasks);
    ensureSceneOptions(state.availableScenes);
    if (modelResult) {
        ensureModelOptions(modelResult.models);
        lastModelsRefreshAt = Date.now();
    }
    setConnectionPill(fivemStatus, Boolean(state.runtime && state.runtime.fivemConnected));
    setLiveValue(runtimeStatus, formatRuntimeStatus(state.runtime && state.runtime.status));
    setLiveValue(activeScene, (state.runtime && state.runtime.activeSceneName) || "None");
    setLiveValue(lastCommand, formatCommand(state.lastCommand));
    renderQueue(state.pendingCommands);
    renderInferenceStatus(state, inference);
    renderProcessorState(inference);
    renderTranslationTuningState(translationState);
    renderActuatorState(actuator);
    renderErrorSurface(translationState, state.runtime, inference, actuator);

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
    pageTraining.hidden = pageName !== "training";
    if (pageName === "inspector" && inspectorRuns.length === 0) {
        refreshInspectorRuns().catch((error) => {
            setInspectorBanner(error instanceof Error ? error.message : "failed to load runs", "error");
        });
    }
    if (pageName === "training") {
        ensureTrainingPageReady().catch((error) => {
            setTrainingBanner(error instanceof Error ? error.message : "failed to load training page", "error");
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
        "label.aux.future_yaw_delta",
        "label.control.Steering",
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

const TRAINING_DRAFT_STORAGE_KEY = "fsd-training-last-values-v1";

function formatTrainingNumber(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return "--";
    }
    if (Math.abs(value) >= 1) {
        return value.toFixed(4);
    }
    return value.toFixed(6);
}

function formatTrainingTimestamp(value) {
    if (!value) {
        return "--";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return String(value);
    }
    return date.toLocaleString();
}

function cloneTrainingMap(map) {
    return Object.fromEntries(Object.entries(map || {}).map(([key, value]) => [key, Number(value || 0)]));
}

function cloneTrainingStateInputs(stateInputs) {
    const out = {};
    for (const key of TRAINING_STATE_INPUT_ORDER) {
        const item = stateInputs && stateInputs[key] || {};
        out[key] = {
            enabled: Boolean(item.enabled),
            heads: Array.isArray(item.heads) ? item.heads.map((value) => String(value).trim()).filter(Boolean) : [],
        };
        if (typeof item.cap === "number" || item.cap) {
            out[key].cap = Number(item.cap || 0);
        }
    }
    return out;
}

function buildTrainingDraftFromConfig(config) {
    return {
        name: "",
        notes: "",
        epochs: Number(config.epochs || 0),
        learningRate: Number(config.learningRate || 0),
        widthMultiplier: Number(config.widthMultiplier || 1.5),
        trainRunIds: Array.isArray(config.trainRunIds) ? [...config.trainRunIds] : [],
        valRunIds: Array.isArray(config.valRunIds) ? [...config.valRunIds] : [],
        lossWeights: cloneTrainingMap(config.lossWeights),
        consistency: cloneTrainingMap(config.consistency),
        yawLossWeighting: {
            enabled: Boolean(config.yawLossWeighting && config.yawLossWeighting.enabled),
            values: cloneTrainingMap(config.yawLossWeighting),
        },
        turnOversampling: {
            enabled: Boolean(config.turnOversampling && config.turnOversampling.enabled),
            values: cloneTrainingMap(config.turnOversampling),
        },
        stateInputs: cloneTrainingStateInputs(config.stateInputs),
    };
}

function loadTrainingDraft() {
    try {
        const raw = window.localStorage.getItem(TRAINING_DRAFT_STORAGE_KEY);
        if (!raw) {
            return null;
        }
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") {
            return null;
        }
        return parsed;
    } catch {
        return null;
    }
}

function saveTrainingDraft(spec) {
    try {
        window.localStorage.setItem(TRAINING_DRAFT_STORAGE_KEY, JSON.stringify(spec));
    } catch {
        // ignore local storage failures
    }
}

function mergeTrainingDraft(config, draft) {
    const merged = buildTrainingDraftFromConfig(config);
    if (!draft || typeof draft !== "object") {
        return merged;
    }
    merged.name = typeof draft.name === "string" ? draft.name : "";
    merged.notes = typeof draft.notes === "string" ? draft.notes : "";
    if (typeof draft.epochs === "number" && Number.isFinite(draft.epochs) && draft.epochs >= 1) {
        merged.epochs = Math.trunc(draft.epochs);
    }
    if (typeof draft.learningRate === "number" && Number.isFinite(draft.learningRate) && draft.learningRate > 0) {
        merged.learningRate = draft.learningRate;
    }
    if (typeof draft.widthMultiplier === "number" && Number.isFinite(draft.widthMultiplier) && draft.widthMultiplier > 0) {
        merged.widthMultiplier = draft.widthMultiplier;
    }
    if (Array.isArray(draft.trainRunIds)) {
        merged.trainRunIds = draft.trainRunIds.map((value) => String(value).trim()).filter(Boolean);
    }
    if (Array.isArray(draft.valRunIds)) {
        merged.valRunIds = draft.valRunIds.map((value) => String(value).trim()).filter(Boolean);
    }
    for (const key of config.allowedLossWeightKeys || []) {
        if (draft.lossWeights && typeof draft.lossWeights[key] === "number" && Number.isFinite(draft.lossWeights[key])) {
            merged.lossWeights[key] = draft.lossWeights[key];
        }
    }
    for (const key of config.allowedConsistencyKeys || []) {
        if (draft.consistency && typeof draft.consistency[key] === "number" && Number.isFinite(draft.consistency[key])) {
            merged.consistency[key] = draft.consistency[key];
        }
    }
    if (draft.yawLossWeighting && typeof draft.yawLossWeighting === "object") {
        merged.yawLossWeighting.enabled = Boolean(draft.yawLossWeighting.enabled);
        for (const [key, value] of Object.entries(draft.yawLossWeighting.values || {})) {
            if (Number.isFinite(value)) {
                merged.yawLossWeighting.values[key] = Number(value);
            }
        }
    }
    if (draft.turnOversampling && typeof draft.turnOversampling === "object") {
        merged.turnOversampling.enabled = Boolean(draft.turnOversampling.enabled);
        for (const [key, value] of Object.entries(draft.turnOversampling.values || {})) {
            if (Number.isFinite(value)) {
                merged.turnOversampling.values[key] = Number(value);
            }
        }
    }
    if (draft.stateInputs && typeof draft.stateInputs === "object") {
        for (const sourceKey of TRAINING_STATE_INPUT_ORDER) {
            const source = draft.stateInputs[sourceKey];
            if (!source || typeof source !== "object") {
                continue;
            }
            if (typeof source.enabled === "boolean") {
                merged.stateInputs[sourceKey].enabled = source.enabled;
            }
            if (typeof source.cap === "number" && Number.isFinite(source.cap) && source.cap > 0) {
                merged.stateInputs[sourceKey].cap = source.cap;
            }
            if (Array.isArray(source.heads)) {
                merged.stateInputs[sourceKey].heads = source.heads.map((value) => String(value).trim()).filter(Boolean);
            }
        }
    }
    return merged;
}

function getTrainingInputMap(container) {
    const values = {};
    const inputs = container.querySelectorAll("input[data-training-key]");
    for (const input of inputs) {
        const key = input.dataset.trainingKey;
        const value = Number(input.value);
        if (key && Number.isFinite(value)) {
            values[key] = value;
        }
    }
    return values;
}

function updateTrainingKeyCardState(input, defaultValue) {
    const card = input.closest(".key-card");
    if (!card) {
        return;
    }
    const numericValue = Number(input.value);
    const isOverride = Number.isFinite(numericValue) && Math.abs(numericValue - Number(defaultValue || 0)) > 1e-9;
    card.classList.toggle("override", isOverride);
}

function renderTrainingKeyGrid(container, keys, currentValues, defaultValues, prefix) {
    container.innerHTML = "";
    for (const key of keys) {
        const wrapper = document.createElement("label");
        wrapper.className = "key-card";

        const heading = document.createElement("span");
        heading.className = "status-label";
        heading.textContent = key;

        const input = document.createElement("input");
        input.type = "number";
        input.step = "0.000001";
        input.min = "0";
        input.value = String(currentValues[key] ?? defaultValues[key] ?? 0);
        input.id = `${prefix}-${key}`;
        input.dataset.trainingKey = key;
        input.addEventListener("input", () => {
            updateTrainingKeyCardState(input, defaultValues[key]);
        });

        const note = document.createElement("span");
        note.className = "field-note";
        note.textContent = `Default ${formatTrainingNumber(defaultValues[key] ?? 0)}`;

        wrapper.appendChild(heading);
        wrapper.appendChild(input);
        wrapper.appendChild(note);
        container.appendChild(wrapper);
        updateTrainingKeyCardState(input, defaultValues[key]);
    }
}

function renderTrainingRunCheckboxList(container, selectedRunIds) {
    const selected = new Set((selectedRunIds || []).map((value) => String(value)));
    container.innerHTML = "";
    if (!inspectorRuns.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No runs found.";
        container.appendChild(empty);
        return;
    }
    for (const run of inspectorRuns) {
        const option = document.createElement("label");
        option.className = "training-run-option";

        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = run.runId;
        input.checked = selected.has(run.runId);

        const body = document.createElement("div");
        body.className = "training-run-body";

        const name = document.createElement("span");
        name.className = "training-run-title";
        name.textContent = run.runId;

        const meta = document.createElement("span");
        meta.className = "training-run-meta";
        meta.textContent = `${Number(run.sceneCount || 0)} scenes · ${Number(run.tripCount || 0)} trips`;

        body.appendChild(name);
        body.appendChild(meta);

        const chip = document.createElement("span");
        chip.className = "field-kind";
        chip.textContent = Number(run.reportExists) ? "report" : "runs";

        option.appendChild(input);
        option.appendChild(body);
        option.appendChild(chip);
        container.appendChild(option);
    }
}

function getSelectedTrainingRuns(container) {
    return Array.from(container.querySelectorAll('input[type="checkbox"]:checked'))
        .map((input) => String(input.value || "").trim())
        .filter(Boolean);
}

function buildYawFocusedBatchSpecs() {
    const trainRunIds = getSelectedTrainingRuns(trainingTrainRunList);
    const valRunIds = getSelectedTrainingRuns(trainingValRunList);
    const widthMultiplier = Number(trainingWidthMultiplierInput.value || trainingConfig.widthMultiplier || 1.5);
    const stateInputs = readTrainingStateInputs();

    return [
        {
            name: "yaw-focus-a",
            notes: "High future_yaw_delta weight with strong yaw consistency and oversampling.",
            epochs: Number(trainingEpochsInput.value || trainingConfig.epochs || 15),
            learningRate: 0.001,
            widthMultiplier,
            trainRunIds,
            valRunIds,
            lossWeights: {
                future_yaw_delta: 3.2,
                yaw_rate: 0.8,
                future_speed: 1.2,
                move_intent: 0.7,
                delta_speed: 0.4,
            },
            consistency: {
                yaw_delta_vs_yaw_rate_weight: 2.0,
                yaw_rate_scale_to_degrees: 57.29577951308232,
                future_speed_vs_delta_speed_weight: 0.35,
            },
            yawLossWeighting: {
                enabled: true,
                base_weight: 1.0,
                alpha: 2.4,
                tau: 0.2,
                max_scale: 3.8,
            },
            turnOversampling: {
                enabled: true,
                straight_weight: 0.8,
                light_turn_weight: 1.8,
                medium_turn_weight: 2.8,
                sharp_turn_weight: 3.6,
                light_turn_threshold: 0.05,
                medium_turn_threshold: 0.15,
                sharp_turn_threshold: 0.3,
            },
            stateInputs,
        },
        {
            name: "yaw-focus-b",
            notes: "More aggressive turn weighting, lighter intent, still keeps longitudinal supervision alive.",
            epochs: Number(trainingEpochsInput.value || trainingConfig.epochs || 15),
            learningRate: 0.0008,
            widthMultiplier,
            trainRunIds,
            valRunIds,
            lossWeights: {
                future_yaw_delta: 3.6,
                yaw_rate: 1.0,
                future_speed: 1.0,
                move_intent: 0.6,
                delta_speed: 0.35,
            },
            consistency: {
                yaw_delta_vs_yaw_rate_weight: 2.4,
                yaw_rate_scale_to_degrees: 57.29577951308232,
                future_speed_vs_delta_speed_weight: 0.25,
            },
            yawLossWeighting: {
                enabled: true,
                base_weight: 1.0,
                alpha: 2.8,
                tau: 0.18,
                max_scale: 4.2,
            },
            turnOversampling: {
                enabled: true,
                straight_weight: 0.7,
                light_turn_weight: 1.9,
                medium_turn_weight: 3.0,
                sharp_turn_weight: 4.0,
                light_turn_threshold: 0.05,
                medium_turn_threshold: 0.15,
                sharp_turn_threshold: 0.3,
            },
            stateInputs,
        },
        {
            name: "yaw-focus-c",
            notes: "Slightly lower LR with balanced yaw emphasis for stability if sharp-turn oversampling is noisy.",
            epochs: Number(trainingEpochsInput.value || trainingConfig.epochs || 15),
            learningRate: 0.0006,
            widthMultiplier,
            trainRunIds,
            valRunIds,
            lossWeights: {
                future_yaw_delta: 2.9,
                yaw_rate: 0.9,
                future_speed: 1.3,
                move_intent: 0.75,
                delta_speed: 0.45,
            },
            consistency: {
                yaw_delta_vs_yaw_rate_weight: 1.8,
                yaw_rate_scale_to_degrees: 57.29577951308232,
                future_speed_vs_delta_speed_weight: 0.3,
            },
            yawLossWeighting: {
                enabled: true,
                base_weight: 1.0,
                alpha: 2.0,
                tau: 0.22,
                max_scale: 3.4,
            },
            turnOversampling: {
                enabled: true,
                straight_weight: 0.9,
                light_turn_weight: 1.6,
                medium_turn_weight: 2.6,
                sharp_turn_weight: 3.2,
                light_turn_threshold: 0.05,
                medium_turn_threshold: 0.15,
                sharp_turn_threshold: 0.3,
            },
            stateInputs,
        },
        {
            name: "yaw-focus-d",
            notes: "Hardest yaw-focused sweep: strongest yaw head, strongest yaw consistency, still keeps intent present.",
            epochs: Number(trainingEpochsInput.value || trainingConfig.epochs || 15),
            learningRate: 0.0009,
            widthMultiplier,
            trainRunIds,
            valRunIds,
            lossWeights: {
                future_yaw_delta: 4.0,
                yaw_rate: 1.1,
                future_speed: 0.9,
                move_intent: 0.65,
                delta_speed: 0.3,
            },
            consistency: {
                yaw_delta_vs_yaw_rate_weight: 2.7,
                yaw_rate_scale_to_degrees: 57.29577951308232,
                future_speed_vs_delta_speed_weight: 0.2,
            },
            yawLossWeighting: {
                enabled: true,
                base_weight: 1.0,
                alpha: 3.0,
                tau: 0.16,
                max_scale: 4.5,
            },
            turnOversampling: {
                enabled: true,
                straight_weight: 0.65,
                light_turn_weight: 2.0,
                medium_turn_weight: 3.1,
                sharp_turn_weight: 4.3,
                light_turn_threshold: 0.05,
                medium_turn_threshold: 0.15,
                sharp_turn_threshold: 0.3,
            },
            stateInputs,
        },
    ];
}

function humanizeTrainingStateInputKey(key) {
    return String(key || "")
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        .replace(/^\w/, (match) => match.toUpperCase());
}

function renderTrainingStateInputs(stateInputs, allowedHeads) {
    trainingStateInputsContainer.innerHTML = "";
    for (const key of TRAINING_STATE_INPUT_ORDER) {
        const item = stateInputs && stateInputs[key] || {};
        const card = document.createElement("div");
        card.className = "field";
        card.dataset.stateInputKey = key;

        const title = document.createElement("span");
        title.className = "status-label";
        title.textContent = humanizeTrainingStateInputKey(key);
        card.appendChild(title);

        const enabledLabel = document.createElement("label");
        enabledLabel.className = "status-label";
        const enabledInput = document.createElement("input");
        enabledInput.type = "checkbox";
        enabledInput.dataset.role = "enabled";
        enabledInput.checked = Boolean(item.enabled);
        enabledLabel.appendChild(enabledInput);
        enabledLabel.appendChild(document.createTextNode(" Enabled"));
        card.appendChild(enabledLabel);

        if (typeof item.cap === "number") {
            const capInput = document.createElement("input");
            capInput.type = "number";
            capInput.step = "0.001";
            capInput.min = "0";
            capInput.dataset.role = "cap";
            capInput.value = String(item.cap || 0);
            capInput.placeholder = "Cap";
            card.appendChild(capInput);
        }

        const headsSelect = document.createElement("select");
        headsSelect.multiple = true;
        headsSelect.dataset.role = "heads";
        for (const head of allowedHeads || []) {
            const option = document.createElement("option");
            option.value = head;
            option.textContent = head;
            option.selected = Array.isArray(item.heads) && item.heads.includes(head);
            headsSelect.appendChild(option);
        }
        card.appendChild(headsSelect);
        trainingStateInputsContainer.appendChild(card);
    }
}

function readTrainingStateInputs() {
    const stateInputs = {};
    for (const card of trainingStateInputsContainer.querySelectorAll("[data-state-input-key]")) {
        const key = card.dataset.stateInputKey;
        if (!key) {
            continue;
        }
        const enabledInput = card.querySelector('[data-role="enabled"]');
        const capInput = card.querySelector('[data-role="cap"]');
        const headsSelect = card.querySelector('[data-role="heads"]');
        const item = {
            enabled: Boolean(enabledInput && enabledInput.checked),
            heads: Array.from(headsSelect && headsSelect.selectedOptions || []).map((option) => option.value),
        };
        if (capInput) {
            const cap = Number(capInput.value);
            if (!Number.isFinite(cap) || cap <= 0) {
                throw new Error(`${humanizeTrainingStateInputKey(key)} cap must be a positive number`);
            }
            item.cap = cap;
        }
        stateInputs[key] = item;
    }
    return stateInputs;
}

function applyTrainingDraftToForm(draft) {
    if (!trainingConfig) {
        return;
    }
    const resolved = mergeTrainingDraft(trainingConfig, draft);
    trainingNameInput.value = resolved.name || "";
    trainingNotesInput.value = resolved.notes || "";
    trainingEpochsInput.value = String(resolved.epochs || trainingConfig.epochs || 0);
    trainingLearningRateInput.value = String(resolved.learningRate || trainingConfig.learningRate || 0);
    trainingWidthMultiplierInput.value = String(resolved.widthMultiplier || trainingConfig.widthMultiplier || 1.5);
    renderTrainingKeyGrid(
        trainingLossWeights,
        trainingConfig.allowedLossWeightKeys || [],
        resolved.lossWeights || {},
        trainingConfig.lossWeights || {},
        "training-loss",
    );
    renderTrainingKeyGrid(
        trainingConsistency,
        trainingConfig.allowedConsistencyKeys || [],
        resolved.consistency || {},
        trainingConfig.consistency || {},
        "training-consistency",
    );
    trainingYawLossWeightingEnabled.checked = Boolean(resolved.yawLossWeighting && resolved.yawLossWeighting.enabled);
    renderTrainingKeyGrid(
        trainingYawLossWeighting,
        ["base_weight", "alpha", "tau", "max_scale"],
        resolved.yawLossWeighting && resolved.yawLossWeighting.values || {},
        trainingConfig.yawLossWeighting || {},
        "training-yaw-loss-weighting",
    );
    trainingTurnOversamplingEnabled.checked = Boolean(resolved.turnOversampling && resolved.turnOversampling.enabled);
    renderTrainingKeyGrid(
        trainingTurnOversampling,
        [
            "straight_weight",
            "light_turn_weight",
            "medium_turn_weight",
            "sharp_turn_weight",
            "light_turn_threshold",
            "medium_turn_threshold",
            "sharp_turn_threshold",
        ],
        resolved.turnOversampling && resolved.turnOversampling.values || {},
        trainingConfig.turnOversampling || {},
        "training-turn-oversampling",
    );
    renderTrainingRunCheckboxList(trainingTrainRunList, resolved.trainRunIds || []);
    renderTrainingRunCheckboxList(trainingValRunList, resolved.valRunIds || []);
    renderTrainingStateInputs(resolved.stateInputs, trainingConfig.allowedStateInputHeads || []);
}

function renderTrainingRuntimeMeta(config) {
    trainingRuntimeMeta.innerHTML = "";
    const items = [
        { label: "Base Config", value: config.configPath || "--" },
        { label: "Python", value: config.pythonBin || "--" },
        { label: "Train Script", value: config.trainScript || "--" },
        { label: "Jobs Directory", value: config.jobsDir || "--" },
        { label: "History Limit", value: String(config.historyLimit || "--") },
    ];
    for (const item of items) {
        const card = document.createElement("div");
        card.className = "runtime-meta-item";
        const title = document.createElement("strong");
        title.textContent = item.label;
        const value = document.createElement("span");
        value.textContent = item.value;
        card.appendChild(title);
        card.appendChild(value);
        trainingRuntimeMeta.appendChild(card);
    }
}

function readTrainingFormSpec() {
    const epochs = Number(trainingEpochsInput.value);
    if (!Number.isFinite(epochs) || epochs < 1 || !Number.isInteger(epochs)) {
        throw new Error("epochs must be an integer >= 1");
    }
    const learningRate = Number(trainingLearningRateInput.value);
    if (!Number.isFinite(learningRate) || learningRate <= 0) {
        throw new Error("learning rate must be a positive number");
    }
    const widthMultiplier = Number(trainingWidthMultiplierInput.value);
    if (!Number.isFinite(widthMultiplier) || widthMultiplier <= 0) {
        throw new Error("width multiplier must be a positive number");
    }
    const turnOversampling = normalizeTurnOversamplingValues({
        enabled: trainingTurnOversamplingEnabled.checked,
        ...getTrainingInputMap(trainingTurnOversampling),
    });
    return {
        name: trainingNameInput.value.trim(),
        notes: trainingNotesInput.value.trim(),
        epochs,
        learningRate,
        widthMultiplier,
        trainRunIds: getSelectedTrainingRuns(trainingTrainRunList),
        valRunIds: getSelectedTrainingRuns(trainingValRunList),
        lossWeights: getTrainingInputMap(trainingLossWeights),
        consistency: getTrainingInputMap(trainingConsistency),
        yawLossWeighting: {
            enabled: trainingYawLossWeightingEnabled.checked,
            ...getTrainingInputMap(trainingYawLossWeighting),
        },
        turnOversampling,
        stateInputs: readTrainingStateInputs(),
    };
}

function normalizeTurnOversamplingValues(values) {
    if (!values || typeof values !== "object") {
        return values;
    }
    const normalized = { ...values };
    const light = Number(normalized.light_turn_threshold);
    const medium = Number(normalized.medium_turn_threshold);
    const sharp = Number(normalized.sharp_turn_threshold);
    if (Number.isFinite(light) && Number.isFinite(medium) && medium < light) {
        normalized.medium_turn_threshold = light;
    }
    const nextMedium = Number(normalized.medium_turn_threshold);
    if (Number.isFinite(nextMedium) && Number.isFinite(sharp) && sharp < nextMedium) {
        normalized.sharp_turn_threshold = nextMedium;
    }
    return normalized;
}

function summarizeTrainingOverrides(job) {
    if (!trainingConfig || !job) {
        return "No overrides";
    }
    const overrides = [];
    if (typeof job.epochs === "number" && Math.trunc(job.epochs) !== Math.trunc(Number(trainingConfig.epochs || 0))) {
        overrides.push(`epochs ${job.epochs}`);
    }
    if (typeof job.learningRate === "number" && Math.abs(job.learningRate - Number(trainingConfig.learningRate || 0)) > 1e-9) {
        overrides.push(`lr ${formatTrainingNumber(job.learningRate)}`);
    }
    if (typeof job.widthMultiplier === "number" && Math.abs(job.widthMultiplier - Number(trainingConfig.widthMultiplier || 1.5)) > 1e-9) {
        overrides.push(`width ${formatTrainingNumber(job.widthMultiplier)}`);
    }
    const trainRunIds = Array.isArray(job.trainRunIds) ? job.trainRunIds : [];
    const valRunIds = Array.isArray(job.valRunIds) ? job.valRunIds : [];
    if (JSON.stringify(trainRunIds) !== JSON.stringify(trainingConfig.trainRunIds || [])) {
        overrides.push(`train runs ${trainRunIds.length}`);
    }
    if (JSON.stringify(valRunIds) !== JSON.stringify(trainingConfig.valRunIds || [])) {
        overrides.push(`val runs ${valRunIds.length}`);
    }
    for (const key of trainingConfig.allowedLossWeightKeys || []) {
        const value = job.lossWeights && typeof job.lossWeights[key] === "number" ? job.lossWeights[key] : undefined;
        if (typeof value === "number" && Math.abs(value - Number(trainingConfig.lossWeights[key] || 0)) > 1e-9) {
            overrides.push(`${key} ${formatTrainingNumber(value)}`);
        }
    }
    for (const key of trainingConfig.allowedConsistencyKeys || []) {
        const value = job.consistency && typeof job.consistency[key] === "number" ? job.consistency[key] : undefined;
        if (typeof value === "number" && Math.abs(value - Number(trainingConfig.consistency[key] || 0)) > 1e-9) {
            overrides.push(`${key} ${formatTrainingNumber(value)}`);
        }
    }
    if (job.turnOversampling && typeof job.turnOversampling.enabled === "boolean" && job.turnOversampling.enabled !== Boolean(trainingConfig.turnOversampling && trainingConfig.turnOversampling.enabled)) {
        overrides.push(`turn sampler ${job.turnOversampling.enabled ? "on" : "off"}`);
    }
    if (job.yawLossWeighting && typeof job.yawLossWeighting.enabled === "boolean" && job.yawLossWeighting.enabled !== Boolean(trainingConfig.yawLossWeighting && trainingConfig.yawLossWeighting.enabled)) {
        overrides.push(`yaw weighting ${job.yawLossWeighting.enabled ? "on" : "off"}`);
    }
    for (const key of TRAINING_STATE_INPUT_ORDER) {
        const jobItem = job.stateInputs && job.stateInputs[key] ? job.stateInputs[key] : null;
        const baseItem = trainingConfig.stateInputs && trainingConfig.stateInputs[key] ? trainingConfig.stateInputs[key] : {};
        if (!jobItem) {
            continue;
        }
        const jobHeads = JSON.stringify(jobItem.heads || []);
        const baseHeads = JSON.stringify(baseItem.heads || []);
        if (Boolean(jobItem.enabled) !== Boolean(baseItem.enabled) || jobHeads !== baseHeads) {
            overrides.push(`${humanizeTrainingStateInputKey(key)} ${jobItem.enabled ? (jobItem.heads || []).join("+") || "on" : "off"}`);
        }
    }
    return overrides.length ? overrides.join(" · ") : "Using base defaults";
}

function isTrainingJobTerminal(job) {
    return ["completed", "failed", "canceled", "stopped"].includes(String(job && job.status || ""));
}

function canRequeueTrainingJob(job) {
    return ["failed", "stopped"].includes(String(job && job.status || ""));
}

function trainingStatusTone(status) {
    if (status === "completed" || status === "running") {
        return "ok";
    }
    if (status === "failed" || status === "canceled" || status === "stopped") {
        return "bad";
    }
    return "";
}

function renderTrainingStatusPill(status) {
    const tone = trainingStatusTone(status);
    return `<span class="pill ${tone}">${status || "unknown"}</span>`;
}

function createTrainingJobRow(job, options = {}) {
    const item = document.createElement("li");
    item.className = "job-row";

    const header = document.createElement("div");
    header.className = "job-row-header";

    const titleBlock = document.createElement("div");
    const title = document.createElement("div");
    title.className = "job-title";
    title.textContent = job.name || job.id;
    const meta = document.createElement("div");
    meta.className = "job-meta";
    meta.textContent = `${job.id} · created ${formatTrainingTimestamp(job.createdAt)}`;
    titleBlock.appendChild(title);
    titleBlock.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "job-row-actions";
    actions.innerHTML = renderTrainingStatusPill(job.status);

    if (options.showCancel) {
        const cancelButton = document.createElement("button");
        cancelButton.className = "ghost";
        cancelButton.type = "button";
        cancelButton.textContent = "Cancel";
        cancelButton.addEventListener("click", async () => {
            try {
                setTrainingBanner("Canceling queued job...", "");
                await postJSON(`/training/jobs/${encodeURIComponent(job.id)}/cancel`, {}, "failed to cancel training job");
                await refreshTrainingStateOnly();
                setTrainingBanner("Queued training job canceled.", "success");
            } catch (error) {
                setTrainingBanner(error instanceof Error ? error.message : "failed to cancel training job", "error");
            }
        });
        actions.appendChild(cancelButton);
    }

    if (options.showDelete && isTrainingJobTerminal(job)) {
        const deleteButton = document.createElement("button");
        deleteButton.className = "ghost";
        deleteButton.type = "button";
        deleteButton.textContent = "Delete";
        deleteButton.addEventListener("click", async () => {
            await deleteTrainingJob(job.id);
        });
        actions.appendChild(deleteButton);
    }

    if (options.showRequeue && canRequeueTrainingJob(job)) {
        const requeueButton = document.createElement("button");
        requeueButton.className = "ghost";
        requeueButton.type = "button";
        requeueButton.textContent = "Requeue";
        requeueButton.addEventListener("click", async () => {
            await requeueTrainingJob(job);
        });
        actions.appendChild(requeueButton);
    }

    const inspectButton = document.createElement("button");
    inspectButton.className = "ghost";
    inspectButton.type = "button";
    inspectButton.textContent = "Inspect";
    inspectButton.addEventListener("click", async () => {
        await loadSelectedTrainingJob(job.id);
    });
    actions.appendChild(inspectButton);

    header.appendChild(titleBlock);
    header.appendChild(actions);

    const summary = document.createElement("div");
    summary.className = "job-summary";
    summary.textContent = summarizeTrainingOverrides(job);

    item.appendChild(header);
    item.appendChild(summary);
    return item;
}

async function deleteTrainingJob(jobId) {
    try {
        setTrainingBanner("Deleting training history item...", "");
        await postJSON(`/training/jobs/${encodeURIComponent(jobId)}/delete`, {}, "failed to delete training history item");
        if (trainingSelectedJobId === jobId) {
            trainingSelectedJobId = "";
            trainingSelectedJob = null;
            updateScrollableLog(trainingSelectedLog, "No job selected.");
        }
        await refreshTrainingStateOnly();
        renderSelectedTrainingJob();
        setTrainingBanner("Training history item deleted.", "success");
    } catch (error) {
        setTrainingBanner(error instanceof Error ? error.message : "failed to delete training history item", "error");
    }
}

async function requeueTrainingJob(job) {
    if (!job || !job.id) {
        return;
    }
    try {
        setBusy(true);
        setTrainingBanner("Requeueing training job...", "");
        const result = await postJSON(`/training/jobs/${encodeURIComponent(job.id)}/requeue`, {}, "failed to requeue training job");
        await refreshTrainingStateOnly();
        const queuedJob = result && result.job ? result.job : null;
        setTrainingBanner(
            queuedJob && queuedJob.id
                ? `Training job requeued as ${queuedJob.id}.`
                : "Training job requeued.",
            "success",
        );
    } catch (error) {
        setTrainingBanner(error instanceof Error ? error.message : "failed to requeue training job", "error");
    } finally {
        setBusy(false);
    }
}

function filteredTrainingRecentJobs() {
    const recentJobs = trainingState && Array.isArray(trainingState.recentJobs) ? trainingState.recentJobs : [];
    const search = trainingRecentSearch.value.trim().toLowerCase();
    const statusFilter = trainingRecentStatus.value;
    return recentJobs.filter((job) => {
        if (statusFilter && statusFilter !== "all" && job.status !== statusFilter) {
            return false;
        }
        if (!search) {
            return true;
        }
        const haystack = [
            job.id,
            job.name,
            job.error,
            job.runDir,
            job.runMetricsPath,
        ]
            .filter(Boolean)
            .join(" ")
            .toLowerCase();
        return haystack.includes(search);
    });
}

function renderTrainingQueue() {
    trainingQueueList.innerHTML = "";
    const queuedJobs = trainingState && Array.isArray(trainingState.queuedJobs) ? trainingState.queuedJobs : [];
    trainingQueueEmpty.hidden = queuedJobs.length > 0;
    for (const job of queuedJobs) {
        trainingQueueList.appendChild(createTrainingJobRow(job, { showCancel: true }));
    }
}

function renderTrainingDetailGrid(container, job) {
    container.innerHTML = "";
    if (!job) {
        return;
    }
    const fields = [
        ["Status", job.status || "--"],
        ["Name", job.name || job.id || "--"],
        ["Created", formatTrainingTimestamp(job.createdAt)],
        ["Started", formatTrainingTimestamp(job.startedAt)],
        ["Finished", formatTrainingTimestamp(job.finishedAt)],
        ["Run Dir", job.runDir || "--"],
        ["Run Metrics", job.runMetricsPath || "--"],
        ["Config Path", job.configPath || "--"],
        ["Log Path", job.logPath || "--"],
        ["Overrides", summarizeTrainingOverrides(job)],
        ["Command", Array.isArray(job.command) ? job.command.join(" ") : "--"],
        ["Error", job.error || "--"],
    ];
    for (const [label, value] of fields) {
        const card = document.createElement("div");
        card.className = "job-detail-item";
        const heading = document.createElement("strong");
        heading.textContent = label;
        const text = document.createElement("span");
        text.textContent = value;
        card.appendChild(heading);
        card.appendChild(text);
        container.appendChild(card);
    }
}

function updateScrollableLog(preElement, nextText) {
    const shouldStick = preElement.scrollHeight - preElement.scrollTop - preElement.clientHeight < 24;
    preElement.textContent = nextText || "";
    if (shouldStick) {
        preElement.scrollTop = preElement.scrollHeight;
    }
}

async function refreshActiveTrainingLog() {
    const activeJob = trainingState && trainingState.activeJob ? trainingState.activeJob : null;
    if (!activeJob) {
        updateScrollableLog(trainingActiveLog, "No active log.");
        return;
    }
    try {
        const log = await fetchTrainingJobLog(activeJob.id, 200);
        updateScrollableLog(trainingActiveLog, log || "No active log output yet.");
    } catch (error) {
        updateScrollableLog(trainingActiveLog, error instanceof Error ? error.message : "failed to load active job log");
    }
}

function renderTrainingActive() {
    const activeJob = trainingState && trainingState.activeJob ? trainingState.activeJob : null;
    trainingStopActiveButton.disabled = !activeJob || busy;
    trainingActiveEmpty.hidden = Boolean(activeJob);
    renderTrainingDetailGrid(trainingActiveSummary, activeJob);
    if (!activeJob) {
        updateScrollableLog(trainingActiveLog, "No active log.");
    }
}

function renderTrainingRecent() {
    trainingRecentList.innerHTML = "";
    const recentJobs = filteredTrainingRecentJobs();
    trainingRecentEmpty.hidden = recentJobs.length > 0;
    for (const job of recentJobs) {
        trainingRecentList.appendChild(createTrainingJobRow(job, { showDelete: true, showRequeue: true }));
    }
}

function findTrainingJobInState(jobId) {
    if (!trainingState || !jobId) {
        return null;
    }
    const matches = [];
    if (trainingState.activeJob) {
        matches.push(trainingState.activeJob);
    }
    if (Array.isArray(trainingState.queuedJobs)) {
        matches.push(...trainingState.queuedJobs);
    }
    if (Array.isArray(trainingState.recentJobs)) {
        matches.push(...trainingState.recentJobs);
    }
    return matches.find((job) => job.id === jobId) || null;
}

function renderSelectedTrainingJob() {
    const fallback = findTrainingJobInState(trainingSelectedJobId);
    const job = trainingSelectedJob || fallback;
    trainingSelectedEmpty.hidden = Boolean(job);
    trainingRequeueSelectedButton.disabled = !job || !canRequeueTrainingJob(job) || busy;
    trainingDeleteSelectedButton.disabled = !job || !isTrainingJobTerminal(job) || busy;
    renderTrainingDetailGrid(trainingSelectedSummary, job);
    if (!job && !trainingSelectedLog.textContent) {
        updateScrollableLog(trainingSelectedLog, "No job selected.");
    }
}

async function loadSelectedTrainingJob(jobId) {
    trainingSelectedJobId = jobId;
    trainingSelectedJob = null;
    renderSelectedTrainingJob();
    updateScrollableLog(trainingSelectedLog, "Loading selected job log...");
    try {
        const [job, log] = await Promise.all([
            fetchTrainingJob(jobId),
            fetchTrainingJobLog(jobId),
        ]);
        trainingSelectedJob = job;
        renderSelectedTrainingJob();
        updateScrollableLog(trainingSelectedLog, log || "No log output captured.");
    } catch (error) {
        updateScrollableLog(trainingSelectedLog, error instanceof Error ? error.message : "failed to load selected job");
        setTrainingBanner(error instanceof Error ? error.message : "failed to load selected job", "error");
    }
}

async function refreshTrainingStateOnly() {
    if (trainingRefreshInFlight) {
        return;
    }
    trainingRefreshInFlight = true;
    try {
        trainingState = await fetchTrainingState();
        renderTrainingQueue();
        renderTrainingActive();
        renderTrainingRecent();
        if (trainingSelectedJobId) {
            const stateJob = findTrainingJobInState(trainingSelectedJobId);
            if (stateJob) {
                trainingSelectedJob = {
                    ...(trainingSelectedJob || {}),
                    ...stateJob,
                };
            }
            renderSelectedTrainingJob();
        }
        await refreshActiveTrainingLog();
    } finally {
        trainingRefreshInFlight = false;
    }
}

async function ensureTrainingPageReady(forceConfig = false) {
    if (!forceConfig && trainingReady && trainingConfig) {
        await refreshTrainingStateOnly();
        return;
    }
    const [config, state, runsResult] = await Promise.all([
        fetchTrainingConfig(),
        fetchTrainingState(),
        fetchDataRuns(),
    ]);
    inspectorRuns = Array.isArray(runsResult.runs) ? runsResult.runs : [];
    trainingConfig = config;
    trainingState = state;
    renderTrainingRuntimeMeta(config);
    applyTrainingDraftToForm(mergeTrainingDraft(config, loadTrainingDraft()));
    renderTrainingQueue();
    renderTrainingActive();
    renderTrainingRecent();
    renderSelectedTrainingJob();
    await refreshActiveTrainingLog();
    trainingReady = true;
}

sceneSelect.addEventListener("change", () => {
    selectedSceneName = sceneSelect.value;
});

modelSelect.addEventListener("change", () => {
    selectedModelPath = modelSelect.value;
});

for (const field of TRANSLATION_TUNING_FIELDS) {
    const input = field.input;
    if (!input) {
        continue;
    }
    input.addEventListener("input", () => {
        translationTuningDirty = true;
        if (translationTuningState && translationTuningState.live) {
            const base = translationTuningDraft || cloneTranslationTuning(translationTuningState.live);
            translationTuningDraft = {
                ...base,
                [field.key]: parseTuningInputValue(input, base[field.key]),
            };
        }
        if (translationTuningStatus) {
            translationTuningStatus.textContent = "Live edits pending apply.";
        }
        updateTranslationTuningActions();
    });
}

translationTuningApplyButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setBanner("Applying live actuator tuning...", "");
        const tuning = await postJSON("/actuator/tuning/apply", buildTranslationTuningPayload(), "failed to apply actuator tuning");
        translationTuningDirty = false;
        translationTuningDraft = cloneTranslationTuning(tuning.live);
        renderTranslationTuningState(tuning);
        setBanner("Live actuator tuning applied.", "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to apply actuator tuning", "error");
    } finally {
        setBusy(false);
    }
});

translationTuningResetButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setBanner("Resetting live actuator tuning...", "");
        const tuning = await postJSON("/actuator/tuning/reset", {}, "failed to reset actuator tuning");
        translationTuningDirty = false;
        translationTuningDraft = cloneTranslationTuning(tuning.live);
        renderTranslationTuningState(tuning);
        setBanner("Live actuator tuning reset.", "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to reset actuator tuning", "error");
    } finally {
        setBusy(false);
    }
});

translationTuningSaveButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setBanner("Saving actuator tuning...", "");
        const tuning = await postJSON("/actuator/tuning/save", {}, "failed to save actuator tuning");
        translationTuningDirty = false;
        translationTuningDraft = cloneTranslationTuning(tuning.live);
        renderTranslationTuningState(tuning);
        setBanner("Actuator tuning saved.", "success");
        await refresh();
    } catch (error) {
        setBanner(error instanceof Error ? error.message : "failed to save actuator tuning", "error");
    } finally {
        setBusy(false);
    }
});

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

trainingQueueJobButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setTrainingBanner("Queueing training job...", "");
        const spec = readTrainingFormSpec();
        await postJSON("/training/jobs", spec, "failed to queue training job");
        saveTrainingDraft(spec);
        await ensureTrainingPageReady(true);
        setTrainingBanner("Training job queued.", "success");
    } catch (error) {
        setTrainingBanner(error instanceof Error ? error.message : "failed to queue training job", "error");
    } finally {
        setBusy(false);
    }
});

trainingResetDefaultsButton.addEventListener("click", () => {
    if (!trainingConfig) {
        return;
    }
    applyTrainingDraftToForm(buildTrainingDraftFromConfig(trainingConfig));
    setTrainingBanner("Training form reset to backend defaults.", "success");
});

trainingUseLastButton.addEventListener("click", () => {
    if (!trainingConfig) {
        return;
    }
    applyTrainingDraftToForm(mergeTrainingDraft(trainingConfig, loadTrainingDraft()));
    setTrainingBanner("Reapplied the last queued values from this browser.", "success");
});

trainingQueueBatchButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        const body = trainingBatchJson.value.trim();
        if (!body) {
            throw new Error("paste one job object or an array of job objects first");
        }
        setTrainingBanner("Queueing training batch...", "");
        const response = await fetch("/training/jobs", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body,
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(result.error || "failed to queue training batch");
        }
        await ensureTrainingPageReady();
        setTrainingBanner(`Queued ${Array.isArray(result.jobs) ? result.jobs.length : 0} training jobs.`, "success");
    } catch (error) {
        setTrainingBanner(error instanceof Error ? error.message : "failed to queue training batch", "error");
    } finally {
        setBusy(false);
    }
});

trainingLoadYawBatchButton.addEventListener("click", () => {
    const specs = buildYawFocusedBatchSpecs();
    trainingBatchJson.value = JSON.stringify(specs, null, 2);
    setTrainingBanner("Loaded 4 yaw-focused batch samples into the JSON box.", "success");
});

trainingStopActiveButton.addEventListener("click", async () => {
    const activeJob = trainingState && trainingState.activeJob ? trainingState.activeJob : null;
    if (!activeJob) {
        return;
    }
    try {
        setBusy(true);
        setTrainingBanner("Stopping active training job...", "");
        await postJSON(`/training/jobs/${encodeURIComponent(activeJob.id)}/stop`, {}, "failed to stop active training job");
        await refreshTrainingStateOnly();
        setTrainingBanner("Active training job stop requested.", "success");
    } catch (error) {
        setTrainingBanner(error instanceof Error ? error.message : "failed to stop active training job", "error");
    } finally {
        setBusy(false);
    }
});

trainingRequeueSelectedButton.addEventListener("click", async () => {
    const fallback = findTrainingJobInState(trainingSelectedJobId);
    const job = trainingSelectedJob || fallback;
    if (!job || !canRequeueTrainingJob(job)) {
        return;
    }
    await requeueTrainingJob(job);
});

trainingRecentSearch.addEventListener("input", () => {
    renderTrainingRecent();
});

trainingRecentStatus.addEventListener("change", () => {
    renderTrainingRecent();
});

trainingClearHistoryButton.addEventListener("click", async () => {
    try {
        setBusy(true);
        setTrainingBanner("Clearing terminal training history...", "");
        const result = await postJSON("/training/history/clear", {}, "failed to clear training history");
        const deletedJobs = Array.isArray(result.jobs) ? result.jobs : [];
        if (trainingSelectedJobId && deletedJobs.some((job) => String(job && job.id || "") === trainingSelectedJobId)) {
            trainingSelectedJobId = "";
            trainingSelectedJob = null;
            trainingSelectedLog = "";
        }
        await refreshTrainingStateOnly();
        const deletedCount = Number.isFinite(Number(result.deletedCount)) ? Number(result.deletedCount) : deletedJobs.length;
        setTrainingBanner(`Cleared ${deletedCount} terminal training jobs.`, "success");
    } catch (error) {
        setTrainingBanner(error instanceof Error ? error.message : "failed to clear training history", "error");
    } finally {
        setBusy(false);
    }
});

trainingDeleteSelectedButton.addEventListener("click", async () => {
    if (!trainingSelectedJobId) {
        return;
    }
    await deleteTrainingJob(trainingSelectedJobId);
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
        const state = await fetchState();
        const telemetry = state && state.telemetry ? state.telemetry : null;
        const egoActive = Boolean(state && state.runtime && state.runtime.activeSceneName === "ego-control");
        if (!egoActive || !telemetry) {
            setBanner("Starting ego control for telemetry...", "");
            await sendCommand("startEgo");
            await waitForTelemetry();
        }
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
    if (activePage !== "control" || document.hidden) {
        return;
    }
    refreshActuator();
}, 100);

setInterval(() => {
    if (activePage !== "control" || document.hidden) {
        return;
    }
    refresh(true).catch((error) => {
        setBanner(error instanceof Error ? error.message : "failed to refresh state", "error");
    });
}, 750);

setInterval(() => {
    if (activePage !== "training" || document.hidden) {
        return;
    }
    refreshTrainingStateOnly().catch((error) => {
        setTrainingBanner(error instanceof Error ? error.message : "failed to refresh training state", "error");
    });
}, 1500);
