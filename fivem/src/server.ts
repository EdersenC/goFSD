//srv/server.ts

import fetch from "node-fetch";
import {log} from "./helper";
import {defaultScene, defaultSceneId, getLocalScene} from "./datasets";
import {normalizeScenePayload, SceneType} from "./sceneManger";
import {WaypointCompleted} from "./egoService";
import {isActuatorStateNeutral, isInferenceCleanupComplete} from "./expertNeutral";
import {RECONNECT_CAPTURE_STOP_REQUEST} from "./captureLifecycle";
import {StopSignJob, StopSignPose} from "./stop-sign/types";
import {
    ControlSessionFrontier,
    matchesControlSafetySyncAck,
} from "./control-session-frontier";
import {
    ControlPollSingleFlight,
    dispatchConfirmationAdvancesCursor,
    parseControlDispatchConfirmation,
    runWithAbortTimeout,
} from "./control-dispatch";

const SERVER_BUILD_ID = "2026-08-11-stop-sign-probe-v6";
const CAPTURE_SOURCE_ID = (process.env.CAPTURE_SOURCE_ID || "monitor-2").trim();
console.log(`[server] loaded build=${SERVER_BUILD_ID}`);

type AggregatedTrip = Omit<WaypointCompleted, "vehicleData" | "chunkIndex" | "isTripComplete"> & {
    vehicleData: WaypointCompleted["vehicleData"]
    receivedFinalChunk: boolean
};

type TripTelemetrySummary = {
    sampleCount: number
    collisionEventCount: number
    offroadEventCount: number
    wrongWayEventCount: number
    reversingEventCount: number
    handbrakeEventCount: number
    junctionSampleCount: number
    trafficLightStopSampleCount: number
    leadVehicleSampleCount: number
    routeGpsValidSampleCount: number
    routeGpsMissingSampleCount: number
    onRoadSampleCount: number
    highwaySampleCount: number
    avgNearbyVehicleCount30m: number
    avgNearbyPedCount20m: number
};

type CaptureRequest = {
    requestId: string
    runId: string
    tripIndex?: number
    sceneId?: string
    sceneVariant?: string
    sceneName?: string
}

type ValidatedCaptureRequest = Omit<CaptureRequest, "tripIndex"> & {
    tripIndex: number
}

type CaptureResponse = {
    requestId: string
    success: boolean
    error?: string
    outputFile?: string
    logFile?: string
    outputBytes?: number
}

type ExpertNeutralizeRequest = {
    requestId: string
}

type ExpertNeutralizeResponse = {
    requestId: string
    success: boolean
    error?: string
}

type TripFinalizeResponse = {
    requestId: string
    success: boolean
    error?: string
    runId?: string
    sceneId?: string
    sceneVariant?: string
    tripIndex?: number
    tripKey?: string
    tripDir?: string
    runFile?: string
    metadataFile?: string
    sampleCount?: number
}

type FinalizedTripResult = {
    response: TripFinalizeResponse
    finalizedAt: number
}

type RunIdParts = {
    safeRunId: string
    runFolder: string
    humanTime: string
}

type RunStoragePaths = {
    runsDir: string
    sceneDir: string
    sceneDirRelative: string
    runDirRelative: string
    runFile: string
    runTiming: RunIdParts
}

type TripStoragePaths = {
    tripDir: string
    tripDirRelative: string
    videoFile: string
    videoFileRelative: string
    logFile: string
    logFileRelative: string
    metadataFile: string
}

type ControlCommandType =
    | "startScene"
    | "runAllScenes"
    | "endScene"
    | "endAllScenes"
    | "setStopSignTarget"
    | "clearStopSignTarget"
    | "setStopSignCatalogWaypoint"
    | "probeStopSignTarget"
    | "startStopSignBatch";
type InferenceCommandType = "startEgo" | "stopEgo";

type ControlCommand = {
    id: string
    type: ControlCommandType | InferenceCommandType
    safetyEpoch: number
    sceneName?: string
    planFingerprint?: string
    stopSignBatchId?: string
    stopSignJobs?: StopSignJob[]
    stopSignCatalogPosition?: {x: number, y: number, z: number}
    stopSignProbe?: unknown
    createdAt?: string
}

type ControlStatusUpdate = {
    status: "idle" | "runningScene" | "runningAllScenes" | "stopping" | "error"
    activeSceneName?: string
    lastError?: string
    appliedSafetyEpoch: number
    inFlightSafetyStarts: number
    safetyStatusSequence: number
    stopSignBatch?: {
        batchId: string
        planFingerprint: string
        state: "running" | "completed" | "stopped" | "failed"
        jobId?: string
        jobIndex: number
        jobCount: number
        completedJobs: number
        attemptIndex: number
        attemptCount: number
        phase: string
        startedAtMs: number
        updatedAtMs: number
        lastAttemptOutcome?: {
            success: boolean
            status: string
            failureReason: string
            durationMs: number
            stoppedAtDistanceM: number | null
            dwellDurationMs: number
            crossedStopLineBeforeDwell: boolean
        }
    }
}

type ControlSafetyEpochSyncAck = {
    requestId: string
    sessionId: string
    safetyEpoch: number
}

type ControlSessionResetResponse = {
    sessionId: string
    safetyEpoch: number
}

type ControlTelemetryUpdate = {
    currentSpeed: number
    currentYaw: number
    yawRate: number
    steering: number
    acceleration: number
    brakePressureAvg: number
    vehicleModelHash: number
    vehicleExists: boolean
    isInVehicle: boolean
    positionX?: number
    positionY?: number
    positionZ?: number
    velocityX?: number
    velocityY?: number
    velocityZ?: number
    pitchDeg?: number
    rollDeg?: number
    gear?: number
    rpm?: number
    wheelAngle?: number
    wheelSteeringFullLock?: number
    onGround?: boolean
    collisionState?: string
    routeDirectionCode: number
    routeDirectionDistanceM: number
    routeDirectionUnknown: number
    routeDirectionKeepStraight: number
    routeDirectionTurnLeft: number
    routeDirectionTurnRight: number
    routeDirectionRerouteWrongWay: number
    routeForwardDelta: number
    routeHeadingError: number
    routeDistance: number
    leadVehicleDistance: number
    hasLeadVehicle: boolean
    stopSignTargetConfigured: boolean
    stopSignPose?: StopSignPose
    stopLinePose?: StopSignPose
    stopSignEgoStopPose?: StopSignPose
    stopSignDistanceM: number
    stopLineDistanceM: number
    stopSignLongitudinalErrorM: number
    stopSignLateralErrorM: number
    stopSignHeadingErrorDeg: number
    stopSignPhase: string
    stopSignDwellElapsedMs: number
    stopSignDwellTargetMs: number
    stopSignStopped: boolean
    stopSignAttemptIndex: number
    stopSignAttemptCount: number
    timestampMs?: number
    gameTimeMs?: number
}

type AvailableScene = {
    name: string
    label: string
}

const activeTripByStream = new Map<string, string>();
const pendingTrips = new Map<string, AggregatedTrip>();
const seenTripTelemetry = new Set<string>();
const finalizedTripResults = new Map<string, FinalizedTripResult>();
const finalizeInFlightTrips = new Map<string, Promise<TripFinalizeResponse>>();
const flushedTripRetentionMs = 10 * 60_000;
const tripFinalizeWaitMs = 15_000;
const tripTelemetryPollMs = 50;
const controlSafetySyncTimeoutMs = 15_000;
const controlDispatchRequestTimeoutMs = 2_000;
const controlReconnectRequestTimeoutMs = 2_000;
const controlSessionFrontier = new ControlSessionFrontier();
const controlPollSingleFlight = new ControlPollSingleFlight();
let availableScenesSyncInFlight = false;
let sceneListRequestInFlight = false;
let controlConnectTask: Promise<void> | null = null;
let controlSafetySyncSequence = 0;
let pendingControlSafetySync: {
    requestId: string
    sessionId: string
    safetyEpoch: number
    playerSource: number
    timer: ReturnType<typeof setTimeout>
    resolve: (acknowledged: boolean) => void
} | null = null;
let lastControlTelemetryDebugAt = 0;

class ApiRequestError extends Error {
    constructor(
        message: string,
        readonly status: number
    ) {
        super(message);
    }
}

function formatAttemptProgress(attemptIndex: unknown, attemptCount: unknown): string {
    const count = Math.trunc(Number(attemptCount ?? 0));
    if (count <= 0) {
        return "--/--";
    }
    const index = Math.trunc(Number(attemptIndex ?? 0));
    if (index <= 0) {
        return `--/${count}`;
    }
    return `${Math.min(index, count)}/${count}`;
}

onNet("capture:startRequest", async (request: CaptureRequest) => {
    const playerSource = (global as any).source;
    const response: CaptureResponse = {
        requestId: String(request?.requestId ?? ""),
        success: false
    };

    try {
        const validated = validateCaptureRequest(request);
        const sceneInfo = resolveSceneInfo(validated.sceneId, validated.sceneVariant, validated.sceneName);
        console.log(`[server] capture:start request run=${validated.runId} scene=${sceneInfo.sceneId} variant=${sceneInfo.sceneVariant} trip=${validated.tripIndex}`);
        const runStorage = buildRunStoragePaths(
            resolveDataRoot(),
            sceneInfo.sceneId,
            sceneInfo.sceneVariant,
            validated.runId
        );
        const tripStorage = buildTripStoragePaths(resolveDataRoot(), runStorage, validated.tripIndex);

        const result = await captureApiRequest("/capture/start", {
            cropToWindow: false,
            sourceId: CAPTURE_SOURCE_ID,
            outputFile: tripStorage.videoFileRelative
        });

        response.success = true;
        response.outputFile = result.outputFile;
        response.logFile = result.logFile;
    } catch (err: any) {
        console.error(`[server] capture:start failed requestId=${response.requestId}: ${err?.message ?? err}`);
        response.error = err?.message ?? "failed to start capture";
    }

    emitNet("capture:startResponse", playerSource, response);
});

onNet("expert:neutralizeRequest", async (request: ExpertNeutralizeRequest) => {
    const playerSource = (global as any).source;
    const response: ExpertNeutralizeResponse = {
        requestId: String(request?.requestId ?? ""),
        success: false,
    };
    try {
        if (!response.requestId) {
            throw new Error("missing expert neutralization requestId");
        }
        await neutralizeExpertBackendInputs();
        response.success = true;
    } catch (error: any) {
        response.error = error?.message ?? "failed to neutralize expert inputs";
        console.error(`[expert] input neutralization failed: ${response.error}`);
    }
    emitNet("expert:neutralizeResponse", playerSource, response);
});

onNet("capture:stopRequest", async (request: CaptureRequest) => {
    const playerSource = (global as any).source;
    const response: CaptureResponse = {
        requestId: String(request?.requestId ?? ""),
        success: false
    };

    try {
        const validated = validateCaptureRequest(request);
        const sceneInfo = resolveSceneInfo(validated.sceneId, validated.sceneVariant, validated.sceneName);
        const tripKey = buildTripKey(
            validated.runId,
            sceneInfo.sceneId,
            sceneInfo.sceneVariant,
            validated.tripIndex ?? 0
        );
        pruneFinalizedTripResults();
        if (!finalizedTripResults.has(tripKey)) {
            throw new Error(`trip ${tripKey} has not been finalized; refusing to stop capture before finalize-trip ack`);
        }
        console.log(`[server] capture:stop request run=${validated.runId} scene=${sceneInfo.sceneId} variant=${sceneInfo.sceneVariant} trip=${validated.tripIndex} finalized=yes`);
        const result = await captureApiRequest("/capture/stop", {
            runId: validated.runId,
            tripIndex: validated.tripIndex ?? 0,
            sceneId: validated.sceneId ?? "",
            sceneVariant: validated.sceneVariant ?? "",
            sceneName: validated.sceneName ?? ""
        });
        response.success = true;
        response.outputFile = result.outputFile;
        response.logFile = result.logFile;
        response.outputBytes = typeof result?.outputBytes === "number" ? result.outputBytes : undefined;
        console.log(`[server] capture:stop success run=${validated.runId} scene=${sceneInfo.sceneId} variant=${sceneInfo.sceneVariant} trip=${validated.tripIndex} output=${response.outputFile ?? ""} bytes=${response.outputBytes ?? -1}`);
    } catch (err: any) {
        console.error(`[server] capture:stop failed requestId=${response.requestId}: ${err?.message ?? err}`);
        response.error = err?.message ?? "failed to stop capture";
    }

    emitNet("capture:stopResponse", playerSource, response);
});

onNet("capture:abortRequest", async (request: CaptureRequest) => {
    const playerSource = (global as any).source;
    const response: CaptureResponse = {
        requestId: String(request?.requestId ?? ""),
        success: false
    };

    try {
        const validated = validateCaptureRequest(request);
        const sceneInfo = resolveSceneInfo(validated.sceneId, validated.sceneVariant, validated.sceneName);
        console.log(`[server] capture:abort request run=${validated.runId} scene=${sceneInfo.sceneId} variant=${sceneInfo.sceneVariant} trip=${validated.tripIndex}`);
        const result = await captureApiRequest("/capture/stop", {
            runId: validated.runId,
            tripIndex: validated.tripIndex ?? 0,
            sceneId: validated.sceneId ?? "",
            sceneVariant: validated.sceneVariant ?? "",
            sceneName: validated.sceneName ?? "",
            abortOnly: true
        });
        response.success = true;
        response.outputFile = result.outputFile;
        response.logFile = result.logFile;
        response.outputBytes = typeof result?.outputBytes === "number" ? result.outputBytes : undefined;
        console.log(`[server] capture:abort success run=${validated.runId} scene=${sceneInfo.sceneId} variant=${sceneInfo.sceneVariant} trip=${validated.tripIndex} output=${response.outputFile ?? ""} bytes=${response.outputBytes ?? -1}`);
    } catch (err: any) {
        const message = String(err?.message ?? err ?? "");
        if (message.toLowerCase().includes("capture is not running")) {
            console.log(`[server] capture:abort no-op requestId=${response.requestId}; capture was already stopped`);
            response.success = true;
        } else {
            console.error(`[server] capture:abort failed requestId=${response.requestId}: ${message}`);
            response.error = err?.message ?? "failed to abort capture";
        }
    }

    emitNet("capture:abortResponse", playerSource, response);
});

onNet("capture:finalizeTripRequest", async (request: CaptureRequest) => {
    const playerSource = (global as any).source;
    const response: TripFinalizeResponse = {
        requestId: String(request?.requestId ?? ""),
        success: false
    };

    try {
        const validated = validateCaptureRequest(request);
        const dataRoot = resolveDataRoot();
        const sceneInfo = resolveSceneInfo(validated.sceneId, validated.sceneVariant, validated.sceneName);
        console.log(`[server] capture:finalize request run=${validated.runId} scene=${sceneInfo.sceneId} variant=${sceneInfo.sceneVariant} trip=${validated.tripIndex}`);
        const runStorage = buildRunStoragePaths(
            dataRoot,
            sceneInfo.sceneId,
            sceneInfo.sceneVariant,
            validated.runId
        );
        const manifestFile = resolveManifestFile(dataRoot);
        const fs = require("fs");
        const tripKey = buildTripKey(
            validated.runId,
            sceneInfo.sceneId,
            sceneInfo.sceneVariant,
            validated.tripIndex ?? 0
        );
        const finalized = await finalizeTrip(
            validated.requestId,
            tripKey,
            runStorage,
            manifestFile,
            fs,
            dataRoot
        );
        Object.assign(response, finalized);
        console.log(`[server] capture:finalize success trip=${response.tripKey ?? "unknown"} tripDir=${response.tripDir ?? ""} samples=${response.sampleCount ?? -1}`);
    } catch (err: any) {
        console.error(`[server] capture:finalize failed requestId=${response.requestId}: ${err?.message ?? err}`);
        response.error = err?.message ?? "failed to finalize trip";
    }

    emitNet("capture:finalizeTripResponse", playerSource, response);
});

onNet("control:statusUpdate", async (update: ControlStatusUpdate) => {
    if (!controlSessionFrontier.isRequestedPlayerSource(parseControlPlayerSource((global as any).source))) {
        return;
    }
    try {
        const appliedSafetyEpoch = update?.appliedSafetyEpoch ?? 0;
        const response = await apiRequest("/control/status", "POST", {
            status: update?.status ?? "idle",
            activeSceneName: update?.activeSceneName ?? "",
            lastError: update?.lastError ?? "",
            stopSignBatch: update?.stopSignBatch,
            appliedSafetyEpoch,
            inFlightSafetyStarts: update?.inFlightSafetyStarts ?? 0,
            safetyStatusSequence: update?.safetyStatusSequence ?? 0,
        });
        const backendAppliedSafetyEpoch = Number(response?.appliedSafetyEpoch);
        if (
            Number.isSafeInteger(appliedSafetyEpoch)
            && appliedSafetyEpoch > 0
            && Number.isSafeInteger(backendAppliedSafetyEpoch)
            && backendAppliedSafetyEpoch !== appliedSafetyEpoch
            && controlSessionFrontier.requestBackendReconnect()
        ) {
            cancelPendingControlSafetySync();
            console.warn(
                `[control] backend safety epoch reset detected `
                + `client=${appliedSafetyEpoch} backend=${backendAppliedSafetyEpoch}; reconnecting`,
            );
            void connectControlSession();
        }
    } catch (err: any) {
        console.error(`[control] failed to push status update: ${err?.message ?? err}`);
    }
});

onNet("control:telemetryUpdate", async (update: ControlTelemetryUpdate) => {
    if (!controlSessionFrontier.isRequestedPlayerSource(parseControlPlayerSource((global as any).source))) {
        return;
    }
    const now = Date.now();
    if (now - lastControlTelemetryDebugAt >= 2000) {
        lastControlTelemetryDebugAt = now;
        console.log(
            `[control] telemetry speed=${Number(update?.currentSpeed ?? 0).toFixed(2)} ` +
            `yawRate=${Number(update?.yawRate ?? 0).toFixed(2)} ` +
            `steer=${Number(update?.steering ?? 0).toFixed(2)} ` +
            `accel=${Number(update?.acceleration ?? 0).toFixed(2)} ` +
            `navCode=${Math.trunc(Number(update?.routeDirectionCode ?? 0))} ` +
            `navOneHot=[u=${Math.trunc(Number(update?.routeDirectionUnknown ?? 0))}, ` +
            `s=${Math.trunc(Number(update?.routeDirectionKeepStraight ?? 0))}, ` +
            `l=${Math.trunc(Number(update?.routeDirectionTurnLeft ?? 0))}, ` +
            `r=${Math.trunc(Number(update?.routeDirectionTurnRight ?? 0))}, ` +
            `w=${Math.trunc(Number(update?.routeDirectionRerouteWrongWay ?? 0))}] ` +
            `routeFwd=${Number(update?.routeForwardDelta ?? 0).toFixed(2)} ` +
            `routeHeading=${Number(update?.routeHeadingError ?? 0).toFixed(2)} ` +
            `routeDistance=${Number(update?.routeDistance ?? 0).toFixed(2)} ` +
            `hasLead=${String(Boolean(update?.hasLeadVehicle))} ` +
            `leadDistance=${Number(update?.leadVehicleDistance ?? 0).toFixed(2)} ` +
            `stopSign=${String(update?.stopSignPhase ?? "idle")} ` +
            `attempt=${formatAttemptProgress(update?.stopSignAttemptIndex, update?.stopSignAttemptCount)}`
        );
    }
    try {
        await apiRequest("/control/telemetry", "POST", {
            currentSpeed: update?.currentSpeed ?? 0,
            currentYaw: update?.currentYaw ?? 0,
            yawRate: update?.yawRate ?? 0,
            steering: update?.steering ?? 0,
            acceleration: update?.acceleration ?? 0,
            brakePressureAvg: update?.brakePressureAvg ?? 0,
            vehicleModelHash: update?.vehicleModelHash ?? 0,
            vehicleExists: Boolean(update?.vehicleExists),
            isInVehicle: Boolean(update?.isInVehicle),
            positionX: update?.positionX,
            positionY: update?.positionY,
            positionZ: update?.positionZ,
            velocityX: update?.velocityX,
            velocityY: update?.velocityY,
            velocityZ: update?.velocityZ,
            pitchDeg: update?.pitchDeg,
            rollDeg: update?.rollDeg,
            gear: update?.gear,
            rpm: update?.rpm,
            wheelAngle: update?.wheelAngle,
            wheelSteeringFullLock: update?.wheelSteeringFullLock,
            onGround: update?.onGround,
            collisionState: update?.collisionState ?? "",
            routeDirectionCode: update?.routeDirectionCode ?? 0,
            routeDirectionDistanceM: update?.routeDirectionDistanceM ?? 0,
            routeDirectionUnknown: update?.routeDirectionUnknown ?? 0,
            routeDirectionKeepStraight: update?.routeDirectionKeepStraight ?? 0,
            routeDirectionTurnLeft: update?.routeDirectionTurnLeft ?? 0,
            routeDirectionTurnRight: update?.routeDirectionTurnRight ?? 0,
            routeDirectionRerouteWrongWay: update?.routeDirectionRerouteWrongWay ?? 0,
            routeForwardDelta: update?.routeForwardDelta ?? 0,
            routeHeadingError: update?.routeHeadingError ?? 0,
            routeDistance: update?.routeDistance ?? 0,
            leadVehicleDistance: update?.leadVehicleDistance ?? 0,
            hasLeadVehicle: Boolean(update?.hasLeadVehicle),
            stopSignTargetConfigured: Boolean(update?.stopSignTargetConfigured),
            stopSignPose: apiStopSignPose(update?.stopSignPose),
            stopLinePose: apiStopSignPose(update?.stopLinePose),
            stopSignEgoStopPose: apiStopSignPose(update?.stopSignEgoStopPose),
            stopSignDistanceM: update?.stopSignDistanceM ?? 0,
            stopLineDistanceM: update?.stopLineDistanceM ?? 0,
            stopSignLongitudinalErrorM: update?.stopSignLongitudinalErrorM ?? 0,
            stopSignLateralErrorM: update?.stopSignLateralErrorM ?? 0,
            stopSignHeadingErrorDeg: update?.stopSignHeadingErrorDeg ?? 0,
            stopSignPhase: update?.stopSignPhase ?? "idle",
            stopSignDwellElapsedMs: update?.stopSignDwellElapsedMs ?? 0,
            stopSignDwellTargetMs: update?.stopSignDwellTargetMs ?? 0,
            stopSignStopped: Boolean(update?.stopSignStopped),
            stopSignAttemptIndex: update?.stopSignAttemptIndex ?? 0,
            stopSignAttemptCount: update?.stopSignAttemptCount ?? 0,
            timestampMs: update?.timestampMs ?? 0,
            gameTimeMs: update?.gameTimeMs ?? 0
        });
    } catch (err: any) {
        console.error(`[control] failed to push telemetry update: ${err?.message ?? err}`);
    }
});

function apiStopSignPose(pose: StopSignPose | undefined) {
    if (!pose || ![pose.x, pose.y, pose.z, pose.heading].every(Number.isFinite)) {
        return undefined;
    }
    return {...pose};
}

onNet("control:availableScenesResponse", async (scenes: AvailableScene[]) => {
    if (!controlSessionFrontier.isRequestedPlayerSource(parseControlPlayerSource((global as any).source))) {
        return;
    }
    try {
        sceneListRequestInFlight = false;
        await syncAvailableScenes(Array.isArray(scenes) ? scenes : []);
    } catch (err: any) {
        console.error(`[control] failed to handle available scenes response: ${err?.message ?? err}`);
    }
});

onNet("control:registerClient", async () => {
    const playerSource = parseControlPlayerSource((global as any).source);
    if (playerSource === null) {
        console.error("[control] rejected registration without a valid player source");
        return;
    }
    const claim = controlSessionFrontier.beginReconnect(playerSource);
    if (claim.kind === "rejected") {
        console.warn(
            `[control] rejected client source=${playerSource}; `
            + `source=${claim.ownerPlayerSource} already owns the control session`,
        );
        return;
    }
    const generation = claim.generation;
    cancelPendingControlSafetySync();
    console.log(`[control] register client source=${String(playerSource ?? "unknown")} generation=${generation} build=${SERVER_BUILD_ID}`);
    await stopCaptureOnReconnect();
    await connectControlSession();
    requestAvailableScenesFromClient();
});

onNet("control:clientHeartbeat", () => {
    const playerSource = parseControlPlayerSource((global as any).source);
    if (!controlSessionFrontier.noteOwnerHeartbeat(playerSource)) {
        return;
    }
    if (controlSessionFrontier.needsConnect()) {
        void connectControlSession();
    }
});

onNet("control:safetyEpochSyncAck", (ack: ControlSafetyEpochSyncAck) => {
    const playerSource = parseControlPlayerSource((global as any).source);
    const pending = pendingControlSafetySync;
    if (
        pending === null
        || !matchesControlSafetySyncAck(pending, playerSource, ack)
    ) {
        console.warn(`[control] ignored unmatched safety epoch sync acknowledgement from source=${String(playerSource ?? "unknown")}`);
        return;
    }

    settlePendingControlSafetySync(true);
});

on("playerDropped", () => {
    const playerSource = parseControlPlayerSource((global as any).source);
    if (!controlSessionFrontier.releaseOwner(playerSource)) {
        return;
    }

    cancelPendingControlSafetySync();
    sceneListRequestInFlight = false;
    console.warn(`[control] released dropped owner source=${String(playerSource)}; polling is paused`);
    void stopCaptureOnReconnect();
});

onNet("ego:vehicleData", async (data: WaypointCompleted) => {
    const fs = require("fs");
    const path = require("path");
    const dataRoot = resolveDataRoot();
    const runStorage = buildRunStoragePaths(
        dataRoot,
        data.sceneId,
        data.sceneVariant,
        data.runId
    );
    const runsDir = runStorage.runsDir;
    const runFile = runStorage.runFile;
    const manifestFile = resolveManifestFile(dataRoot);
    const manifestDir = path.dirname(manifestFile);
    const tripStorage = buildTripStoragePaths(dataRoot, runStorage, data.tripIndex);
    const streamKey = buildTripStreamKey(data.runId, data.sceneId, data.sceneVariant);
    const tripKey = buildTripKey(data.runId, data.sceneId, data.sceneVariant, data.tripIndex);
    pruneFinalizedTripResults();

    console.log(`[server] received chunk ${data.chunkIndex} for trip ${data.tripIndex} run ${data.runId} with ${data.vehicleData.length} points complete=${data.isTripComplete}`);

    if (finalizedTripResults.has(tripKey)) {
        console.log(`[server] ignoring telemetry chunk ${data.chunkIndex} for already-finalized trip ${tripKey}`);
        return;
    }

    if (!fs.existsSync(runsDir)) {
        fs.mkdirSync(runsDir, { recursive: true });
        console.log("Directory created:", runsDir);
    }

    if (!fs.existsSync(runStorage.sceneDir)) {
        fs.mkdirSync(runStorage.sceneDir, { recursive: true });
        console.log("Directory created:", runStorage.sceneDir);
    }

    if (!fs.existsSync(tripStorage.tripDir)) {
        fs.mkdirSync(tripStorage.tripDir, { recursive: true });
        console.log("Directory created:", tripStorage.tripDir);
    }

    if (!fs.existsSync(manifestDir)) {
        fs.mkdirSync(manifestDir, { recursive: true });
        console.log("Directory created:", manifestDir);
    }

    if (!fs.existsSync(manifestFile)) {
        try {
            fs.writeFileSync(manifestFile, "");
            console.log("File created:", manifestFile);
        } catch (err) {
            console.error("Error creating file:", err);
            return;
        }
    }

    if (!fs.existsSync(runFile)) {
        try {
            fs.writeFileSync(runFile, "");
            console.log(`[server] created scene manifest ${runFile}`);
        } catch (err) {
            console.error(`[server] failed to create scene manifest ${runFile}:`, err);
            return;
        }
    }

    const previousTripKey = activeTripByStream.get(streamKey);
    if (previousTripKey && previousTripKey !== tripKey) {
        console.error(
            `[server] refusing telemetry for ${tripKey} while previous trip ${previousTripKey} is still active; finalize-trip barrier was bypassed`
        );
        return;
    }

    activeTripByStream.set(streamKey, tripKey);

    const existingTrip = pendingTrips.get(tripKey);
    if (existingTrip) {
        existingTrip.vehicleData.push(...data.vehicleData);
        existingTrip.endTime = data.endTime;
        existingTrip.chunkDurationMs = Math.max(0, data.endTime - existingTrip.syncTime);
        existingTrip.stopSignGoal = data.stopSignGoal ?? existingTrip.stopSignGoal;
        existingTrip.stopSignOutcome = data.stopSignOutcome ?? existingTrip.stopSignOutcome;
    } else {
        const {vehicleData, chunkIndex: _chunkIndex, isTripComplete: _isTripComplete, ...tripMetadata} = data;
        pendingTrips.set(tripKey, {
            ...tripMetadata,
            chunkDurationMs: Math.max(0, data.endTime - data.syncTime),
            vehicleData: [...vehicleData],
            receivedFinalChunk: false
        });
    }
    seenTripTelemetry.add(tripKey);

    if (data.isTripComplete) {
        const trip = pendingTrips.get(tripKey);
        if (trip) {
            trip.receivedFinalChunk = true;
        }
        console.log(`[server] received final chunk for trip ${tripKey}; awaiting explicit finalize request`);
    }
});

function buildTripStreamKey(runId: string, sceneId: string, sceneVariant: string): string {
    return `${runId}:${sceneId}:${sceneVariant}`;
}

function buildTripKey(runId: string, sceneId: string, sceneVariant: string, tripIndex: number): string {
    return `${buildTripStreamKey(runId, sceneId, sceneVariant)}:${tripIndex}`;
}

function getTripFinalizeState(tripKey: string): "ready" | "pending" | "finalized" | "missing" {
    pruneFinalizedTripResults();
    const cached = finalizedTripResults.get(tripKey);
    if (cached) {
        return "finalized";
    }
    const trip = pendingTrips.get(tripKey);
    if (trip?.receivedFinalChunk) {
        return "ready";
    }
    if (trip || seenTripTelemetry.has(tripKey)) {
        return "pending";
    }
    return "missing";
}

async function waitForTripFinalizationReady(
    tripKey: string,
    timeoutMs = tripFinalizeWaitMs
): Promise<"ready" | "pending" | "finalized" | "missing"> {
    const deadline = Date.now() + Math.max(0, timeoutMs);
    let state = getTripFinalizeState(tripKey);
    while ((state === "missing" || state === "pending") && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, tripTelemetryPollMs));
        state = getTripFinalizeState(tripKey);
    }
    return state;
}

function coerceBoolean(value: unknown): boolean {
    if (typeof value === "boolean") {
        return value;
    }
    if (typeof value === "number") {
        return value !== 0;
    }
    return false;
}

function coerceNumber(value: unknown): number | undefined {
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function buildTripTelemetrySummary(vehicleData: WaypointCompleted["vehicleData"]): TripTelemetrySummary {
    let collisionEventCount = 0;
    let offroadEventCount = 0;
    let wrongWayEventCount = 0;
    let reversingEventCount = 0;
    let handbrakeEventCount = 0;
    let junctionSampleCount = 0;
    let trafficLightStopSampleCount = 0;
    let leadVehicleSampleCount = 0;
    let routeGpsValidSampleCount = 0;
    let routeGpsMissingSampleCount = 0;
    let onRoadSampleCount = 0;
    let highwaySampleCount = 0;
    let nearbyVehicleTotal = 0;
    let nearbyPedTotal = 0;
    let nearbyVehicleSamples = 0;
    let nearbyPedSamples = 0;

    for (const point of vehicleData) {
        if (coerceBoolean(point.eventCollision)) {
            collisionEventCount++;
        }
        if (coerceBoolean(point.eventOffroad)) {
            offroadEventCount++;
        }
        if (coerceBoolean(point.eventWrongWay)) {
            wrongWayEventCount++;
        }
        if (coerceBoolean(point.eventReversing)) {
            reversingEventCount++;
        }
        if (coerceBoolean(point.eventHandbrake)) {
            handbrakeEventCount++;
        }
        if (coerceBoolean(point.isInJunction)) {
            junctionSampleCount++;
        }
        if (coerceBoolean(point.isStoppedAtTrafficLights)) {
            trafficLightStopSampleCount++;
        }
        if (coerceBoolean(point.hasLeadVehicle)) {
            leadVehicleSampleCount++;
        }
        if (coerceBoolean(point.routeGpsValid)) {
            routeGpsValidSampleCount++;
        } else {
            routeGpsMissingSampleCount++;
        }
        if (coerceBoolean(point.isOnRoad)) {
            onRoadSampleCount++;
        }
        if (coerceBoolean(point.isHighway)) {
            highwaySampleCount++;
        }

        const nearbyVehicleCount = coerceNumber(point.nearbyVehicleCount30m);
        if (nearbyVehicleCount !== undefined) {
            nearbyVehicleTotal += nearbyVehicleCount;
            nearbyVehicleSamples++;
        }
        const nearbyPedCount = coerceNumber(point.nearbyPedCount20m);
        if (nearbyPedCount !== undefined) {
            nearbyPedTotal += nearbyPedCount;
            nearbyPedSamples++;
        }
    }

    return {
        sampleCount: vehicleData.length,
        collisionEventCount,
        offroadEventCount,
        wrongWayEventCount,
        reversingEventCount,
        handbrakeEventCount,
        junctionSampleCount,
        trafficLightStopSampleCount,
        leadVehicleSampleCount,
        routeGpsValidSampleCount,
        routeGpsMissingSampleCount,
        onRoadSampleCount,
        highwaySampleCount,
        avgNearbyVehicleCount30m: nearbyVehicleSamples > 0 ? nearbyVehicleTotal / nearbyVehicleSamples : 0,
        avgNearbyPedCount20m: nearbyPedSamples > 0 ? nearbyPedTotal / nearbyPedSamples : 0,
    };
}

function sanitizePathSegment(value: string): string {
    return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function parseControlPlayerSource(value: unknown): number | null {
    const parsed = Number(value);
    if (!Number.isSafeInteger(parsed) || parsed <= 0) {
        return null;
    }
    return parsed;
}

function resolveDataRoot(): string {
    return normalizeDataRoot(process.env.VEHICLE_DATA_DIR)
        ?? normalizeDataRoot(process.env.FSD_DATA_ROOT)
        ?? (process.platform === "win32"
            ? "S:\\fsd_fivem_data"
            : "/mnt/s/fsd_fivem_data");
}

function normalizeDataRoot(value?: string): string | undefined {
    if (!value) {
        return undefined;
    }

    const trimmed = value.trim().replace(/^["']|["']$/g, "");
    if (!trimmed) {
        return undefined;
    }

    if (/^[a-zA-Z]:[^\\/]/.test(trimmed)) {
        return `${trimmed.slice(0, 2)}\\${trimmed.slice(2)}`;
    }

    return trimmed;
}

function resolveManifestFile(dataRoot: string): string {
    const path = require("path");
    return process.env.VEHICLE_DATA_FILE
        ?? path.join(dataRoot, "runs.jsonl");
}

function parseRunIdParts(runId: string): RunIdParts {
    const safeRunId = sanitizePathSegment(runId);
    const match = safeRunId.match(/^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})(AM|PM)(?:_[a-z0-9]+)?$/i);
    if (!match) {
        return {
            safeRunId,
            runFolder: safeRunId || "legacy",
            humanTime: safeRunId
        };
    }

    const date = match[1]!;
    const hour = match[2]!;
    const minute = match[3]!;
    const second = match[4]!;
    const amPm = match[5]!.toUpperCase();
    const runFolder = safeRunId;
    return {
        safeRunId,
        runFolder,
        humanTime: `${date} ${hour}:${minute}:${second} ${amPm}`
    };
}

function buildRunStoragePaths(dataRoot: string, sceneId: string, sceneVariant: string, runId: string): RunStoragePaths {
    const path = require("path");
    const runTiming = parseRunIdParts(runId);
    const sceneFolder = sanitizePathSegment(`${sceneId}_${sceneVariant}`);

    const runsDir = path.join(
        dataRoot,
        "runs",
        runTiming.runFolder
    );
    const sceneDir = path.join(runsDir, sceneFolder);
    const runDirRelative = `runs/${runTiming.runFolder}`;
    const sceneDirRelative = `runs/${runTiming.runFolder}/${sceneFolder}`;
    const runFile = path.join(sceneDir, "run.jsonl");
    return {
        runsDir,
        sceneDir,
        runDirRelative,
        sceneDirRelative,
        runFile,
        runTiming
    };
}

function buildTripStoragePaths(dataRoot: string, runStorage: RunStoragePaths, tripIndex: number): TripStoragePaths {
    const path = require("path");
    const tripSuffix = String(tripIndex).padStart(3, "0");
    const tripDirRelative = `${runStorage.sceneDirRelative}/trip-${tripSuffix}`;
    const videoFileRelative = `${tripDirRelative}/video.mkv`;
    return {
        tripDir: path.join(runStorage.sceneDir, `trip-${tripSuffix}`),
        tripDirRelative,
        videoFile: path.join(runStorage.sceneDir, `trip-${tripSuffix}`, "video.mkv"),
        videoFileRelative,
        logFile: path.join(runStorage.sceneDir, `trip-${tripSuffix}`, "video.log"),
        logFileRelative: `${tripDirRelative}/video.log`,
        metadataFile: path.join(runStorage.sceneDir, `trip-${tripSuffix}`, "metadata.json")
    };
}

function validateCaptureRequest(request: CaptureRequest): ValidatedCaptureRequest {
    const requestId = String(request?.requestId ?? "").trim();
    const runId = String(request?.runId ?? "").trim();
    const tripIndex = typeof request?.tripIndex === "number"
        ? request.tripIndex
        : Number(request?.tripIndex);

    if (!requestId) {
        throw new Error("missing requestId");
    }
    if (!runId) {
        throw new Error("missing runId");
    }
    if (!Number.isInteger(tripIndex) || tripIndex < 0) {
        throw new Error("missing tripIndex");
    }

    return {
        requestId,
        runId,
        tripIndex,
        sceneId: request?.sceneId,
        sceneVariant: request?.sceneVariant,
        sceneName: request?.sceneName
    };
}

function resolveSceneInfo(sceneId?: string, sceneVariant?: string, sceneName?: string) {
    const parsed = parseSceneName(sceneName);
    return {
        sceneId: sanitizePathSegment((sceneId || parsed.sceneId || "unknown-scene").trim()),
        sceneVariant: sanitizePathSegment((sceneVariant || parsed.sceneVariant || "default").trim())
    };
}

function parseSceneName(sceneName?: string): { sceneId: string; sceneVariant: string } {
    if (!sceneName || typeof sceneName !== "string") {
        return { sceneId: "unknown-scene", sceneVariant: "default" };
    }

    const [sceneId, sceneVariant] = sceneName.split(":");
    return {
        sceneId: sceneId || "unknown-scene",
        sceneVariant: sceneVariant || "default"
    };
}

async function captureApiRequest(endpoint: string, body: any): Promise<any> {
    return apiRequest(endpoint, "POST", body);
}

async function apiRequest(
    endpoint: string,
    method: "GET" | "POST",
    body?: any,
    signal?: AbortSignal,
): Promise<any> {
    const baseUrl = (process.env.CAPTURE_API_URL || "http://127.0.0.1:8080").replace(/\/+$/, "");
    const url = `${baseUrl}${endpoint}`;
    const response = await fetch(url, method === "GET"
        ? {
            method,
            signal,
        }
        : {
            method,
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(body ?? {}),
            signal,
        });

    const text = await response.text();
    let parsed: any = {};
    if (text) {
        try {
            parsed = JSON.parse(text);
        } catch {
            parsed = { error: text };
        }
    }

    if (!response.ok) {
        const message = parsed?.error || `capture api ${endpoint} failed with status ${response.status}`;
        throw new ApiRequestError(message, response.status);
    }

    return parsed;
}

function controlDispatchApiRequest(endpoint: string, method: "GET" | "POST", body?: any): Promise<any> {
    return runWithAbortTimeout(
        `control dispatch ${method} ${endpoint}`,
        controlDispatchRequestTimeoutMs,
        (signal) => apiRequest(endpoint, method, body, signal),
    );
}

function controlReconnectApiRequest(endpoint: string, method: "GET" | "POST", body?: any): Promise<any> {
    return runWithAbortTimeout(
        `control reconnect ${method} ${endpoint}`,
        controlReconnectRequestTimeoutMs,
        (signal) => apiRequest(endpoint, method, body, signal),
    );
}

async function neutralizeExpertBackendInputs() {
    await stopInferenceForExpert();
    await disableActuatorForExpert();
    console.log("[expert] inference stopped and virtual actuator confirmed neutral");
}

async function stopInferenceForExpert() {
    const deadline = Date.now() + 10_000;
    while (Date.now() < deadline) {
        const status = await apiRequest("/inference/status", "GET");
        const state = String(status?.state ?? "").trim().toLowerCase();
        if (state === "idle" || state === "succeeded" || state === "failed" || state === "error") {
            if (isInferenceCleanupComplete(status)) {
                return;
            }
            await delayServer(100);
            continue;
        }
        if (state === "stopping") {
            await delayServer(100);
            continue;
        }
        if (state !== "starting" && state !== "running") {
            throw new Error(`cannot neutralize expert inputs from unknown inference state ${state || "<empty>"}`);
        }
        try {
            await apiRequest("/inference/stop", "POST", {});
        } catch (error) {
            if (!isApiStatus(error, 404)) {
                throw error;
            }
            return;
        }
        await delayServer(100);
    }
    throw new Error("inference did not reach idle before expert collection deadline");
}

async function disableActuatorForExpert() {
    const disabled = false;
    try {
        await apiRequest("/actuator/command", "POST", {
            steer: 0,
            throttle: 0,
            brakePressureAvg: 0,
            handbrake: false,
            enabled: disabled,
            inputMode: "normalized",
        });
    } catch (error) {
        if (isUnavailableActuatorError(error)) {
            console.log("[expert] virtual actuator unavailable; no controller can contaminate collection");
            return;
        }
        throw error;
    }

    const deadline = Date.now() + 1_000;
    while (Date.now() < deadline) {
        await delayServer(25);
        let state: unknown;
        try {
            state = await apiRequest("/actuator/state", "GET");
        } catch (error) {
            if (isUnavailableActuatorError(error)) {
                return;
            }
            throw error;
        }
        if (isActuatorStateNeutral(state)) {
            return;
        }
    }
    throw new Error("virtual actuator did not confirm a neutral applied state before expert collection");
}

function isUnavailableActuatorError(error: unknown): boolean {
    return isApiStatus(error, 501) || isApiStatus(error, 503);
}

function isApiStatus(error: unknown, status: number): boolean {
    return error instanceof ApiRequestError && error.status === status;
}

function delayServer(durationMs: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, durationMs));
}

async function syncAvailableScenes(scenes: AvailableScene[]) {
    if (availableScenesSyncInFlight) {
        return;
    }

    availableScenesSyncInFlight = true;
    try {
        await apiRequest("/control/scenes", "POST", {
            scenes
        });
    } catch (err: any) {
        console.error(`[control] failed to sync available scenes: ${err?.message ?? err}`);
    } finally {
        availableScenesSyncInFlight = false;
    }
}

async function connectControlSession() {
    if (controlConnectTask !== null) {
        await controlConnectTask;
        return;
    }

    const task = runControlConnectLoop();
    controlConnectTask = task;
    try {
        await task;
    } finally {
        if (controlConnectTask === task) {
            controlConnectTask = null;
        }
    }
}

async function runControlConnectLoop() {
    while (controlSessionFrontier.needsConnect()) {
        const lease = controlSessionFrontier.beginConnect();
        if (!lease) {
            return;
        }

        let result: ControlSessionResetResponse;
        try {
            result = validateControlSessionResetResponse(
                await controlReconnectApiRequest("/control/connect", "POST", {}),
            );
        } catch (err: any) {
            controlSessionFrontier.failConnect(lease);
            console.error(`[control] failed to reset control session: ${err?.message ?? err}`);
            return;
        }

        const completion = controlSessionFrontier.completeConnect(lease);
        if (completion.kind === "stale") {
            console.warn(`[control] discarded superseded connect response session=${result.sessionId} generation=${lease.generation}`);
            continue;
        }

        const playerSource = completion.playerSource;

        let acknowledged = false;
        try {
            acknowledged = await requestControlSafetyEpochSync(
                completion.generation,
                result,
                playerSource,
            );
        } catch (err: any) {
            cancelPendingControlSafetySync();
            controlSessionFrontier.failSafetySync(completion.generation);
            console.error(`[control] failed to send safety epoch sync: ${err?.message ?? err}`);
            return;
        }
        if (!acknowledged) {
            const superseded = !controlSessionFrontier.isCurrentGeneration(completion.generation);
            controlSessionFrontier.failSafetySync(completion.generation);
            if (superseded) {
                continue;
            }
            console.error(`[control] safety epoch sync was not acknowledged; command polling remains paused`);
            return;
        }
        if (!controlSessionFrontier.completeSafetySync(completion.generation, playerSource)) {
            console.warn(`[control] discarded superseded safety epoch sync session=${result.sessionId} generation=${completion.generation}`);
            continue;
        }

        console.log(
            `[control] connected to session ${result.sessionId} safetyEpoch=${result.safetyEpoch} `
            + `generation=${completion.generation} build=${SERVER_BUILD_ID}`,
        );
    }
}

function validateControlSessionResetResponse(value: any): ControlSessionResetResponse {
    const sessionId = String(value?.sessionId ?? "").trim();
    const safetyEpoch = Number(value?.safetyEpoch);
    if (!sessionId) {
        throw new Error("control connect response is missing sessionId");
    }
    if (!Number.isSafeInteger(safetyEpoch) || safetyEpoch < 1) {
        throw new Error(`control connect response has invalid safetyEpoch ${String(value?.safetyEpoch)}`);
    }
    return {sessionId, safetyEpoch};
}

function requestControlSafetyEpochSync(
    generation: number,
    reset: ControlSessionResetResponse,
    playerSource: number,
): Promise<boolean> {
    cancelPendingControlSafetySync();
    const requestId = `${reset.sessionId}:sync-${generation}-${++controlSafetySyncSequence}`;

    return new Promise((resolve) => {
        const timer = setTimeout(() => {
            if (pendingControlSafetySync?.requestId !== requestId) {
                return;
            }
            console.error(
                `[control] timed out synchronizing safetyEpoch=${reset.safetyEpoch} `
                + `session=${reset.sessionId} source=${playerSource}`,
            );
            settlePendingControlSafetySync(false);
        }, controlSafetySyncTimeoutMs);
        pendingControlSafetySync = {
            requestId,
            sessionId: reset.sessionId,
            safetyEpoch: reset.safetyEpoch,
            playerSource,
            timer,
            resolve,
        };
        emitNet("control:safetyEpochSync", playerSource, {
            requestId,
            sessionId: reset.sessionId,
            safetyEpoch: reset.safetyEpoch,
        });
    });
}

function cancelPendingControlSafetySync() {
    settlePendingControlSafetySync(false);
}

function settlePendingControlSafetySync(acknowledged: boolean) {
    const pending = pendingControlSafetySync;
    if (pending === null) {
        return;
    }
    pendingControlSafetySync = null;
    clearTimeout(pending.timer);
    pending.resolve(acknowledged);
}

async function stopCaptureOnReconnect() {
    try {
        await controlReconnectApiRequest("/capture/stop", "POST", RECONNECT_CAPTURE_STOP_REQUEST);
        console.log("[control] aborted active capture during reconnect");
    } catch (err: any) {
        const message = String(err?.message ?? err ?? "");
        if (message.toLowerCase().includes("capture is not running")) {
            return;
        }
        console.error(`[control] failed to stop capture during reconnect: ${message}`);
    }
}

function requestAvailableScenesFromClient() {
    if (sceneListRequestInFlight) {
        return;
    }

    const playerSource = controlSessionFrontier.getReadyPlayerSource();
    if (playerSource === null) {
        return;
    }

    sceneListRequestInFlight = true;
    emitNet("control:requestAvailableScenes", playerSource);
    setTimeout(() => {
        sceneListRequestInFlight = false;
    }, 3000);
}

async function pollControlCommands() {
    await controlPollSingleFlight.run(async () => {
        const lease = controlSessionFrontier.beginPoll();
        if (!lease) {
            return;
        }

        try {
            const query = lease.lastSeenCommandId
                ? `?lastSeenCommandId=${encodeURIComponent(lease.lastSeenCommandId)}`
                : "";
            const result = await controlDispatchApiRequest(`/control/poll${query}`, "GET");
            const command = result?.command as ControlCommand | undefined;
            if (!command?.id) {
                controlSessionFrontier.acceptPoll(lease);
                return;
            }

            if (!controlSessionFrontier.isPollCurrent(lease)) {
                console.warn(`[control] discarded command id=${command.id} from a superseded poll generation=${lease.generation}`);
                return;
            }

            const confirmation = parseControlDispatchConfirmation(
                await controlDispatchApiRequest("/control/dispatch/confirm", "POST", {commandId: command.id}),
                command.id,
            );
            if (!controlSessionFrontier.isPollCurrent(lease)) {
                console.warn(`[control] discarded confirmation id=${command.id} from a superseded poll generation=${lease.generation}`);
                return;
            }
            if (!dispatchConfirmationAdvancesCursor(confirmation)) {
                console.warn(`[control] command id=${command.id} disappeared before confirmation; retrying from the prior cursor`);
                return;
            }
            if (!confirmation.confirmed) {
                if (!controlSessionFrontier.acceptDispatchConfirmation(lease, command.id)) {
                    console.warn(`[control] discarded rejection id=${command.id} from an inactive poll generation=${lease.generation}`);
                    return;
                }
                console.warn(`[control] skipped command id=${command.id} after backend rejection reason=${confirmation.reason}`);
                return;
            }

            const confirmedCommand = confirmation.command as ControlCommand;
            if (!controlSessionFrontier.dispatchConfirmedCommand(lease, confirmedCommand.id, () => {
                console.log(`[control] dispatching command id=${confirmedCommand.id} type=${confirmedCommand.type} scene=${confirmedCommand.sceneName ?? ""} to source=${lease.playerSource}`);
                emitNet("control:executeCommand", lease.playerSource, confirmedCommand);
            })) {
                console.warn(`[control] discarded confirmed command id=${command.id} from an inactive poll generation=${lease.generation}`);
            }
        } catch (err: any) {
            console.error(`[control] poll failed: ${err?.message ?? err}`);
        }
    });
}

setImmediate(() => {
    requestAvailableScenesFromClient();
    void pollControlCommands();
});

setInterval(() => {
    void pollControlCommands();
}, 250);

setInterval(() => {
    requestAvailableScenesFromClient();
}, 10000);

async function finalizeTrip(
    requestId: string,
    tripKey: string,
    runStorage: RunStoragePaths,
    manifestFile: string,
    fs: any,
    dataRoot: string
): Promise<TripFinalizeResponse> {
    pruneFinalizedTripResults();
    const cached = finalizedTripResults.get(tripKey);
    if (cached) {
        return {
            ...cached.response,
            requestId
        };
    }

    const inFlight = finalizeInFlightTrips.get(tripKey);
    if (inFlight) {
        const finalized = await inFlight;
        return {
            ...finalized,
            requestId
        };
    }

    const promise = (async () => {
        const state = await waitForTripFinalizationReady(tripKey);
        if (state === "finalized") {
            const finalized = finalizedTripResults.get(tripKey);
            if (!finalized) {
                throw new Error(`trip ${tripKey} reported finalized but no finalized result was cached`);
            }
            return finalized.response;
        }
        if (state !== "ready") {
            throw new Error(`trip ${tripKey} did not become ready for finalization within ${tripFinalizeWaitMs}ms`);
        }
        const finalized = flushTrip(tripKey, runStorage, manifestFile, fs, dataRoot);
        if (!finalized) {
            throw new Error(`trip ${tripKey} became ready for finalization but no pending trip data was found`);
        }
        return finalized;
    })();

    finalizeInFlightTrips.set(tripKey, promise);
    try {
        const finalized = await promise;
        return {
            ...finalized,
            requestId
        };
    } finally {
        finalizeInFlightTrips.delete(tripKey);
    }
}

function flushTrip(
    tripKey: string,
    runStorage: RunStoragePaths,
    manifestFile: string,
    fs: any,
    dataRoot: string
): TripFinalizeResponse | null {
    const trip = pendingTrips.get(tripKey);
    if (!trip) {
        return null;
    }

    const tripStorage = buildTripStoragePaths(dataRoot, runStorage, trip.tripIndex);
    const runFile = runStorage.runFile;

    if (!fs.existsSync(tripStorage.tripDir)) {
        fs.mkdirSync(tripStorage.tripDir, { recursive: true });
    }

    console.log(`[server] flush start trip=${tripKey} runFile=${runFile} tripDir=${tripStorage.tripDir} samples=${trip.vehicleData.length}`);

    const telemetrySummary = buildTripTelemetrySummary(trip.vehicleData);
    const tripMetadata = {
        runId: trip.runId,
        runFolder: runStorage.runTiming.runFolder,
        runDir: runStorage.runsDir,
        runDirRelative: runStorage.runDirRelative,
        runFile,
        sceneFolder: pathBasename(runStorage.sceneDir),
        runLocalTime: runStorage.runTiming.humanTime,
        sceneId: trip.sceneId,
        sceneVariant: trip.sceneVariant,
        tripIndex: trip.tripIndex,
        chunkDurationMs: trip.chunkDurationMs,
        syncTime: trip.syncTime,
        endTime: trip.endTime,
        tripSeed: trip.tripProfile?.seed ?? "",
        weatherType: trip.tripProfile?.weatherType ?? "",
        timeOfDay: trip.tripProfile?.timeBucket ?? "",
        time: trip.tripProfile?.time ?? null,
        vehicleModel: trip.tripProfile?.vehicleModel ?? trip.vehicle?.model ?? "",
        vehicleColor: trip.tripProfile?.vehicleColorName ?? "",
        tripProfile: trip.tripProfile ?? null,
        stopSignGoal: trip.stopSignGoal ?? null,
        stopSignOutcome: trip.stopSignOutcome ?? null,
        fromDestination: trip.fromDestination,
        toDestination: trip.toDestination,
        vehicleDataPoints: trip.vehicleData.length,
        telemetrySchemaVersion: 5,
        telemetrySummary,
        videoFile: tripStorage.videoFile,
        logFile: tripStorage.logFile
    };

    fs.writeFileSync(tripStorage.metadataFile, JSON.stringify(tripMetadata, null, 2) + "\n");

    fs.appendFileSync(runFile, JSON.stringify(trip) + "\n");

    const manifestLine = JSON.stringify({
        runId: trip.runId,
        runFolder: runStorage.runTiming.runFolder,
        runDir: runStorage.runsDir,
        runDirRelative: runStorage.runDirRelative,
        sceneFolder: pathBasename(runStorage.sceneDir),
        runLocalTime: runStorage.runTiming.humanTime,
        sceneId: trip.sceneId,
        sceneVariant: trip.sceneVariant,
        tripIndex: trip.tripIndex,
        chunkDurationMs: trip.chunkDurationMs,
        syncTime: trip.syncTime,
        endTime: trip.endTime,
        tripSeed: trip.tripProfile?.seed ?? "",
        weatherType: trip.tripProfile?.weatherType ?? "",
        timeOfDay: trip.tripProfile?.timeBucket ?? "",
        time: trip.tripProfile?.time ?? null,
        vehicleModel: trip.tripProfile?.vehicleModel ?? trip.vehicle?.model ?? "",
        vehicleColor: trip.tripProfile?.vehicleColorName ?? "",
        tripProfile: trip.tripProfile ?? null,
        stopSignGoal: trip.stopSignGoal ?? null,
        stopSignOutcome: trip.stopSignOutcome ?? null,
        fromDestination: trip.fromDestination,
        toDestination: trip.toDestination,
        vehicleDataPoints: trip.vehicleData.length,
        telemetrySchemaVersion: 5,
        telemetrySummary,
        file: runFile,
        tripDir: tripStorage.tripDir,
        videoFile: tripStorage.videoFile,
        logFile: tripStorage.logFile,
        metadataFile: tripStorage.metadataFile
    }) + "\n";

    fs.appendFileSync(manifestFile, manifestLine);
    pendingTrips.delete(tripKey);
    seenTripTelemetry.delete(tripKey);
    activeTripByStream.delete(buildTripStreamKey(trip.runId, trip.sceneId, trip.sceneVariant));
    const finalizedAt = Date.now();
    const response: TripFinalizeResponse = {
        requestId: "",
        success: true,
        runId: trip.runId,
        sceneId: trip.sceneId,
        sceneVariant: trip.sceneVariant,
        tripIndex: trip.tripIndex,
        tripKey,
        tripDir: tripStorage.tripDir,
        runFile,
        metadataFile: tripStorage.metadataFile,
        sampleCount: trip.vehicleData.length
    };
    finalizedTripResults.set(tripKey, {
        response,
        finalizedAt
    });
    console.log(`[server] flush success trip=${tripKey} stored tripIndex=${trip.tripIndex} run=${trip.runId} samples=${trip.vehicleData.length} runFile=${runFile}`);
    return response;
}

function pathBasename(targetPath: string): string {
    const path = require("path");
    return path.basename(targetPath);
}

function pruneFinalizedTripResults(now = Date.now()) {
    for (const [tripKey, finalized] of finalizedTripResults.entries()) {
        if (now - finalized.finalizedAt > flushedTripRetentionMs) {
            finalizedTripResults.delete(tripKey);
        }
    }
}
