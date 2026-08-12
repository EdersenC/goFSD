// src/client.ts
import {SceneManager, SceneType} from "./sceneManger";
import {log} from "./helper";
import {
    innerCityDrivingScenes,
    type InnerCityDrivingSceneVariant
} from "./datasets/inner-city-driving";
import {CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS} from "./controlTelemetry";
import {resolveStopCommandStatus} from "./control-status";
import {parseStopSignJobs} from "./stop-sign/batch";
import {STOP_SIGN_SCENE_NAME} from "./stop-sign/runner";
import {StopSignOutcome, StopSignPose} from "./stop-sign/types";
import {setStopSignCatalogWaypoint} from "./stop-sign/catalog-waypoint";
import {
    canAcknowledgeControlSafetyEpochSync,
    ControlSafetyBarrier,
    ControlSafetyEpochSync,
    parseControlSafetyEpochSync,
    runGuardedSafetyStart,
    SafetyStartLease,
} from "./control-safety";
import {
    applyJoinedPlayerSetup,
    applyPlayerGodMode,
    disablePlayerGodMode,
    type PlayerSetupOperations,
} from "./player-setup";
import {
    teleportEntityToWaypoint,
    type TeleportPosition,
    type WaypointTeleportOperations,
} from "./waypoint-teleport";

const CLIENT_BUILD_ID = "2026-08-11-streamed-stop-sign-probe-v17";
log(`[client] loaded build=${CLIENT_BUILD_ID}`);



const sceneManager = new SceneManager();
const playerSetupOperations: PlayerSetupOperations = {
    playerId: () => PlayerId(),
    playerPedId: () => PlayerPedId(),
    entityExists: (entity) => DoesEntityExist(entity),
    setPlayerInvincible: (player, enabled) => SetPlayerInvincible(player, enabled),
    setEntityInvincible: (entity, enabled) => SetEntityInvincible(entity, enabled),
    setEntityCanBeDamaged: (entity, enabled) => SetEntityCanBeDamaged(entity, enabled),
    setPedDropsWeaponsWhenDead: (ped, enabled) => SetPedDropsWeaponsWhenDead(ped, enabled),
    setPedInfiniteAmmoClip: (ped, enabled) => SetPedInfiniteAmmoClip(ped, enabled),
    hashWeaponName: (weaponName) => GetHashKey(weaponName),
    isWeaponValid: (weaponHash) => IsWeaponValid(weaponHash),
    giveWeaponToPed: (ped, weaponHash, ammo) => GiveWeaponToPed(ped, weaponHash, ammo, false, false),
    setPedInfiniteAmmo: (ped, enabled, weaponHash) => SetPedInfiniteAmmo(ped, enabled, weaponHash),
};
const innerCitySceneNames = Object.keys(innerCityDrivingScenes) as InnerCityDrivingSceneVariant[];
const canonicalInnerCitySceneName = "inner-city-driving:default";
const innerCitySceneBaseLabel = "Inner City Driving";

type ControlCommandType =
    | "startScene"
    | "runAllScenes"
    | "endScene"
    | "endAllScenes"
    | "startEgo"
    | "stopEgo"
    | "setStopSignTarget"
    | "clearStopSignTarget"
    | "setStopSignCatalogWaypoint"
    | "probeStopSignTarget"
    | "teleportStopSignStart"
    | "startStopSignBatch";
type ControlRuntimeStatus = "idle" | "runningScene" | "runningAllScenes" | "stopping" | "error";
type ControlStopSignBatchProgress = {
    batchId: string
    planFingerprint: string
    state: "running" | "completed" | "stopped" | "failed"
    jobId?: string
    jobIndex: number
    jobCount: number
    completedJobs: number
    startedAtMs: number
    updatedAtMs: number
    attemptIndex: number
    attemptCount: number
    phase: string
    lastAttemptOutcome?: StopSignOutcome
};

type ControlCommand = {
    id: string
    type: ControlCommandType
    safetyEpoch: number
    sceneName?: string
    planFingerprint?: string
    stopSignBatchId?: string
    stopSignJobs?: unknown
    stopSignCatalogPosition?: unknown
    stopSignProbe?: unknown
    stopSignStartPose?: unknown
}

type AvailableScene = {
    name: string
    label: string
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
    stopSignConfirmationElapsedMs: number
    stopSignConfirmationTargetMs: number
    stopSignStopped: boolean
    stopSignAttemptIndex: number
    stopSignAttemptCount: number
    timestampMs: number
    gameTimeMs: number
}

const CONTROL_REGISTER_INTERVAL_MS = 5000;
let stopSignBatchActive = false;
let stopSignBatchStopRequested = false;
const controlSafetyBarrier = new ControlSafetyBarrier();

type ReportedControlStatus = {
    status: ControlRuntimeStatus
    activeSceneName: string
    lastError: string
    stopSignBatch?: ControlStopSignBatchProgress
};

let safetyStatusSequence = 0;
let pendingControlSafetySyncAck: {
    sync: ControlSafetyEpochSync
    cleanupComplete: boolean
} | null = null;
let acknowledgedControlSafetySync: ControlSafetyEpochSync | null = null;
let reportedControlStatus: ReportedControlStatus = {
    status: "idle",
    activeSceneName: "",
    lastError: "",
};

function registerInnerCityScenes() {
    for (const variant of innerCitySceneNames) {
        const sceneName = variant === "default"
            ? canonicalInnerCitySceneName
            : `inner-city-driving:${variant}`;
        sceneManager.addScene(sceneName, innerCityDrivingScenes[variant]);
    }
}

function humanizeSceneVariant(variant: string): string {
    return variant
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        .replace(/^\w/, (match) => match.toUpperCase());
}

function buildAvailableScenes(): AvailableScene[] {
    return innerCitySceneNames.map((variant) => ({
        name: variant === "default"
            ? canonicalInnerCitySceneName
            : `inner-city-driving:${variant}`,
        label: `${innerCitySceneBaseLabel} - ${humanizeSceneVariant(variant)}`
    }));
}

function publishAvailableScenes() {
    emitNet("control:availableScenesResponse", buildAvailableScenes());
}

function registerControlClient(reason: string) {
    log(`[client] registering control client session (${reason})`);
    emitNet("control:registerClient");
}

function sendControlClientHeartbeat() {
    emitNet("control:clientHeartbeat");
}

function publishTelemetry(update: ControlTelemetryUpdate) {
    emitNet("control:telemetryUpdate", update);
}

function reportControlStatus(
    status: ControlRuntimeStatus,
    activeSceneName = "",
    lastError = "",
    stopSignBatch?: ControlStopSignBatchProgress,
) {
    reportedControlStatus = {status, activeSceneName, lastError, stopSignBatch};
    publishControlStatus();
}

function publishControlStatus() {
    const safety = controlSafetyBarrier.snapshot();
    emitNet("control:statusUpdate", {
        ...reportedControlStatus,
        status: safety.hasStaleSafetyStarts ? "stopping" : reportedControlStatus.status,
        appliedSafetyEpoch: safety.appliedSafetyEpoch,
        inFlightSafetyStarts: safety.inFlightSafetyStarts,
        safetyStatusSequence: ++safetyStatusSequence,
    });
    acknowledgeControlSafetySyncIfSettled(safety);
}

function acknowledgeControlSafetySyncIfSettled(
    safety = controlSafetyBarrier.snapshot(),
) {
    const pending = pendingControlSafetySyncAck;
    if (pending === null || !canAcknowledgeControlSafetyEpochSync(pending.sync, pending.cleanupComplete, safety)) {
        return;
    }

    pendingControlSafetySyncAck = null;
    acknowledgedControlSafetySync = pending.sync;
    emitNet("control:safetyEpochSyncAck", pending.sync);
    log(
        `[client] applied control safety epoch ${pending.sync.safetyEpoch} `
        + `for session ${pending.sync.sessionId}`,
    );
}

async function executeSceneByName(sceneName: string, safetyStart?: SafetyStartLease) {
    safetyStart?.throwIfStale();
    reportControlStatus("runningScene", sceneName);
    try {
        await sceneManager.executeScene(sceneName, {
            isCanceled: () => safetyStart?.isStale() ?? false,
        });
        safetyStart?.throwIfStale();
        reportControlStatus("idle");
    } catch (error: any) {
        const message = error?.message ?? `Failed to execute scene "${sceneName}"`;
        reportControlStatus("error", sceneName, message);
        throw error;
    }
}

async function executeAllScenesControlled(safetyStart?: SafetyStartLease) {
    safetyStart?.throwIfStale();
    reportControlStatus("runningAllScenes");
    try {
        await sceneManager.executeAllScenes(() => safetyStart?.isStale() ?? false);
        safetyStart?.throwIfStale();
        reportControlStatus("idle");
    } catch (error: any) {
        const message = error?.message ?? "Failed to execute all scenes";
        reportControlStatus("error", "", message);
        throw error;
    }
}

async function executeEgoControl(safetyStart?: SafetyStartLease) {
    safetyStart?.throwIfStale();
    reportControlStatus("runningScene", "ego-control");
    log("[client] starting ego control");
    try {
        await sceneManager.startEgoControl(() => safetyStart?.isStale() ?? false);
        safetyStart?.throwIfStale();
        log("[client] ego control started");
    } catch (error: any) {
        const message = error?.message ?? "Failed to start ego control";
        reportControlStatus("error", "ego-control", message);
        log(`[client] ego control failed: ${message}`);
        throw error;
    }
}

function stopEgoControl() {
    const stopSignOperationActive = requestStopSignWorkStop();
    const stoppedSynchronously = sceneManager.endAllScenes();
    reportControlStatus(resolveStopCommandStatus(stoppedSynchronously, stopSignOperationActive));
}

function setStopSignTarget() {
    const target = sceneManager.setStopSignTarget();
    reportControlStatus("runningScene", "ego-control");
    log(
        `[stop-sign] target ready sign=(${target.signPose.x.toFixed(2)}, ${target.signPose.y.toFixed(2)}) `
        + `line=(${target.stopLinePose.x.toFixed(2)}, ${target.stopLinePose.y.toFixed(2)}) `
        + `exit=(${target.exitPose.x.toFixed(2)}, ${target.exitPose.y.toFixed(2)})`
    );
}

function clearStopSignTarget() {
    sceneManager.clearStopSignTarget();
    reportControlStatus("runningScene", "ego-control");
}

async function executeStopSignProbe(command: ControlCommand, safetyStart: SafetyStartLease) {
    safetyStart.throwIfStale();
    reportControlStatus("runningScene", "stop-sign-probe");
    const result = await sceneManager.probeStopSignTarget(
        command.stopSignProbe,
        () => safetyStart.throwIfStale(),
    );
    safetyStart.throwIfStale();
    reportControlStatus("runningScene", "ego-control");
    log(
        `[stop-sign] probe ready id=${result.probe.catalogId} `
        + `nodeDistance=${result.probe.distanceFromCatalogM.toFixed(2)}m `
        + `pose=(${result.target.signPose.x.toFixed(2)}, ${result.target.signPose.y.toFixed(2)}, `
        + `${result.target.signPose.z.toFixed(2)}, h=${result.target.signPose.heading.toFixed(1)})`,
    );
}

async function executeStopSignStartTeleport(command: ControlCommand, safetyStart: SafetyStartLease) {
    safetyStart.throwIfStale();
    reportControlStatus("runningScene", "saved-stop-sign-start");
    const pose = await sceneManager.teleportStopSignStart(
        command.stopSignStartPose,
        () => safetyStart.throwIfStale(),
    );
    safetyStart.throwIfStale();
    reportControlStatus("runningScene", "ego-control");
    log(
        `[stop-sign] setup car moved to saved Start `
        + `(${pose.x.toFixed(2)}, ${pose.y.toFixed(2)}, ${pose.z.toFixed(2)}, h=${pose.heading.toFixed(1)})`,
    );
}

async function executeStopSignBatchControl(
    batchIdValue: unknown,
    planFingerprintValue: unknown,
    jobsValue: unknown,
    safetyStart?: SafetyStartLease,
) {
    safetyStart?.throwIfStale();
    if (stopSignBatchActive) {
        throw new Error("A stop-sign collection batch is already active");
    }
    const batchId = typeof batchIdValue === "string" ? batchIdValue.trim() : "";
    if (!batchId) {
        throw new Error("startStopSignBatch command missing stopSignBatchId");
    }
    const planFingerprint = typeof planFingerprintValue === "string" ? planFingerprintValue.trim() : "";
    if (!/^sha256:[0-9a-f]{64}$/.test(planFingerprint)) {
        throw new Error("startStopSignBatch command missing a valid planFingerprint");
    }
    const jobs = parseStopSignJobs(jobsValue);
    stopSignBatchActive = true;
    stopSignBatchStopRequested = false;
    const startedAtMs = Date.now();
    let currentJobId = "";
    let currentJobIndex = 0;
    let completedJobs = 0;
    let attemptIndex = 0;
    let attemptCount = 0;
    let lastAttemptOutcome: StopSignOutcome | undefined;
    const progress = (state: ControlStopSignBatchProgress["state"]): ControlStopSignBatchProgress => ({
        batchId,
        planFingerprint,
        state,
        jobId: currentJobId || undefined,
        jobIndex: currentJobIndex,
        jobCount: jobs.length,
        completedJobs,
        attemptIndex,
        attemptCount,
        phase: sceneManager.currentEgoControlTelemetry().stopSignPhase,
        lastAttemptOutcome: lastAttemptOutcome ? {...lastAttemptOutcome} : undefined,
        startedAtMs,
        updatedAtMs: Date.now(),
    });
    const publishProgress = (state: ControlStopSignBatchProgress["state"]) => {
        reportControlStatus(
            state === "running" ? "runningAllScenes" : (state === "failed" ? "error" : "runningScene"),
            state === "running" ? `stop-sign-batch:${batchId}` : "ego-control",
            "",
            progress(state),
        );
    };
    publishProgress("running");
    try {
        completedJobs = await sceneManager.startStopSignBatch(batchId, jobs, {
            stopRequested: () => stopSignBatchStopRequested || (safetyStart?.isStale() ?? false),
            onJobStart: (job, jobIndex) => {
                safetyStart?.throwIfStale();
                currentJobId = job.id;
                currentJobIndex = jobIndex;
                attemptIndex = 0;
                attemptCount = job.attemptCount;
                publishProgress("running");
            },
            onAttemptComplete: (_job, completedAttempt, outcome) => {
                attemptIndex = completedAttempt;
                lastAttemptOutcome = {...outcome};
                publishProgress("running");
            },
            onJobComplete: (_job, jobIndex) => {
                completedJobs = jobIndex;
                publishProgress("running");
            },
        });
        safetyStart?.throwIfStale();
        log(`[stop-sign] batch=${batchId} completedJobs=${completedJobs}`);
        publishProgress(stopSignBatchStopRequested ? "stopped" : "completed");
    } catch (error: any) {
        const message = error?.message ?? `Failed to execute stop-sign batch "${batchId}"`;
        reportControlStatus("error", `stop-sign-batch:${batchId}`, message, progress("failed"));
        throw error;
    } finally {
        stopSignBatchActive = false;
        stopSignBatchStopRequested = false;
    }
}

function requestEndScene() {
    const stopSignOperationActive = requestStopSignWorkStop();
    const stoppedSynchronously = sceneManager.endScene();
    reportControlStatus(resolveStopCommandStatus(stoppedSynchronously, stopSignOperationActive));
}

function requestEndAllScenes() {
    const stopSignOperationActive = requestStopSignWorkStop();
    const stoppedSynchronously = sceneManager.endAllScenes();
    reportControlStatus(resolveStopCommandStatus(stoppedSynchronously, stopSignOperationActive));
}

function requestStopSignWorkStop(): boolean {
    if (stopSignBatchActive) {
        stopSignBatchStopRequested = true;
    }
    return stopSignBatchActive;
}

function requireCommandSafetyEpoch(command: ControlCommand): number {
    if (!Number.isSafeInteger(command.safetyEpoch) || command.safetyEpoch < 1) {
        throw new Error(`Control command ${command.id || "unknown"} is missing a valid safety epoch`);
    }
    return command.safetyEpoch;
}

function applyEmergencyCommandEpoch(command: ControlCommand) {
    controlSafetyBarrier.applyEmergencyStop(requireCommandSafetyEpoch(command));
    publishControlStatus();
}

function acceptNonStartCommand(command: ControlCommand): boolean {
    const accepted = controlSafetyBarrier.acceptNonStartCommand(requireCommandSafetyEpoch(command));
    publishControlStatus();
    if (!accepted) {
        log(`[client] ignored stale control command id=${command.id} epoch=${command.safetyEpoch}`);
    }
    return accepted;
}

function cleanupInterruptedSafetyStart() {
    requestStopSignWorkStop();
    sceneManager.forceSafeCleanup("safety epoch superseded the active start");
    reportControlStatus("idle");
}

async function executeGuardedControlStart(
    command: ControlCommand,
    work: (safetyStart: SafetyStartLease) => Promise<void>,
) {
    const result = await runGuardedSafetyStart(
        controlSafetyBarrier,
        requireCommandSafetyEpoch(command),
        work,
        cleanupInterruptedSafetyStart,
        publishControlStatus,
    );
    if (result.kind === "canceled") {
        log(`[client] canceled stale control start id=${command.id} type=${command.type} epoch=${command.safetyEpoch}`);
    }
}

registerInnerCityScenes();
registerControlClient("startup");
reportControlStatus("idle");
publishAvailableScenes();

setInterval(() => {
    sendControlClientHeartbeat();
}, CONTROL_REGISTER_INTERVAL_MS);

let lastTelemetrySentAt = 0;
let lastRouteForwardDelta = 0;
let lastRouteHeadingError = 0;
let lastRouteDistance = 0;
let lastLeadVehicleDistance = 100;
let lastTelemetryDebugLogAt = 0;
setTick(() => {
    const now = GetGameTimer();
    if (now-lastTelemetrySentAt < CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS) {
        return;
    }
    const telemetry = sceneManager.currentEgoControlTelemetry();
    const routeForwardDelta = telemetry.routeForwardDelta ?? lastRouteForwardDelta;
    const routeHeadingError = telemetry.routeHeadingError ?? lastRouteHeadingError;
    const routeDistance = telemetry.routeDistance ?? lastRouteDistance;
    const leadVehicleDistance = telemetry.hasLeadVehicle
        ? (telemetry.leadVehicleDistance ?? lastLeadVehicleDistance)
        : 100;
    if (telemetry.routeForwardDelta !== null) {
        lastRouteForwardDelta = telemetry.routeForwardDelta;
    }
    if (telemetry.routeHeadingError !== null) {
        lastRouteHeadingError = telemetry.routeHeadingError;
    }
    if (telemetry.routeDistance !== null) {
        lastRouteDistance = telemetry.routeDistance;
    }
    if (telemetry.hasLeadVehicle && telemetry.leadVehicleDistance !== null) {
        lastLeadVehicleDistance = telemetry.leadVehicleDistance;
    }
    lastTelemetrySentAt = now;
    publishTelemetry({
        currentSpeed: telemetry.currentSpeed,
        currentYaw: telemetry.currentYaw,
        yawRate: telemetry.yawRate,
        steering: telemetry.steering,
        acceleration: telemetry.acceleration,
        brakePressureAvg: telemetry.brakePressureAvg,
        vehicleModelHash: telemetry.vehicleModelHash,
        vehicleExists: telemetry.vehicleExists,
        isInVehicle: telemetry.isInVehicle,
        positionX: telemetry.positionX,
        positionY: telemetry.positionY,
        positionZ: telemetry.positionZ,
        velocityX: telemetry.velocityX,
        velocityY: telemetry.velocityY,
        velocityZ: telemetry.velocityZ,
        pitchDeg: telemetry.pitchDeg,
        rollDeg: telemetry.rollDeg,
        gear: telemetry.gear,
        rpm: telemetry.rpm,
        wheelAngle: telemetry.wheelAngle,
        wheelSteeringFullLock: telemetry.wheelSteeringFullLock,
        onGround: telemetry.onGround,
        collisionState: telemetry.collisionState,
        routeDirectionCode: telemetry.routeDirectionCode,
        routeDirectionDistanceM: telemetry.routeDirectionDistanceM,
        routeDirectionUnknown: telemetry.routeDirectionUnknown,
        routeDirectionKeepStraight: telemetry.routeDirectionKeepStraight,
        routeDirectionTurnLeft: telemetry.routeDirectionTurnLeft,
        routeDirectionTurnRight: telemetry.routeDirectionTurnRight,
        routeDirectionRerouteWrongWay: telemetry.routeDirectionRerouteWrongWay,
        routeForwardDelta,
        routeHeadingError,
        routeDistance,
        leadVehicleDistance,
        hasLeadVehicle: telemetry.hasLeadVehicle,
        stopSignTargetConfigured: telemetry.stopSignTargetConfigured,
        stopSignPose: telemetry.stopSignPose,
        stopLinePose: telemetry.stopLinePose,
        stopSignEgoStopPose: telemetry.stopSignEgoStopPose,
        stopSignDistanceM: telemetry.stopSignDistanceM,
        stopLineDistanceM: telemetry.stopLineDistanceM,
        stopSignLongitudinalErrorM: telemetry.stopSignLongitudinalErrorM,
        stopSignLateralErrorM: telemetry.stopSignLateralErrorM,
        stopSignHeadingErrorDeg: telemetry.stopSignHeadingErrorDeg,
        stopSignPhase: telemetry.stopSignPhase,
        stopSignConfirmationElapsedMs: telemetry.stopSignConfirmationElapsedMs,
        stopSignConfirmationTargetMs: telemetry.stopSignConfirmationTargetMs,
        stopSignStopped: telemetry.stopSignStopped,
        stopSignAttemptIndex: telemetry.stopSignAttemptIndex,
        stopSignAttemptCount: telemetry.stopSignAttemptCount,
        timestampMs: Date.now(),
        gameTimeMs: telemetry.gameTimeMs,
    });
    if (now - lastTelemetryDebugLogAt >= 2000) {
        lastTelemetryDebugLogAt = now;
        console.log(
            `[client] telemetry publish speed=${telemetry.currentSpeed.toFixed(2)} ` +
            `yawRate=${telemetry.yawRate.toFixed(2)} ` +
            `steer=${telemetry.steering.toFixed(2)} ` +
            `accel=${telemetry.acceleration.toFixed(2)} ` +
            `brakeAvg=${telemetry.brakePressureAvg.toFixed(2)} ` +
            `navCode=${telemetry.routeDirectionCode} ` +
            `navOneHot=[u=${Number(telemetry.routeDirectionUnknown || 0).toFixed(0)}, ` +
            `s=${Number(telemetry.routeDirectionKeepStraight || 0).toFixed(0)}, ` +
            `l=${Number(telemetry.routeDirectionTurnLeft || 0).toFixed(0)}, ` +
            `r=${Number(telemetry.routeDirectionTurnRight || 0).toFixed(0)}, ` +
            `w=${Number(telemetry.routeDirectionRerouteWrongWay || 0).toFixed(0)}] ` +
            `routeFwd=${routeForwardDelta.toFixed(2)} ` +
            `routeHeading=${routeHeadingError.toFixed(2)} ` +
            `routeDistance=${routeDistance.toFixed(2)} ` +
            `hasLead=${String(telemetry.hasLeadVehicle)} ` +
            `leadDistance=${leadVehicleDistance.toFixed(2)}`
        );
    }
});

RegisterCommand("startScene", async (_source: number, args: string[]) => {
    const requestedVariant = args[0] as InnerCityDrivingSceneVariant | undefined;
    const variant = requestedVariant && requestedVariant in innerCityDrivingScenes
        ? requestedVariant
        : "default";
    const sceneName = variant === "default"
        ? canonicalInnerCitySceneName
        : `inner-city-driving:${variant}`;

    await executeSceneByName(sceneName);
}, false);

RegisterCommand("runAllScenes", async () => {
    await executeAllScenesControlled();
}, false);

RegisterCommand("endScene", () => {
    requestEndScene();
}, false);

RegisterCommand("endAllScenes", () => {
    requestEndAllScenes();
}, false);

RegisterCommand("startEgo", async (_source: number, args: string[]) => {
    await executeEgoControl();
}, false);

RegisterCommand("stopEgo", () => {
    stopEgoControl();
}, false);

RegisterCommand("setStopSignTarget", () => {
    try {
        setStopSignTarget();
    } catch (error: any) {
        log(`[stop-sign] target calibration failed: ${error?.message ?? error}`);
    }
}, false);

RegisterCommand("clearStopSignTarget", () => {
    try {
        clearStopSignTarget();
    } catch (error: any) {
        log(`[stop-sign] target clear failed: ${error?.message ?? error}`);
    }
}, false);

RegisterCommand("listSceneVariants", () => {
    const variants = innerCitySceneNames.join(", ");
    emit("chat:addMessage", {
        args: ["Scenes", variants],
    });
}, false);

onNet("demo:responseScenes", async (scene: SceneType) => {
    sceneManager.addScene("remote-scene", scene);
    console.log(scene)
    // sceneManager.shuffleWaypoints("remote-scene");
    await executeAllScenesControlled()
});

onNet("control:executeCommand", async (command: ControlCommand) => {
    try {
        log(`[client] received control command id=${command?.id ?? "unknown"} type=${command?.type ?? "unknown"} scene=${command?.sceneName ?? ""}`);
        switch (command?.type) {
            case "startScene":
                if (!command.sceneName) {
                    throw new Error("startScene command missing sceneName");
                }
                await executeGuardedControlStart(command, (safetyStart) => (
                    executeSceneByName(command.sceneName!, safetyStart)
                ));
                break;
            case "runAllScenes":
                await executeGuardedControlStart(command, executeAllScenesControlled);
                break;
            case "endScene":
                applyEmergencyCommandEpoch(command);
                requestEndScene();
                break;
            case "endAllScenes":
                applyEmergencyCommandEpoch(command);
                requestEndAllScenes();
                break;
            case "startEgo":
                await executeGuardedControlStart(command, executeEgoControl);
                break;
            case "stopEgo":
                applyEmergencyCommandEpoch(command);
                stopEgoControl();
                break;
            case "setStopSignTarget":
                if (!acceptNonStartCommand(command)) break;
                setStopSignTarget();
                break;
            case "clearStopSignTarget":
                if (!acceptNonStartCommand(command)) break;
                clearStopSignTarget();
                break;
            case "setStopSignCatalogWaypoint": {
                if (!acceptNonStartCommand(command)) break;
                const position = setStopSignCatalogWaypoint(command.stopSignCatalogPosition, {
                    setNewWaypoint: (x, y) => SetNewWaypoint(x, y),
                });
                log(`[client] stop-sign catalog waypoint set x=${position.x.toFixed(2)} y=${position.y.toFixed(2)} z=${position.z.toFixed(2)}; use /tpwaypoint to travel`);
                break;
            }
            case "probeStopSignTarget":
                await executeGuardedControlStart(command, (safetyStart) => executeStopSignProbe(command, safetyStart));
                break;
            case "teleportStopSignStart":
                await executeGuardedControlStart(command, (safetyStart) => executeStopSignStartTeleport(command, safetyStart));
                break;
            case "startStopSignBatch":
                await executeGuardedControlStart(command, (safetyStart) => (
                    executeStopSignBatchControl(
                        command.stopSignBatchId,
                        command.planFingerprint,
                        command.stopSignJobs,
                        safetyStart,
                    )
                ));
                break;
            default:
                throw new Error(`Unsupported control command: ${command?.type ?? "unknown"}`);
        }
    } catch (error: any) {
        const message = error?.message ?? "Failed to execute control command";
        reportControlStatus("error", command?.sceneName ?? "", message);
        log(`[client] command failed: ${message}`);
    }
});

onNet("control:safetyEpochSync", (payload: unknown) => {
    try {
        const sync = parseControlSafetyEpochSync(payload);
        if (isSameControlSafetySync(sync, acknowledgedControlSafetySync)) {
            emitNet("control:safetyEpochSyncAck", sync);
            return;
        }
        if (isSameControlSafetySync(sync, pendingControlSafetySyncAck?.sync ?? null)) {
            acknowledgeControlSafetySyncIfSettled();
            return;
        }
        if (sync.safetyEpoch < controlSafetyBarrier.snapshot().appliedSafetyEpoch) {
            throw new Error(`Control safety epoch ${sync.safetyEpoch} is stale`);
        }

        pendingControlSafetySyncAck = {sync, cleanupComplete: false};
        controlSafetyBarrier.applyEmergencyStop(sync.safetyEpoch);
        publishControlStatus();
        cleanupInterruptedSafetyStart();
        if (isSameControlSafetySync(sync, pendingControlSafetySyncAck?.sync ?? null)) {
            pendingControlSafetySyncAck.cleanupComplete = true;
        }
        acknowledgeControlSafetySyncIfSettled();
    } catch (error: any) {
        log(`[client] rejected control safety epoch synchronization: ${error?.message ?? error}`);
    }
});

function isSameControlSafetySync(
    left: ControlSafetyEpochSync,
    right: ControlSafetyEpochSync | null,
): boolean {
    return right !== null
        && left.requestId === right.requestId
        && left.sessionId === right.sessionId
        && left.safetyEpoch === right.safetyEpoch;
}

onNet("control:requestAvailableScenes", () => {
    publishAvailableScenes();
});

on("onClientResourceStop", (resourceName: string) => {
    if (resourceName !== GetCurrentResourceName()) {
        return;
    }
    clearTick(playerGodModeTickId);
    sceneManager.forceSafeCleanup("FiveM client resource stopped");
    sceneManager.shutdownWorldIsolation();
    disablePlayerGodMode(playerSetupOperations);
});

let playerSetupGeneration = 0;

async function applyJoinedPlayerSetupWhenReady() {
    const generation = ++playerSetupGeneration;
    for (let attempt = 0; attempt < 20; attempt += 1) {
        if (generation !== playerSetupGeneration) {
            return;
        }
        const result = applyJoinedPlayerSetup(playerSetupOperations);
        if (result !== null) {
            console.log(`[player-setup] god mode enabled and ${result.grantedWeaponCount} weapons granted`);
            return;
        }
        await wait(250);
    }
    console.error("[player-setup] player ped did not become ready after spawn");
}

on("playerSpawned", () => {
    void applyJoinedPlayerSetupWhenReady();
});

on("onClientResourceStart", (resourceName: string) => {
    if (resourceName === GetCurrentResourceName()) {
        void applyJoinedPlayerSetupWhenReady();
    }
});

let nextPlayerGodModeRefreshAt = 0;
const playerGodModeTickId = setTick(() => {
    const now = GetGameTimer();
    if (now < nextPlayerGodModeRefreshAt) {
        return;
    }
    nextPlayerGodModeRefreshAt = now + 500;
    applyPlayerGodMode(playerSetupOperations);
});

setTimeout(() => {
    void applyJoinedPlayerSetupWhenReady();
}, 500);

RegisterCommand("coords", () => {
    const [x, y, z] = GetEntityCoords(PlayerPedId(), true) as [number, number, number];
    const formatted = `${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}`;

    console.log(`[coords] ${formatted}`);
    emit("chat:addMessage", {
        args: ["Coords", formatted],
    });
}, false);

const waypointTeleportOperations: WaypointTeleportOperations = {
    getEntityCoords: (entity) => GetEntityCoords(entity, true) as TeleportPosition,
    getEntityHeading: (entity) => GetEntityHeading(entity),
    freezeEntity: (entity, frozen) => FreezeEntityPosition(entity, frozen),
    setEntityCoords: (entity, [x, y, z]) => SetEntityCoordsNoOffset(entity, x, y, z, false, false, true),
    setEntityHeading: (entity, heading) => SetEntityHeading(entity, heading),
    setFocus: ([x, y, z]) => SetFocusPosAndVel(x, y, z, 0, 0, 0),
    clearFocus: () => ClearFocus(),
    requestCollision: ([x, y, z]) => RequestCollisionAtCoord(x, y, z),
    getGroundZ: ([x, y, z]) => GetGroundZFor_3dCoord(x, y, z, false),
    getClosestVehicleNode: ([x, y, z]) => {
        const [found, position, heading] = GetClosestVehicleNodeWithHeading(x, y, z, 1, 3, 0);
        return [found, position as TeleportPosition, heading];
    },
    hasCollisionLoadedAroundEntity: (entity) => HasCollisionLoadedAroundEntity(entity),
    setVehicleOnGround: (vehicle) => SetVehicleOnGroundProperly(vehicle),
    wait,
};

let waypointTeleportInProgress = false;

async function teleportToWaypoint() {
    if (waypointTeleportInProgress) {
        emit("chat:addMessage", {args: ["Teleport", "A waypoint teleport is already in progress"]});
        return;
    }
    const waypointBlip = GetFirstBlipInfoId(8);
    if (!waypointBlip || !DoesBlipExist(waypointBlip)) {
        emit("chat:addMessage", {
            args: ["Teleport", "Set a waypoint first"],
        });
        return;
    }

    const ped = PlayerPedId();
    const inVehicle = IsPedInAnyVehicle(ped, false);
    const vehicle = inVehicle ? GetVehiclePedIsIn(ped, false) : 0;
    const isDriver = vehicle !== 0 && GetPedInVehicleSeat(vehicle, -1) === ped;
    const entityToMove = isDriver ? vehicle : ped;
    const [x, y] = GetBlipInfoIdCoord(waypointBlip) as unknown as TeleportPosition;

    waypointTeleportInProgress = true;
    try {
        const result = await teleportEntityToWaypoint(
            entityToMove,
            isDriver,
            [x, y, 0],
            waypointTeleportOperations,
        );
        if (!result.success) {
            emit("chat:addMessage", {args: ["Teleport", result.reason ?? "Teleport failed"]});
            return;
        }
        const fallbackMessage = result.usedVehicleNodeFallback ? " (nearest safe road)" : "";
        const collisionMessage = result.collisionLoaded ? "" : "; collision may still be streaming";
        emit("chat:addMessage", {
            args: ["Teleport", `Teleported to waypoint${fallbackMessage}${collisionMessage}`],
        });
    } finally {
        waypointTeleportInProgress = false;
    }
}

RegisterCommand("tpwaypoint", () => {
    void teleportToWaypoint();
}, false);

RegisterCommand("tp", () => {
    void teleportToWaypoint();
}, false);

function wait(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
