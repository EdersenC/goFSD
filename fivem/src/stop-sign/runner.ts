import {requestCapture, requestTripFinalize} from "../captureControl";
import {CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS} from "../controlTelemetry";
import {
    DrivingStyle,
    Ego,
    EgoService,
    VehicleColor,
    VehicleModel,
} from "../egoService";
import {WeatherType} from "../environment";
import {isValidEntity, wait} from "../helper";
import {requestExpertInputNeutralization} from "../expertControl";
import {syncFlash} from "../syncFlash";
import {TripProfileSnapshot} from "../tripProfiles";
import {
    classifyStopSignPhase,
    planStopSignExpert,
    STOP_SIGN_CONTROL_INTERVAL_MS,
    STOP_SIGN_GO_DISTANCE_M,
    STOP_SIGN_STOP_POSITION_TOLERANCE_M,
    STOP_SIGN_STOP_SPEED_MPS,
} from "./expert";
import {gtaForwardVector, poseBehind, relativeStopLinePose} from "./geometry";
import {
    StopSignBehaviorPhase,
    StopSignGoal,
    StopSignJob,
    StopSignOutcome,
    StopSignPose,
    StopSignRuntimePhase,
    StopSignTelemetry,
} from "./types";

export const STOP_SIGN_SCENE_ID = "stop-sign";
export const STOP_SIGN_SCENE_VARIANT = "temporal-v1";
export const STOP_SIGN_SCENE_NAME = `${STOP_SIGN_SCENE_ID}:${STOP_SIGN_SCENE_VARIANT}`;

const captureWarmupMs = 750;
const maximumAttemptDurationMs = 90_000;
const spawnSettleMs = 1000;

export type StopSignBatchHooks = {
    stopRequested: () => boolean
    onJobStart?: (job: StopSignJob, jobIndex: number) => void
    onJobComplete?: (job: StopSignJob, jobIndex: number) => void
    onAttemptComplete?: (job: StopSignJob, attemptIndex: number, outcome: StopSignOutcome) => void
};

export class StopSignRunner {
    private target: {signPose: StopSignPose; stopLinePose: StopSignPose; egoStopPose: StopSignPose} | null = null;
    private phase: StopSignRuntimePhase = "idle";
    private running = false;
    private localStopRequested = false;
    private attemptIndex = 0;
    private attemptCount = 0;
    private latestTelemetry = emptyTelemetry();
    private targetDwellMs = 5000;
    private evaluationArmed = false;
    private evaluationDwellStartedAtMs: number | null = null;

    constructor(private readonly egoService: EgoService) {}

    calibrateTargetFromCurrentEgo(stopDistanceM = 3, egoCenterOffsetM = 2.5) {
        if (this.running) {
            throw new Error("Cannot calibrate a stop sign while collection is active");
        }
        const ego = this.egoService.oldEgo;
        if (!ego || !isValidEntity(ego.vehicle.id)) {
            throw new Error("A valid setup car is required before marking a stop sign");
        }
        const signPose = entityPose(ego.vehicle.id);
        if (!signPose) {
            throw new Error("Could not read the setup car pose for stop-sign calibration");
        }
        const speedMps = GetEntitySpeed(ego.vehicle.id);
        if (!Number.isFinite(speedMps) || speedMps > STOP_SIGN_STOP_SPEED_MPS) {
            throw new Error("Stop-sign calibration requires a stationary setup car");
        }
        const stopLinePose = poseBehind(signPose, stopDistanceM);
        const egoStopPose = poseBehind(stopLinePose, egoCenterOffsetM);
        return this.configureTarget(signPose, stopLinePose, egoStopPose);
    }

    configureTarget(signPose: StopSignPose, stopLinePose: StopSignPose, egoStopPose: StopSignPose, dwellMs = 5000) {
        this.target = {
            signPose: clonePose(signPose),
            stopLinePose: clonePose(stopLinePose),
            egoStopPose: clonePose(egoStopPose),
        };
        this.phase = "idle";
        this.targetDwellMs = Math.max(500, Math.trunc(dwellMs));
        this.evaluationArmed = false;
        this.evaluationDwellStartedAtMs = null;
        this.latestTelemetry = telemetryForTarget(this.target, this.phase, 0, 0, null, 0, this.targetDwellMs);
        return cloneTarget(this.target);
    }

    clearTarget() {
        if (this.running) {
            throw new Error("Cannot clear the stop-sign target while collection is active");
        }
        this.target = null;
        this.phase = "idle";
        this.evaluationArmed = false;
        this.evaluationDwellStartedAtMs = null;
        this.latestTelemetry = emptyTelemetry();
    }

    isRunning() {
        return this.running;
    }

    requestStop() {
        if (!this.running) {
            return;
        }
        this.localStopRequested = true;
        this.phase = "stopping";
        const vehicle = this.egoService.oldEgo?.vehicle.id ?? 0;
        holdVehicle(vehicle);
    }

    currentTelemetry(): StopSignTelemetry {
        this.refreshIdleEvaluationTelemetry();
        return {...this.latestTelemetry};
    }

    async runBatch(batchId: string, jobs: StopSignJob[], hooks: StopSignBatchHooks): Promise<number> {
        if (this.running) {
            throw new Error("A stop-sign batch is already active");
        }
        if (!batchId.trim() || jobs.length === 0) {
            throw new Error("Stop-sign collection requires a batch id and at least one job");
        }
        await requestExpertInputNeutralization();
        const runId = createRunId(batchId);
        let completedJobs = 0;
        let tripIndex = 0;
        this.running = true;
        this.localStopRequested = false;
        try {
            for (const [jobOffset, job] of jobs.entries()) {
                if (this.stopWasRequested(hooks)) {
                    break;
                }
                hooks.onJobStart?.(job, jobOffset + 1);
                this.configureTarget(job.signPose, job.stopLinePose, job.egoStopPose, job.dwellMs);
                applyEnvironment(job);
                const ego = buildStopSignEgo(job);
                this.phase = "spawning";
                await this.egoService.executeEgoAt(
                    ego,
                    toSpawnPoint(job.startPose),
                    STOP_SIGN_SCENE_NAME,
                    runId,
                    () => this.stopWasRequested(hooks),
                );
                requireEgo(ego);
                applyVehicleVariant(ego.vehicle.id, job);
                await ensurePlayerIsDriver(ego.vehicle.id);
                SetEntityMaxSpeed(ego.vehicle.id, Math.max(job.targetSpeedMps + 2, 12));
                SetVehicleMaxSpeed(ego.vehicle.id, Math.max(job.targetSpeedMps + 2, 12));

                for (let attempt = 1; attempt <= job.attemptCount; attempt += 1) {
                    if (this.stopWasRequested(hooks)) {
                        break;
                    }
                    this.attemptIndex = attempt;
                    this.attemptCount = job.attemptCount;
                    const outcome = await this.runAttempt(ego, job, runId, tripIndex);
                    hooks.onAttemptComplete?.(job, attempt, outcome);
                    tripIndex += 1;
                    await wait(200);
                }
                if (this.stopWasRequested(hooks)) {
                    break;
                }
                completedJobs += 1;
                hooks.onJobComplete?.(job, jobOffset + 1);
            }
            return completedJobs;
        } finally {
            holdVehicle(this.egoService.oldEgo?.vehicle.id ?? 0);
            this.egoService.releaseCurrentEgoForManualControl();
            this.running = false;
            this.localStopRequested = false;
            if (this.phase === "stopping") {
                this.phase = "failed";
            }
        }
    }

    private async runAttempt(ego: Ego, job: StopSignJob, runId: string, tripIndex: number): Promise<StopSignOutcome> {
        await resetVehicleAtPose(ego.vehicle.id, job.startPose);
        await ensurePlayerIsDriver(ego.vehicle.id);
        const goal = buildGoal(job, this.attemptIndex);
        const capturePayload = {
            runId,
            tripIndex,
            sceneId: STOP_SIGN_SCENE_ID,
            sceneVariant: STOP_SIGN_SCENE_VARIANT,
            sceneName: STOP_SIGN_SCENE_NAME,
        };
        let captureStarted = false;
        let captureStopped = false;
        try {
            await requestCapture("start", capturePayload);
            captureStarted = true;
            await wait(captureWarmupMs);
            const syncTime = await syncFlash(750);
            this.egoService.beginStopSignAttemptRecording(ego, {
                ...capturePayload,
                syncTime,
                fromDestination: poseCoords(job.startPose),
                toDestination: poseCoords(job.signPose),
                tripProfile: buildTripProfile(job),
                goal,
            });
            this.phase = "recording";
            const outcome = await this.monitorAttempt(ego, job);
            holdVehicle(ego.vehicle.id);
            this.egoService.finishStopSignAttemptRecording(ego, outcome);
            const finalized = this.egoService.requireFinalizedTrip(
                await requestTripFinalize(capturePayload),
                tripIndex,
                {sceneId: STOP_SIGN_SCENE_ID, sceneVariant: STOP_SIGN_SCENE_VARIANT},
            );
            await requestCapture("stop", {...finalized, sceneName: STOP_SIGN_SCENE_NAME});
            captureStopped = true;
            return outcome;
        } catch (error) {
            this.egoService.clearRecordingContext();
            throw error;
        } finally {
            if (captureStarted && !captureStopped) {
                try {
                    await requestCapture("abort", capturePayload);
                } catch (abortError: any) {
                    console.error(`[stop-sign] capture abort failed: ${abortError?.message ?? abortError}`);
                }
            }
        }
    }

    private async monitorAttempt(ego: Ego, job: StopSignJob): Promise<StopSignOutcome> {
        const startedAtMs = GetGameTimer();
        let dwellStartedAtMs: number | null = null;
        let stoppedAtDistanceM: number | null = null;
        let previousDesiredSpeedMps = 0;
        while (true) {
            const nowMs = GetGameTimer();
            if (this.localStopRequested) {
                return outcome(false, "stopped", "Collection stopped", startedAtMs, nowMs, stoppedAtDistanceM, dwellStartedAtMs, false);
            }
            if (nowMs - startedAtMs > maximumAttemptDurationMs) {
                return outcome(false, "timeout", "Attempt timed out", startedAtMs, nowMs, stoppedAtDistanceM, dwellStartedAtMs, false);
            }
            if (!isValidEntity(ego.vehicle.id)) {
                return outcome(false, "invalid_vehicle", "Vehicle no longer exists", startedAtMs, nowMs, stoppedAtDistanceM, dwellStartedAtMs, false);
            }
            if (HasEntityCollidedWithAnything(ego.vehicle.id)) {
                return outcome(false, "collision", "Vehicle collided during the stop-sign clip", startedAtMs, nowMs, stoppedAtDistanceM, dwellStartedAtMs, false);
            }

            const pose = entityPose(ego.vehicle.id);
            if (!pose) {
                return outcome(false, "invalid_vehicle", "Vehicle pose is unavailable", startedAtMs, nowMs, stoppedAtDistanceM, dwellStartedAtMs, false);
            }
            const speedMps = Math.max(0, GetEntitySpeed(ego.vehicle.id));
            const centerError = relativeStopLinePose(pose, job.egoStopPose);
            const frontDistanceM = signedFrontBumperDistance(ego.vehicle.id, pose, job.stopLinePose);
            const withinStopPose = Math.abs(centerError.longitudinalM) <= STOP_SIGN_STOP_POSITION_TOLERANCE_M
                && Math.abs(centerError.lateralM) <= 0.8;
            if (dwellStartedAtMs === null && frontDistanceM < -0.1) {
                return outcome(false, "crossed_without_stop", "Front bumper crossed the stop line before the dwell", startedAtMs, nowMs, stoppedAtDistanceM, null, true);
            }
            if (dwellStartedAtMs === null && speedMps <= STOP_SIGN_STOP_SPEED_MPS && -centerError.longitudinalM > 3) {
                return outcome(false, "stopped_too_early", "Vehicle stopped more than 3 m before the ego stop pose", startedAtMs, nowMs, stoppedAtDistanceM, null, false);
            }
            if (dwellStartedAtMs === null && withinStopPose && speedMps <= STOP_SIGN_STOP_SPEED_MPS) {
                dwellStartedAtMs = nowMs;
                stoppedAtDistanceM = frontDistanceM;
            }

            const dwellElapsedMs = dwellStartedAtMs === null ? 0 : nowMs - dwellStartedAtMs;
            let phase: StopSignBehaviorPhase;
            if (dwellStartedAtMs !== null && dwellElapsedMs < job.dwellMs) {
                phase = "stop_hold";
            } else if (dwellStartedAtMs !== null) {
                phase = "release";
            } else {
                phase = classifyStopSignPhase(
                    Math.max(0, -centerError.longitudinalM),
                    speedMps,
                    job.targetSpeedMps,
                );
            }
            this.phase = phase;
            this.latestTelemetry = telemetryForTarget(
                {signPose: job.signPose, stopLinePose: job.stopLinePose, egoStopPose: job.egoStopPose},
                phase,
                this.attemptIndex,
                this.attemptCount,
                centerError,
                dwellElapsedMs,
                job.dwellMs,
                frontDistanceM,
            );
            const supervision = planStopSignExpert({
                phase,
                remainingDistanceM: Math.max(0, -centerError.longitudinalM),
                measuredSpeedMps: speedMps,
                targetSpeedMps: job.targetSpeedMps,
                previousDesiredSpeedMps,
                dtSeconds: STOP_SIGN_CONTROL_INTERVAL_MS / 1000,
                lateralErrorM: centerError.lateralM,
                headingErrorDeg: centerError.headingErrorDeg,
            });
            previousDesiredSpeedMps = supervision.desiredSpeedMps;
            applyExpertControl(ego.vehicle.id, speedMps, supervision);
            this.egoService.collectStopSignData(ego, this.latestTelemetry, supervision);

            if (phase === "release" && centerError.longitudinalM >= STOP_SIGN_GO_DISTANCE_M) {
                this.phase = "complete";
                this.latestTelemetry = {...this.latestTelemetry, stopSignPhase: "complete"};
                return outcome(true, "succeeded", "", startedAtMs, nowMs, stoppedAtDistanceM, dwellStartedAtMs, false);
            }
            await wait(STOP_SIGN_CONTROL_INTERVAL_MS);
        }
    }

    private stopWasRequested(hooks: StopSignBatchHooks) {
        return this.localStopRequested || hooks.stopRequested();
    }

    private refreshIdleEvaluationTelemetry() {
        if (this.running || !this.target || this.phase === "complete") {
            return;
        }
        const vehicle = this.egoService.oldEgo?.vehicle.id ?? 0;
        const pose = entityPose(vehicle);
        if (!pose) {
            return;
        }
        const speedMps = Math.max(0, GetEntitySpeed(vehicle));
        const error = relativeStopLinePose(pose, this.target.egoStopPose);
        const frontDistanceM = signedFrontBumperDistance(vehicle, pose, this.target.stopLinePose);
        if (!this.evaluationArmed && error.longitudinalM <= -5) {
            this.evaluationArmed = true;
        }

        const nowMs = GetGameTimer();
        let dwellElapsedMs = this.evaluationDwellStartedAtMs === null
            ? 0
            : Math.max(0, nowMs - this.evaluationDwellStartedAtMs);
        const withinStopPose = Math.abs(error.longitudinalM) <= STOP_SIGN_STOP_POSITION_TOLERANCE_M
            && Math.abs(error.lateralM) <= 0.8;
        if (this.evaluationArmed && this.evaluationDwellStartedAtMs === null && withinStopPose && speedMps <= STOP_SIGN_STOP_SPEED_MPS) {
            this.evaluationDwellStartedAtMs = nowMs;
            dwellElapsedMs = 0;
        }

        if (!this.evaluationArmed) {
            this.phase = "idle";
        } else if (this.evaluationDwellStartedAtMs !== null && dwellElapsedMs < this.targetDwellMs) {
            this.phase = "stop_hold";
        } else if (this.evaluationDwellStartedAtMs !== null && error.longitudinalM < STOP_SIGN_GO_DISTANCE_M) {
            this.phase = "release";
        } else if (this.evaluationDwellStartedAtMs !== null) {
            this.phase = "complete";
        } else if (frontDistanceM < -0.1) {
            this.phase = "failed";
        } else {
            this.phase = classifyStopSignPhase(Math.max(0, -error.longitudinalM), speedMps, 8);
        }
        this.latestTelemetry = telemetryForTarget(
            this.target,
            this.phase,
            0,
            0,
            error,
            dwellElapsedMs,
            this.targetDwellMs,
            frontDistanceM,
        );
    }
}

function applyExpertControl(vehicle: number, measuredSpeedMps: number, supervision: {
    phase: StopSignBehaviorPhase
    desiredWheelSteerNormalized: number
    throttle: number
    brake: number
}) {
    SetVehicleSteerBias(vehicle, supervision.desiredWheelSteerNormalized);
    if (supervision.phase === "stop_hold") {
        SetVehicleForwardSpeed(vehicle, 0);
        SetVehicleBrake(vehicle, true);
        SetVehicleHandbrake(vehicle, true);
        return;
    }
    SetVehicleHandbrake(vehicle, false);
    SetVehicleBrake(vehicle, supervision.brake > 0.01);
    const dtSeconds = STOP_SIGN_CONTROL_INTERVAL_MS / 1000;
    const nextSpeedMps = Math.max(
        0,
        measuredSpeedMps + supervision.throttle * 1.8 * dtSeconds - supervision.brake * 3 * dtSeconds,
    );
    SetVehicleForwardSpeed(vehicle, nextSpeedMps);
}

async function resetVehicleAtPose(vehicle: number, pose: StopSignPose) {
    if (!isValidEntity(vehicle)) {
        throw new Error("Cannot reset an invalid stop-sign vehicle");
    }
    SetEntityRecordsCollisions(vehicle, false);
    FreezeEntityPosition(vehicle, true);
    holdVehicle(vehicle);
    RequestCollisionAtCoord(pose.x, pose.y, pose.z);
    SetEntityCoordsNoOffset(vehicle, pose.x, pose.y, pose.z + 0.75, false, false, true);
    SetEntityHeading(vehicle, pose.heading);
    SetEntityVelocity(vehicle, 0, 0, 0);
    SetVehicleEngineOn(vehicle, true, true, false);
    SetVehicleUndriveable(vehicle, false);
    SetVehicleOnGroundProperly(vehicle);
    await wait(spawnSettleMs);
    SetVehicleOnGroundProperly(vehicle);
    SetEntityRecordsCollisions(vehicle, true);
    FreezeEntityPosition(vehicle, false);
    SetVehicleHandbrake(vehicle, false);
}

async function ensurePlayerIsDriver(vehicle: number) {
    const player = PlayerPedId();
    const deadline = GetGameTimer() + 1500;
    while (GetGameTimer() <= deadline) {
        if (IsPedInVehicle(player, vehicle, false) && GetPedInVehicleSeat(vehicle, -1) === player) {
            return;
        }
        TaskWarpPedIntoVehicle(player, vehicle, -1);
        SetPedIntoVehicle(player, vehicle, -1);
        await wait(50);
    }
    throw new Error("Could not seat the player in the stop-sign collection vehicle");
}

function buildStopSignEgo(job: StopSignJob): Ego {
    return {
        vehicle: {
            id: 0,
            model: job.vehicle.model || VehicleModel.Sultan,
            color: VehicleColor.Blue,
            maxSpeed: job.targetSpeedMps,
            drivingStyle: DrivingStyle.Cautious,
        },
        waypoints: [],
    };
}

function applyEnvironment(job: StopSignJob) {
    SetWeatherTypeNowPersist(job.weather);
    NetworkOverrideClockTime(job.time.hour, job.time.minute, 0);
    PauseClock(true);
}

function applyVehicleVariant(vehicle: number, job: StopSignJob) {
    const color = job.vehicle.color;
    if (!color) {
        return;
    }
    SetVehicleCustomPrimaryColour(vehicle, color.r, color.g, color.b);
    SetVehicleCustomSecondaryColour(vehicle, color.r, color.g, color.b);
}

function buildTripProfile(job: StopSignJob): TripProfileSnapshot {
    return {
        seed: job.seed,
        weatherType: job.weather as WeatherType,
        time: {hour: job.time.hour, minute: job.time.minute, second: 0, persistent: true},
        timeBucket: `${String(job.time.hour).padStart(2, "0")}:${String(job.time.minute).padStart(2, "0")}`,
        vehicleModel: job.vehicle.model || VehicleModel.Sultan,
        vehicleColor: VehicleColor.Blue,
        vehicleColorName: job.vehicle.color
            ? `rgb(${job.vehicle.color.r},${job.vehicle.color.g},${job.vehicle.color.b})`
            : "Blue",
    };
}

function buildGoal(job: StopSignJob, attemptIndex: number): StopSignGoal {
    return {
        task: "stop-sign",
        contract: "stop-sign-goal.v1",
        releasePolicy: "scripted_dwell_release_v0",
        signPose: clonePose(job.signPose),
        stopLinePose: clonePose(job.stopLinePose),
        egoStopPose: clonePose(job.egoStopPose),
        startPose: clonePose(job.startPose),
        targetSpeedMps: job.targetSpeedMps,
        dwellMs: job.dwellMs,
        attemptIndex,
        attemptCount: job.attemptCount,
        seed: job.seed,
        variationId: job.variationId,
    };
}

function telemetryForTarget(
    target: {signPose: StopSignPose; stopLinePose: StopSignPose; egoStopPose: StopSignPose},
    phase: StopSignRuntimePhase,
    attemptIndex: number,
    attemptCount: number,
    error: ReturnType<typeof relativeStopLinePose> | null,
    dwellElapsedMs: number,
    dwellTargetMs: number,
    frontDistanceM = 0,
): StopSignTelemetry {
    return {
        stopSignTargetConfigured: true,
        stopSignPose: clonePose(target.signPose),
        stopLinePose: clonePose(target.stopLinePose),
        stopSignEgoStopPose: clonePose(target.egoStopPose),
        stopSignDistanceM: Math.hypot(target.signPose.x - target.egoStopPose.x, target.signPose.y - target.egoStopPose.y),
        stopLineDistanceM: frontDistanceM,
        stopSignLongitudinalErrorM: error?.longitudinalM ?? 0,
        stopSignLateralErrorM: error?.lateralM ?? 0,
        stopSignHeadingErrorDeg: error?.headingErrorDeg ?? 0,
        stopSignPhase: phase,
        stopSignDwellElapsedMs: dwellElapsedMs,
        stopSignDwellTargetMs: dwellTargetMs,
        stopSignStopped: phase === "stop_hold",
        stopSignAttemptIndex: attemptIndex,
        stopSignAttemptCount: attemptCount,
    };
}

function emptyTelemetry(): StopSignTelemetry {
    return {
        stopSignTargetConfigured: false,
        stopSignDistanceM: 0,
        stopLineDistanceM: 0,
        stopSignLongitudinalErrorM: 0,
        stopSignLateralErrorM: 0,
        stopSignHeadingErrorDeg: 0,
        stopSignPhase: "idle",
        stopSignDwellElapsedMs: 0,
        stopSignDwellTargetMs: 0,
        stopSignStopped: false,
        stopSignAttemptIndex: 0,
        stopSignAttemptCount: 0,
    };
}

function signedFrontBumperDistance(vehicle: number, pose: StopSignPose, stopLine: StopSignPose): number {
    const [, max] = GetModelDimensions(GetEntityModel(vehicle));
    const frontOverhangM = Array.isArray(max) && Number.isFinite(Number(max[1])) ? Math.max(1, Number(max[1])) : 2.5;
    const [forwardX, forwardY] = gtaForwardVector(pose.heading);
    const frontPose = {
        ...pose,
        x: pose.x + forwardX * frontOverhangM,
        y: pose.y + forwardY * frontOverhangM,
    };
    return -relativeStopLinePose(frontPose, stopLine).longitudinalM;
}

function entityPose(entity: number): StopSignPose | null {
    if (!isValidEntity(entity)) {
        return null;
    }
    const coords = GetEntityCoords(entity, false) as [number, number, number];
    const heading = GetEntityHeading(entity);
    if (!Array.isArray(coords) || coords.length < 3 || !coords.every(Number.isFinite) || !Number.isFinite(heading)) {
        return null;
    }
    return {x: coords[0], y: coords[1], z: coords[2], heading};
}

function holdVehicle(vehicle: number) {
    if (!isValidEntity(vehicle)) {
        return;
    }
    SetVehicleForwardSpeed(vehicle, 0);
    SetVehicleBrake(vehicle, true);
    SetVehicleHandbrake(vehicle, true);
    SetVehicleSteerBias(vehicle, 0);
}

function requireEgo(ego: Ego) {
    if (!isValidEntity(ego.vehicle.id)) {
        throw new Error("Stop-sign collection vehicle is invalid");
    }
}

function outcome(
    success: boolean,
    status: StopSignOutcome["status"],
    failureReason: string,
    startedAtMs: number,
    nowMs: number,
    stoppedAtDistanceM: number | null,
    dwellStartedAtMs: number | null,
    crossedStopLineBeforeDwell: boolean,
): StopSignOutcome {
    return {
        success,
        status,
        failureReason,
        durationMs: Math.max(0, nowMs - startedAtMs),
        stoppedAtDistanceM,
        dwellDurationMs: dwellStartedAtMs === null ? 0 : Math.max(0, nowMs - dwellStartedAtMs),
        crossedStopLineBeforeDwell,
    };
}

function toSpawnPoint(pose: StopSignPose) {
    return {coords: poseCoords(pose), heading: pose.heading};
}

function poseCoords(pose: StopSignPose): [number, number, number] {
    return [pose.x, pose.y, pose.z];
}

function clonePose(pose: StopSignPose): StopSignPose {
    return {...pose};
}

function cloneTarget(target: {signPose: StopSignPose; stopLinePose: StopSignPose; egoStopPose: StopSignPose}) {
    return {
        signPose: clonePose(target.signPose),
        stopLinePose: clonePose(target.stopLinePose),
        egoStopPose: clonePose(target.egoStopPose),
    };
}

function createRunId(batchId: string) {
    const now = new Date();
    const stamp = [now.getFullYear(), now.getMonth() + 1, now.getDate(), now.getHours(), now.getMinutes(), now.getSeconds()]
        .map((value) => String(value).padStart(2, "0"))
        .join("-");
    return `stop-sign_${stamp}_${sanitizeId(batchId)}_${Math.random().toString(36).slice(2, 8)}`;
}

function sanitizeId(value: string) {
    return value.trim().replace(/[^a-zA-Z0-9_-]+/g, "-").slice(0, 40) || "batch";
}
