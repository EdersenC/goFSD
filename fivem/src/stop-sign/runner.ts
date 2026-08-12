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
import {gtaForwardVector, poseAhead, poseBehind, relativeStopLinePose} from "./geometry";
import {
    advanceEarlyStopProgress,
    nextFixedIntervalDeadlineMs,
    validateStopSignAttemptVehicleState,
} from "./attempt-progress";
import {
    StopSignBehaviorPhase,
    StopSignGoal,
    StopSignJob,
    StopSignOutcome,
    StopSignPose,
    StopSignRuntimePhase,
    StopSignTelemetry,
} from "./types";
import {positionVehicleAtStopSignPose} from "./vehicle-positioning";

export const STOP_SIGN_SCENE_ID = "stop-sign";
export const STOP_SIGN_SCENE_VARIANT = "continuous-v2";
export const STOP_SIGN_SCENE_NAME = `${STOP_SIGN_SCENE_ID}:${STOP_SIGN_SCENE_VARIANT}`;

const captureWarmupMs = 750;
const maximumAttemptDurationMs = 90_000;
const exitWaypointLeadM = 125;

export type StopSignBatchHooks = {
    stopRequested: () => boolean
    onJobStart?: (job: StopSignJob, jobIndex: number) => void
    onJobComplete?: (job: StopSignJob, jobIndex: number) => void
    onAttemptComplete?: (job: StopSignJob, attemptIndex: number, outcome: StopSignOutcome) => void
};

export class StopSignRunner {
    private target: {signPose: StopSignPose; stopLinePose: StopSignPose; egoStopPose: StopSignPose; exitPose: StopSignPose} | null = null;
    private phase: StopSignRuntimePhase = "idle";
    private running = false;
    private localStopRequested = false;
    private attemptIndex = 0;
    private attemptCount = 0;
    private latestTelemetry = emptyTelemetry();
    private targetStopConfirmationMs = 250;
    private evaluationArmed = false;
    private evaluationStopStartedAtMs: number | null = null;

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
        const exitPose = poseAhead(signPose, STOP_SIGN_GO_DISTANCE_M);
        return this.configureTarget(signPose, stopLinePose, egoStopPose, 250, exitPose);
    }

    configureTarget(
        signPose: StopSignPose,
        stopLinePose: StopSignPose,
        egoStopPose: StopSignPose,
        stopConfirmationMs = 250,
        exitPose = poseAhead(signPose, STOP_SIGN_GO_DISTANCE_M),
    ) {
        this.target = {
            signPose: clonePose(signPose),
            stopLinePose: clonePose(stopLinePose),
            egoStopPose: clonePose(egoStopPose),
            exitPose: clonePose(exitPose),
        };
        this.phase = "idle";
        this.targetStopConfirmationMs = Math.max(100, Math.trunc(stopConfirmationMs));
        this.evaluationArmed = false;
        this.evaluationStopStartedAtMs = null;
        this.latestTelemetry = telemetryForTarget(this.target, this.phase, 0, 0, null, 0, this.targetStopConfirmationMs);
        return cloneTarget(this.target);
    }

    clearTarget() {
        if (this.running) {
            throw new Error("Cannot clear the stop-sign target while collection is active");
        }
        this.target = null;
        this.phase = "idle";
        this.evaluationArmed = false;
        this.evaluationStopStartedAtMs = null;
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
                this.configureTarget(job.signPose, job.stopLinePose, job.egoStopPose, job.stopConfirmationMs, job.exitPose);
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
        const routeWaypoint = poseAhead(job.exitPose, exitWaypointLeadM);
        SetNewWaypoint(routeWaypoint.x, routeWaypoint.y);
        await prepareAttemptVehicle(this.egoService, ego, job.startPose);
        return this.recordContinuousAttempt(ego, job, runId, tripIndex);
    }

    private async recordContinuousAttempt(
        ego: Ego,
        job: StopSignJob,
        runId: string,
        tripIndex: number,
    ): Promise<StopSignOutcome> {
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
                toDestination: poseCoords(job.exitPose),
                tripProfile: buildTripProfile(job),
                goal,
            });
            this.phase = "recording";
            const outcome = await this.monitorContinuousAttempt(ego, job);
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

    private async monitorContinuousAttempt(ego: Ego, job: StopSignJob): Promise<StopSignOutcome> {
        const startedAtMs = GetGameTimer();
        let brakingStartedAtMs: number | null = null;
        let stoppedAtMs: number | null = null;
        let stopConfirmedAtMs: number | null = null;
        let releaseStartedAtMs: number | null = null;
        let stoppedAtDistanceM: number | null = null;
        let departureObserved = false;
        let previousDesiredSpeedMps = 0;
        let nextControlDeadlineMs = startedAtMs;
        while (true) {
            const nowMs = GetGameTimer();
            const failure = attemptFailure(ego.vehicle.id, startedAtMs, nowMs, this.localStopRequested, job.stopConfirmationMs);
            if (failure) return failure;
            const pose = entityPose(ego.vehicle.id);
            if (!pose) return outcome(false, "invalid_vehicle", "Vehicle pose is unavailable", startedAtMs, nowMs, null, null, job.stopConfirmationMs, false);
            const speedMps = Math.max(0, GetEntitySpeed(ego.vehicle.id));
            const centerError = relativeStopLinePose(pose, job.egoStopPose);
            const startError = relativeStopLinePose(pose, job.startPose);
            const exitError = relativeStopLinePose(pose, job.exitPose);
            const remainingStopDistanceM = Math.max(0, -centerError.longitudinalM);
            const frontDistanceM = signedFrontBumperDistance(ego.vehicle.id, pose, job.stopLinePose);
            if (releaseStartedAtMs === null && stoppedAtMs === null && frontDistanceM < -0.1) {
                return outcome(false, "crossed_without_stop", "Front bumper crossed the stop line", startedAtMs, nowMs, null, null, job.stopConfirmationMs, true);
            }
            const earlyStopProgress = advanceEarlyStopProgress({
                departureObserved,
                speedMps,
                distanceFromStartM: startError.distanceM,
                stopStartedAtMs: stoppedAtMs,
                remainingDistanceM: remainingStopDistanceM,
            });
            departureObserved = earlyStopProgress.departureObserved;
            if (releaseStartedAtMs === null && earlyStopProgress.stoppedTooEarly) {
                return outcome(false, "stopped_too_early", "Vehicle stopped more than 3 m before Stop", startedAtMs, nowMs, null, null, job.stopConfirmationMs, false);
            }
            const withinStopPose = Math.abs(centerError.longitudinalM) <= STOP_SIGN_STOP_POSITION_TOLERANCE_M
                && Math.abs(centerError.lateralM) <= .8;
            if (releaseStartedAtMs === null && stoppedAtMs === null && withinStopPose && speedMps <= STOP_SIGN_STOP_SPEED_MPS) {
                stoppedAtMs = nowMs;
                stoppedAtDistanceM = frontDistanceM;
            }
            const confirmationMs = stoppedAtMs === null ? 0 : nowMs - stoppedAtMs;
            let phase: StopSignBehaviorPhase;
            if (releaseStartedAtMs !== null) {
                phase = "release";
            } else if (stoppedAtMs !== null && confirmationMs >= job.stopConfirmationMs) {
                stopConfirmedAtMs = nowMs;
                releaseStartedAtMs = nowMs;
                previousDesiredSpeedMps = 0;
                phase = "release";
            } else if (stoppedAtMs !== null) {
                phase = "stop_hold";
            } else {
                phase = classifyStopSignPhase(remainingStopDistanceM, speedMps, job.targetSpeedMps);
                if (phase === "decelerate" && brakingStartedAtMs === null) {
                    brakingStartedAtMs = nowMs;
                }
            }
            this.phase = phase;
            this.latestTelemetry = telemetryForTarget(
                {signPose: job.signPose, stopLinePose: job.stopLinePose, egoStopPose: job.egoStopPose},
                phase,
                this.attemptIndex,
                this.attemptCount,
                centerError,
                confirmationMs,
                job.stopConfirmationMs,
                frontDistanceM,
            );
            const trackingError = phase === "release" ? exitError : centerError;
            const supervision = planStopSignExpert({
                phase,
                remainingDistanceM: Math.max(0, -trackingError.longitudinalM),
                measuredSpeedMps: speedMps,
                targetSpeedMps: job.targetSpeedMps,
                previousDesiredSpeedMps,
                dtSeconds: STOP_SIGN_CONTROL_INTERVAL_MS / 1000,
                lateralErrorM: trackingError.lateralM,
                headingErrorDeg: trackingError.headingErrorDeg,
            });
            previousDesiredSpeedMps = supervision.desiredSpeedMps;
            applyExpertControl(ego.vehicle.id, supervision);
            this.egoService.collectStopSignData(ego, this.latestTelemetry, supervision);
            if (phase === "release" && reachedExitPose(exitError)) {
                this.phase = "complete";
                this.latestTelemetry = {...this.latestTelemetry, stopSignPhase: "complete"};
                return outcome(
                    true,
                    "succeeded",
                    "",
                    startedAtMs,
                    nowMs,
                    stoppedAtDistanceM,
                    stoppedAtMs,
                    job.stopConfirmationMs,
                    false,
                    {
                        approachStartedGameTimeMs: startedAtMs,
                        brakeStopStartedGameTimeMs: brakingStartedAtMs,
                        stopConfirmedGameTimeMs: stopConfirmedAtMs,
                        releaseStartedGameTimeMs: releaseStartedAtMs,
                        completedGameTimeMs: nowMs,
                    },
                );
            }
            nextControlDeadlineMs = nextFixedIntervalDeadlineMs(nextControlDeadlineMs, GetGameTimer(), STOP_SIGN_CONTROL_INTERVAL_MS);
            await wait(Math.max(0, nextControlDeadlineMs - GetGameTimer()));
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
        const exitError = relativeStopLinePose(pose, this.target.exitPose);
        const frontDistanceM = signedFrontBumperDistance(vehicle, pose, this.target.stopLinePose);
        if (!this.evaluationArmed && error.longitudinalM <= -5) {
            this.evaluationArmed = true;
        }

        const nowMs = GetGameTimer();
        let stopConfirmationElapsedMs = this.evaluationStopStartedAtMs === null
            ? 0
            : Math.max(0, nowMs - this.evaluationStopStartedAtMs);
        const withinStopPose = Math.abs(error.longitudinalM) <= STOP_SIGN_STOP_POSITION_TOLERANCE_M
            && Math.abs(error.lateralM) <= 0.8;
        if (this.evaluationArmed && this.evaluationStopStartedAtMs === null && withinStopPose && speedMps <= STOP_SIGN_STOP_SPEED_MPS) {
            this.evaluationStopStartedAtMs = nowMs;
            stopConfirmationElapsedMs = 0;
        }

        if (!this.evaluationArmed) {
            this.phase = "idle";
        } else if (this.evaluationStopStartedAtMs !== null && stopConfirmationElapsedMs < this.targetStopConfirmationMs) {
            this.phase = "stop_hold";
        } else if (this.evaluationStopStartedAtMs !== null && !reachedExitPose(exitError)) {
            this.phase = "release";
        } else if (this.evaluationStopStartedAtMs !== null) {
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
            stopConfirmationElapsedMs,
            this.targetStopConfirmationMs,
            frontDistanceM,
        );
    }
}

function attemptFailure(
    vehicle: number,
    startedAtMs: number,
    nowMs: number,
    stopRequested: boolean,
    stopConfirmationMs: number,
): StopSignOutcome | null {
    if (stopRequested) {
        return outcome(false, "stopped", "Collection stopped", startedAtMs, nowMs, null, null, stopConfirmationMs, false);
    }
    if (nowMs - startedAtMs > maximumAttemptDurationMs) {
        return outcome(false, "timeout", "Continuous attempt timed out", startedAtMs, nowMs, null, null, stopConfirmationMs, false);
    }
    if (!isValidEntity(vehicle)) {
        return outcome(false, "invalid_vehicle", "Vehicle no longer exists", startedAtMs, nowMs, null, null, stopConfirmationMs, false);
    }
    if (HasEntityCollidedWithAnything(vehicle)) {
        return outcome(false, "collision", "Vehicle collided during the continuous attempt", startedAtMs, nowMs, null, null, stopConfirmationMs, false);
    }
    return null;
}

function applyExpertControl(vehicle: number, supervision: {
    phase: StopSignBehaviorPhase
    desiredWheelSteerNormalized: number
    desiredSpeedMps: number
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
    // The planner already rate-limits desiredSpeedMps from its previous command.
    // Applying a second measured-speed limiter here makes gravity win on grades.
    SetVehicleForwardSpeed(vehicle, supervision.desiredSpeedMps);
}

async function prepareAttemptVehicle(egoService: EgoService, ego: Ego, pose: StopSignPose) {
    await positionVehicleAtStopSignPose(ego.vehicle.id, pose);
    await ensurePlayerIsDriver(ego.vehicle.id);
    SetVehicleEngineOn(ego.vehicle.id, true, true, false);
    SetVehicleUndriveable(ego.vehicle.id, false);
    egoService.startCaptureCamera(ego);
    await wait(100);

    const player = PlayerPedId();
    const readinessError = validateStopSignAttemptVehicleState({
        exists: isValidEntity(ego.vehicle.id),
        playerIsDriver: GetPedInVehicleSeat(ego.vehicle.id, -1) === player,
        engineRunning: GetIsVehicleEngineRunning(ego.vehicle.id),
        onAllWheels: IsVehicleOnAllWheels(ego.vehicle.id),
        speedMps: GetEntitySpeed(ego.vehicle.id),
        pitchDeg: GetEntityPitch(ego.vehicle.id),
        rollDeg: GetEntityRoll(ego.vehicle.id),
        collided: HasEntityCollidedWithAnything(ego.vehicle.id),
    });
    if (readinessError) {
        throw new Error(`Stop-sign attempt preflight failed: ${readinessError}`);
    }
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
        contract: "stop-sign-goal.v3",
        releasePolicy: "scripted_continuous_release_v2",
        captureMode: "continuous",
        logicalClipStages: ["approach", "brake_stop", "release"],
        catalogId: job.catalogId,
        catalogPosition: job.catalogPosition ? {...job.catalogPosition} : undefined,
        signPose: clonePose(job.signPose),
        stopLinePose: clonePose(job.stopLinePose),
        egoStopPose: clonePose(job.egoStopPose),
        startPose: clonePose(job.startPose),
        exitPose: clonePose(job.exitPose),
        exitDistanceM: job.exitDistanceM,
        targetSpeedMps: job.targetSpeedMps,
        stopConfirmationMs: job.stopConfirmationMs,
        attemptIndex,
        attemptCount: job.attemptCount,
        seed: job.seed,
        variationId: job.variationId,
    };
}

function reachedExitPose(error: ReturnType<typeof relativeStopLinePose>): boolean {
    const crossedExitPlane = error.longitudinalM >= 0;
    const nearExitPose = error.distanceM <= 1;
    return nearExitPose || (crossedExitPlane && Math.abs(error.lateralM) <= 1.5);
}

function telemetryForTarget(
    target: {signPose: StopSignPose; stopLinePose: StopSignPose; egoStopPose: StopSignPose},
    phase: StopSignRuntimePhase,
    attemptIndex: number,
    attemptCount: number,
    error: ReturnType<typeof relativeStopLinePose> | null,
    stopConfirmationElapsedMs: number,
    stopConfirmationTargetMs: number,
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
        stopSignConfirmationElapsedMs: stopConfirmationElapsedMs,
        stopSignConfirmationTargetMs: stopConfirmationTargetMs,
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
        stopSignConfirmationElapsedMs: 0,
        stopSignConfirmationTargetMs: 0,
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
    stopStartedAtMs: number | null,
    stopConfirmationTargetMs: number,
    crossedStopLineBeforeStop: boolean,
    stageTransitions?: StopSignOutcome["stageTransitions"],
): StopSignOutcome {
    return {
        success,
        status,
        failureReason,
        durationMs: Math.max(0, nowMs - startedAtMs),
        stoppedAtDistanceM,
        stopConfirmationDurationMs: stopStartedAtMs === null
            ? 0
            : Math.min(stopConfirmationTargetMs, Math.max(0, nowMs - stopStartedAtMs)),
        crossedStopLineBeforeStop,
        stageTransitions,
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

function cloneTarget(target: {signPose: StopSignPose; stopLinePose: StopSignPose; egoStopPose: StopSignPose; exitPose: StopSignPose}) {
    return {
        signPose: clonePose(target.signPose),
        stopLinePose: clonePose(target.stopLinePose),
        egoStopPose: clonePose(target.egoStopPose),
        exitPose: clonePose(target.exitPose),
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
