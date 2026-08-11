import {requestCapture, requestTripFinalize} from "../captureControl";
import {
    DrivingStyle,
    Ego,
    EgoService,
    VehicleColor,
    VehicleModel,
} from "../egoService";
import {isValidEntity, wait} from "../helper";
import {syncFlash} from "../syncFlash";
import {requestExpertInputNeutralization} from "../expertControl";
import {CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS} from "../controlTelemetry";
import {
    buildForwardBayCurriculum,
    buildForwardBayEvaluationPlan,
    resolveStraightParkingStartOffset,
    resolveParkingAttemptCount,
} from "./curriculum";
import {
    PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS,
    PARKING_EVALUATION_SPAWN_SETTLE_MS,
    PARKING_POST_FLASH_CLEAN_FRAME_BUFFER_MS,
    suppressParkingVehicleControls,
} from "./controls";
import {gtaForwardVector, isVehicleFootprintInsideBay, relativePose} from "./geometry";
import {ParkingFixtureSet} from "./fixtures";
import {drawParkingTargetMarker, PARKING_TARGET_MARKER_ID} from "./marker";
import {
    clearParkingIsolationNativeZone,
    enforceParkingIsolationThisFrame,
    PARKING_ISOLATION_SWEEP_INTERVAL_MS,
    PersistentParkingIsolationZone,
    sweepParkingIsolation,
} from "./isolation";
import {
    planStraightStopExpert,
    STRAIGHT_STOP_EXPERT_INTERVAL_MS,
} from "./expert";
import {
    emptyParkingState,
    MAX_PARKING_TILT_DEG,
    ParkingEvaluationOutcomeTracker,
    ParkingHazardLatch,
    ParkingOutcomeTracker,
    terminalParkingPhase,
} from "./outcome";
import {
    ParkingAttemptPlan,
    ParkingExpertSupervision,
    ParkingGoal,
    ParkingOutcome,
    ParkingPhase,
    ParkingPose,
    ParkingSetup,
    ParkingState,
    ParkingTarget,
    ParkingTelemetry,
    ParkingVector3,
    VehicleFootprint,
} from "./types";

export const PARKING_SCENE_ID = "parking-forward-bay";
export const PARKING_SCENE_VARIANT = "straight-stop-v2";
export const PARKING_SCENE_NAME = `${PARKING_SCENE_ID}:${PARKING_SCENE_VARIANT}`;
export const PARKING_EVALUATION_SCENE_NAME = "parking-evaluation";
export const PARKING_MAX_SPEED_MPS = 2.22;

const headingConventionDotThreshold = 0.99;
const maximumCalibrationSpeedMps = 0.1;
const reversingSpeedThresholdMps = -0.1;
const maximumStartPosePositionErrorM = 0.3;
const maximumStartPoseHeadingErrorDeg = 2;
const parkingFixturesEnabled = false;
const captureWarmupMs = 1250;
const stableStartSampleCount = 3;

const defaultParkingTarget: Omit<ParkingTarget, "pose"> = {
    bay: {
        widthM: 3.4,
        lengthM: 6.5,
    },
    tolerances: {
        maxCenterDistanceM: 0.65,
        maxHeadingErrorDeg: 5,
        maxSpeedMps: 0.15,
        settledDurationMs: 1000,
    },
};

const fallbackVehicleFootprint: VehicleFootprint = {
    halfWidthM: 1,
    halfLengthM: 2.4,
};

type AttemptHazardMonitor = {
    collisionDetected: () => boolean
    reversingDetected: () => boolean
    stop: () => void
};

type ParkingSteeringLock = {
    set: (desiredWheelSteerNormalized: number) => void
    stop: () => void
};

export class ParkingRunner {
    private target: ParkingTarget | null = null;
    private startPose: ParkingPose | null = null;
    private running = false;
    private stopRequested = false;
    private phase: ParkingPhase = "idle";
    private attemptIndex = 0;
    private attemptCount = 0;
    private latestState: ParkingState = emptyParkingState();
    private populationTickId: number | null = null;
    private playerIsolationTickId: number | null = null;
    private evaluationTickId: number | null = null;
    private targetMarkerTickId: number | null = null;
    private driverSeatGuardTickId: number | null = null;
    private nextIsolationSweepAtMs = 0;
    private nextPlayerIsolationSweepAtMs = 0;
    private worldIsolationActive = false;
    private readonly ownedParkingVehicles = new Set<number>();
    private readonly playerIsolationNativeZone = new PersistentParkingIsolationZone();
    private readonly fixtures: ParkingFixtureSet;

    constructor(private readonly egoService: EgoService) {
        this.fixtures = new ParkingFixtureSet({
            spawn: (plan) => this.egoService.spawnVehicleAt(
                plan.model,
                plan.color as VehicleColor,
                plan.pose
            ),
            protect: (vehicle) => this.egoService.makeVehicleGodMode(vehicle),
        });
        this.startPersistentTrafficIsolationAroundPlayer();
    }

    calibrateTargetFromCurrentEgo(): ParkingTarget {
        if (this.running) {
            throw new Error("Cannot calibrate a parking target while a parking run is active");
        }
        const ego = this.egoService.oldEgo;
        if (!ego || !isValidEntity(ego.vehicle.id)) {
            throw new Error("A valid managed ego vehicle is required; run startEgo before setParkingTarget");
        }

        const pose = entityPose(ego.vehicle.id);
        if (!pose) {
            throw new Error("Unable to read the current ego pose for parking calibration");
        }
        requireStableUserCalibrationVehicle(ego.vehicle.id);
        requireMatchingNativeForwardVector(ego.vehicle.id, pose.heading);
        const target = this.configureTarget(pose);
        console.log(
            `[parking] target calibrated at (${pose.coords.map((value) => value.toFixed(3)).join(", ")}) `
            + `heading=${pose.heading.toFixed(2)} marker=${PARKING_TARGET_MARKER_ID}`
        );
        return target;
    }

    configureTarget(poseValue: ParkingPose): ParkingTarget {
        if (this.running) {
            throw new Error("Cannot configure a parking target while a parking run is active");
        }
        const pose = validatedParkingPose(poseValue, "target");
        this.stopEvaluationState();
        this.startPose = null;
        this.target = {
            pose,
            bay: {...defaultParkingTarget.bay},
            tolerances: {...defaultParkingTarget.tolerances},
        };
        this.startTargetMarker();
        this.rememberCurrentEgoAsOwned();
        this.sweepIsolationNow();
        this.egoService.configureParkingRouteTarget(this.target.pose.coords);
        this.phase = "idle";
        this.latestState = emptyParkingState();
        return cloneTarget(this.target);
    }

    calibrateStartFromCurrentEgo(): ParkingSetup {
        if (this.running) {
            throw new Error("Cannot calibrate a parking start while a parking operation is active");
        }
        if (!this.target) {
            throw new Error("Save the parking end goal before saving the start position");
        }
        const ego = this.egoService.oldEgo;
        if (!ego || !isValidEntity(ego.vehicle.id)) {
            throw new Error("A valid managed ego vehicle is required; run startEgo before setParkingStart");
        }
        const pose = entityPose(ego.vehicle.id);
        if (!pose) {
            throw new Error("Unable to read the current ego pose for parking start calibration");
        }
        requireStableUserCalibrationVehicle(ego.vehicle.id);
        requireMatchingNativeForwardVector(ego.vehicle.id, pose.heading);
        const setup = this.configureStart(pose);
        console.log(
            `[parking] start calibrated at (${pose.coords.map((value) => value.toFixed(3)).join(", ")}) `
            + `offset=[${setup.startOffset.longitudinalM.toFixed(2)}, ${setup.startOffset.lateralM.toFixed(2)}, `
            + `${setup.startOffset.headingDeg.toFixed(2)}]`
        );
        return setup;
    }

    configureStart(poseValue: ParkingPose): ParkingSetup {
        if (this.running) {
            throw new Error("Cannot configure a parking start while a parking operation is active");
        }
        if (!this.target) {
            throw new Error("Configure the parking end goal before configuring the start position");
        }
        const pose = validatedParkingPose(poseValue, "start");
        const startOffset = resolveStraightParkingStartOffset(pose, this.target.pose);
        this.stopEvaluationState();
        this.startPose = clonePose(pose);
        this.rememberCurrentEgoAsOwned();
        this.egoService.configureParkingRouteTarget(this.target.pose.coords);
        this.phase = "ready";
        const ego = this.egoService.oldEgo;
        const footprint = ego && isValidEntity(ego.vehicle.id)
            ? vehicleFootprint(ego.vehicle.id)
            : fallbackVehicleFootprint;
        this.latestState = parkingStateAtPose(
            this.target,
            this.startPose,
            footprint
        );
        return {
            target: cloneTarget(this.target),
            startPose: clonePose(this.startPose),
            startOffset: {...startOffset},
        };
    }

    clearTarget() {
        if (this.running) {
            throw new Error("Cannot clear the parking target while a parking run is active");
        }
        this.stopEvaluationState();
        this.stopTargetMarker();
        this.ownedParkingVehicles.clear();
        this.target = null;
        this.startPose = null;
        this.egoService.clearParkingRouteTarget();
        this.phase = "idle";
        this.attemptIndex = 0;
        this.attemptCount = 0;
        this.latestState = emptyParkingState();
        console.log("[parking] start and end setup cleared");
    }

    hasTarget(): boolean {
        return this.target !== null;
    }

    hasStart(): boolean {
        return this.startPose !== null;
    }

    hasSetup(): boolean {
        return this.target !== null && this.startPose !== null;
    }

    isRunning(): boolean {
        return this.running;
    }

    startPersistentTrafficIsolationAroundPlayer() {
        const center = playerPosition();
        this.rememberCurrentEgoAsOwned();
        if (this.playerIsolationTickId === null) {
            this.playerIsolationTickId = setTick(() => {
                const currentCenter = playerPosition();
                if (!currentCenter) {
                    return;
                }
                enforceParkingIsolationThisFrame(currentCenter);
                const nowMs = GetGameTimer();
                if (nowMs < this.nextPlayerIsolationSweepAtMs) {
                    return;
                }
                this.nextPlayerIsolationSweepAtMs = nowMs + PARKING_ISOLATION_SWEEP_INTERVAL_MS;
                this.playerIsolationNativeZone.sync(currentCenter);
                this.sweepIsolationAt(currentCenter);
            });
        }

        if (!center) {
            return {removedVehicles: 0, removedPeds: 0};
        }
        enforceParkingIsolationThisFrame(center);
        this.playerIsolationNativeZone.sync(center);
        const initialSweep = this.sweepIsolationAt(center);
        this.nextPlayerIsolationSweepAtMs = GetGameTimer() + PARKING_ISOLATION_SWEEP_INTERVAL_MS;
        return initialSweep;
    }

    stopPersistentTrafficIsolation() {
        if (this.playerIsolationTickId !== null) {
            clearTick(this.playerIsolationTickId);
            this.playerIsolationTickId = null;
        }
        this.nextPlayerIsolationSweepAtMs = 0;
        this.playerIsolationNativeZone.clear();
        this.clearNativeZoneIfIsolationInactive();
    }

    shutdownWorldIsolation() {
        this.stopParkingWorldIsolation();
        this.stopPersistentTrafficIsolation();
        this.stopTargetMarker();
        clearParkingIsolationNativeZone();
    }

    requestStop() {
        if (!this.running) {
            return;
        }
        this.stopRequested = true;
        this.phase = "stopping";
        const ego = this.egoService.oldEgo;
        if (ego && isValidEntity(ego.vehicle.id)) {
            SetVehicleForwardSpeed(ego.vehicle.id, 0);
            SetVehicleHandbrake(ego.vehicle.id, true);
        }
    }

    async run(attemptCountValue: unknown, requestedSeed?: string) {
        if (this.running) {
            throw new Error("A parking run is already active");
        }
        if (!this.target || !this.startPose) {
            throw new Error("Parking setup is incomplete; save the end goal and start position first");
        }

        const attemptCount = resolveParkingAttemptCount(attemptCountValue);
        const runId = createRunId();
        const seed = requestedSeed?.trim() || `${PARKING_SCENE_ID}:${runId}`;
        const target = cloneTarget(this.target);
        const startPose = clonePose(this.startPose);
        const plans = buildForwardBayCurriculum(target.pose, startPose, attemptCount);

        this.running = true;
        this.stopRequested = false;
        try {
            await requestExpertInputNeutralization();
            if (this.stopRequested) {
                this.phase = "failed";
                return;
            }

            this.stopEvaluationState();
            this.attemptCount = attemptCount;
            this.attemptIndex = 0;
            this.latestState = emptyParkingState();
            this.phase = "spawning";
            console.log(`[parking] starting run=${runId} attempts=${attemptCount} seed=${seed}`);
            this.startParkingWorldIsolation({suppressVehicleControls: true});
            ClearGpsPlayerWaypoint();
            this.egoService.disposeCurrentEgo();
            await this.prepareParkingScene(target);
            if (this.stopRequested) {
                throw new Error("Parking collection setup was stopped");
            }
            const ego = buildParkingEgo();
            await this.egoService.executeEgoAt(
                ego,
                startPose,
                PARKING_SCENE_NAME,
                runId,
                () => this.stopRequested,
            );
            requireParkingEgo(ego, "collection");
            if (this.stopRequested) {
                throw new Error("Parking collection setup was stopped");
            }
            await ensurePlayerIsParkingDriver(ego.vehicle.id);
            if (this.stopRequested) {
                throw new Error("Parking collection setup was stopped");
            }
            this.startDriverSeatGuard(ego.vehicle.id);
            configureParkingEgoSpeed(ego.vehicle.id);
            this.egoService.configureParkingRouteTarget(target.pose.coords);
            for (const plan of plans) {
                if (this.stopRequested) {
                    break;
                }
                this.attemptIndex = plan.attemptIndex;
                this.latestState = emptyParkingState();
                const outcome = await this.runAttempt(ego, target, plan, attemptCount, runId, seed);
                console.log(
                    `[parking] attempt=${plan.attemptIndex} status=${outcome.status} `
                    + `distance=${outcome.finalDistance.toFixed(3)} heading=${outcome.finalHeadingError.toFixed(2)}`
                );
                if (this.stopRequested) {
                    break;
                }
                await wait(250);
            }
        } catch (error) {
            this.phase = "failed";
            throw error;
        } finally {
            try {
                this.stopEvaluationMonitor();
                this.stopDriverSeatGuard();
                this.stopParkingWorldIsolation();
                this.egoService.clearParkingRouteTarget();
            } finally {
                try {
                    this.egoService.releaseCurrentEgoForManualControl();
                } finally {
                    this.fixtures.clear();
                    if (this.phase === "stopping") {
                        this.phase = "failed";
                    }
                }
            }
            this.running = false;
            this.stopRequested = false;
        }
        console.log(`[parking] run complete run=${runId}`);
    }

    async prepareEvaluation(requestedSeed?: string): Promise<ParkingGoal> {
        if (this.running) {
            throw new Error("A parking operation is already active");
        }
        if (!this.target || !this.startPose) {
            throw new Error("Parking setup is incomplete; save the end goal and start position first");
        }

        const target = cloneTarget(this.target);
        const evaluation = buildForwardBayEvaluationPlan(
            target.pose,
            clonePose(this.startPose),
            requestedSeed
        );
        const goal = buildParkingGoal(target, evaluation.plan, 1, evaluation.seed);
        const runId = `${PARKING_SCENE_ID}-evaluation`;

        this.running = true;
        this.stopRequested = false;
        try {
            await requestExpertInputNeutralization();
            if (this.stopRequested) {
                throw new Error("Parking evaluation setup was stopped");
            }

            this.stopEvaluationState();
            this.phase = "spawning";
            this.attemptIndex = 0;
            this.attemptCount = 1;
            this.latestState = emptyParkingState();
            this.startParkingWorldIsolation({suppressVehicleControls: false});
            this.egoService.disposeCurrentEgo();
            await this.prepareParkingScene(target);
            if (this.stopRequested) {
                throw new Error("Parking evaluation setup was stopped");
            }

            const ego = buildParkingEgo();
            await this.egoService.executeEgoAt(
                ego,
                evaluation.plan.startPose,
                PARKING_EVALUATION_SCENE_NAME,
                runId,
                () => this.stopRequested,
            );
            requireParkingEgo(ego, "evaluation");
            if (this.stopRequested) {
                throw new Error("Parking evaluation setup was stopped");
            }

            configureParkingEgoSpeed(ego.vehicle.id);
            await resetParkingEgoAtStart(ego.vehicle.id, evaluation.plan.startPose);
            await ensurePlayerIsParkingDriver(ego.vehicle.id);
            this.startDriverSeatGuard(ego.vehicle.id);
            if (this.stopRequested) {
                throw new Error("Parking evaluation setup was stopped");
            }
            this.egoService.configureParkingRouteTarget(target.pose.coords);
            this.latestState = parkingStateAtPose(
                target,
                evaluation.plan.startPose,
                vehicleFootprint(ego.vehicle.id)
            );
            SetEntityRecordsCollisions(ego.vehicle.id, true);
            SetVehicleHandbrake(ego.vehicle.id, false);
            FreezeEntityPosition(ego.vehicle.id, false);
            this.phase = "ready";
            this.startEvaluationMonitor(ego, goal, vehicleFootprint(ego.vehicle.id));
            console.log(
                `[parking] evaluation ready seed=${evaluation.seed} `
                + `offset=[${evaluation.plan.startOffset.longitudinalM.toFixed(2)}, `
                + `${evaluation.plan.startOffset.lateralM.toFixed(2)}, `
                + `${evaluation.plan.startOffset.headingDeg.toFixed(2)}]`
            );
            return goal;
        } catch (error) {
            this.stopEvaluationState();
            this.egoService.disposeCurrentEgo();
            throw error;
        } finally {
            this.running = false;
            this.stopRequested = false;
        }
    }

    stopEvaluation() {
        if (this.running) {
            this.requestStop();
            return;
        }
        this.stopEvaluationState();
    }

    currentTelemetry(): ParkingTelemetry {
        if (!this.target) {
            return emptyParkingTelemetry("idle");
        }

        let state = this.latestState;
        const ego = this.egoService.oldEgo;
        if (!this.running && this.evaluationTickId === null && ego && isValidEntity(ego.vehicle.id)) {
            const pose = entityPose(ego.vehicle.id);
            if (pose) {
                state = parkingStateAtPose(this.target, pose, vehicleFootprint(ego.vehicle.id));
            }
        }
        return telemetryFromState(
            state,
            this.target.pose,
            this.startPose,
            this.phase,
            this.attemptIndex,
            this.attemptCount
        );
    }

    private async runAttempt(
        ego: Ego,
        target: ParkingTarget,
        plan: ParkingAttemptPlan,
        attemptCount: number,
        runId: string,
        seed: string
    ): Promise<ParkingOutcome> {
        this.phase = "spawning";
        requireParkingEgo(ego, `attempt ${plan.attemptIndex}`);
        await resetParkingEgoAtStart(ego.vehicle.id, plan.startPose);
        await ensurePlayerIsParkingDriver(ego.vehicle.id);

        const goal = buildParkingGoal(target, plan, attemptCount, seed);
        const capturePayload = {
            runId,
            tripIndex: plan.attemptIndex,
            sceneId: PARKING_SCENE_ID,
            sceneVariant: PARKING_SCENE_VARIANT,
            sceneName: PARKING_SCENE_NAME,
        };
        const footprint = vehicleFootprint(ego.vehicle.id);
        let hazards: AttemptHazardMonitor | null = null;
        let captureStartRequested = false;
        let captureStopped = false;

        try {
            captureStartRequested = true;
            await requestCapture("start", capturePayload);
            await wait(captureWarmupMs);

            const syncTime = await syncFlash(1000);
            await wait(PARKING_POST_FLASH_CLEAN_FRAME_BUFFER_MS);
            this.egoService.beginParkingAttemptRecording(ego, {
                runId,
                sceneId: PARKING_SCENE_ID,
                sceneVariant: PARKING_SCENE_VARIANT,
                tripIndex: plan.attemptIndex,
                syncTime,
                fromDestination: plan.startPose.coords,
                toDestination: target.pose.coords,
                goal,
            });
            this.phase = "recording";

            const tracker = new ParkingOutcomeTracker(goal, footprint, GetGameTimer());
            SetEntityRecordsCollisions(ego.vehicle.id, true);
            SetVehicleHandbrake(ego.vehicle.id, false);
            FreezeEntityPosition(ego.vehicle.id, false);
            hazards = monitorAttemptHazards(ego.vehicle.id);
            const steeringLock = startParkingSteeringLock(ego.vehicle.id);
            this.phase = this.stopRequested ? "stopping" : "parking";
            let outcome: ParkingOutcome;
            try {
                outcome = await this.monitorAttempt(ego, tracker, hazards, steeringLock);
            } finally {
                steeringLock.stop();
            }

            if (isValidEntity(ego.vehicle.id)) {
                SetVehicleForwardSpeed(ego.vehicle.id, 0);
                SetVehicleHandbrake(ego.vehicle.id, true);
                FreezeEntityPosition(ego.vehicle.id, true);
            }
            this.egoService.finishParkingAttemptRecording(ego, outcome);
            const finalizeResponse = await requestTripFinalize(capturePayload);
            const finalizedTrip = this.egoService.requireFinalizedTrip(
                finalizeResponse,
                plan.attemptIndex,
                {sceneId: PARKING_SCENE_ID, sceneVariant: PARKING_SCENE_VARIANT}
            );
            await requestCapture("stop", {
                ...finalizedTrip,
                sceneName: PARKING_SCENE_NAME,
            });
            captureStopped = true;
            return outcome;
        } catch (error) {
            this.egoService.clearRecordingContext();
            throw error;
        } finally {
            hazards?.stop();
            if (captureStartRequested && !captureStopped) {
                try {
                    await requestCapture("abort", capturePayload);
                } catch (abortError: any) {
                    console.error(
                        `[parking] capture abort failed attempt=${plan.attemptIndex}: `
                        + `${abortError?.message ?? abortError}`
                    );
                }
            }
        }
    }

    private async monitorAttempt(
        ego: Ego,
        tracker: ParkingOutcomeTracker,
        hazards: AttemptHazardMonitor,
        steeringLock: ParkingSteeringLock
    ): Promise<ParkingOutcome> {
        let previousDesiredSpeedMps = 0;
        let previousStepAtMs = GetGameTimer() - STRAIGHT_STOP_EXPERT_INTERVAL_MS;
        while (true) {
            const vehicleValid = isValidEntity(ego.vehicle.id);
            const pose = vehicleValid ? entityPose(ego.vehicle.id) : null;
            const speedMps = vehicleValid ? GetEntitySpeed(ego.vehicle.id) : 0;
            const pitchDeg = vehicleValid ? GetEntityPitch(ego.vehicle.id) : 0;
            const rollDeg = vehicleValid ? GetEntityRoll(ego.vehicle.id) : 0;
            const result = tracker.observe({
                nowMs: GetGameTimer(),
                pose,
                speedMps,
                collision: hazards.collisionDetected(),
                reversing: hazards.reversingDetected(),
                onGround: vehicleValid && IsVehicleOnAllWheels(ego.vehicle.id),
                pitchDeg,
                rollDeg,
                stopRequested: this.stopRequested,
            });
            this.latestState = result.state;
            if (result.outcome) {
                this.egoService.collectEgoData(
                    ego,
                    this.currentTelemetry(),
                    terminalExpertSupervision(result.outcome)
                );
                this.phase = terminalParkingPhase(result.outcome);
                return result.outcome;
            }

            this.phase = result.state.settledDurationMs > 0 ? "settling" : "parking";
            const nowMs = GetGameTimer();
            const command = planStraightStopExpert({
                remainingDistanceM: Math.max(0, -result.state.longitudinalError),
                measuredSpeedMps: Math.max(0, speedMps),
                previousDesiredSpeedMps,
                dtSeconds: Math.max(1, nowMs - previousStepAtMs) / 1000,
                lateralErrorM: result.state.lateralError,
                headingErrorDeg: result.state.headingError,
            });
            previousDesiredSpeedMps = command.desiredSpeedMps;
            previousStepAtMs = nowMs;
            steeringLock.set(command.desiredWheelSteerNormalized);
            applyStraightStopExpertCommand(ego.vehicle.id, command);
            const supervision: ParkingExpertSupervision = {
                desiredWheelSteerNormalized: command.desiredWheelSteerNormalized,
                desiredSpeedMps: command.desiredSpeedMps,
                stopProbability: command.stopRequested ? 1 : 0,
            };
            this.egoService.collectEgoData(ego, this.currentTelemetry(), supervision);
            await wait(STRAIGHT_STOP_EXPERT_INTERVAL_MS);
        }
    }

    private async prepareParkingScene(target: ParkingTarget) {
        this.fixtures.clear();
        this.sweepIsolationAt(target.pose.coords);
        await wait(200);
        if (parkingFixturesEnabled) {
            await this.fixtures.replace(target);
        }
    }

    private startEvaluationMonitor(ego: Ego, goal: ParkingGoal, footprint: VehicleFootprint) {
        this.stopEvaluationMonitor();
        const vehicle = ego.vehicle.id;
        const tracker = new ParkingEvaluationOutcomeTracker(goal, footprint);
        const hazards = new ParkingHazardLatch(reversingSpeedThresholdMps);
        let lastObservedAt = 0;
        let terminal = false;

        this.evaluationTickId = setTick(() => {
            if (terminal) {
                holdEvaluationVehicle(vehicle);
                return;
            }

            const vehicleValid = isValidEntity(vehicle);
            if (vehicleValid) {
                const velocity = toVector3(GetEntitySpeedVector(vehicle, true));
                hazards.observe(
                    HasEntityCollidedWithAnything(vehicle),
                    velocity?.[1] ?? 0
                );
            }

            const nowMs = GetGameTimer();
            if (nowMs - lastObservedAt < CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS) {
                return;
            }
            lastObservedAt = nowMs;

            const result = tracker.observe({
                nowMs,
                pose: vehicleValid ? entityPose(vehicle) : null,
                speedMps: vehicleValid ? GetEntitySpeed(vehicle) : 0,
                collision: hazards.collisionDetected(),
                reversing: hazards.reversingDetected(),
                onGround: vehicleValid && IsVehicleOnAllWheels(vehicle),
                pitchDeg: vehicleValid ? GetEntityPitch(vehicle) : 0,
                rollDeg: vehicleValid ? GetEntityRoll(vehicle) : 0,
                stopRequested: false,
            });

            if (!result.outcome) {
                this.latestState = result.state;
                return;
            }

            holdEvaluationVehicle(vehicle);
            this.latestState = result.state;
            this.phase = terminalParkingPhase(result.outcome);
            terminal = true;
            console.log(
                `[parking] evaluation ${result.outcome.status} `
                + `distance=${result.outcome.finalDistance.toFixed(3)} `
                + `heading=${result.outcome.finalHeadingError.toFixed(2)}`
            );
        });
    }

    private stopEvaluationMonitor() {
        if (this.evaluationTickId === null) {
            return;
        }
        clearTick(this.evaluationTickId);
        this.evaluationTickId = null;
    }

    private stopEvaluationState() {
        this.stopEvaluationMonitor();
        this.stopDriverSeatGuard();
        this.stopParkingWorldIsolation();
        this.fixtures.clear();
        this.egoService.clearParkingRouteTarget();
        this.latestState = emptyParkingState();
        this.attemptIndex = 0;
        this.attemptCount = 0;
        this.phase = this.hasSetup() ? "ready" : "idle";
    }

    private startTargetMarker() {
        this.stopTargetMarker();
        this.nextIsolationSweepAtMs = 0;
        this.targetMarkerTickId = setTick(() => {
            if (!this.target) {
                return;
            }
            drawParkingTargetMarker(this.target.pose, this.target.bay);
            enforceParkingIsolationThisFrame(this.target.pose.coords);
            const nowMs = GetGameTimer();
            if (nowMs >= this.nextIsolationSweepAtMs) {
                this.nextIsolationSweepAtMs = nowMs + PARKING_ISOLATION_SWEEP_INTERVAL_MS;
                this.sweepIsolationAt(this.target.pose.coords);
            }
        });
    }

    private stopTargetMarker() {
        if (this.targetMarkerTickId !== null) {
            clearTick(this.targetMarkerTickId);
            this.targetMarkerTickId = null;
        }
        this.clearNativeZoneIfIsolationInactive();
    }

    private startParkingWorldIsolation(options: {suppressVehicleControls: boolean}) {
        this.stopParkingWorldIsolation();
        this.worldIsolationActive = true;
        enforceFixedParkingDaylight();
        emit("chat:clear");
        this.populationTickId = setTick(() => {
            const center = this.target?.pose.coords ?? playerPosition();
            if (center) {
                enforceParkingIsolationThisFrame(center);
            }
            if (options.suppressVehicleControls) {
                suppressParkingVehicleControls();
            }
            HideHudAndRadarThisFrame();
        });
    }

    private stopParkingWorldIsolation() {
        if (!this.worldIsolationActive && this.populationTickId === null) {
            return;
        }
        this.worldIsolationActive = false;
        if (this.populationTickId !== null) {
            clearTick(this.populationTickId);
            this.populationTickId = null;
        }
        this.clearNativeZoneIfIsolationInactive();
        enforceFixedParkingDaylight();
    }

    private clearNativeZoneIfIsolationInactive() {
        if (
            this.playerIsolationTickId !== null
            || this.targetMarkerTickId !== null
            || this.populationTickId !== null
        ) {
            return;
        }
        clearParkingIsolationNativeZone();
    }

    private startDriverSeatGuard(vehicle: number) {
        this.stopDriverSeatGuard();
        this.driverSeatGuardTickId = setTick(() => {
            if (!isValidEntity(vehicle)) {
                return;
            }
            const playerPed = PlayerPedId();
            if (!isPlayerParkingDriver(playerPed, vehicle)) {
                SetPedIntoVehicle(playerPed, vehicle, -1);
            }
        });
    }

    private stopDriverSeatGuard() {
        if (this.driverSeatGuardTickId === null) {
            return;
        }
        clearTick(this.driverSeatGuardTickId);
        this.driverSeatGuardTickId = null;
    }

    private rememberCurrentEgoAsOwned() {
        const ego = this.egoService.oldEgo;
        if (ego && isValidEntity(ego.vehicle.id)) {
            this.ownedParkingVehicles.add(ego.vehicle.id);
        }
    }

    private sweepIsolationNow() {
        if (this.target) {
            this.sweepIsolationAt(this.target.pose.coords);
        }
    }

    private sweepIsolationAt(center: ParkingVector3) {
        for (const vehicle of [...this.ownedParkingVehicles]) {
            if (!isValidEntity(vehicle)) {
                this.ownedParkingVehicles.delete(vehicle);
            }
        }
        this.rememberCurrentEgoAsOwned();
        const removed = sweepParkingIsolation(center, this.ownedParkingVehicles);
        if (removed.removedVehicles > 0 || removed.removedPeds > 0) {
            console.log(
                `[parking] isolation cleared vehicles=${removed.removedVehicles} peds=${removed.removedPeds}`
            );
        }
        return removed;
    }
}

function holdEvaluationVehicle(vehicle: number) {
    if (!isValidEntity(vehicle)) {
        return;
    }
    SetVehicleForwardSpeed(vehicle, 0);
    SetVehicleHandbrake(vehicle, true);
    SetVehicleSteerBias(vehicle, 0);
    FreezeEntityPosition(vehicle, true);
}

function requireParkingEgo(ego: Ego, context: string) {
    if (!isValidEntity(ego.vehicle.id)) {
        throw new Error(`Parking ${context} vehicle is not valid`);
    }
}

function isPlayerParkingDriver(playerPed: number, vehicle: number): boolean {
    return IsPedInVehicle(playerPed, vehicle, false)
        && GetPedInVehicleSeat(vehicle, -1) === playerPed;
}

async function ensurePlayerIsParkingDriver(vehicle: number) {
    if (!isValidEntity(vehicle)) {
        throw new Error("Cannot seat the player in an invalid parking vehicle");
    }

    const playerPed = PlayerPedId();
    const deadlineMs = GetGameTimer() + 1000;
    while (GetGameTimer() <= deadlineMs) {
        if (isPlayerParkingDriver(playerPed, vehicle)) {
            return;
        }
        TaskWarpPedIntoVehicle(playerPed, vehicle, -1);
        SetPedIntoVehicle(playerPed, vehicle, -1);
        await wait(50);
    }
    throw new Error("Parking collection could not place the player in the driver seat");
}

function configureParkingEgoSpeed(vehicle: number) {
    SetEntityMaxSpeed(vehicle, PARKING_MAX_SPEED_MPS);
    SetVehicleMaxSpeed(vehicle, PARKING_MAX_SPEED_MPS);
}

function startParkingSteeringLock(vehicle: number): ParkingSteeringLock {
    let desiredWheelSteerNormalized = 0;
    SetVehicleSteerBias(vehicle, desiredWheelSteerNormalized);
    const tickId = setTick(() => {
        if (isValidEntity(vehicle)) {
            SetVehicleSteerBias(vehicle, desiredWheelSteerNormalized);
        }
    });
    return {
        set: (value) => {
            if (!Number.isFinite(value) || value < -1 || value > 1) {
                throw new RangeError("Parking steering command must be finite and within [-1, 1]");
            }
            desiredWheelSteerNormalized = value;
            if (isValidEntity(vehicle)) {
                SetVehicleSteerBias(vehicle, value);
            }
        },
        stop: () => clearTick(tickId),
    };
}

function applyStraightStopExpertCommand(
    vehicle: number,
    command: {
        desiredSpeedMps: number
        desiredWheelSteerNormalized: number
        stopRequested: boolean
    }
) {
    if (!isValidEntity(vehicle)) {
        throw new Error("Cannot apply a straight-stop command to an invalid parking vehicle");
    }
    SetVehicleSteerBias(vehicle, command.desiredWheelSteerNormalized);
    if (command.stopRequested) {
        SetVehicleForwardSpeed(vehicle, 0);
        SetVehicleHandbrake(vehicle, true);
        return;
    }
    SetVehicleHandbrake(vehicle, false);
    SetVehicleForwardSpeed(vehicle, command.desiredSpeedMps);
}

function terminalExpertSupervision(outcome: ParkingOutcome): ParkingExpertSupervision {
    return {
        desiredWheelSteerNormalized: 0,
        desiredSpeedMps: 0,
        stopProbability: outcome.success ? 1 : 0,
    };
}

async function resetParkingEgoAtStart(vehicle: number, expectedPose: ParkingPose) {
    if (!isValidEntity(vehicle)) {
        throw new Error("Cannot reset an invalid parking vehicle");
    }
    SetEntityRecordsCollisions(vehicle, false);
    FreezeEntityPosition(vehicle, true);
    SetVehicleHandbrake(vehicle, true);
    SetVehicleEngineOn(vehicle, true, true, false);
    SetVehicleUndriveable(vehicle, false);
    SetEntityVelocity(vehicle, 0, 0, 0);
    SetVehicleForwardSpeed(vehicle, 0);
    SetEntityCoordsNoOffset(
        vehicle,
        expectedPose.coords[0],
        expectedPose.coords[1],
        expectedPose.coords[2] + 0.75,
        false,
        false,
        true
    );
    SetEntityHeading(vehicle, expectedPose.heading);
    SetVehicleOnGroundProperly(vehicle);

    const deadlineMs = GetGameTimer() + PARKING_EVALUATION_SPAWN_SETTLE_MS;
    let consecutiveStableSamples = 0;
    let lastFailure = "vehicle did not settle";
    while (GetGameTimer() <= deadlineMs) {
        SetVehicleOnGroundProperly(vehicle);
        lastFailure = parkingStartInstability(vehicle, expectedPose);
        if (!lastFailure) {
            consecutiveStableSamples += 1;
            if (consecutiveStableSamples >= stableStartSampleCount) {
                await wait(PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS);
                const finalFailure = parkingStartInstability(vehicle, expectedPose);
                if (finalFailure) {
                    throw new Error(`Parking start became unstable after settling: ${finalFailure}`);
                }
                return;
            }
        } else {
            consecutiveStableSamples = 0;
        }
        await wait(CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS);
    }
    throw new Error(`Parking start failed to settle at the saved pose: ${lastFailure}`);
}

function parkingStartInstability(vehicle: number, expectedPose: ParkingPose): string {
    if (!isValidEntity(vehicle)) {
        return "vehicle no longer exists";
    }
    const speedMps = GetEntitySpeed(vehicle);
    if (!Number.isFinite(speedMps) || Math.abs(speedMps) > maximumCalibrationSpeedMps) {
        return `speed=${Number.isFinite(speedMps) ? speedMps.toFixed(3) : "invalid"}m/s`;
    }
    if (!IsVehicleOnAllWheels(vehicle)) {
        return "vehicle is not on all wheels";
    }
    const pitchDeg = GetEntityPitch(vehicle);
    const rollDeg = GetEntityRoll(vehicle);
    if (!Number.isFinite(pitchDeg) || !Number.isFinite(rollDeg)) {
        return "vehicle pitch or roll is invalid";
    }
    if (Math.abs(pitchDeg) > MAX_PARKING_TILT_DEG || Math.abs(rollDeg) > MAX_PARKING_TILT_DEG) {
        return `pitch=${pitchDeg.toFixed(2)}deg roll=${rollDeg.toFixed(2)}deg`;
    }
    const actualPose = entityPose(vehicle);
    if (!actualPose) {
        return "vehicle pose is unavailable";
    }
    const error = relativePose(actualPose, expectedPose);
    if (
        error.distanceM > maximumStartPosePositionErrorM
        || Math.abs(error.headingDeg) > maximumStartPoseHeadingErrorDeg
    ) {
        return `position=${error.distanceM.toFixed(3)}m heading=${error.headingDeg.toFixed(2)}deg`;
    }
    return "";
}

function enforceFixedParkingDaylight() {
    SetWeatherTypeNowPersist("EXTRASUNNY");
    NetworkOverrideClockTime(12, 30, 0);
    PauseClock(true);
}

function buildParkingEgo(): Ego {
    return {
        vehicle: {
            id: 0,
            model: VehicleModel.Sultan,
            color: VehicleColor.Blue,
            maxSpeed: PARKING_MAX_SPEED_MPS,
            drivingStyle: DrivingStyle.Cautious,
        },
        waypoints: [],
    };
}

function buildParkingGoal(
    target: ParkingTarget,
    plan: ParkingAttemptPlan,
    attemptCount: number,
    seed: string
): ParkingGoal {
    return {
        task: "parking",
        maneuver: "forward-bay",
        visualCueId: PARKING_TARGET_MARKER_ID,
        target: clonePose(target.pose),
        bay: {...target.bay},
        tolerances: {...target.tolerances},
        attemptIndex: plan.attemptIndex,
        attemptCount,
        seed,
        startPose: clonePose(plan.startPose),
        startOffset: {...plan.startOffset},
    };
}

function parkingStateAtPose(
    target: ParkingTarget,
    pose: ParkingPose,
    footprint: VehicleFootprint
): ParkingState {
    const relative = relativePose(pose, target.pose);
    return {
        longitudinalError: relative.longitudinalM,
        lateralError: relative.lateralM,
        headingError: relative.headingDeg,
        distance: relative.distanceM,
        insideBay: isVehicleFootprintInsideBay(pose, target.pose, target.bay, footprint),
        aligned: Math.abs(relative.headingDeg) <= target.tolerances.maxHeadingErrorDeg,
        parked: false,
        settledDurationMs: 0,
    };
}

function telemetryFromState(
    state: ParkingState,
    targetPose: ParkingPose,
    startPose: ParkingPose | null,
    phase: ParkingPhase,
    attemptIndex: number,
    attemptCount: number
): ParkingTelemetry {
    return {
        parkingTargetConfigured: true,
        parkingStartConfigured: startPose !== null,
        parkingTargetPose: clonePose(targetPose),
        parkingStartPose: startPose ? clonePose(startPose) : undefined,
        parkingLongitudinalError: state.longitudinalError,
        parkingLateralError: state.lateralError,
        parkingHeadingError: state.headingError,
        parkingDistance: state.distance,
        parkingInsideBay: state.insideBay,
        parkingAligned: state.aligned,
        parkingParked: state.parked,
        parkingAttemptIndex: attemptIndex,
        parkingAttemptCount: attemptCount,
        parkingPhase: phase,
    };
}

function emptyParkingTelemetry(phase: ParkingPhase): ParkingTelemetry {
    return {
        parkingTargetConfigured: false,
        parkingStartConfigured: false,
        parkingLongitudinalError: 0,
        parkingLateralError: 0,
        parkingHeadingError: 0,
        parkingDistance: 0,
        parkingInsideBay: false,
        parkingAligned: false,
        parkingParked: false,
        parkingAttemptIndex: 0,
        parkingAttemptCount: 0,
        parkingPhase: phase,
    };
}

function playerPosition(): ParkingVector3 | null {
    return toVector3(GetEntityCoords(PlayerPedId(), false));
}

function entityPose(entity: number): ParkingPose | null {
    if (!isValidEntity(entity)) {
        return null;
    }
    const coords = toVector3(GetEntityCoords(entity, false));
    const heading = GetEntityHeading(entity);
    if (!coords || !Number.isFinite(heading)) {
        return null;
    }
    return {coords, heading};
}

function requireStableUserCalibrationVehicle(vehicle: number) {
    const speedMps = GetEntitySpeed(vehicle);
    if (!Number.isFinite(speedMps) || Math.abs(speedMps) > maximumCalibrationSpeedMps) {
        throw new Error(
            `Parking calibration requires a stationary vehicle at or below ${maximumCalibrationSpeedMps.toFixed(2)} m/s`
        );
    }
    if (!IsVehicleOnAllWheels(vehicle)) {
        throw new Error("Parking calibration requires the vehicle to be on all wheels");
    }
    const pitchDeg = GetEntityPitch(vehicle);
    const rollDeg = GetEntityRoll(vehicle);
    if (!Number.isFinite(pitchDeg) || !Number.isFinite(rollDeg)) {
        throw new Error("Parking calibration requires finite vehicle pitch and roll");
    }
    if (Math.abs(pitchDeg) > MAX_PARKING_TILT_DEG || Math.abs(rollDeg) > MAX_PARKING_TILT_DEG) {
        throw new Error(
            `Parking calibration requires an upright vehicle within ${MAX_PARKING_TILT_DEG} degrees of level`
        );
    }
}

function monitorAttemptHazards(vehicle: number): AttemptHazardMonitor {
    const hazards = new ParkingHazardLatch(reversingSpeedThresholdMps);
    const tickId = setTick(() => {
        if (!isValidEntity(vehicle)) {
            return;
        }
        const localVelocity = toVector3(GetEntitySpeedVector(vehicle, true));
        hazards.observe(
            HasEntityCollidedWithAnything(vehicle),
            localVelocity?.[1] ?? 0
        );
    });
    return {
        collisionDetected: () => hazards.collisionDetected(),
        reversingDetected: () => hazards.reversingDetected(),
        stop: () => clearTick(tickId),
    };
}

function requireMatchingNativeForwardVector(vehicle: number, heading: number) {
    const nativeForward = toVector3(GetEntityForwardVector(vehicle));
    if (!nativeForward) {
        throw new Error("Parking heading invariant violated: native forward vector was not finite");
    }
    const nativeHorizontalLength = Math.hypot(nativeForward[0], nativeForward[1]);
    if (!Number.isFinite(nativeHorizontalLength) || nativeHorizontalLength <= 0) {
        throw new Error("Parking heading invariant violated: native forward vector had zero horizontal length");
    }
    const expectedForward = gtaForwardVector(heading);
    const dot = (
        (expectedForward[0] * nativeForward[0])
        + (expectedForward[1] * nativeForward[1])
    ) / nativeHorizontalLength;
    if (!Number.isFinite(dot) || dot < headingConventionDotThreshold) {
        throw new Error(
            `Parking heading invariant violated: heading ${heading.toFixed(3)} predicts `
            + `forward=(${expectedForward[0].toFixed(4)},${expectedForward[1].toFixed(4)}) but `
            + `native=(${nativeForward[0].toFixed(4)},${nativeForward[1].toFixed(4)}), dot=${dot.toFixed(5)}`
        );
    }
}

function vehicleFootprint(vehicle: number): VehicleFootprint {
    if (!isValidEntity(vehicle)) {
        return {...fallbackVehicleFootprint};
    }
    const [minRaw, maxRaw] = GetModelDimensions(GetEntityModel(vehicle));
    const min = toVector3(minRaw);
    const max = toVector3(maxRaw);
    if (!min || !max) {
        return {...fallbackVehicleFootprint};
    }
    const halfWidthM = Math.max(Math.abs(min[0]), Math.abs(max[0]));
    const halfLengthM = Math.max(Math.abs(min[1]), Math.abs(max[1]));
    if (!Number.isFinite(halfWidthM) || !Number.isFinite(halfLengthM)) {
        return {...fallbackVehicleFootprint};
    }
    return {
        halfWidthM: Math.max(0.5, halfWidthM),
        halfLengthM: Math.max(1, halfLengthM),
    };
}

function toVector3(value: unknown): ParkingVector3 | null {
    if (!Array.isArray(value) || value.length < 3) {
        return null;
    }
    const coords: ParkingVector3 = [Number(value[0]), Number(value[1]), Number(value[2])];
    return coords.every((component) => Number.isFinite(component)) ? coords : null;
}

function clonePose(pose: ParkingPose): ParkingPose {
    return {
        coords: [...pose.coords] as ParkingVector3,
        heading: pose.heading,
    };
}

function validatedParkingPose(pose: ParkingPose, label: string): ParkingPose {
    if (
        !pose
        || !Array.isArray(pose.coords)
        || pose.coords.length !== 3
        || !pose.coords.every(Number.isFinite)
        || !Number.isFinite(pose.heading)
        || pose.heading < 0
        || pose.heading >= 360
    ) {
        throw new Error(`Parking ${label} pose must contain finite coordinates and heading in [0, 360)`);
    }
    return clonePose(pose);
}

function cloneTarget(target: ParkingTarget): ParkingTarget {
    return {
        pose: clonePose(target.pose),
        bay: {...target.bay},
        tolerances: {...target.tolerances},
    };
}

function createRunId(): string {
    const now = new Date();
    const hours24 = now.getHours();
    const meridiem = hours24 >= 12 ? "PM" : "AM";
    const hours12 = hours24 % 12 || 12;
    const suffix = Math.random().toString(36).slice(2, 8);
    return `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}_`
        + `${pad2(hours12)}-${pad2(now.getMinutes())}-${pad2(now.getSeconds())}${meridiem}_${suffix}`;
}

function pad2(value: number): string {
    return String(value).padStart(2, "0");
}
