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
    resolveParkingAttemptCount,
} from "./curriculum";
import {
    PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS,
    PARKING_EVALUATION_SPAWN_SETTLE_MS,
    PARKING_POST_FLASH_CLEAN_FRAME_BUFFER_MS,
    suppressParkingVehicleControls,
} from "./controls";
import {gtaForwardVector, isVehicleFootprintInsideBay, relativePose} from "./geometry";
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
    ParkingGoal,
    ParkingOutcome,
    ParkingPhase,
    ParkingPose,
    ParkingState,
    ParkingTarget,
    ParkingTelemetry,
    ParkingVector3,
    VehicleFootprint,
} from "./types";

export const PARKING_SCENE_ID = "parking-forward-bay";
export const PARKING_SCENE_VARIANT = "default";
export const PARKING_SCENE_NAME = `${PARKING_SCENE_ID}:${PARKING_SCENE_VARIANT}`;
export const PARKING_EVALUATION_SCENE_NAME = "parking-evaluation";
export const PARKING_MAX_SPEED_MPS = 2.22;

const headingConventionDotThreshold = 0.99;
const maximumCalibrationSpeedMps = 0.1;
const reversingSpeedThresholdMps = -0.1;

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

export class ParkingRunner {
    private target: ParkingTarget | null = null;
    private running = false;
    private stopRequested = false;
    private phase: ParkingPhase = "idle";
    private attemptIndex = 0;
    private attemptCount = 0;
    private latestState: ParkingState = emptyParkingState();
    private populationTickId: number | null = null;
    private evaluationTickId: number | null = null;
    private worldIsolationActive = false;

    constructor(private readonly egoService: EgoService) {}

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
        requireStableCalibrationVehicle(ego.vehicle.id);
        requireMatchingNativeForwardVector(ego.vehicle.id, pose.heading);
        this.stopEvaluationState();
        this.target = {
            pose,
            bay: {...defaultParkingTarget.bay},
            tolerances: {...defaultParkingTarget.tolerances},
        };
        this.egoService.configureParkingRouteTarget(this.target.pose.coords);
        this.phase = "ready";
        this.latestState = emptyParkingState();
        console.log(
            `[parking] target calibrated at (${pose.coords.map((value) => value.toFixed(3)).join(", ")}) `
            + `heading=${pose.heading.toFixed(2)}`
        );
        return cloneTarget(this.target);
    }

    clearTarget() {
        if (this.running) {
            throw new Error("Cannot clear the parking target while a parking run is active");
        }
        this.stopEvaluationState();
        this.target = null;
        this.egoService.clearParkingRouteTarget();
        this.phase = "idle";
        this.attemptIndex = 0;
        this.attemptCount = 0;
        this.latestState = emptyParkingState();
        console.log("[parking] target cleared");
    }

    hasTarget(): boolean {
        return this.target !== null;
    }

    isRunning(): boolean {
        return this.running;
    }

    requestStop() {
        if (!this.running) {
            return;
        }
        this.stopRequested = true;
        this.phase = "stopping";
        const ego = this.egoService.oldEgo;
        if (ego && isValidEntity(ego.vehicle.id)) {
            ClearPedTasksImmediately(PlayerPedId());
            SetVehicleForwardSpeed(ego.vehicle.id, 0);
            SetVehicleHandbrake(ego.vehicle.id, true);
        }
    }

    async run(attemptCountValue: unknown, requestedSeed?: string) {
        if (this.running) {
            throw new Error("A parking run is already active");
        }
        if (!this.target) {
            throw new Error("Parking target is not configured; run setParkingTarget first");
        }

        const attemptCount = resolveParkingAttemptCount(attemptCountValue);
        const runId = createRunId();
        const seed = requestedSeed?.trim() || `${PARKING_SCENE_ID}:${runId}`;
        const target = cloneTarget(this.target);
        const plans = buildForwardBayCurriculum(target.pose, attemptCount, seed);

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
            for (const plan of plans) {
                if (this.stopRequested) {
                    break;
                }
                this.attemptIndex = plan.attemptIndex;
                this.latestState = emptyParkingState();
                const outcome = await this.runAttempt(target, plan, attemptCount, runId, seed);
                console.log(
                    `[parking] attempt=${plan.attemptIndex} status=${outcome.status} `
                    + `distance=${outcome.finalDistance.toFixed(3)} heading=${outcome.finalHeadingError.toFixed(2)}`
                );
                this.egoService.disposeCurrentEgo();
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
                this.stopParkingWorldIsolation();
                this.egoService.clearParkingRouteTarget();
            } finally {
                try {
                    this.egoService.disposeCurrentEgo();
                } finally {
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
        if (!this.target) {
            throw new Error("Parking target is not configured; run setParkingTarget first");
        }

        const target = cloneTarget(this.target);
        const evaluation = buildForwardBayEvaluationPlan(target.pose, requestedSeed);
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
            this.clearParkingArea(target.pose.coords);
            await wait(200);

            const ego = buildParkingEgo();
            await this.egoService.executeEgoAt(
                ego,
                evaluation.plan.startPose,
                PARKING_EVALUATION_SCENE_NAME,
                runId
            );
            if (!isValidEntity(ego.vehicle.id)) {
                throw new Error("Failed to spawn the parking evaluation ego");
            }
            if (this.stopRequested) {
                throw new Error("Parking evaluation setup was stopped");
            }

            SetEntityMaxSpeed(ego.vehicle.id, PARKING_MAX_SPEED_MPS);
            SetVehicleMaxSpeed(ego.vehicle.id, PARKING_MAX_SPEED_MPS);
            SetEntityRecordsCollisions(ego.vehicle.id, false);
            FreezeEntityPosition(ego.vehicle.id, true);
            SetVehicleForwardSpeed(ego.vehicle.id, 0);
            SetVehicleHandbrake(ego.vehicle.id, true);
            SetVehicleOnGroundProperly(ego.vehicle.id);
            await wait(PARKING_EVALUATION_SPAWN_SETTLE_MS);
            SetVehicleOnGroundProperly(ego.vehicle.id);
            requireStableCalibrationVehicle(ego.vehicle.id);
            await wait(PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS);
            SetVehicleOnGroundProperly(ego.vehicle.id);
            requireStableCalibrationVehicle(ego.vehicle.id);
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
            this.phase,
            this.attemptIndex,
            this.attemptCount
        );
    }

    private async runAttempt(
        target: ParkingTarget,
        plan: ParkingAttemptPlan,
        attemptCount: number,
        runId: string,
        seed: string
    ): Promise<ParkingOutcome> {
        this.phase = "spawning";
        this.clearParkingArea(target.pose.coords);
        await wait(200);

        const ego = buildParkingEgo();
        await this.egoService.executeEgoAt(ego, plan.startPose, PARKING_SCENE_NAME, runId);
        if (!isValidEntity(ego.vehicle.id)) {
            throw new Error(`Failed to spawn parking ego for attempt ${plan.attemptIndex}`);
        }
        SetEntityMaxSpeed(ego.vehicle.id, PARKING_MAX_SPEED_MPS);
        SetVehicleMaxSpeed(ego.vehicle.id, PARKING_MAX_SPEED_MPS);

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
            await wait(500);

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
            hazards = monitorAttemptHazards(ego.vehicle.id);
            const [targetX, targetY, targetZ] = target.pose.coords;
            if (!this.stopRequested) {
                TaskVehiclePark(
                    PlayerPedId(),
                    ego.vehicle.id,
                    targetX,
                    targetY,
                    targetZ,
                    target.pose.heading,
                    1,
                    20.0,
                    true
                );
            }
            this.phase = this.stopRequested ? "stopping" : "parking";
            const outcome = await this.monitorAttempt(ego, tracker, hazards);

            if (isValidEntity(ego.vehicle.id)) {
                ClearPedTasksImmediately(PlayerPedId());
                SetVehicleForwardSpeed(ego.vehicle.id, 0);
                SetVehicleHandbrake(ego.vehicle.id, true);
                this.egoService.collectEgoData(ego, this.currentTelemetry());
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
        hazards: AttemptHazardMonitor
    ): Promise<ParkingOutcome> {
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
                this.phase = terminalParkingPhase(result.outcome);
                return result.outcome;
            }

            this.phase = result.state.settledDurationMs > 0 ? "settling" : "parking";
            this.egoService.collectEgoData(ego, this.currentTelemetry());
            await wait(CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS);
        }
    }

    private clearParkingArea(coords: ParkingVector3) {
        ClearAreaOfVehicles(coords[0], coords[1], coords[2], 35, false, false, false, false, false);
        ClearAreaOfPeds(coords[0], coords[1], coords[2], 35, true);
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
        this.stopParkingWorldIsolation();
        this.egoService.clearParkingRouteTarget();
        this.latestState = emptyParkingState();
        this.attemptIndex = 0;
        this.attemptCount = 0;
        this.phase = this.target ? "ready" : "idle";
    }

    private startParkingWorldIsolation(options: {suppressVehicleControls: boolean}) {
        this.stopParkingWorldIsolation();
        this.worldIsolationActive = true;
        SetWeatherTypeNowPersist("EXTRASUNNY");
        NetworkOverrideClockTime(12, 30, 0);
        PauseClock(true);
        emit("chat:clear");
        this.populationTickId = setTick(() => {
            SetPedDensityMultiplierThisFrame(0);
            SetScenarioPedDensityMultiplierThisFrame(0, 0);
            SetVehicleDensityMultiplierThisFrame(0);
            SetRandomVehicleDensityMultiplierThisFrame(0);
            SetParkedVehicleDensityMultiplierThisFrame(0);
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
        PauseClock(false);
        NetworkClearClockTimeOverride();
        ClearWeatherTypePersist();
        ClearOverrideWeather();
    }
}

function holdEvaluationVehicle(vehicle: number) {
    if (!isValidEntity(vehicle)) {
        return;
    }
    ClearPedTasksImmediately(PlayerPedId());
    SetVehicleForwardSpeed(vehicle, 0);
    SetVehicleHandbrake(vehicle, true);
    FreezeEntityPosition(vehicle, true);
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
    phase: ParkingPhase,
    attemptIndex: number,
    attemptCount: number
): ParkingTelemetry {
    return {
        parkingTargetConfigured: true,
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

function requireStableCalibrationVehicle(vehicle: number) {
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
