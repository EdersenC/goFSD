import {isVehicleFootprintInsideBay, relativePose} from "./geometry";
import {
    ParkingGoal,
    ParkingOutcome,
    ParkingOutcomeStatus,
    ParkingPhase,
    ParkingPose,
    ParkingState,
    VehicleFootprint,
} from "./types";
import {
    MAX_STRAIGHT_APPROACH_HEADING_ERROR_DEG,
    MAX_STRAIGHT_APPROACH_LATERAL_ERROR_M,
    MAX_STRAIGHT_TRACKING_HEADING_ERROR_DEG,
    MAX_STRAIGHT_TRACKING_LATERAL_ERROR_M,
} from "./constraints";

export const DEFAULT_PARKING_TIMEOUT_MS = 45_000;
export const MAX_PARKING_TILT_DEG = 5;
export const PARKING_EVALUATION_ARMING_SPEED_MPS = 0.05;
export {
    MAX_STRAIGHT_APPROACH_HEADING_ERROR_DEG,
    MAX_STRAIGHT_APPROACH_LATERAL_ERROR_M,
    MAX_STRAIGHT_TRACKING_HEADING_ERROR_DEG,
    MAX_STRAIGHT_TRACKING_LATERAL_ERROR_M,
} from "./constraints";

export type ParkingObservation = {
    nowMs: number
    pose: ParkingPose | null
    speedMps: number
    collision: boolean
    reversing: boolean
    onGround: boolean
    pitchDeg: number
    rollDeg: number
    stopRequested: boolean
};

export type ParkingObservationResult = {
    state: ParkingState
    outcome: ParkingOutcome | null
};

export class ParkingHazardLatch {
    private collision = false;
    private reversing = false;

    constructor(private readonly reversingThresholdMps: number) {}

    observe(collision: boolean, localForwardSpeedMps: number) {
        this.collision = this.collision || collision;
        this.reversing = this.reversing || (
            Number.isFinite(localForwardSpeedMps)
            && localForwardSpeedMps < this.reversingThresholdMps
        );
    }

    collisionDetected(): boolean {
        return this.collision;
    }

    reversingDetected(): boolean {
        return this.reversing;
    }
}

export function terminalParkingPhase(outcome: ParkingOutcome): ParkingPhase {
    return outcome.success ? "succeeded" : "failed";
}

export function evaluateParkingState(
    goal: ParkingGoal,
    pose: ParkingPose,
    footprint: VehicleFootprint,
    speedMps: number,
    settledDurationMs: number
): ParkingState {
    const relative = relativePose(pose, goal.target);
    const insideBay = isVehicleFootprintInsideBay(pose, goal.target, goal.bay, footprint);
    const aligned = Math.abs(relative.headingDeg) <= goal.tolerances.maxHeadingErrorDeg;
    const settled = insideBay
        && aligned
        && relative.distanceM <= goal.tolerances.maxCenterDistanceM
        && Math.abs(speedMps) <= goal.tolerances.maxSpeedMps;
    return {
        longitudinalError: relative.longitudinalM,
        lateralError: relative.lateralM,
        headingError: relative.headingDeg,
        distance: relative.distanceM,
        insideBay,
        aligned,
        parked: settled && settledDurationMs >= goal.tolerances.settledDurationMs,
        settledDurationMs,
    };
}

export class ParkingOutcomeTracker {
    private settledAtMs: number | null = null;
    private lastState: ParkingState = emptyParkingState();

    constructor(
        private readonly goal: ParkingGoal,
        private readonly footprint: VehicleFootprint,
        private readonly startedAtMs: number,
        private readonly timeoutMs = DEFAULT_PARKING_TIMEOUT_MS
    ) {}

    observe(observation: ParkingObservation): ParkingObservationResult {
        if (!observation.pose) {
            return this.finish("invalid_vehicle", observation, "ego vehicle is no longer valid");
        }
        if (!Number.isFinite(observation.speedMps)) {
            return this.finish("invalid_vehicle", observation, "ego vehicle speed was not finite");
        }

        const unsettledState = evaluateParkingState(
            this.goal,
            observation.pose,
            this.footprint,
            observation.speedMps,
            0
        );
        const canSettle = unsettledState.insideBay
            && unsettledState.aligned
            && unsettledState.distance <= this.goal.tolerances.maxCenterDistanceM
            && Math.abs(observation.speedMps) <= this.goal.tolerances.maxSpeedMps;
        this.settledAtMs = canSettle ? this.settledAtMs ?? observation.nowMs : null;
        const settledDurationMs = this.settledAtMs === null ? 0 : observation.nowMs - this.settledAtMs;
        this.lastState = evaluateParkingState(
            this.goal,
            observation.pose,
            this.footprint,
            observation.speedMps,
            settledDurationMs
        );

        if (observation.collision) {
            return this.finish("collision", observation, "vehicle collision detected");
        }
        if (observation.reversing) {
            return this.finish("reversing", observation, "reverse motion detected during forward-bay expert attempt");
        }
        if (!observation.onGround) {
            return this.finish("off_ground", observation, "vehicle left the ground during parking attempt");
        }
        if (
            !Number.isFinite(observation.pitchDeg)
            || !Number.isFinite(observation.rollDeg)
            || Math.abs(observation.pitchDeg) > MAX_PARKING_TILT_DEG
            || Math.abs(observation.rollDeg) > MAX_PARKING_TILT_DEG
        ) {
            return this.finish(
                "not_upright",
                observation,
                `vehicle exceeded ${MAX_PARKING_TILT_DEG} degree parking tilt limit`
            );
        }
        if (
            Math.abs(this.lastState.lateralError) > MAX_STRAIGHT_TRACKING_LATERAL_ERROR_M
            || Math.abs(this.lastState.headingError) > MAX_STRAIGHT_TRACKING_HEADING_ERROR_DEG
        ) {
            return this.finish(
                "left_straight_corridor",
                observation,
                "vehicle left the centered straight-approach corridor"
            );
        }
        if (observation.stopRequested) {
            return this.finish("stopped", observation, "parking run stopped by command");
        }
        if (this.lastState.parked) {
            return this.finish("succeeded", observation, "");
        }
        if (observation.nowMs - this.startedAtMs >= this.timeoutMs) {
            return this.finish("timeout", observation, `parking attempt exceeded ${this.timeoutMs}ms`);
        }
        return {state: this.lastState, outcome: null};
    }

    currentState(): ParkingState {
        return {...this.lastState};
    }

    private finish(
        status: ParkingOutcomeStatus,
        observation: ParkingObservation,
        failureReason: string
    ): ParkingObservationResult {
        const success = status === "succeeded";
        this.lastState = {
            ...this.lastState,
            parked: success && this.lastState.parked,
        };
        const outcome: ParkingOutcome = {
            success,
            status,
            failureReason,
            durationMs: Math.max(0, observation.nowMs - this.startedAtMs),
            finalLongitudinalError: this.lastState.longitudinalError,
            finalLateralError: this.lastState.lateralError,
            finalHeadingError: this.lastState.headingError,
            finalDistance: this.lastState.distance,
            finalInsideBay: this.lastState.insideBay,
            finalAligned: this.lastState.aligned,
            settledDurationMs: this.lastState.settledDurationMs,
            collision: observation.collision,
        };
        return {state: this.lastState, outcome};
    }
}

/** Defers evaluation timeout and settlement until the model first moves the vehicle. */
export class ParkingEvaluationOutcomeTracker {
    private tracker: ParkingOutcomeTracker | null = null;
    private state: ParkingState = emptyParkingState();

    constructor(
        private readonly goal: ParkingGoal,
        private readonly footprint: VehicleFootprint,
        private readonly timeoutMs = DEFAULT_PARKING_TIMEOUT_MS
    ) {}

    observe(observation: ParkingObservation): ParkingObservationResult {
        if (
            !this.tracker
            && (
                requiresImmediateEvaluationOutcome(observation)
                || shouldArmParkingEvaluation(observation.speedMps)
            )
        ) {
            this.tracker = new ParkingOutcomeTracker(
                this.goal,
                this.footprint,
                observation.nowMs,
                this.timeoutMs
            );
        }
        if (this.tracker) {
            const result = this.tracker.observe(observation);
            this.state = result.state;
            return result;
        }
        if (observation.pose) {
            this.state = evaluateParkingState(
                this.goal,
                observation.pose,
                this.footprint,
                observation.speedMps,
                0
            );
        }
        return {state: this.state, outcome: null};
    }

    isArmed(): boolean {
        return this.tracker !== null;
    }
}

export function shouldArmParkingEvaluation(speedMps: number): boolean {
    return Number.isFinite(speedMps)
        && Math.abs(speedMps) >= PARKING_EVALUATION_ARMING_SPEED_MPS;
}

function requiresImmediateEvaluationOutcome(observation: ParkingObservation): boolean {
    return observation.pose === null
        || !Number.isFinite(observation.speedMps)
        || observation.collision
        || observation.reversing
        || !observation.onGround
        || !Number.isFinite(observation.pitchDeg)
        || !Number.isFinite(observation.rollDeg)
        || Math.abs(observation.pitchDeg) > MAX_PARKING_TILT_DEG
        || Math.abs(observation.rollDeg) > MAX_PARKING_TILT_DEG;
}

export function emptyParkingState(): ParkingState {
    return {
        longitudinalError: 0,
        lateralError: 0,
        headingError: 0,
        distance: 0,
        insideBay: false,
        aligned: false,
        parked: false,
        settledDurationMs: 0,
    };
}
