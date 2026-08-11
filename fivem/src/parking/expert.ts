export const STRAIGHT_STOP_EXPERT_INTERVAL_MS = 50;
export const STRAIGHT_STOP_MAX_DT_SECONDS = 0.15;
export const STRAIGHT_STOP_MAX_SPEED_MPS = 2.22;
export const STRAIGHT_STOP_ACCELERATION_MPS2 = 0.8;
export const STRAIGHT_STOP_BRAKING_MPS2 = 0.9;
export const STRAIGHT_STOP_DISTANCE_OFFSET_M = 0.05;
export const STRAIGHT_STOP_GUARD_TIME_SECONDS = 0.1;
export const STRAIGHT_STOP_HOLD_DISTANCE_M = 0.08;
export const STRAIGHT_STOP_HOLD_SPEED_MPS = 0.12;
export const STRAIGHT_STOP_CROSS_TRACK_GAIN = 0.6;
export const STRAIGHT_STOP_STEERING_SOFTENING_MPS = 1.0;
export const STRAIGHT_STOP_MAX_WHEEL_STEER_NORMALIZED = 0.35;

const sultanSteeringFullLockRadians = 0.733038306;
const degreesToRadians = Math.PI / 180;

export type StraightStopExpertInput = {
    remainingDistanceM: number
    measuredSpeedMps: number
    previousDesiredSpeedMps: number
    dtSeconds: number
    lateralErrorM: number
    headingErrorDeg: number
};

export type StraightStopExpertOutput = {
    desiredSpeedMps: number
    desiredWheelSteerNormalized: number
    stopRequested: boolean
};

/** Plans one deterministic, straight-only expert speed step toward the bay center. */
export function planStraightStopExpert(input: StraightStopExpertInput): StraightStopExpertOutput {
    validateInput(input);

    if (
        input.remainingDistanceM <= STRAIGHT_STOP_HOLD_DISTANCE_M
        && input.measuredSpeedMps <= STRAIGHT_STOP_HOLD_SPEED_MPS
    ) {
        return {desiredSpeedMps: 0, desiredWheelSteerNormalized: 0, stopRequested: true};
    }

    const guardedDistanceM = Math.max(
        0,
        input.remainingDistanceM
            - STRAIGHT_STOP_DISTANCE_OFFSET_M
            - (input.measuredSpeedMps * STRAIGHT_STOP_GUARD_TIME_SECONDS)
    );
    const brakingLimitedSpeedMps = Math.sqrt(
        2 * STRAIGHT_STOP_BRAKING_MPS2 * guardedDistanceM
    );
    const accelerationLimitedSpeedMps = input.previousDesiredSpeedMps
        + (STRAIGHT_STOP_ACCELERATION_MPS2 * input.dtSeconds);
    const desiredSpeedMps = Math.min(
        STRAIGHT_STOP_MAX_SPEED_MPS,
        accelerationLimitedSpeedMps,
        brakingLimitedSpeedMps
    );

    if (!Number.isFinite(desiredSpeedMps) || desiredSpeedMps < 0) {
        throw new Error("straight-stop expert produced an invalid desired speed");
    }
    return {
        desiredSpeedMps,
        desiredWheelSteerNormalized: positionTrackingSteer(input),
        stopRequested: false,
    };
}

/** Stanley-style path tracking toward the saved bay pose. Positive steer is left in FiveM. */
function positionTrackingSteer(input: StraightStopExpertInput): number {
    const headingCorrectionRadians = input.headingErrorDeg * degreesToRadians;
    const crossTrackCorrectionRadians = Math.atan2(
        STRAIGHT_STOP_CROSS_TRACK_GAIN * input.lateralErrorM,
        input.measuredSpeedMps + STRAIGHT_STOP_STEERING_SOFTENING_MPS
    );
    return clamp(
        (headingCorrectionRadians + crossTrackCorrectionRadians) / sultanSteeringFullLockRadians,
        -STRAIGHT_STOP_MAX_WHEEL_STEER_NORMALIZED,
        STRAIGHT_STOP_MAX_WHEEL_STEER_NORMALIZED
    );
}

function validateInput(input: StraightStopExpertInput) {
    requireFiniteNonNegative(input.remainingDistanceM, "remainingDistanceM");
    requireFiniteNonNegative(input.measuredSpeedMps, "measuredSpeedMps");
    requireFiniteNonNegative(input.previousDesiredSpeedMps, "previousDesiredSpeedMps");
    requireFinite(input.lateralErrorM, "lateralErrorM");
    requireFinite(input.headingErrorDeg, "headingErrorDeg");
    if (input.previousDesiredSpeedMps > STRAIGHT_STOP_MAX_SPEED_MPS) {
        throw new RangeError(
            `previousDesiredSpeedMps must not exceed ${STRAIGHT_STOP_MAX_SPEED_MPS}`
        );
    }
    if (
        !Number.isFinite(input.dtSeconds)
        || input.dtSeconds <= 0
        || input.dtSeconds > STRAIGHT_STOP_MAX_DT_SECONDS
    ) {
        throw new RangeError(
            `dtSeconds must be finite and within (0, ${STRAIGHT_STOP_MAX_DT_SECONDS}]`
        );
    }
}

function requireFiniteNonNegative(value: number, name: string) {
    if (!Number.isFinite(value) || value < 0) {
        throw new RangeError(`${name} must be finite and non-negative`);
    }
}

function requireFinite(value: number, name: string) {
    if (!Number.isFinite(value)) {
        throw new RangeError(`${name} must be finite`);
    }
}

function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
}
