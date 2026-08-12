import {StopSignBehaviorPhase, StopSignExpertSupervision} from "./types";

export const STOP_SIGN_CONTROL_INTERVAL_MS = 50;
// These are smooth collection motion targets, not GTA vehicle limits.
export const STOP_SIGN_BRAKING_DECELERATION_MPS2 = 3.8;
export const STOP_SIGN_LAUNCH_ACCELERATION_MPS2 = 3.5;
export const STOP_SIGN_STOP_SPEED_MPS = 0.12;
export const STOP_SIGN_STOP_POSITION_TOLERANCE_M = 0.65;
export const STOP_SIGN_EARLY_STOP_DISTANCE_M = 3;
export const STOP_SIGN_STOP_INTENT_MARGIN_M = 1.4;
export const STOP_SIGN_GO_DISTANCE_M = 8;

export type StopSignExpertInput = {
    phase: StopSignBehaviorPhase
    remainingDistanceM: number
    measuredSpeedMps: number
    targetSpeedMps: number
    previousDesiredSpeedMps: number
    dtSeconds: number
    lateralErrorM: number
    headingErrorDeg: number
};

export type StopSignPhaseClassificationInput = Pick<StopSignExpertInput,
    | "remainingDistanceM"
    | "measuredSpeedMps"
    | "targetSpeedMps"
    | "previousDesiredSpeedMps"
    | "dtSeconds"
>;

/**
 * Produces the complete supervision contract for a single 20 Hz expert step.
 * Distance and phase are labels/metadata; the RGB policy must not receive them as inputs.
 */
export function planStopSignExpert(input: StopSignExpertInput): StopSignExpertSupervision {
    validateInput(input);

    if (input.phase === "stop_hold") {
        return supervision(input, 0, 0, 0, 1, 1, 0);
    }

    const desiredSpeedMps = input.phase === "release"
        ? accelerateToward(input.previousDesiredSpeedMps, input.targetSpeedMps, input.dtSeconds)
        : approachDesiredSpeed(input);
    const speedErrorMps = desiredSpeedMps - input.measuredSpeedMps;
    const throttle = clamp(speedErrorMps * 0.85, 0, 1);
    const brake = clamp(-speedErrorMps * 0.75, 0, 1);
    const steering = trackingSteer(input.lateralErrorM, input.headingErrorDeg, input.measuredSpeedMps);
    const stopProbability = input.phase === "decelerate"
        ? clamp(1 - input.remainingDistanceM / Math.max(1, stopIntentReferenceDistance(input.measuredSpeedMps)), 0, 1)
        : 0;

    return supervision(
        input,
        desiredSpeedMps,
        steering,
        throttle,
        brake,
        stopProbability,
        input.phase === "release" ? 1 : 0,
    );
}

export function classifyStopSignPhase(
    input: StopSignPhaseClassificationInput,
): Exclude<StopSignBehaviorPhase, "stop_hold" | "release"> {
    validatePhaseClassificationInput(input);
    const desiredSpeedMps = approachDesiredSpeed(input);
    if (desiredSpeedMps < input.previousDesiredSpeedMps - 1e-6) {
        return "decelerate";
    }
    if (input.measuredSpeedMps < input.targetSpeedMps * 0.82) {
        return "accelerate";
    }
    return "cruise_approach";
}

function stopIntentReferenceDistance(speedMps: number): number {
    if (!Number.isFinite(speedMps) || speedMps < 0) {
        throw new RangeError("speedMps must be finite and non-negative");
    }
    return (speedMps * speedMps) / (2 * STOP_SIGN_BRAKING_DECELERATION_MPS2)
        + STOP_SIGN_STOP_INTENT_MARGIN_M;
}

function approachDesiredSpeed(input: StopSignPhaseClassificationInput): number {
    const brakingLimitedSpeedMps = Math.sqrt(
        Math.max(0, 2 * STOP_SIGN_BRAKING_DECELERATION_MPS2 * Math.max(0, input.remainingDistanceM - 0.15)),
    );
    const accelerationLimitedSpeedMps = accelerateToward(
        input.previousDesiredSpeedMps,
        input.targetSpeedMps,
        input.dtSeconds,
    );
    return Math.min(input.targetSpeedMps, brakingLimitedSpeedMps, accelerationLimitedSpeedMps);
}

function validatePhaseClassificationInput(input: StopSignPhaseClassificationInput) {
    for (const [label, value] of Object.entries(input)) {
        if (!Number.isFinite(value)) {
            throw new RangeError(`${label} must be finite`);
        }
    }
    if (input.remainingDistanceM < 0 || input.measuredSpeedMps < 0 || input.targetSpeedMps <= 0 || input.previousDesiredSpeedMps < 0) {
        throw new RangeError("distance and speeds must be non-negative, with a positive targetSpeedMps");
    }
    if (input.dtSeconds <= 0 || input.dtSeconds > 0.2) {
        throw new RangeError("dtSeconds must be within (0, 0.2]");
    }
}

function accelerateToward(current: number, target: number, dtSeconds: number): number {
    return Math.min(target, current + STOP_SIGN_LAUNCH_ACCELERATION_MPS2 * dtSeconds);
}

function trackingSteer(lateralErrorM: number, headingErrorDeg: number, speedMps: number): number {
    const headingCorrection = headingErrorDeg * Math.PI / 180;
    const crossTrackCorrection = Math.atan2(0.55 * lateralErrorM, speedMps + 1);
    return clamp((headingCorrection + crossTrackCorrection) / 0.733038306, -0.35, 0.35);
}

function supervision(
    input: StopSignExpertInput,
    desiredSpeedMps: number,
    desiredWheelSteerNormalized: number,
    throttle: number,
    brake: number,
    stopProbability: number,
    goProbability: number,
): StopSignExpertSupervision {
    return {
        phase: input.phase,
        desiredWheelSteerNormalized,
        desiredSpeedMps,
        throttle,
        brake,
        stopProbability,
        goProbability,
        distanceToStopLineM: input.remainingDistanceM,
    };
}

function validateInput(input: StopSignExpertInput) {
    for (const [label, value] of Object.entries(input)) {
        if (label === "phase") {
            continue;
        }
        if (!Number.isFinite(value)) {
            throw new RangeError(`${label} must be finite`);
        }
    }
    if (input.remainingDistanceM < 0 || input.measuredSpeedMps < 0 || input.targetSpeedMps <= 0) {
        throw new RangeError("distance and speeds must be non-negative, with a positive targetSpeedMps");
    }
    if (input.dtSeconds <= 0 || input.dtSeconds > 0.2) {
        throw new RangeError("dtSeconds must be within (0, 0.2]");
    }
}

function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
}
