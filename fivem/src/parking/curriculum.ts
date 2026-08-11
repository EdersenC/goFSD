import {relativePose} from "./geometry";
import {
    MAX_STRAIGHT_APPROACH_HEADING_ERROR_DEG,
    MAX_STRAIGHT_APPROACH_LATERAL_ERROR_M,
    MAX_STRAIGHT_START_DISTANCE_M,
    MAX_STRAIGHT_START_VERTICAL_ERROR_M,
    MIN_STRAIGHT_START_DISTANCE_M,
} from "./constraints";
import {
    ParkingAttemptPlan,
    ParkingPose,
    ParkingStartOffset,
} from "./types";

export const MIN_PARKING_ATTEMPTS = 1;
export const MAX_PARKING_ATTEMPTS = 50;
export const DEFAULT_PARKING_EVALUATION_SEED = "parking-forward-bay:evaluation";

export type ParkingEvaluationPlan = {
    seed: string
    plan: ParkingAttemptPlan
};

export function resolveParkingAttemptCount(value: unknown): number {
    const parsed = typeof value === "number" ? value : Number(value);
    if (!Number.isInteger(parsed) || parsed < MIN_PARKING_ATTEMPTS || parsed > MAX_PARKING_ATTEMPTS) {
        throw new Error(
            `attemptCount must be an integer from ${MIN_PARKING_ATTEMPTS} through ${MAX_PARKING_ATTEMPTS}`
        );
    }
    return parsed;
}

export function resolveStraightParkingStartOffset(
    startPose: ParkingPose,
    targetPose: ParkingPose
): ParkingStartOffset {
    requireFinitePose(startPose, "start");
    requireFinitePose(targetPose, "target");

    const relative = relativePose(startPose, targetPose);
    const startDistanceM = -relative.longitudinalM;
    if (
        startDistanceM < MIN_STRAIGHT_START_DISTANCE_M
        || startDistanceM > MAX_STRAIGHT_START_DISTANCE_M
    ) {
        throw new Error(
            `Parking start must be ${MIN_STRAIGHT_START_DISTANCE_M.toFixed(1)}-${MAX_STRAIGHT_START_DISTANCE_M.toFixed(1)}m `
            + `behind the target; measured ${startDistanceM.toFixed(2)}m`
        );
    }
    if (Math.abs(relative.lateralM) > MAX_STRAIGHT_APPROACH_LATERAL_ERROR_M) {
        throw new Error(
            `Parking start must be centered within ${MAX_STRAIGHT_APPROACH_LATERAL_ERROR_M.toFixed(2)}m; `
            + `measured ${relative.lateralM.toFixed(2)}m lateral error`
        );
    }
    if (Math.abs(relative.headingDeg) > MAX_STRAIGHT_APPROACH_HEADING_ERROR_DEG) {
        throw new Error(
            `Parking start heading must be within ${MAX_STRAIGHT_APPROACH_HEADING_ERROR_DEG.toFixed(0)} degrees; `
            + `measured ${relative.headingDeg.toFixed(2)} degrees`
        );
    }
    const verticalErrorM = startPose.coords[2] - targetPose.coords[2];
    if (Math.abs(verticalErrorM) > MAX_STRAIGHT_START_VERTICAL_ERROR_M) {
        throw new Error(
            `Parking start and target must be on the same level within ${MAX_STRAIGHT_START_VERTICAL_ERROR_M.toFixed(1)}m; `
            + `measured ${verticalErrorM.toFixed(2)}m`
        );
    }

    return {
        longitudinalM: relative.longitudinalM,
        lateralM: relative.lateralM,
        // relativePose reports target minus start; poseFromLocalOffset stores start minus target.
        headingDeg: -relative.headingDeg,
    };
}

export function buildForwardBayCurriculum(
    targetPose: ParkingPose,
    startPose: ParkingPose,
    attemptCount: number
): ParkingAttemptPlan[] {
    const count = resolveParkingAttemptCount(attemptCount);
    const startOffset = resolveStraightParkingStartOffset(startPose, targetPose);
    return Array.from({length: count}, (_, attemptIndex) => ({
        attemptIndex,
        startOffset: {...startOffset},
        startPose: clonePose(startPose),
    }));
}

export function buildForwardBayEvaluationPlan(
    targetPose: ParkingPose,
    startPose: ParkingPose,
    requestedSeed?: string
): ParkingEvaluationPlan {
    const seed = requestedSeed?.trim() || DEFAULT_PARKING_EVALUATION_SEED;
    const plan = buildForwardBayCurriculum(targetPose, startPose, 1)[0];
    if (!plan) {
        throw new Error("parking evaluation plan invariant violated");
    }
    return {seed, plan};
}

function requireFinitePose(pose: ParkingPose, label: string) {
    if (
        !Array.isArray(pose.coords)
        || pose.coords.length !== 3
        || !pose.coords.every(Number.isFinite)
        || !Number.isFinite(pose.heading)
    ) {
        throw new Error(`Parking ${label} pose must contain finite coordinates and heading`);
    }
}

function clonePose(pose: ParkingPose): ParkingPose {
    return {coords: [...pose.coords], heading: pose.heading};
}
