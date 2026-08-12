import {StopSignPose} from "./types";

export type StopSignRelativePose = {
    longitudinalM: number
    lateralM: number
    headingErrorDeg: number
    distanceM: number
};

export type StopSignTrackingError = {
    lateralErrorM: number
    headingErrorDeg: number
    blend: number
};

const maximumApproachTangentBlendM = 24;
const minimumApproachTangentBlendM = 8;
const approachTangentBlendRouteFraction = 0.4;

export function gtaForwardVector(heading: number): [number, number] {
    const radians = heading * Math.PI / 180;
    return [-Math.sin(radians), Math.cos(radians)];
}

export function gtaRightVector(heading: number): [number, number] {
    const radians = heading * Math.PI / 180;
    return [Math.cos(radians), Math.sin(radians)];
}

export function poseBehind(reference: StopSignPose, distanceM: number): StopSignPose {
    if (!Number.isFinite(distanceM) || distanceM < 0) {
        throw new RangeError("distanceM must be finite and non-negative");
    }
    const [forwardX, forwardY] = gtaForwardVector(reference.heading);
    return {
        x: reference.x - forwardX * distanceM,
        y: reference.y - forwardY * distanceM,
        z: reference.z,
        heading: normalizeHeading(reference.heading),
    };
}

export function poseAhead(reference: StopSignPose, distanceM: number): StopSignPose {
    if (!Number.isFinite(distanceM) || distanceM < 0) {
        throw new RangeError("distanceM must be finite and non-negative");
    }
    const [forwardX, forwardY] = gtaForwardVector(reference.heading);
    return {
        x: reference.x + forwardX * distanceM,
        y: reference.y + forwardY * distanceM,
        z: reference.z,
        heading: normalizeHeading(reference.heading),
    };
}

export function relativeStopLinePose(pose: StopSignPose, stopLine: StopSignPose): StopSignRelativePose {
    const [forwardX, forwardY] = gtaForwardVector(stopLine.heading);
    const [rightX, rightY] = gtaRightVector(stopLine.heading);
    const deltaX = pose.x - stopLine.x;
    const deltaY = pose.y - stopLine.y;
    const longitudinalM = deltaX * forwardX + deltaY * forwardY;
    const lateralM = deltaX * rightX + deltaY * rightY;
    return {
        longitudinalM,
        lateralM,
        headingErrorDeg: shortestHeadingDelta(stopLine.heading, pose.heading),
        distanceM: Math.hypot(longitudinalM, lateralM),
    };
}

/**
 * Follows the captured Start tangent before smoothly joining the Stop tangent.
 * Aiming at the Stop tangent from frame one can demand full lock when a road curves
 * between the two saved poses, even though the vehicle is correctly aligned at Start.
 */
export function approachTrackingError(
    pose: StopSignPose,
    startPose: StopSignPose,
    stopPose: StopSignPose,
): StopSignTrackingError {
    const startError = relativeStopLinePose(pose, startPose);
    const stopError = relativeStopLinePose(pose, stopPose);
    const routeDistanceM = Math.hypot(stopPose.x - startPose.x, stopPose.y - startPose.y);
    const blendDistanceM = Math.min(
        maximumApproachTangentBlendM,
        Math.max(minimumApproachTangentBlendM, routeDistanceM * approachTangentBlendRouteFraction),
    );
    const progress = clamp(startError.distanceM / blendDistanceM, 0, 1);
    const blend = progress * progress * (3 - 2 * progress);
    return {
        lateralErrorM: lerp(startError.lateralM, stopError.lateralM, blend),
        headingErrorDeg: lerp(startError.headingErrorDeg, stopError.headingErrorDeg, blend),
        blend,
    };
}

export function normalizeHeading(heading: number): number {
    return ((heading % 360) + 360) % 360;
}

export function shortestHeadingDelta(target: number, source: number): number {
    return ((target - source + 540) % 360) - 180;
}

function lerp(start: number, end: number, amount: number): number {
    return start + (end - start) * amount;
}

function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
}
