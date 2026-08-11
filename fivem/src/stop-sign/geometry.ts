import {StopSignPose} from "./types";

export type StopSignRelativePose = {
    longitudinalM: number
    lateralM: number
    headingErrorDeg: number
    distanceM: number
};

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

export function normalizeHeading(heading: number): number {
    return ((heading % 360) + 360) % 360;
}

export function shortestHeadingDelta(target: number, source: number): number {
    return ((target - source + 540) % 360) - 180;
}
