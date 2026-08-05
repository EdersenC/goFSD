import {
    ParkingBay,
    ParkingPose,
    ParkingVector3,
    VehicleFootprint,
} from "./types";

export type ParkingRelativePose = {
    longitudinalM: number
    lateralM: number
    headingDeg: number
    distanceM: number
};

export function normalizeHeadingDegrees(heading: number): number {
    let normalized = heading % 360;
    if (normalized < 0) {
        normalized += 360;
    }
    return normalized;
}

export function headingDeltaDegrees(targetHeading: number, sourceHeading: number): number {
    let delta = normalizeHeadingDegrees(targetHeading) - normalizeHeadingDegrees(sourceHeading);
    while (delta > 180) {
        delta -= 360;
    }
    while (delta < -180) {
        delta += 360;
    }
    return delta;
}

// GTA headings use 0 degrees at +Y and 90 degrees at -X.
export function gtaForwardVector(heading: number): ParkingVector3 {
    const radians = heading * (Math.PI / 180);
    return [-Math.sin(radians), Math.cos(radians), 0];
}

export function gtaRightVector(heading: number): ParkingVector3 {
    const radians = heading * (Math.PI / 180);
    return [Math.cos(radians), Math.sin(radians), 0];
}

export function gtaHeadingFromVector(vector: ParkingVector3): number {
    return normalizeHeadingDegrees(Math.atan2(-vector[0], vector[1]) * (180 / Math.PI));
}

export function dotProduct(a: ParkingVector3, b: ParkingVector3): number {
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

export function subtractVectors(a: ParkingVector3, b: ParkingVector3): ParkingVector3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vectorLength(vector: ParkingVector3): number {
    return Math.sqrt(dotProduct(vector, vector));
}

export function distanceBetween(a: ParkingVector3, b: ParkingVector3): number {
    return vectorLength(subtractVectors(a, b));
}

export function poseFromLocalOffset(
    target: ParkingPose,
    longitudinalM: number,
    lateralM: number,
    headingOffsetDeg = 0
): ParkingPose {
    const forward = gtaForwardVector(target.heading);
    const right = gtaRightVector(target.heading);
    return {
        coords: [
            target.coords[0] + (forward[0] * longitudinalM) + (right[0] * lateralM),
            target.coords[1] + (forward[1] * longitudinalM) + (right[1] * lateralM),
            target.coords[2],
        ],
        heading: normalizeHeadingDegrees(target.heading + headingOffsetDeg),
    };
}

export function relativePose(pose: ParkingPose, target: ParkingPose): ParkingRelativePose {
    const delta = subtractVectors(pose.coords, target.coords);
    const forward = gtaForwardVector(target.heading);
    const right = gtaRightVector(target.heading);
    const longitudinalM = dotProduct(delta, forward);
    const lateralM = dotProduct(delta, right);
    return {
        longitudinalM,
        lateralM,
        headingDeg: headingDeltaDegrees(target.heading, pose.heading),
        distanceM: Math.sqrt((longitudinalM * longitudinalM) + (lateralM * lateralM)),
    };
}

export function isVehicleFootprintInsideBay(
    pose: ParkingPose,
    target: ParkingPose,
    bay: ParkingBay,
    footprint: VehicleFootprint
): boolean {
    const longitudinalOffsets = [-footprint.halfLengthM, footprint.halfLengthM];
    const lateralOffsets = [-footprint.halfWidthM, footprint.halfWidthM];
    const halfBayLength = bay.lengthM / 2;
    const halfBayWidth = bay.widthM / 2;

    for (const longitudinalM of longitudinalOffsets) {
        for (const lateralM of lateralOffsets) {
            const corner = poseFromLocalOffset(pose, longitudinalM, lateralM).coords;
            const relativeCorner = relativePose({coords: corner, heading: pose.heading}, target);
            if (
                Math.abs(relativeCorner.longitudinalM) > halfBayLength
                || Math.abs(relativeCorner.lateralM) > halfBayWidth
            ) {
                return false;
            }
        }
    }
    return true;
}
