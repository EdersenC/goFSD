import {poseFromLocalOffset} from "./geometry";
import {ParkingBay, ParkingPose, ParkingVector3} from "./types";

export const PARKING_TARGET_MARKER_ID = "orange-bay-highlight-v1";

const markerHeightOffsetM = 0.04;
const fillColor = {red: 255, green: 128, blue: 0, alpha: 78};
const edgeColor = {red: 255, green: 176, blue: 48, alpha: 235};

export type ParkingMarkerCorners = readonly [
    ParkingVector3,
    ParkingVector3,
    ParkingVector3,
    ParkingVector3,
];

export function parkingMarkerCorners(target: ParkingPose, bay: ParkingBay): ParkingMarkerCorners {
    const halfLengthM = bay.lengthM / 2;
    const halfWidthM = bay.widthM / 2;
    return [
        raised(poseFromLocalOffset(target, -halfLengthM, -halfWidthM).coords),
        raised(poseFromLocalOffset(target, -halfLengthM, halfWidthM).coords),
        raised(poseFromLocalOffset(target, halfLengthM, halfWidthM).coords),
        raised(poseFromLocalOffset(target, halfLengthM, -halfWidthM).coords),
    ];
}

export function drawParkingTargetMarker(target: ParkingPose, bay: ParkingBay) {
    const [approachLeft, approachRight, farRight, farLeft] = parkingMarkerCorners(target, bay);
    drawTriangle(approachLeft, approachRight, farRight);
    drawTriangle(approachLeft, farRight, farLeft);
    // Both windings keep the cue visible across camera angles and uneven ground.
    drawTriangle(farRight, approachRight, approachLeft);
    drawTriangle(farLeft, farRight, approachLeft);
    drawEdge(approachLeft, approachRight);
    drawEdge(approachRight, farRight);
    drawEdge(farRight, farLeft);
    drawEdge(farLeft, approachLeft);
}

function raised(coords: ParkingVector3): ParkingVector3 {
    return [coords[0], coords[1], coords[2] + markerHeightOffsetM];
}

function drawTriangle(first: ParkingVector3, second: ParkingVector3, third: ParkingVector3) {
    DrawPoly(
        ...first,
        ...second,
        ...third,
        fillColor.red,
        fillColor.green,
        fillColor.blue,
        fillColor.alpha
    );
}

function drawEdge(first: ParkingVector3, second: ParkingVector3) {
    DrawLine(
        ...first,
        ...second,
        edgeColor.red,
        edgeColor.green,
        edgeColor.blue,
        edgeColor.alpha
    );
}
