import {StopSignPose} from "./types";

export type StopSignCatalogPosition = {
    x: number
    y: number
    z: number
};

export type StopSignLocationProbeRequest = {
    catalogId: string
    catalogPosition: StopSignCatalogPosition
    headingOffsetDeg?: number
};

export type StopSignLocationProbeResult = {
    catalogId: string
    catalogPosition: StopSignCatalogPosition
    roadNodePose: StopSignPose
    settledPose: StopSignPose
    distanceFromCatalogM: number
    collisionLoaded: boolean
};

export type StopSignLocationProbeOperations = {
    closestVehicleNode: (position: StopSignCatalogPosition) => [boolean, [number, number, number], number]
    entityCoords: (entity: number) => [number, number, number]
    entityHeading: (entity: number) => number
    entityPitch: (entity: number) => number
    entityRoll: (entity: number) => number
    entitySpeed: (entity: number) => number
    entityOnGround: (entity: number) => boolean
    freezeEntity: (entity: number, frozen: boolean) => void
    setEntityCoords: (entity: number, position: [number, number, number]) => void
    setEntityHeading: (entity: number, heading: number) => void
    setVehicleForwardSpeed: (vehicle: number, speedMps: number) => void
    setVehicleOnGround: (vehicle: number) => boolean
    setFocus: (position: [number, number, number]) => void
    clearFocus: () => void
    requestCollision: (position: [number, number, number]) => void
    collisionLoaded: (entity: number) => boolean
    wait: (milliseconds: number) => Promise<void>
};

const MAX_CATALOG_TO_ROAD_NODE_METERS = 30;
const MAX_LEVEL_ANGLE_DEGREES = 5;
const MAX_SETTLED_SPEED_MPS = 0.1;
const COLLISION_ATTEMPTS = 40;
const COLLISION_WAIT_MS = 50;
const SETTLE_WAIT_MS = 600;
const NODE_HEIGHT_CLEARANCE_M = 1;

export async function placeVehicleAtStopSignLocation(
    vehicle: number,
    rawRequest: unknown,
    operations: StopSignLocationProbeOperations,
    assertCurrent: () => void = () => undefined,
): Promise<StopSignLocationProbeResult> {
    if (!Number.isSafeInteger(vehicle) || vehicle <= 0) {
        throw new Error("Stop-sign probe requires a valid managed vehicle");
    }
    const request = parseStopSignLocationProbeRequest(rawRequest);
    assertCurrent();

    const [foundNode, nodePosition, nodeHeading] = operations.closestVehicleNode(request.catalogPosition);
    if (!foundNode || !isFinitePosition(nodePosition) || !Number.isFinite(nodeHeading)) {
        throw new Error(`Stop-sign probe could not resolve a drivable road node for ${request.catalogId}`);
    }
    const distanceFromCatalogM = distance3D(request.catalogPosition, nodePosition);
    if (distanceFromCatalogM > MAX_CATALOG_TO_ROAD_NODE_METERS) {
        throw new Error(
            `Stop-sign probe road node is ${distanceFromCatalogM.toFixed(1)}m from ${request.catalogId}; maximum is ${MAX_CATALOG_TO_ROAD_NODE_METERS}m`,
        );
    }

    const heading = normalizeHeading(nodeHeading + request.headingOffsetDeg);
    const destination: [number, number, number] = [
        nodePosition[0],
        nodePosition[1],
        nodePosition[2] + NODE_HEIGHT_CLEARANCE_M,
    ];

    operations.freezeEntity(vehicle, true);
    operations.setFocus(destination);
    try {
        assertCurrent();
        operations.requestCollision(destination);
        operations.setEntityCoords(vehicle, destination);
        operations.setEntityHeading(vehicle, heading);
        operations.setVehicleForwardSpeed(vehicle, 0);
        operations.setVehicleOnGround(vehicle);
        const collisionLoaded = await waitForCollision(vehicle, operations, assertCurrent);
        operations.setVehicleOnGround(vehicle);
        operations.setEntityHeading(vehicle, heading);
        return await settleAndValidate(
            vehicle,
            request,
            nodePosition,
            heading,
            distanceFromCatalogM,
            collisionLoaded,
            operations,
            assertCurrent,
        );
    } finally {
        operations.clearFocus();
        operations.freezeEntity(vehicle, false);
    }
}

export function parseStopSignLocationProbeRequest(raw: unknown): Required<StopSignLocationProbeRequest> {
    if (!isRecord(raw)) {
        throw new Error("stopSignProbe must be an object");
    }
    const catalogId = String(raw.catalogId ?? "").trim();
    if (!catalogId || catalogId.length > 120) {
        throw new Error("stopSignProbe.catalogId must contain between 1 and 120 characters");
    }
    const catalogPosition = parseCatalogPosition(raw.catalogPosition);
    const headingOffsetDeg = raw.headingOffsetDeg == null ? 0 : Number(raw.headingOffsetDeg);
    if (headingOffsetDeg !== 0 && headingOffsetDeg !== 180) {
        throw new Error("stopSignProbe.headingOffsetDeg must be either 0 or 180");
    }
    return {catalogId, catalogPosition, headingOffsetDeg};
}

async function waitForCollision(
    vehicle: number,
    operations: StopSignLocationProbeOperations,
    assertCurrent: () => void,
): Promise<boolean> {
    for (let attempt = 0; attempt < COLLISION_ATTEMPTS; attempt += 1) {
        assertCurrent();
        if (operations.collisionLoaded(vehicle)) {
            return true;
        }
        await operations.wait(COLLISION_WAIT_MS);
    }
    return operations.collisionLoaded(vehicle);
}

async function settleAndValidate(
    vehicle: number,
    request: Required<StopSignLocationProbeRequest>,
    nodePosition: [number, number, number],
    heading: number,
    distanceFromCatalogM: number,
    collisionLoaded: boolean,
    operations: StopSignLocationProbeOperations,
    assertCurrent: () => void,
): Promise<StopSignLocationProbeResult> {
    operations.freezeEntity(vehicle, false);
    await operations.wait(SETTLE_WAIT_MS);
    assertCurrent();
    operations.setVehicleOnGround(vehicle);
    operations.setVehicleForwardSpeed(vehicle, 0);
    operations.setEntityHeading(vehicle, heading);
    await operations.wait(100);
    assertCurrent();

    if (!collisionLoaded) {
        throw new Error(`Stop-sign probe collision did not load for ${request.catalogId}`);
    }
    const settledPose = readEntityPose(vehicle, operations);
    const pitch = operations.entityPitch(vehicle);
    const roll = operations.entityRoll(vehicle);
    const speedMps = operations.entitySpeed(vehicle);
    if (![pitch, roll, speedMps].every(Number.isFinite)) {
        throw new Error(`Stop-sign probe received invalid settled telemetry for ${request.catalogId}`);
    }
    if (!operations.entityOnGround(vehicle)) {
        throw new Error(`Stop-sign probe vehicle is not on the ground at ${request.catalogId}`);
    }
    if (Math.abs(pitch) > MAX_LEVEL_ANGLE_DEGREES || Math.abs(roll) > MAX_LEVEL_ANGLE_DEGREES) {
        throw new Error(
            `Stop-sign probe vehicle is not level at ${request.catalogId}: pitch=${pitch.toFixed(1)} roll=${roll.toFixed(1)}`,
        );
    }
    if (Math.abs(speedMps) > MAX_SETTLED_SPEED_MPS) {
        throw new Error(`Stop-sign probe vehicle did not settle at ${request.catalogId}: speed=${speedMps.toFixed(2)}m/s`);
    }

    return {
        catalogId: request.catalogId,
        catalogPosition: {...request.catalogPosition},
        roadNodePose: {x: nodePosition[0], y: nodePosition[1], z: nodePosition[2], heading},
        settledPose,
        distanceFromCatalogM,
        collisionLoaded,
    };
}

function readEntityPose(entity: number, operations: StopSignLocationProbeOperations): StopSignPose {
    const coords = operations.entityCoords(entity);
    const heading = operations.entityHeading(entity);
    if (!isFinitePosition(coords) || !Number.isFinite(heading)) {
        throw new Error("Stop-sign probe could not read the settled vehicle pose");
    }
    return {x: coords[0], y: coords[1], z: coords[2], heading: normalizeHeading(heading)};
}

function parseCatalogPosition(raw: unknown): StopSignCatalogPosition {
    if (!isRecord(raw)) {
        throw new Error("stopSignProbe.catalogPosition must be an object");
    }
    const position = {x: Number(raw.x), y: Number(raw.y), z: Number(raw.z)};
    if (![position.x, position.y, position.z].every(Number.isFinite)) {
        throw new Error("stopSignProbe.catalogPosition must contain finite x, y, and z coordinates");
    }
    if (Math.abs(position.x) > 10_000 || Math.abs(position.y) > 10_000 || position.z < -1_000 || position.z > 3_000) {
        throw new Error("stopSignProbe.catalogPosition is outside supported GTA world bounds");
    }
    return position;
}

function distance3D(a: StopSignCatalogPosition, b: [number, number, number]): number {
    return Math.hypot(a.x - b[0], a.y - b[1], a.z - b[2]);
}

function normalizeHeading(value: number): number {
    return ((value % 360) + 360) % 360;
}

function isFinitePosition(value: unknown): value is [number, number, number] {
    return Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every(Number.isFinite);
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
