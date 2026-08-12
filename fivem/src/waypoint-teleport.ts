export type TeleportPosition = [number, number, number];

export type WaypointTeleportOperations = {
    getEntityCoords: (entity: number) => TeleportPosition
    getEntityHeading: (entity: number) => number
    freezeEntity: (entity: number, frozen: boolean) => void
    setEntityCoords: (entity: number, position: TeleportPosition) => void
    setEntityHeading: (entity: number, heading: number) => void
    setFocus: (position: TeleportPosition) => void
    clearFocus: () => void
    requestCollision: (position: TeleportPosition) => void
    getGroundZ: (position: TeleportPosition) => [boolean, number]
    getClosestVehicleNode: (position: TeleportPosition) => [boolean, TeleportPosition, number]
    hasCollisionLoadedAroundEntity: (entity: number) => boolean
    setVehicleOnGround: (vehicle: number) => boolean
    wait: (milliseconds: number) => Promise<void>
};

export type WaypointTeleportResult = {
    success: boolean
    usedVehicleNodeFallback: boolean
    collisionLoaded: boolean
    reason?: string
};

const GROUND_PROBE_MAX_Z = 1000;
const GROUND_PROBE_MIN_Z = -100;
const GROUND_PROBE_STEP = 50;
const GROUND_PROBE_PASSES = 2;
const GROUND_PROBE_WAIT_MS = 40;
const COLLISION_WAIT_ATTEMPTS = 30;
const COLLISION_WAIT_MS = 50;
const DESTINATION_CLEARANCE_METERS = 1;

export async function teleportEntityToWaypoint(
    entity: number,
    isVehicle: boolean,
    waypoint: TeleportPosition,
    operations: WaypointTeleportOperations,
): Promise<WaypointTeleportResult> {
    if (entity === 0 || !isFinitePosition(waypoint)) {
        return failedResult("Invalid entity or waypoint");
    }

    const originalPosition = operations.getEntityCoords(entity);
    const originalHeading = operations.getEntityHeading(entity);
    if (!isFinitePosition(originalPosition) || !Number.isFinite(originalHeading)) {
        return failedResult("Could not read the current entity position");
    }

    operations.freezeEntity(entity, true);
    operations.setFocus([waypoint[0], waypoint[1], GROUND_PROBE_MAX_Z]);

    let movedFromOriginalPosition = false;
    try {
        operations.setEntityCoords(entity, [waypoint[0], waypoint[1], GROUND_PROBE_MAX_Z]);
        movedFromOriginalPosition = true;

        const resolvedDestination = await resolveDestination(waypoint, operations);
        if (resolvedDestination === null) {
            restoreEntity(entity, originalPosition, originalHeading, operations);
            movedFromOriginalPosition = false;
            return failedResult("Could not find safe ground at the waypoint");
        }

        const destination: TeleportPosition = [
            resolvedDestination.position[0],
            resolvedDestination.position[1],
            resolvedDestination.position[2] + DESTINATION_CLEARANCE_METERS,
        ];
        operations.requestCollision(destination);
        operations.setEntityCoords(entity, destination);
        operations.setEntityHeading(entity, originalHeading);
        if (isVehicle) {
            operations.setVehicleOnGround(entity);
        }

        const collisionLoaded = await waitForCollision(entity, operations);
        if (isVehicle) {
            operations.setVehicleOnGround(entity);
        }

        return {
            success: true,
            usedVehicleNodeFallback: resolvedDestination.usedVehicleNodeFallback,
            collisionLoaded,
        };
    } catch (error) {
        if (movedFromOriginalPosition) {
            restoreEntity(entity, originalPosition, originalHeading, operations);
        }
        return failedResult(error instanceof Error ? error.message : "Teleport failed unexpectedly");
    } finally {
        operations.clearFocus();
        operations.freezeEntity(entity, false);
    }
}

async function resolveDestination(
    waypoint: TeleportPosition,
    operations: WaypointTeleportOperations,
): Promise<{position: TeleportPosition; usedVehicleNodeFallback: boolean} | null> {
    for (let pass = 0; pass < GROUND_PROBE_PASSES; pass += 1) {
        for (let z = GROUND_PROBE_MAX_Z; z >= GROUND_PROBE_MIN_Z; z -= GROUND_PROBE_STEP) {
            const probe: TeleportPosition = [waypoint[0], waypoint[1], z];
            operations.requestCollision(probe);
            await operations.wait(GROUND_PROBE_WAIT_MS);
            const [foundGround, groundZ] = operations.getGroundZ(probe);
            if (foundGround && Number.isFinite(groundZ)) {
                return {
                    position: [waypoint[0], waypoint[1], groundZ],
                    usedVehicleNodeFallback: false,
                };
            }
        }
    }

    const [foundNode, nodePosition] = operations.getClosestVehicleNode(waypoint);
    if (!foundNode || !isFinitePosition(nodePosition)) {
        return null;
    }
    return {position: nodePosition, usedVehicleNodeFallback: true};
}

async function waitForCollision(
    entity: number,
    operations: WaypointTeleportOperations,
): Promise<boolean> {
    for (let attempt = 0; attempt < COLLISION_WAIT_ATTEMPTS; attempt += 1) {
        if (operations.hasCollisionLoadedAroundEntity(entity)) {
            return true;
        }
        await operations.wait(COLLISION_WAIT_MS);
    }
    return operations.hasCollisionLoadedAroundEntity(entity);
}

function restoreEntity(
    entity: number,
    originalPosition: TeleportPosition,
    originalHeading: number,
    operations: WaypointTeleportOperations,
) {
    operations.requestCollision(originalPosition);
    operations.setEntityCoords(entity, originalPosition);
    operations.setEntityHeading(entity, originalHeading);
}

function failedResult(reason: string): WaypointTeleportResult {
    return {
        success: false,
        usedVehicleNodeFallback: false,
        collisionLoaded: false,
        reason,
    };
}

function isFinitePosition(position: TeleportPosition): boolean {
    return position.length === 3 && position.every(Number.isFinite);
}
