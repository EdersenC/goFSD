export type StopSignCatalogPosition = {
    x: number
    y: number
    z: number
};

export type CatalogWaypointOperations = {
    setNewWaypoint: (x: number, y: number) => void
};

export function setStopSignCatalogWaypoint(
    value: unknown,
    operations: CatalogWaypointOperations,
): StopSignCatalogPosition {
    const position = parseStopSignCatalogPosition(value);
    operations.setNewWaypoint(position.x, position.y);
    return position;
}

export function parseStopSignCatalogPosition(value: unknown): StopSignCatalogPosition {
    if (!isRecord(value)) {
        throw new Error("setStopSignCatalogWaypoint command missing stopSignCatalogPosition");
    }
    const position = {
        x: requireCoordinate("x", value.x),
        y: requireCoordinate("y", value.y),
        z: requireCoordinate("z", value.z),
    };
    if (
        position.x < -10_000 || position.x > 10_000
        || position.y < -10_000 || position.y > 10_000
        || position.z < -1_000 || position.z > 3_000
    ) {
        throw new Error("stopSignCatalogPosition is outside the supported GTA world bounds");
    }
    return position;
}

function requireCoordinate(label: string, value: unknown): number {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`stopSignCatalogPosition.${label} must be finite`);
    }
    return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
