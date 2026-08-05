export const SULTAN_STEERING_FULL_LOCK_RADIANS = 0.733038306;

const minimumHandlingSteeringLockDegrees = 5;
const maximumHandlingSteeringLockDegrees = 90;
const degreesToRadians = Math.PI / 180;

export type VehicleSteeringSample = {
    normalized: number
    wheelAngleRadians: number
    fullLockRadians: number
};

/** Converts physical wheel telemetry into the controller's normalized steer contract. */
export function normalizeVehicleSteering(
    wheelAngleRadians: number,
    handlingSteeringLockDegrees: number
): VehicleSteeringSample {
    const safeWheelAngle = Number.isFinite(wheelAngleRadians) ? wheelAngleRadians : 0;
    const fullLockRadians = resolveSteeringFullLockRadians(handlingSteeringLockDegrees);
    return {
        normalized: clamp(safeWheelAngle / fullLockRadians, -1, 1),
        wheelAngleRadians: safeWheelAngle,
        fullLockRadians,
    };
}

export function resolveSteeringFullLockRadians(handlingSteeringLockDegrees: number): number {
    if (
        !Number.isFinite(handlingSteeringLockDegrees)
        || handlingSteeringLockDegrees < minimumHandlingSteeringLockDegrees
        || handlingSteeringLockDegrees > maximumHandlingSteeringLockDegrees
    ) {
        return SULTAN_STEERING_FULL_LOCK_RADIANS;
    }
    return handlingSteeringLockDegrees * degreesToRadians;
}

function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
}
