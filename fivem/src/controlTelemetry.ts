export type EgoControlTelemetry = {
    currentSpeed: number
    currentYaw: number
    yawRate: number
    steering: number
    acceleration: number
    brakePressureAvg: number
    vehicleExists: boolean
    isInVehicle: boolean
    positionX?: number
    positionY?: number
    positionZ?: number
    velocityX?: number
    velocityY?: number
    velocityZ?: number
    pitchDeg?: number
    rollDeg?: number
    gear?: number
    rpm?: number
    /** Raw physical front-wheel angle in radians. */
    wheelAngle?: number
    /** Per-vehicle physical full-lock angle in radians. */
    wheelSteeringFullLock?: number
    onGround?: boolean
    collisionState?: string
    routeDirectionCode: number
    routeDirectionDistanceM: number
    routeDirectionUnknown: number
    routeDirectionKeepStraight: number
    routeDirectionTurnLeft: number
    routeDirectionTurnRight: number
    routeDirectionRerouteWrongWay: number
    routeForwardDelta: number | null
    routeHeadingError: number | null
    routeDistance: number | null
    hasLeadVehicle: boolean
    leadVehicleDistance: number | null
    gameTimeMs: number
};

/** Shared cadence for expert samples and the live-control telemetry history. */
export const CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS = 50;

export function createNoEgoControlTelemetry(gameTimeMs: number): EgoControlTelemetry {
    return {
        currentSpeed: 0,
        currentYaw: 0,
        yawRate: 0,
        steering: 0,
        acceleration: 0,
        brakePressureAvg: 0,
        vehicleExists: false,
        isInVehicle: false,
        collisionState: "",
        routeDirectionCode: 0,
        routeDirectionDistanceM: 0,
        routeDirectionUnknown: 1,
        routeDirectionKeepStraight: 0,
        routeDirectionTurnLeft: 0,
        routeDirectionTurnRight: 0,
        routeDirectionRerouteWrongWay: 0,
        routeForwardDelta: 0,
        routeHeadingError: 0,
        routeDistance: 0,
        hasLeadVehicle: false,
        leadVehicleDistance: null,
        gameTimeMs,
    };
}
