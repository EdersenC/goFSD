import {STOP_SIGN_STOP_SPEED_MPS} from "./expert";

const MAX_ATTEMPT_ROAD_GRADE_DEG = 15;
const MAX_ATTEMPT_ROLL_DEG = 5;

export function updateDepartureObserved(
    alreadyObserved: boolean,
    speedMps: number,
    longitudinalFromStartM: number,
    lateralFromStartM: number,
): boolean {
    return alreadyObserved
        || speedMps >= 0.5
        || Math.hypot(longitudinalFromStartM, lateralFromStartM) >= 1;
}

/** Maintains a fixed-period loop without accumulating work-time drift. */
export function nextFixedIntervalDeadlineMs(previousDeadlineMs: number, nowMs: number, intervalMs: number): number {
    if (![previousDeadlineMs, nowMs, intervalMs].every(Number.isFinite) || intervalMs <= 0) {
        throw new RangeError("fixed interval scheduling requires finite timestamps and a positive interval");
    }
    const scheduled = previousDeadlineMs + intervalMs;
    return scheduled > nowMs ? scheduled : nowMs + intervalMs;
}

export function shouldReportStoppedTooEarly(
    departureObserved: boolean,
    dwellStartedAtMs: number | null,
    speedMps: number,
    remainingDistanceM: number,
): boolean {
    return departureObserved
        && dwellStartedAtMs === null
        && speedMps <= STOP_SIGN_STOP_SPEED_MPS
        && remainingDistanceM > 3;
}

export type StopSignAttemptVehicleState = {
    exists: boolean
    playerIsDriver: boolean
    engineRunning: boolean
    onAllWheels: boolean
    speedMps: number
    pitchDeg: number
    rollDeg: number
    collided: boolean
};

/** Rejects a bad spawn before capture starts so setup failures never become training clips. */
export function validateStopSignAttemptVehicleState(state: StopSignAttemptVehicleState): string | null {
    if (!state.exists) {
        return "collection vehicle does not exist";
    }
    if (!state.playerIsDriver) {
        return "player is not in the managed driver seat";
    }
    if (!state.engineRunning) {
        return "collection vehicle engine is not running";
    }
    if (!state.onAllWheels) {
        return "collection vehicle is not on all wheels";
    }
    if (![state.speedMps, state.pitchDeg, state.rollDeg].every(Number.isFinite)) {
        return "collection vehicle readiness telemetry is invalid";
    }
    if (Math.abs(state.speedMps) > 0.1) {
        return `collection vehicle did not settle: speed=${state.speedMps.toFixed(2)}m/s`;
    }
    if (Math.abs(state.pitchDeg) > MAX_ATTEMPT_ROAD_GRADE_DEG || Math.abs(state.rollDeg) > MAX_ATTEMPT_ROLL_DEG) {
        return `collection vehicle is not level: pitch=${state.pitchDeg.toFixed(1)} roll=${state.rollDeg.toFixed(1)}`;
    }
    if (state.collided) {
        return "collection vehicle spawned in contact with another object";
    }
    return null;
}
