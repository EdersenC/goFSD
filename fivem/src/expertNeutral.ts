type ActuatorAppliedState = {
    enabled?: unknown
    steer?: unknown
    throttle?: unknown
    brakePressureAvg?: unknown
    handbrake?: unknown
};

const terminalInferenceStates = new Set(["idle", "succeeded", "failed", "error"]);

type InferenceStatus = {
    state?: unknown
    active?: unknown
    stoppedAt?: unknown
};

export function isInferenceCleanupComplete(value: unknown): boolean {
    if (!value || typeof value !== "object") {
        return false;
    }
    const status = value as InferenceStatus;
    const state = typeof status.state === "string" ? status.state.trim().toLowerCase() : "";
    if (!terminalInferenceStates.has(state)) {
        return false;
    }
    if (typeof status.active === "boolean") {
        return !status.active;
    }

    // Compatibility with older backends: terminal sessions set stoppedAt only
    // after process cleanup. A pristine idle backend has no session to clean up.
    return state === "idle"
        || (typeof status.stoppedAt === "string" && status.stoppedAt.trim().length > 0);
}

export function isActuatorStateNeutral(value: unknown): boolean {
    if (!value || typeof value !== "object") {
        return false;
    }
    const applied = (value as {applied?: ActuatorAppliedState}).applied;
    if (!applied || applied.enabled !== false || applied.handbrake === true) {
        return false;
    }
    return isZero(applied.steer)
        && isZero(applied.throttle)
        && isZero(applied.brakePressureAvg);
}

function isZero(value: unknown): boolean {
    return typeof value === "number" && Number.isFinite(value) && Math.abs(value) <= 0.0001;
}
