export type ParkingControlStatus = "idle" | "runningScene" | "stopping";

export function resolveStopCommandStatus(
    stoppedSynchronously: boolean,
    parkingOperationActive: boolean
): ParkingControlStatus {
    return stoppedSynchronously && !parkingOperationActive ? "idle" : "stopping";
}

export function resolveParkingOperationCompletionStatus(
    stopRequested: boolean,
    finalHoldApplied: boolean
): ParkingControlStatus {
    if (!stopRequested) {
        return "runningScene";
    }
    return finalHoldApplied ? "idle" : "stopping";
}
