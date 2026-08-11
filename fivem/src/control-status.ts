export type ControlStopStatus = "idle" | "stopping";

export function resolveStopCommandStatus(
    stoppedSynchronously: boolean,
    asynchronousOperationActive: boolean,
): ControlStopStatus {
    return stoppedSynchronously && !asynchronousOperationActive ? "idle" : "stopping";
}
