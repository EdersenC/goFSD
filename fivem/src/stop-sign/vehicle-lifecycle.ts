const defaultMaximumSpawnAttempts = 3;

export function canReuseStopSignVehicle(
    currentModel: string | null,
    requestedModel: string,
    currentVehicleIsValidAndOwned: boolean,
): boolean {
    if (!currentVehicleIsValidAndOwned || !currentModel) {
        return false;
    }
    return currentModel.trim().toLowerCase() === requestedModel.trim().toLowerCase();
}

export async function spawnStopSignVehicleWithRetries<T>(
    spawn: (attempt: number) => Promise<T>,
    stopRequested: () => boolean,
    waitBeforeRetry: (failedAttempt: number) => Promise<void>,
    maximumAttempts = defaultMaximumSpawnAttempts,
): Promise<T> {
    if (!Number.isInteger(maximumAttempts) || maximumAttempts < 1) {
        throw new RangeError("maximumAttempts must be a positive integer");
    }

    let lastError: unknown = new Error("Stop-sign vehicle spawn did not run");
    for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
        try {
            return await spawn(attempt);
        } catch (error) {
            lastError = error;
            if (stopRequested() || attempt === maximumAttempts) {
                throw lastError;
            }
            await waitBeforeRetry(attempt);
        }
    }

    throw lastError;
}
