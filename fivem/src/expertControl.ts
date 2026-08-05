type ExpertNeutralizeResponse = {
    requestId: string
    success: boolean
    error?: string
};

type PendingRequest = {
    resolve: () => void
    reject: (error: Error) => void
    timeoutHandle: ReturnType<typeof setTimeout>
};

const pendingRequests = new Map<string, PendingRequest>();
const expertNeutralizeTimeoutMs = 15_000;

onNet("expert:neutralizeResponse", (response: ExpertNeutralizeResponse) => {
    if (!response || typeof response.requestId !== "string") {
        return;
    }
    const pending = pendingRequests.get(response.requestId);
    if (!pending) {
        return;
    }
    clearTimeout(pending.timeoutHandle);
    pendingRequests.delete(response.requestId);
    if (response.success) {
        pending.resolve();
        return;
    }
    pending.reject(new Error(response.error ?? "failed to neutralize inference and actuator inputs"));
});

export function requestExpertInputNeutralization(): Promise<void> {
    const requestId = `expert-neutralize-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    return new Promise<void>((resolve, reject) => {
        const timeoutHandle = setTimeout(() => {
            pendingRequests.delete(requestId);
            reject(new Error(`expert input neutralization timed out after ${expertNeutralizeTimeoutMs}ms`));
        }, expertNeutralizeTimeoutMs);
        pendingRequests.set(requestId, {resolve, reject, timeoutHandle});
        emitNet("expert:neutralizeRequest", {requestId});
    });
}
