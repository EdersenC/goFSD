export type CaptureAction = "start" | "stop" | "abort";
type CaptureRequestAction = CaptureAction | "finalizeTrip";

export type CaptureRequest = {
    requestId: string
    runId: string
    tripIndex: number
    sceneId: string
    sceneVariant: string
    sceneName: string
}

export type CaptureResponse = {
    requestId: string
    success: boolean
    error?: string
    outputFile?: string
    logFile?: string
}

export type TripFinalizeResponse = {
    requestId: string
    success: boolean
    error?: string
    runId?: string
    sceneId?: string
    sceneVariant?: string
    tripIndex?: number
    tripKey?: string
    tripDir?: string
    runFile?: string
    metadataFile?: string
    sampleCount?: number
}

type PendingCaptureRequest = {
    action: CaptureRequestAction
    resolve: (response: CaptureResponse | TripFinalizeResponse) => void
    reject: (error: Error) => void
    timeoutHandle: ReturnType<typeof setTimeout>
}

const pendingCaptureRequests = new Map<string, PendingCaptureRequest>();
const captureRequestTimeoutMs = 20_000;
const requestEventByAction: Record<CaptureRequestAction, string> = {
    start: "capture:startRequest",
    stop: "capture:stopRequest",
    abort: "capture:abortRequest",
    finalizeTrip: "capture:finalizeTripRequest"
};

onNet("capture:startResponse", (response: CaptureResponse) => {
    resolveCaptureRequest("start", response);
});

onNet("capture:stopResponse", (response: CaptureResponse) => {
    resolveCaptureRequest("stop", response);
});

onNet("capture:abortResponse", (response: CaptureResponse) => {
    resolveCaptureRequest("abort", response);
});

onNet("capture:finalizeTripResponse", (response: TripFinalizeResponse) => {
    resolveCaptureRequest("finalizeTrip", response);
});

function resolveCaptureRequest(
    action: CaptureRequestAction,
    response: CaptureResponse | TripFinalizeResponse
) {
    if (!response || typeof response.requestId !== "string") {
        return;
    }

    const pending = pendingCaptureRequests.get(response.requestId);
    if (!pending || pending.action !== action) {
        return;
    }

    clearTimeout(pending.timeoutHandle);
    pendingCaptureRequests.delete(response.requestId);

    if (response.success) {
        pending.resolve(response);
        return;
    }

    const errorMessage = response.error ?? `capture ${action} failed`;
    pending.reject(new Error(errorMessage));
}

function createRequestId(action: CaptureRequestAction): string {
    return `${action}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function requestCaptureLike<TResponse extends CaptureResponse | TripFinalizeResponse>(
    action: CaptureRequestAction,
    payload: Omit<CaptureRequest, "requestId">
): Promise<TResponse> {
    const requestId = createRequestId(action);
    const request: CaptureRequest = {
        requestId,
        ...payload
    };

    return new Promise<TResponse>((resolve, reject) => {
        const timeoutHandle = setTimeout(() => {
            pendingCaptureRequests.delete(requestId);
            reject(new Error(`capture ${action} timed out after ${captureRequestTimeoutMs}ms`));
        }, captureRequestTimeoutMs);

        pendingCaptureRequests.set(requestId, {
            action,
            resolve,
            reject,
            timeoutHandle
        });

        emitNet(requestEventByAction[action], request);
    });
}

export function requestCapture(
    action: CaptureAction,
    payload: Omit<CaptureRequest, "requestId">
): Promise<CaptureResponse> {
    return requestCaptureLike<CaptureResponse>(action, payload);
}

export function requestTripFinalize(
    payload: Omit<CaptureRequest, "requestId">
): Promise<TripFinalizeResponse> {
    return requestCaptureLike<TripFinalizeResponse>("finalizeTrip", payload);
}
