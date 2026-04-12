export type CaptureAction = "start" | "stop";

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

type PendingCaptureRequest = {
    action: CaptureAction
    resolve: (response: CaptureResponse) => void
    reject: (error: Error) => void
    timeoutHandle: ReturnType<typeof setTimeout>
}

const pendingCaptureRequests = new Map<string, PendingCaptureRequest>();
const captureRequestTimeoutMs = 20_000;

onNet("capture:startResponse", (response: CaptureResponse) => {
    resolveCaptureRequest("start", response);
});

onNet("capture:stopResponse", (response: CaptureResponse) => {
    resolveCaptureRequest("stop", response);
});

function resolveCaptureRequest(action: CaptureAction, response: CaptureResponse) {
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

function createRequestId(action: CaptureAction): string {
    return `${action}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function requestCapture(action: CaptureAction, payload: Omit<CaptureRequest, "requestId">): Promise<CaptureResponse> {
    const requestId = createRequestId(action);
    const request: CaptureRequest = {
        requestId,
        ...payload
    };

    return new Promise<CaptureResponse>((resolve, reject) => {
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

        const eventName = action === "start" ? "capture:startRequest" : "capture:stopRequest";
        emitNet(eventName, request);
    });
}
