import {
    ControlState,
    InferenceModel,
    InferenceStatus,
    InspectorRun,
    LoadInferenceModelResponse,
    ProcessingReconcileResponse,
    ProcessingReadiness,
    ProcessingState,
    QueueTrainingJobResponse,
    QueueStopSignBatchResponse,
    StopSignPlan,
    TrainingConfig,
    TrainingJob,
    TrainingJobSpec,
    TrainingState,
} from "./types";
import {parseStopSignCatalogCsv, type StopSignCatalogLocation} from "./stop-sign/catalog";
import {cloneStopSignPlan} from "./stop-sign-plan";

export class ApiError extends Error {
    constructor(
        message: string,
        readonly status: number,
    ) {
        super(message);
        this.name = "ApiError";
    }
}

async function requestJSON<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(path, init);
    const body = await response.json().catch(() => ({})) as {error?: string};
    if (!response.ok) {
        throw new ApiError(body.error || `${response.status} ${response.statusText}`, response.status);
    }
    return body as T;
}

export function fetchControlState(signal?: AbortSignal): Promise<ControlState> {
    return requestJSON<ControlState>(`/control/state?requestTime=${Date.now()}`, {
        signal,
        cache: "no-store",
    });
}

export async function fetchStopSignCatalog(signal?: AbortSignal): Promise<StopSignCatalogLocation[]> {
    const response = await fetch("/stop-sign-locations.csv", {signal, cache: "no-store"});
    if (!response.ok) {
        throw new ApiError(`Failed to load stop-sign catalog: ${response.status}`, response.status);
    }
    return parseStopSignCatalogCsv(await response.text());
}

export function sendControlCommand(type: string, payload: Record<string, unknown> = {}, signal?: AbortSignal) {
    return requestJSON<{status: string}>("/control/command", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({type, ...payload}),
        signal,
    });
}

export function queueStopSignPlan(plan: StopSignPlan, safetyEpoch: number, signal?: AbortSignal): Promise<QueueStopSignBatchResponse> {
    const {version: _localVersion, ...requestPlan} = cloneStopSignPlan(plan);
    return requestJSON<QueueStopSignBatchResponse>("/control/stop-sign-batches", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...requestPlan, safetyEpoch}),
        signal,
    });
}

export function fetchProcessingReadiness(signal?: AbortSignal): Promise<ProcessingReadiness> {
    return requestJSON<ProcessingReadiness>("/processing/readiness", {signal});
}

export function fetchProcessingState(signal?: AbortSignal): Promise<ProcessingState> {
    return requestJSON<ProcessingState>("/processing/state", {signal});
}

export function reconcileProcessing(): Promise<ProcessingReconcileResponse> {
    return requestJSON<ProcessingReconcileResponse>("/processing/reconcile", {method: "POST"});
}

export async function fetchRuns(signal?: AbortSignal): Promise<InspectorRun[]> {
    const response = await requestJSON<{runs?: InspectorRun[]}>("/data/runs", {signal});
    return Array.isArray(response.runs) ? response.runs : [];
}


export function fetchTrainingConfig(signal?: AbortSignal): Promise<TrainingConfig> {
    return requestJSON<TrainingConfig>("/training/config", {signal});
}

export function fetchTrainingState(signal?: AbortSignal): Promise<TrainingState> {
    return requestJSON<TrainingState>("/training/state", {signal});
}

export function queueTrainingJob(spec: TrainingJobSpec): Promise<QueueTrainingJobResponse> {
    return requestJSON<QueueTrainingJobResponse>("/training/jobs", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(spec),
    });
}

export function stopTrainingJob(jobId: string): Promise<{status: string, job: TrainingJob}> {
    return requestJSON<{status: string, job: TrainingJob}>(`/training/jobs/${encodeURIComponent(jobId)}/stop`, {
        method: "POST",
    });
}

export function cancelTrainingJob(jobId: string): Promise<{status: string, job: TrainingJob}> {
    return requestJSON<{status: string, job: TrainingJob}>(`/training/jobs/${encodeURIComponent(jobId)}/cancel`, {
        method: "POST",
    });
}

export function retryTrainingJob(jobId: string): Promise<{status: string, job: TrainingJob}> {
    return requestJSON<{status: string, job: TrainingJob}>(`/training/jobs/${encodeURIComponent(jobId)}/requeue`, {
        method: "POST",
    });
}

export function fetchInferenceStatus(signal?: AbortSignal): Promise<InferenceStatus> {
    return requestJSON<InferenceStatus>("/inference/status", {signal});
}

export async function fetchInferenceModels(signal?: AbortSignal): Promise<InferenceModel[]> {
    const response = await requestJSON<{models?: InferenceModel[]}>("/inference/models", {signal});
    return Array.isArray(response.models) ? response.models : [];
}

export function loadInferenceModel(checkpoint: string): Promise<LoadInferenceModelResponse> {
    return requestJSON<LoadInferenceModelResponse>("/inference/model/load", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({checkpoint}),
    });
}

export function startInference(signal?: AbortSignal): Promise<InferenceStatus> {
    return requestJSON<InferenceStatus>("/inference/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: "{}",
        signal,
    });
}

export function stopInference(signal?: AbortSignal): Promise<InferenceStatus> {
    return requestJSON<InferenceStatus>("/inference/stop", {method: "POST", signal});
}
