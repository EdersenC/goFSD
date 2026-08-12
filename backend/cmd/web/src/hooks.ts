import {useCallback, useEffect, useRef, useState} from "react";
import {
    fetchControlState,
    fetchInferenceModels,
    fetchInferenceStatus,
    fetchProcessingReadiness,
    fetchProcessingState,
    fetchRuns,
    fetchTrainingConfig,
    fetchTrainingState,
} from "./api";
import {
    ControlState,
    InferenceModel,
    InferenceStatus,
    InspectorRun,
    ProcessingReadiness,
    ProcessingState,
    TrainingConfig,
    TrainingState,
} from "./types";

export type AsyncSnapshot<T> = {
    data?: T
    error?: string
    loading: boolean
    refreshedAtMs?: number
};

export function useControlState(intervalMs = 600) {
    const [snapshot, setSnapshot] = useState<AsyncSnapshot<ControlState>>({loading: true});
    const inFlight = useRef(false);

    const refresh = useCallback(async () => {
        if (inFlight.current) {
            return;
        }
        inFlight.current = true;
        try {
            const data = await fetchControlState();
            setSnapshot({data, loading: false, refreshedAtMs: Date.now()});
        } catch (error) {
            setSnapshot((current) => ({
                ...current,
                error: error instanceof Error ? error.message : "Failed to load control state",
                loading: false,
            }));
        } finally {
            inFlight.current = false;
        }
    }, []);

    useEffect(() => {
        void refresh();
        const timer = window.setInterval(() => {
            void refresh();
        }, intervalMs);
        const refreshOnReturn = () => {
            if (!document.hidden) {
                void refresh();
            }
        };
        document.addEventListener("visibilitychange", refreshOnReturn);
        window.addEventListener("focus", refreshOnReturn);
        return () => {
            window.clearInterval(timer);
            document.removeEventListener("visibilitychange", refreshOnReturn);
            window.removeEventListener("focus", refreshOnReturn);
        };
    }, [intervalMs, refresh]);

    return {...snapshot, refresh};
}

export function useCollectionData(active: boolean) {
    const [readiness, setReadiness] = useState<AsyncSnapshot<ProcessingReadiness>>({loading: false});
    const [processing, setProcessing] = useState<AsyncSnapshot<ProcessingState>>({loading: false});
    const [runs, setRuns] = useState<AsyncSnapshot<InspectorRun[]>>({loading: false});
    const refreshRequest = useRef(0);

    const refresh = useCallback(async () => {
        const request = ++refreshRequest.current;
        setReadiness((current) => ({...current, loading: true}));
        setProcessing((current) => ({...current, loading: true}));
        setRuns((current) => ({...current, loading: true}));
        const [readinessResult, processingResult, runsResult] = await Promise.allSettled([
            fetchProcessingReadiness(),
            fetchProcessingState(),
            fetchRuns(),
        ]);
        if (request !== refreshRequest.current) {
            return;
        }
        setReadiness((current) => snapshotFromResult(readinessResult, current));
        setProcessing((current) => snapshotFromResult(processingResult, current));
        setRuns((current) => snapshotFromResult(runsResult, current));
    }, []);

    useEffect(() => {
        if (!active) {
            return;
        }
        void refresh();
        const timer = window.setInterval(() => {
            if (!document.hidden) {
                void refresh();
            }
        }, 5000);
        return () => window.clearInterval(timer);
    }, [active, refresh]);

    return {readiness, processing, runs, refresh};
}

export function useTrainingWorkspace(active: boolean) {
    const [config, setConfig] = useState<AsyncSnapshot<TrainingConfig>>({loading: false});
    const [state, setState] = useState<AsyncSnapshot<TrainingState>>({loading: false});
    const refreshRequest = useRef(0);

    const refresh = useCallback(async () => {
        const request = ++refreshRequest.current;
        setConfig((current) => ({...current, loading: true}));
        setState((current) => ({...current, loading: true}));
        const [configResult, stateResult] = await Promise.allSettled([
            fetchTrainingConfig(),
            fetchTrainingState(),
        ]);
        if (request !== refreshRequest.current) {
            return;
        }
        setConfig((current) => snapshotFromResult(configResult, current));
        setState((current) => snapshotFromResult(stateResult, current));
    }, []);

    useEffect(() => {
        if (!active) {
            return;
        }
        void refresh();
        const timer = window.setInterval(() => {
            if (!document.hidden) {
                void refresh();
            }
        }, 2500);
        return () => window.clearInterval(timer);
    }, [active, refresh]);

    return {config, state, refresh};
}

export function useInferenceWorkspace(active: boolean) {
    const [status, setStatus] = useState<AsyncSnapshot<InferenceStatus>>({loading: false});
    const [models, setModels] = useState<AsyncSnapshot<InferenceModel[]>>({loading: false});
    const statusRequest = useRef(0);
    const modelsRequest = useRef(0);

    const refreshStatus = useCallback(async () => {
        const request = ++statusRequest.current;
        setStatus((current) => ({...current, loading: true}));
        const result = await Promise.allSettled([fetchInferenceStatus()]);
        if (request === statusRequest.current) {
            setStatus((current) => snapshotFromResult(result[0], current));
        }
    }, []);

    const refreshModels = useCallback(async () => {
        const request = ++modelsRequest.current;
        setModels((current) => ({...current, loading: true}));
        const result = await Promise.allSettled([fetchInferenceModels()]);
        if (request === modelsRequest.current) {
            setModels((current) => snapshotFromResult(result[0], current));
        }
    }, []);

    const refresh = useCallback(async () => {
        await Promise.all([refreshStatus(), refreshModels()]);
    }, [refreshModels, refreshStatus]);

    useEffect(() => {
        if (!active) {
            return;
        }
        void refresh();
        const timer = window.setInterval(() => {
            if (!document.hidden) {
                void refreshStatus();
            }
        }, 1000);
        const modelTimer = window.setInterval(() => {
            if (!document.hidden) {
                void refreshModels();
            }
        }, 10000);
        return () => {
            window.clearInterval(timer);
            window.clearInterval(modelTimer);
        };
    }, [active, refresh, refreshModels, refreshStatus]);

    return {status, models, refresh, refreshStatus, refreshModels};
}

export function snapshotFromResult<T>(
    result: PromiseSettledResult<T>,
    current: AsyncSnapshot<T> = {loading: false},
): AsyncSnapshot<T> {
    if (result.status === "fulfilled") {
        return {data: result.value, loading: false, refreshedAtMs: Date.now()};
    }
    return {...current, error: errorText(result.reason), loading: false};
}

function errorText(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}
