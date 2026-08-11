import EmergencyRounded from "@mui/icons-material/EmergencyRounded";
import MemoryRounded from "@mui/icons-material/MemoryRounded";
import PlayArrowRounded from "@mui/icons-material/PlayArrowRounded";
import RefreshRounded from "@mui/icons-material/RefreshRounded";
import StopRounded from "@mui/icons-material/StopRounded";
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    Container,
    Divider,
    Fab,
    FormControl,
    InputLabel,
    LinearProgress,
    MenuItem,
    Select,
    Snackbar,
    Stack,
    TextField,
    Tooltip,
    Typography,
} from "@mui/material";
import {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {
    ApiError,
    fetchControlState,
    loadInferenceModel,
    queueStopSignPlan,
    queueTrainingJob,
    reconcileProcessing,
    sendControlCommand,
    startInference,
    stopInference,
    stopTrainingJob,
} from "./api";
import {useCollectionData, useControlState, useInferenceWorkspace, useTrainingWorkspace} from "./hooks";
import {
    cloneStopSignPlan,
    createStopSignPlan,
    parseStoredStopSignPlan,
    STOP_SIGN_PLAN_STORAGE_KEY,
    stageStopSignCatalogLocation,
    stopSignPlanStats,
    validateStopSignPlan,
} from "./stop-sign-plan";
import {PhaseRail} from "./stop-sign/PhaseRail";
import {PlanEditor} from "./stop-sign/PlanEditor";
import {TelemetryPanel} from "./stop-sign/TelemetryPanel";
import {StopSignCatalog} from "./stop-sign/StopSignCatalog";
import type {StopSignCatalogLocation} from "./stop-sign/catalog";
import type {InferenceModel, InferenceStatus, ProcessingReadiness, StopSignPlan, TrainingJob, TrainingJobSpec} from "./types";
import {requireCurrentSafetyEpoch, runSafetyStartWithHoldBarrier} from "./workspace/safetyStartBarrier";
import {ArchitectureOverview} from "./ArchitectureOverview";

type Notice = {message: string, severity: "success" | "info" | "warning" | "error"};

export function App() {
    if (window.location.pathname === "/architecture") {
        return <ArchitectureOverview />;
    }
    const [plan, setPlan] = useState<StopSignPlan>(loadPlan);
    const [pending, setPending] = useState<ReadonlySet<string>>(() => new Set());
    const pendingRef = useRef<Set<string>>(new Set());
    const [notice, setNotice] = useState<Notice | null>(null);
    const [trainingName, setTrainingName] = useState("");
    const [epochs, setEpochs] = useState(20);
    const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
    const safetyGeneration = useRef(0);
    const startControllers = useRef<Set<AbortController>>(new Set());

    const control = useControlState();
    const collection = useCollectionData(true);
    const training = useTrainingWorkspace(true);
    const inference = useInferenceWorkspace(true);
    const stats = useMemo(() => stopSignPlanStats(plan), [plan]);
    const planErrors = useMemo(() => validateStopSignPlan(plan), [plan]);
    const activeBatch = control.data?.runtime.stopSignBatch;
    const collectionActive = activeBatch?.state === "running" || control.data?.runtime.status === "runningAllScenes";
    const activeTraining = training.state.data?.activeJob;
    const fivemLinked = Boolean(control.data?.runtime.fivemConnected);
    const fivemControlReady = fivemLinked
        && control.data?.runtime.appliedSafetyEpoch === control.data?.safetyEpoch;

    useEffect(() => {
        document.title = "Stop Sign Lab";
        try {
            window.localStorage.setItem(STOP_SIGN_PLAN_STORAGE_KEY, JSON.stringify(plan));
        } catch {
            setNotice({message: "Plan storage unavailable. Keep this tab open.", severity: "warning"});
        }
    }, [plan]);

    useEffect(() => {
        const models = inference.models.data ?? [];
        if (models.some((model) => model.path === selectedCheckpoint)) {
            return;
        }
        const loaded = inference.status.data?.loadedCheckpoint;
        setSelectedCheckpoint(models.find((model) => model.path === loaded)?.path ?? models.find((model) => model.isBest)?.path ?? models[0]?.path ?? "");
    }, [inference.models.data, inference.status.data?.loadedCheckpoint, selectedCheckpoint]);

    const operate = useCallback(async (id: string, work: () => Promise<void>) => {
        if (pendingRef.current.has(id)) {
            return;
        }
        pendingRef.current = new Set(pendingRef.current).add(id);
        setPending(new Set(pendingRef.current));
        try {
            await work();
        } catch (error) {
            setNotice({message: errorMessage(error), severity: "error"});
        } finally {
            const next = new Set(pendingRef.current);
            next.delete(id);
            pendingRef.current = next;
            setPending(new Set(next));
        }
    }, []);

    const issueStops = useCallback(async (): Promise<void> => {
        const results = await Promise.allSettled([
            stopInferenceUnlessIdle(),
            sendControlCommand("endAllScenes"),
            sendControlCommand("stopEgo"),
        ]);
        const failed = results.filter((result) => result.status === "rejected");
        if (failed.length === results.length) {
            throw new Error("Hold could not reach either runtime. Use the in-game brake.");
        }
    }, []);

    const runControlStart = useCallback(async <T,>(start: (safetyEpoch: number, signal: AbortSignal) => Promise<T>) => {
        const epoch = requireCurrentSafetyEpoch(control.data);
        const generation = safetyGeneration.current;
        const controller = new AbortController();
        startControllers.current.add(controller);
        try {
            return await runSafetyStartWithHoldBarrier(
                () => start(epoch, controller.signal),
                generation,
                () => safetyGeneration.current,
                issueStops,
            );
        } finally {
            startControllers.current.delete(controller);
        }
    }, [control.data, issueStops]);

    const hold = useCallback(() => operate("hold", async () => {
        safetyGeneration.current += 1;
        for (const controller of startControllers.current) {
            controller.abort();
        }
        await issueStops();
        await waitForControlHold();
        setNotice({message: "Hold confirmed. Collection and model control stopped.", severity: "warning"});
        await Promise.allSettled([control.refresh(), inference.refreshStatus()]);
    }), [control, inference, issueStops, operate]);

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.altKey && event.shiftKey && event.key.toLowerCase() === "h" && !event.repeat) {
                event.preventDefault();
                void hold();
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [hold]);

    const startSetupCar = () => operate("setup", async () => {
        if (!fivemControlReady) {
            throw new Error("FiveM control is not synchronized. Run restart FSD in the server console.");
        }
        if (collectionActive || inference.status.data?.active) {
            throw new Error("Stop collection or inference before changing the setup car.");
        }
        const result = await runControlStart((safetyEpoch, signal) => sendControlCommand("startEgo", {safetyEpoch}, signal));
        if (result.kind === "started") {
            setNotice({message: "Setup car queued.", severity: "success"});
        }
        window.setTimeout(() => void control.refresh(), 250);
    });

    const setCatalogWaypoint = (location: StopSignCatalogLocation) => operate("catalog-waypoint", async () => {
        if (!fivemControlReady) {
            throw new Error("FiveM control is not synchronized. Run restart FSD in the server console.");
        }
        await sendControlCommand("setStopSignCatalogWaypoint", {
            stopSignCatalogPosition: {x: location.x, y: location.y, z: location.z},
        });
        setNotice({message: `${location.id} waypoint queued. Use /tpwaypoint in FiveM, then calibrate the lane pose.`, severity: "success"});
    });

    const stageCatalogLocation = (location: StopSignCatalogLocation) => {
        setPlan(stageStopSignCatalogLocation(plan, location.id));
        setNotice({
            message: `${location.id} staged. Its roadside coordinates were not copied; calibrate and apply the live lane pose.`,
            severity: "info",
        });
    };

    const calibrateSign = () => operate("calibrate", async () => {
        if (!control.data?.telemetry?.isInVehicle) {
            throw new Error("Enter the setup car before calibrating the sign pose.");
        }
        await sendControlCommand("setStopSignTarget");
        setNotice({message: "Sign pose calibration queued.", severity: "success"});
        window.setTimeout(() => void control.refresh(), 250);
    });

    const clearCalibration = () => operate("calibrate", async () => {
        await sendControlCommand("clearStopSignTarget");
        setNotice({message: "Live sign calibration cleared.", severity: "info"});
        window.setTimeout(() => void control.refresh(), 250);
    });

    const applyCalibration = (entryIndex: number) => {
        const signPose = control.data?.telemetry?.stopSignPose;
        if (!signPose) {
            setNotice({message: "No accepted sign pose is available.", severity: "warning"});
            return;
        }
        const entries = plan.entries.map((entry, index) => index === entryIndex ? {...entry, signPose: {...signPose}} : entry);
        setPlan(cloneStopSignPlan({...plan, entries}));
        setNotice({message: `${entries[entryIndex]?.id ?? "Sign"} updated from live calibration.`, severity: "success"});
    };

    const queueCollection = () => operate("queue", async () => {
        if (planErrors.length > 0) {
            throw new Error(planErrors[0]);
        }
        if (!fivemControlReady) {
            throw new Error("FiveM control is not synchronized. Run restart FSD in the server console.");
        }
        if (collectionActive) {
            throw new Error("A collection batch is already active.");
        }
        if (inference.status.data?.active || training.state.data?.running) {
            throw new Error("Stop inference or training before collection.");
        }
        const result = await runControlStart((safetyEpoch, signal) => queueStopSignPlan(plan, safetyEpoch, signal));
        if (result.kind === "started") {
            setNotice({message: `${result.value.jobCount} jobs queued · ${stats.attemptCount} attempts.`, severity: "success"});
        }
        window.setTimeout(() => void control.refresh(), 250);
    });

    const endCollection = () => operate("end-collection", async () => {
        await sendControlCommand("endAllScenes");
        setNotice({message: "Collection stop queued.", severity: "warning"});
        window.setTimeout(() => void control.refresh(), 250);
    });

    const prepareData = () => operate("prepare", async () => {
        const result = await reconcileProcessing();
        setNotice({
            message: result.queued > 0 ? `${result.queued} trips queued for processing.` : "Dataset is current.",
            severity: result.failed > 0 ? "warning" : "success",
        });
        window.setTimeout(() => void collection.refresh(), 250);
    });

    const queueTraining = () => operate("train", async () => {
        const readiness = collection.readiness.data;
        const config = training.config.data;
        if (!readiness?.trainingReady || !config) {
            throw new Error("Training readiness has not passed.");
        }
        if (collectionActive || inference.status.data?.active) {
            throw new Error("Stop collection or inference before training.");
        }
        const trainRunIds = [...new Set(readiness.suggestedTrainRunIds)].sort();
        const valRunIds = [...new Set(readiness.suggestedValRunIds)].sort();
        if (trainRunIds.length < 1 || valRunIds.length < 1) {
            throw new Error("Training requires successful clips from at least two physical stop-sign locations.");
        }
        const spec: TrainingJobSpec = {
            name: trainingName.trim() || `stop-sign-${new Date().toISOString().slice(0, 10)}`,
            notes: "Stop-sign temporal policy: launch, approach/brake, stop/dwell, go.",
            epochs,
            trainRunIds,
            valRunIds,
        };
        const result = await queueTrainingJob(spec);
        setNotice({message: `${result.jobs[0]?.name ?? spec.name} queued.`, severity: "success"});
        setTrainingName("");
        window.setTimeout(() => void training.refresh(), 250);
    });

    const stopTraining = () => operate("stop-training", async () => {
        if (!activeTraining?.id) {
            throw new Error("No active training job.");
        }
        await stopTrainingJob(activeTraining.id);
        setNotice({message: "Training stop requested.", severity: "warning"});
        window.setTimeout(() => void training.refresh(), 250);
    });

    const loadModel = () => operate("load-model", async () => {
        if (!selectedCheckpoint) {
            throw new Error("Select a checkpoint.");
        }
        await loadInferenceModel(selectedCheckpoint);
        setNotice({message: "Checkpoint loaded.", severity: "success"});
        await inference.refresh();
    });

    const beginInference = () => operate("inference", async () => {
        const status = inference.status.data;
        if (!status?.safetyReady) {
            throw new Error(status?.safetyBlocker || "Inference safety preflight has not passed.");
        }
        if (collectionActive || training.state.data?.running) {
            throw new Error("Stop collection or training before inference.");
        }
        const generation = safetyGeneration.current;
        const controller = new AbortController();
        startControllers.current.add(controller);
        try {
            const result = await runSafetyStartWithHoldBarrier(
                () => startInference(controller.signal),
                generation,
                () => safetyGeneration.current,
                issueStops,
            );
            if (result.kind === "started") {
                setNotice({message: "Inference started.", severity: "success"});
            }
        } finally {
            startControllers.current.delete(controller);
        }
        await inference.refreshStatus();
    });

    const endInference = () => operate("stop-inference", async () => {
        await stopInferenceUnlessIdle();
        setNotice({message: "Inference stopped.", severity: "warning"});
        await inference.refreshStatus();
    });

    const refreshAll = () => operate("refresh", async () => {
        await Promise.allSettled([control.refresh(), collection.refresh(), training.refresh(), inference.refresh()]);
        setNotice({message: "Status refreshed.", severity: "info"});
    });

    const calibrationPose = control.data?.telemetry?.stopSignPose;
    const phase = control.data?.telemetry?.stopSignPhase ?? activeBatch?.phase;
    return (
        <Box sx={{minHeight: "100vh", pb: 11, background: "radial-gradient(circle at 72% -10%, rgba(87,217,255,.09), transparent 35%), #080b0e"}}>
            <Container maxWidth="xl" sx={{pt: {xs: 2, md: 3}}}>
                <Box component="header" sx={{display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 2, mb: 2.5}}>
                    <Box>
                        <Typography variant="overline" color="secondary.main">Temporal control workspace</Typography>
                        <Typography variant="h1">Stop Sign Lab</Typography>
                    </Box>
                    <Stack direction="row" sx={{gap: 1, alignItems: "center", flexWrap: "wrap", justifyContent: "flex-end"}}>
                        <StatusChip label="API" active={!control.error} />
                        <StatusChip label="FiveM" active={fivemLinked} />
                        <StatusChip label="Control" active={fivemControlReady} />
                        <StatusChip label="Model" active={Boolean(inference.status.data?.controllerReady)} />
                        <Button href="/architecture" color="inherit" size="small">Architecture</Button>
                        <Tooltip title="Refresh all status"><span><Button aria-label="Refresh all status" color="inherit" disabled={pending.has("refresh")} onClick={refreshAll}><RefreshRounded /></Button></span></Tooltip>
                    </Stack>
                </Box>

                <PhaseRail phase={phase} />

                <Box component="main" sx={{display: "grid", gridTemplateColumns: {xs: "1fr", lg: "minmax(0, 1.7fr) minmax(300px, .8fr)"}, gap: 2, mt: 2}}>
                    <Stack sx={{gap: 2}}>
                        <StopSignCatalog
                            connected={fivemControlReady}
                            busy={pending.has("catalog-waypoint")}
                            onSetWaypoint={setCatalogWaypoint}
                            onStageLocation={stageCatalogLocation}
                        />
                        <PlanEditor
                            plan={plan}
                            onChange={(next) => setPlan(cloneStopSignPlan(next))}
                            onCalibrate={calibrateSign}
                            onClearCalibration={clearCalibration}
                            onApplyCalibration={applyCalibration}
                            calibrationReady={Boolean(calibrationPose)}
                            calibrationBusy={pending.has("calibrate")}
                        />
                    </Stack>
                    <Stack sx={{gap: 2}}>
                        <TelemetryPanel control={control.data} />
                        <CollectionControls
                            connected={fivemControlReady}
                            active={collectionActive}
                            valid={planErrors.length === 0}
                            batchProgress={batchProgress(activeBatch)}
                            pending={pending}
                            onStartSetup={startSetupCar}
                            onQueue={queueCollection}
                            onStop={endCollection}
                        />
                    </Stack>
                </Box>

                <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", lg: "repeat(2, minmax(0, 1fr))"}, gap: 2, mt: 2}}>
                    <DataTrainingPanel
                        readiness={collection.readiness.data}
                        processingActive={Boolean(collection.processing.data?.active.length || collection.processing.data?.queued.length)}
                        trainingName={trainingName}
                        epochs={epochs}
                        activeTraining={activeTraining}
                        pending={pending}
                        onName={setTrainingName}
                        onEpochs={setEpochs}
                        onPrepare={prepareData}
                        onTrain={queueTraining}
                        onStopTraining={stopTraining}
                    />
                    <InferencePanel
                        models={inference.models.data ?? []}
                        selected={selectedCheckpoint}
                        status={inference.status.data}
                        pending={pending}
                        onSelect={setSelectedCheckpoint}
                        onLoad={loadModel}
                        onStart={beginInference}
                        onStop={endInference}
                    />
                </Box>
            </Container>

            <Tooltip title="Stops collection, vehicle motion, and inference · Alt+Shift+H">
                <span>
                    <Fab
                        color="error"
                        variant="extended"
                        aria-label="Hold vehicle safely"
                        aria-keyshortcuts="Alt+Shift+H"
                        disabled={pending.has("hold")}
                        onClick={() => void hold()}
                        sx={{position: "fixed", right: 20, bottom: 20, zIndex: 1200}}
                    >
                        <EmergencyRounded sx={{mr: 1}} />
                        {pending.has("hold") ? "Holding…" : "Hold"}
                    </Fab>
                </span>
            </Tooltip>
            <Snackbar open={Boolean(notice)} autoHideDuration={5500} onClose={() => setNotice(null)} anchorOrigin={{vertical: "bottom", horizontal: "center"}}>
                {notice ? <Alert severity={notice.severity} onClose={() => setNotice(null)}>{notice.message}</Alert> : undefined}
            </Snackbar>
        </Box>
    );
}

function CollectionControls({connected, active, valid, batchProgress, pending, onStartSetup, onQueue, onStop}: {
    connected: boolean
    active: boolean
    valid: boolean
    batchProgress: number
    pending: ReadonlySet<string>
    onStartSetup: () => void
    onQueue: () => void
    onStop: () => void
}) {
    return (
        <Card component="section" aria-labelledby="run-control-title">
            <CardContent>
                <Typography id="run-control-title" variant="h2">Run control</Typography>
                <Stack sx={{gap: 1, mt: 1.5}}>
                    <Button variant="outlined" disabled={!connected || pending.has("setup") || active} onClick={onStartSetup}>Start setup car</Button>
                    <Button variant="contained" startIcon={<PlayArrowRounded />} disabled={!connected || !valid || active || pending.has("queue")} onClick={onQueue}>Queue collection</Button>
                    <Button variant="outlined" color="warning" startIcon={<StopRounded />} disabled={!active || pending.has("end-collection")} onClick={onStop}>End collection</Button>
                </Stack>
                <Divider sx={{my: 2}} />
                <Stack direction="row" sx={{justifyContent: "space-between", mb: .7}}>
                    <Typography variant="caption" color="text.secondary">Batch</Typography>
                    <Typography variant="caption">{batchProgress.toFixed(0)}%</Typography>
                </Stack>
                <LinearProgress variant="determinate" value={batchProgress} />
            </CardContent>
        </Card>
    );
}

function DataTrainingPanel({readiness, processingActive, trainingName, epochs, activeTraining, pending, onName, onEpochs, onPrepare, onTrain, onStopTraining}: {
    readiness?: ProcessingReadiness
    processingActive: boolean
    trainingName: string
    epochs: number
    activeTraining?: TrainingJob | null
    pending: ReadonlySet<string>
    onName: (name: string) => void
    onEpochs: (epochs: number) => void
    onPrepare: () => void
    onTrain: () => void
    onStopTraining: () => void
}) {
    return (
        <Card component="section" aria-labelledby="data-train-title">
            <CardContent>
                <Stack direction="row" sx={{justifyContent: "space-between", alignItems: "center"}}>
                    <Box>
                        <Typography variant="overline" color="secondary.main">Dataset</Typography>
                        <Typography id="data-train-title" variant="h2">Process + train</Typography>
                    </Box>
                    <Chip size="small" color={readiness?.trainingReady ? "success" : "default"} label={readiness?.trainingReady ? "ready" : "not ready"} />
                </Stack>
                <Box sx={{display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 1, my: 2}}>
                    <DataMetric label="Locations" value={readiness?.trainingLocationCount ?? 0} />
                    <DataMetric label="Trips" value={readiness?.trainingEligibleTripCount ?? 0} />
                    <DataMetric label="Samples" value={readiness?.trainingSampleCount ?? 0} />
                </Box>
                <Button variant="outlined" color="secondary" disabled={processingActive || pending.has("prepare")} onClick={onPrepare} fullWidth>
                    {processingActive ? "Processing…" : "Process recordings"}
                </Button>
                <Divider sx={{my: 2}} />
                <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", sm: "2fr 1fr"}, gap: 1}}>
                    <TextField label="Job name" value={trainingName} onChange={(event) => onName(event.target.value)} />
                    <TextField label="Epochs" type="number" value={epochs} onChange={(event) => onEpochs(Number(event.target.value))} />
                </Box>
                {activeTraining ? (
                    <Box sx={{mt: 1.5}}>
                        <Stack direction="row" sx={{justifyContent: "space-between", mb: .6}}>
                            <Typography variant="body2" sx={{fontWeight: 750}}>{activeTraining.name}</Typography>
                            <Typography variant="caption">{activeTraining.progressPercent ?? 0}%</Typography>
                        </Stack>
                        <LinearProgress variant="determinate" value={activeTraining.progressPercent ?? 0} />
                        <Button color="warning" startIcon={<StopRounded />} sx={{mt: 1}} disabled={pending.has("stop-training")} onClick={onStopTraining}>Stop training</Button>
                    </Box>
                ) : (
                    <Button startIcon={<MemoryRounded />} variant="contained" sx={{mt: 1.5}} disabled={!readiness?.trainingReady || pending.has("train")} onClick={onTrain} fullWidth>Queue training</Button>
                )}
            </CardContent>
        </Card>
    );
}

function InferencePanel({models, selected, status, pending, onSelect, onLoad, onStart, onStop}: {
    models: InferenceModel[]
    selected: string
    status?: InferenceStatus
    pending: ReadonlySet<string>
    onSelect: (path: string) => void
    onLoad: () => void
    onStart: () => void
    onStop: () => void
}) {
    const loaded = Boolean(selected && status?.loadedCheckpoint === selected);
    return (
        <Card component="section" aria-labelledby="inference-title">
            <CardContent>
                <Stack direction="row" sx={{justifyContent: "space-between", alignItems: "center"}}>
                    <Box>
                        <Typography variant="overline" color="secondary.main">Policy</Typography>
                        <Typography id="inference-title" variant="h2">Inference</Typography>
                    </Box>
                    <Chip size="small" color={status?.active ? "success" : "default"} label={status?.active ? "running" : status?.state ?? "offline"} />
                </Stack>
                <FormControl fullWidth size="small" sx={{mt: 2}}>
                    <InputLabel id="checkpoint-label">Checkpoint</InputLabel>
                    <Select labelId="checkpoint-label" label="Checkpoint" value={selected} onChange={(event) => onSelect(event.target.value)}>
                        {models.map((model) => <MenuItem key={model.path} value={model.path}>{model.label}{model.isBest ? " · best" : ""}</MenuItem>)}
                    </Select>
                </FormControl>
                <Stack direction={{xs: "column", sm: "row"}} sx={{gap: 1, mt: 1.5}}>
                    <Button variant="outlined" disabled={!selected || status?.active || pending.has("load-model")} onClick={onLoad} fullWidth>Load</Button>
                    {status?.active ? (
                        <Button color="warning" variant="contained" disabled={pending.has("stop-inference")} onClick={onStop} fullWidth>Stop inference</Button>
                    ) : (
                        <Button variant="contained" disabled={!loaded || !status?.safetyReady || pending.has("inference")} onClick={onStart} fullWidth>Start inference</Button>
                    )}
                </Stack>
                <Typography variant="caption" color="text.secondary" sx={{display: "block", mt: 2}}>
                    Output: future speed profile + stop intent. Throttle and brake remain diagnostics.
                </Typography>
                <Box sx={{display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 1, mt: 1.5}}>
                    <DataMetric label="Frames" value={status?.framesSeen ?? 0} />
                    <DataMetric label="Predictions" value={status?.predictionsSent ?? 0} />
                    <DataMetric label="Errors" value={status?.predictionErrors ?? 0} />
                </Box>
                {status?.safetyBlocker && <Alert severity="warning" sx={{mt: 1.5}}>{status.safetyBlocker}</Alert>}
            </CardContent>
        </Card>
    );
}

function StatusChip({label, active}: {label: string, active: boolean}) {
    return <Chip size="small" variant="outlined" color={active ? "success" : "default"} label={`${label} ${active ? "on" : "off"}`} />;
}

function DataMetric({label, value}: {label: string, value: number}) {
    return (
        <Box sx={{border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1}}>
            <Typography sx={{fontWeight: 820, fontFamily: "ui-monospace, monospace"}}>{value.toLocaleString()}</Typography>
            <Typography variant="caption" color="text.secondary">{label}</Typography>
        </Box>
    );
}

function batchProgress(batch: {jobCount: number, completedJobs: number, attemptIndex?: number, attemptCount?: number} | undefined): number {
    if (!batch?.jobCount) {
        return 0;
    }
    const activeJobProgress = batch.attemptCount ? Math.min(1, (batch.attemptIndex ?? 0) / batch.attemptCount) : 0;
    return Math.min(100, ((batch.completedJobs + activeJobProgress) / batch.jobCount) * 100);
}

function loadPlan(): StopSignPlan {
    try {
        return parseStoredStopSignPlan(window.localStorage.getItem(STOP_SIGN_PLAN_STORAGE_KEY)) ?? createStopSignPlan();
    } catch {
        return createStopSignPlan();
    }
}

async function stopInferenceUnlessIdle(): Promise<void> {
    try {
        await stopInference();
    } catch (error) {
        if (!(error instanceof ApiError) || (error.status !== 409 && error.status !== 404)) {
            throw error;
        }
    }
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

async function waitForControlHold(timeoutMs = 5000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const control = await fetchControlState();
        const runtimeIdle = control.runtime.status === "idle" || control.runtime.status === "error";
        const consumerSettled = control.runtime.appliedSafetyEpoch === control.safetyEpoch
            && control.runtime.inFlightSafetyStarts === 0;
        if (runtimeIdle && consumerSettled) {
            return;
        }
        await wait(200);
    }
    throw new Error("Hold was sent but FiveM did not confirm an idle safety epoch. Use the in-game brake.");
}

function wait(delayMs: number): Promise<void> {
    return new Promise((resolve) => window.setTimeout(resolve, delayMs));
}
