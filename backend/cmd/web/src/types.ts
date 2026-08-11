export type Pose = {
    x: number
    y: number
    z: number
    heading: number
};

export const STOP_SIGN_PLAN_VERSION = "stop-sign-plan.v2" as const;

export type StopSignTime = {hour: number, minute: number};
export type StopSignColor = {r: number, g: number, b: number};
export type StopSignVehicle = {model?: string, color?: StopSignColor};

export type StopSignVariation = {
    id: string
    stopDistanceM?: number
    egoCenterOffsetM?: number
    startDistanceM?: number
    exitDistanceM?: number
    targetSpeedMps?: number
    dwellMs?: number
    attemptCount?: number
    weather?: string
    time?: StopSignTime
    vehicle?: StopSignVehicle
};

export type StopSignPlanEntry = {
    id: string
    signPose: Pose
    stopDistanceM: number
    egoCenterOffsetM: number
    startDistanceM: number
    exitDistanceM: number
    targetSpeedMps: number
    dwellMs: number
    attemptCount: number
    weather: string
    time: StopSignTime
    vehicle: StopSignVehicle
    variations: StopSignVariation[]
};

export type StopSignPlan = {
    version: typeof STOP_SIGN_PLAN_VERSION
    id: string
    seed: string
    entries: StopSignPlanEntry[]
};

export type StopSignBatchJob = {
    id: string
    entryId: string
    variationId: string
    signPose: Pose
    stopLinePose: Pose
    egoStopPose: Pose
    startPose: Pose
    exitPose: Pose
    stopDistanceM: number
    egoCenterOffsetM: number
    startDistanceM: number
    exitDistanceM: number
    targetSpeedMps: number
    dwellMs: number
    attemptCount: number
    weather: string
    time: StopSignTime
    vehicle: StopSignVehicle
    seed: string
};

export type QueueStopSignBatchResponse = {
    status: "queued"
    planId: string
    planFingerprint: string
    jobCount: number
    jobs: StopSignBatchJob[]
    command: ControlCommand
};

export type BatchProgress = {
    batchId: string
    planFingerprint?: string
    state: "running" | "completed" | "stopped" | "failed"
    jobId?: string
    jobIndex: number
    jobCount: number
    completedJobs: number
    attemptIndex: number
    attemptCount: number
    phase: string
    startedAtMs?: number
    updatedAtMs?: number
    lastAttemptOutcome?: {
        success: boolean
        status: string
        failureReason: string
        durationMs: number
        stoppedAtDistanceM: number | null
        dwellDurationMs: number
        crossedStopLineBeforeDwell: boolean
    }
};

export type RuntimeState = {
    status: "idle" | "runningScene" | "runningAllScenes" | "stopping" | "error"
    activeSceneName?: string
    lastError?: string
    updatedAt?: string
    fivemConnected: boolean
    lastPollAt?: string
    stopSignBatch?: BatchProgress
    appliedSafetyEpoch: number
    inFlightSafetyStarts: number
};

export type RuntimeTelemetry = {
    currentSpeed: number
    currentYaw: number
    acceleration?: number
    steering?: number
    brakePressureAvg?: number
    throttleApplied?: number
    brakeApplied?: number
    vehicleExists: boolean
    isInVehicle: boolean
    positionX?: number
    positionY?: number
    positionZ?: number
    pitchDeg?: number
    rollDeg?: number
    stopSignTargetConfigured?: boolean
    stopSignPose?: Pose
    stopLinePose?: Pose
    stopSignEgoStopPose?: Pose
    stopSignDistanceM?: number
    stopLineDistanceM?: number
    stopSignLongitudinalErrorM?: number
    stopSignLateralErrorM?: number
    stopSignHeadingErrorDeg?: number
    stopSignStopped?: boolean
    stopSignDwellElapsedMs?: number
    stopSignDwellTargetMs?: number
    stopSignAttemptIndex?: number
    stopSignAttemptCount?: number
    stopSignPhase?: string
    receivedAtMs?: number
    updatedAt?: string
};

export type ControlCommand = {
    id: string
    type: string
    createdAt: string
};

export type ControlState = {
    safetyEpoch: number
    runtime: RuntimeState
    telemetry?: RuntimeTelemetry
    lastCommand?: ControlCommand
    pendingCommands: ControlCommand[]
};

export type ProcessingReadiness = {
    configFingerprint: string
    scope: string
    runCount: number
    totalTripCount: number
    totalDatasetCount: number
    selectedTripCount: number
    selectedDatasetCount: number
    currentTripCount: number
    missingDatasetCount: number
    unreadyDatasetCount: number
    excludedTripCount: number
    readyRunIds: string[]
    trainingEligibleTripCount: number
    trainingSampleCount: number
    trainingEligibleRunIds: string[]
    trainingLocationCount: number
    suggestedTrainRunIds: string[]
    suggestedValRunIds: string[]
    trainingEligibilityErrorCount: number
    trainingReady: boolean
};

export type ProcessingJob = {
    tripDir: string
    state: string
    error?: string
    queuedAt: string
    startedAt?: string
    completedAt?: string
};

export type ProcessingState = {
    workerCount: number
    queueCapacity: number
    recovered: number
    recoveryPending: boolean
    recoveryError?: string
    queued: ProcessingJob[]
    active: ProcessingJob[]
    completed: ProcessingJob[]
    failed: ProcessingJob[]
    configFingerprint: string
};

export type ProcessingFailure = {
    trip: string
    error: string
};

export type ProcessingReconcileResponse = {
    selected: number
    current: number
    queued: number
    alreadyQueued: number
    incomplete: number
    failed: number
    failures: ProcessingFailure[]
};

export type InspectorTrip = {
    tripName: string
    tripIndex: number
    rawAvailable: boolean
    processedAvailable: boolean
    processingState: string
    processingError?: string
};

export type InspectorScene = {
    sceneKey: string
    sceneId: string
    sceneVariant: string
    tripCount: number
    trips: InspectorTrip[]
};

export type InspectorRun = {
    runId: string
    sceneCount: number
    tripCount: number
    reportExists: boolean
    scenes: InspectorScene[]
};

export type TrainingConfig = {
    configPath: string
    jobsDir: string
    epochs: number
    learningRate: number
    lossFunction: string
    smoothL1Beta: number
    earlyStoppingMetric: string
    widthMultiplier: number
    trainRunIds: string[]
    valRunIds: string[]
};

export type TrainingJobStatus = "queued" | "running" | "completed" | "failed" | "canceled" | "stopped";
export type TrainingJobPhase = "waiting" | "starting" | "training" | "stopping" | "finished";

export type TrainingJobMetrics = {
    trainLoss?: number
    valLoss?: number
    stopSignScore?: number
    valControlMae?: number
};

export type TrainingJob = {
    id: string
    name: string
    notes: string
    status: TrainingJobStatus
    phase?: TrainingJobPhase
    epochs?: number | null
    learningRate?: number | null
    trainRunIds?: string[] | null
    valRunIds?: string[] | null
    createdAt: string
    startedAt?: string
    finishedAt?: string
    lastUpdatedAt: string
    runDir?: string
    runMetricsPath?: string
    error?: string
    currentEpoch?: number
    totalEpochs?: number
    progressPercent?: number
    latestMetrics?: TrainingJobMetrics
    bestEpoch?: number
    elapsedSeconds?: number
    recoveryMessage?: string
};

export type TrainingState = {
    activeJobId?: string | null
    queuedCount: number
    running: boolean
    queuedJobs: TrainingJob[]
    activeJob?: TrainingJob | null
    recentJobs: TrainingJob[]
    jobsDirectory: string
    recoveryWarnings?: string[]
};

export type TrainingJobSpec = {
    name: string
    notes: string
    epochs: number
    trainRunIds: string[]
    valRunIds: string[]
};

export type QueueTrainingJobResponse = {
    status: "queued"
    jobs: TrainingJob[]
};

export type InferenceModel = {
    label: string
    path: string
    runId?: string
    epoch?: number
    variant?: "raw" | "ema" | string
    isBest: boolean
    updatedAt?: string
};

export type InferencePrediction = {
    sequence: number
    frameIndex: number
    checkpoint?: string
    capturedAt: string
    predictedAt: string
    frameTelemetryAligned: boolean
};

export type InferenceStatus = {
    state: string
    active: boolean
    actuatorReady: boolean
    controllerReady: boolean
    calibrationVerified: boolean
    calibrationId?: string
    safetyReady: boolean
    safetyBlocker?: string
    sourceId?: string
    sourceFps: number
    inferenceHz: number
    windowSize: number
    frameStride: number
    dispatchStride: number
    frameWidth: number
    frameHeight: number
    modelServerUrl?: string
    loadedCheckpoint?: string
    loadedModelDevice?: string
    startedAt?: string
    stoppedAt?: string
    lastPrediction?: InferencePrediction
    framesSeen: number
    predictionsSent: number
    predictionErrors: number
    lastError?: string
};

export type LoadInferenceModelResponse = {
    status: "loaded"
    checkpoint: string
    device: string
    epoch?: number
};
