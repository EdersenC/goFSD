export type StopSignVector3 = [number, number, number];

export type StopSignPose = {
    x: number
    y: number
    z: number
    heading: number
};

export type StopSignCatalogPosition = {
    x: number
    y: number
    z: number
};

export type StopSignVehicleVariant = {
    model?: string
    color?: {r: number; g: number; b: number}
};

export type StopSignTimeVariant = {
    hour: number
    minute: number
};

export type StopSignVariationProfile = {
    contract: "stop-sign-variation-profile.v1"
    baseline: boolean
    configuredMotionVariancePct: number
    changedDimensions: string[]
    changeCount: number
    combinationMagnitudePct: number
    targetSpeedDeltaMps: number
    targetSpeedDeltaPct: number
    startDistanceDeltaM: number
    exitDistanceDeltaM: number
    stopOffsetM: number
    stopHeadingDeltaDeg: number
    startLaneOffsetDeltaM: number
    startHeadingDeltaDeg: number
    exitLaneOffsetDeltaM: number
    exitHeadingDeltaDeg: number
    timeDeltaMinutes: number
    weatherChanged: boolean
    vehicleModelChanged: boolean
    vehicleColorDeltaPct: number
};

export type StopSignJob = {
    id: string
    entryId: string
    variationId: string
    catalogId?: string
    catalogPosition?: StopSignCatalogPosition
    signPose: StopSignPose
    stopLinePose: StopSignPose
    egoStopPose: StopSignPose
    startPose: StopSignPose
    exitPose: StopSignPose
    stopDistanceM: number
    egoCenterOffsetM: number
    startDistanceM: number
    exitDistanceM: number
    targetSpeedMps: number
    stopConfirmationMs: number
    attemptCount: number
    weather: string
    time: StopSignTimeVariant
    vehicle: StopSignVehicleVariant
    seed: string
    variationProfile: StopSignVariationProfile
};

export type StopSignBehaviorPhase =
    | "accelerate"
    | "cruise_approach"
    | "decelerate"
    | "stop_hold"
    | "release";

export type StopSignClipStage = "approach" | "brake_stop" | "release";

export type StopSignRuntimePhase =
    | "idle"
    | "spawning"
    | "recording"
    | StopSignBehaviorPhase
    | "complete"
    | "failed"
    | "stopping";

export type StopSignGoal = {
    task: "stop-sign"
    contract: "stop-sign-goal.v3"
    releasePolicy: "scripted_continuous_release_v2"
    captureMode: "continuous"
    logicalClipStages: StopSignClipStage[]
    catalogId?: string
    catalogPosition?: StopSignCatalogPosition
    signPose: StopSignPose
    stopLinePose: StopSignPose
    egoStopPose: StopSignPose
    startPose: StopSignPose
    exitPose: StopSignPose
    exitDistanceM: number
    routeWaypointPose: StopSignPose
    routeWaypointForwardM: number
    routeWaypointLateralM: number
    targetSpeedMps: number
    stopConfirmationMs: number
    attemptIndex: number
    attemptCount: number
    seed: string
    variationId: string
    variationProfile: StopSignVariationProfile
};

export type StopSignExpertSupervision = {
    phase: StopSignBehaviorPhase
    desiredWheelSteerNormalized: number
    desiredSpeedMps: number
    throttle: number
    brake: number
    stopProbability: number
    goProbability: number
    distanceToStopLineM: number
};

export type StopSignOutcomeStatus =
    | "succeeded"
    | "crossed_without_stop"
    | "stopped_too_early"
    | "collision"
    | "timeout"
    | "stopped"
    | "invalid_vehicle";

export type StopSignOutcome = {
    success: boolean
    status: StopSignOutcomeStatus
    failureReason: string
    durationMs: number
    stoppedAtDistanceM: number | null
    stopConfirmationDurationMs: number
    crossedStopLineBeforeStop: boolean
    stageTransitions?: StopSignStageTransitions
};

export type StopSignStageTransitions = {
    approachStartedGameTimeMs: number
    brakeStopStartedGameTimeMs: number | null
    stopConfirmedGameTimeMs: number | null
    releaseStartedGameTimeMs: number | null
    completedGameTimeMs: number | null
};

export type StopSignTelemetry = {
    stopSignTargetConfigured: boolean
    stopSignPose?: StopSignPose
    stopLinePose?: StopSignPose
    stopSignEgoStopPose?: StopSignPose
    stopSignDistanceM: number
    stopLineDistanceM: number
    stopSignLongitudinalErrorM: number
    stopSignLateralErrorM: number
    stopSignHeadingErrorDeg: number
    stopSignPhase: StopSignRuntimePhase
    stopSignConfirmationElapsedMs: number
    stopSignConfirmationTargetMs: number
    stopSignStopped: boolean
    stopSignAttemptIndex: number
    stopSignAttemptCount: number
};
