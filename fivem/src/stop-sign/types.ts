export type StopSignVector3 = [number, number, number];

export type StopSignPose = {
    x: number
    y: number
    z: number
    heading: number
};

export type StopSignVehicleVariant = {
    model?: string
    color?: {r: number; g: number; b: number}
};

export type StopSignTimeVariant = {
    hour: number
    minute: number
};

export type StopSignJob = {
    id: string
    entryId: string
    variationId: string
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
    dwellMs: number
    attemptCount: number
    weather: string
    time: StopSignTimeVariant
    vehicle: StopSignVehicleVariant
    seed: string
};

export type StopSignBehaviorPhase =
    | "accelerate"
    | "cruise_approach"
    | "decelerate"
    | "stop_hold"
    | "release";

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
    contract: "stop-sign-goal.v1"
    releasePolicy: "scripted_dwell_release_v0"
    signPose: StopSignPose
    stopLinePose: StopSignPose
    egoStopPose: StopSignPose
    startPose: StopSignPose
    exitPose: StopSignPose
    exitDistanceM: number
    targetSpeedMps: number
    dwellMs: number
    attemptIndex: number
    attemptCount: number
    seed: string
    variationId: string
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
    dwellDurationMs: number
    crossedStopLineBeforeDwell: boolean
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
    stopSignDwellElapsedMs: number
    stopSignDwellTargetMs: number
    stopSignStopped: boolean
    stopSignAttemptIndex: number
    stopSignAttemptCount: number
};
