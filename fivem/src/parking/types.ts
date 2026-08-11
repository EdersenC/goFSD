export type ParkingVector3 = [number, number, number];

export type ParkingPose = {
    coords: ParkingVector3
    heading: number
};

export type ParkingBay = {
    widthM: number
    lengthM: number
};

export type ParkingTolerances = {
    maxCenterDistanceM: number
    maxHeadingErrorDeg: number
    maxSpeedMps: number
    settledDurationMs: number
};

export type ParkingTarget = {
    pose: ParkingPose
    bay: ParkingBay
    tolerances: ParkingTolerances
};

export type ParkingStartOffset = {
    longitudinalM: number
    lateralM: number
    headingDeg: number
};

export type ParkingAttemptPlan = {
    attemptIndex: number
    startPose: ParkingPose
    startOffset: ParkingStartOffset
};

export type ParkingGoal = {
    task: "parking"
    maneuver: "forward-bay"
    visualCueId: string
    target: ParkingPose
    bay: ParkingBay
    tolerances: ParkingTolerances
    attemptIndex: number
    attemptCount: number
    seed: string
    startPose: ParkingPose
    startOffset: ParkingStartOffset
};

export type ParkingSetup = {
    target: ParkingTarget
    startPose: ParkingPose
    startOffset: ParkingStartOffset
};

export type ParkingOutcomeStatus =
    | "succeeded"
    | "collision"
    | "left_straight_corridor"
    | "reversing"
    | "off_ground"
    | "not_upright"
    | "timeout"
    | "stopped"
    | "invalid_vehicle";

export type ParkingOutcome = {
    success: boolean
    status: ParkingOutcomeStatus
    failureReason: string
    durationMs: number
    finalLongitudinalError: number
    finalLateralError: number
    finalHeadingError: number
    finalDistance: number
    finalInsideBay: boolean
    finalAligned: boolean
    settledDurationMs: number
    collision: boolean
};

export type ParkingPhase =
    | "idle"
    | "ready"
    | "spawning"
    | "recording"
    | "parking"
    | "settling"
    | "succeeded"
    | "failed"
    | "stopping";

export type VehicleFootprint = {
    halfWidthM: number
    halfLengthM: number
};

export type ParkingState = {
    longitudinalError: number
    lateralError: number
    headingError: number
    distance: number
    insideBay: boolean
    aligned: boolean
    parked: boolean
    settledDurationMs: number
};

/**
 * Labels emitted by the deterministic parking expert for one captured sample.
 * These are supervision only; the trained model never receives them as inputs.
 */
export type ParkingExpertSupervision = {
    desiredWheelSteerNormalized: number
    desiredSpeedMps: number
    stopProbability: number
};

export type ParkingTelemetry = {
    parkingTargetConfigured: boolean
    parkingStartConfigured: boolean
    parkingTargetPose?: ParkingPose
    parkingStartPose?: ParkingPose
    parkingLongitudinalError: number
    parkingLateralError: number
    parkingHeadingError: number
    parkingDistance: number
    parkingInsideBay: boolean
    parkingAligned: boolean
    parkingParked: boolean
    parkingAttemptIndex: number
    parkingAttemptCount: number
    parkingPhase: ParkingPhase
};
