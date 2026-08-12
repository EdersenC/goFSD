import {StopSignVariationProfile} from "./types";

export const STOP_SIGN_VARIATION_PROFILE_CONTRACT = "stop-sign-variation-profile.v2";

const canonicalDimensions = [
    "target_speed",
    "braking_deceleration",
    "release_acceleration",
    "start_distance",
    "exit_distance",
    "stop_pose",
    "stop_heading",
    "start_lane_offset",
    "start_heading",
    "exit_lane_offset",
    "exit_heading",
    "weather",
    "time",
    "vehicle_model",
    "vehicle_color",
] as const;

export function parseStopSignVariationProfile(value: unknown, label: string): StopSignVariationProfile {
    if (!isRecord(value)) {
        throw new Error(`${label} must be an object`);
    }
    if (value.contract !== STOP_SIGN_VARIATION_PROFILE_CONTRACT) {
        throw new Error(`${label}.contract must equal ${STOP_SIGN_VARIATION_PROFILE_CONTRACT}`);
    }
    const profile: StopSignVariationProfile = {
        contract: STOP_SIGN_VARIATION_PROFILE_CONTRACT,
        baseline: requiredBoolean(value.baseline, `${label}.baseline`),
        configuredMotionVariancePct: boundedNumber(value.configuredMotionVariancePct, `${label}.configuredMotionVariancePct`, 0, 25),
        changedDimensions: parseDimensions(value.changedDimensions, `${label}.changedDimensions`),
        changeCount: boundedNumber(value.changeCount, `${label}.changeCount`, 0, canonicalDimensions.length, true),
        combinationMagnitudePct: boundedNumber(value.combinationMagnitudePct, `${label}.combinationMagnitudePct`, 0, 100),
        targetSpeedDeltaMps: boundedNumber(value.targetSpeedDeltaMps, `${label}.targetSpeedDeltaMps`, -15, 15),
        targetSpeedDeltaPct: boundedNumber(value.targetSpeedDeltaPct, `${label}.targetSpeedDeltaPct`, -100, 3000),
        brakingDecelerationDeltaMps2: boundedNumber(value.brakingDecelerationDeltaMps2, `${label}.brakingDecelerationDeltaMps2`, -.7, .7),
        releaseAccelerationDeltaMps2: boundedNumber(value.releaseAccelerationDeltaMps2, `${label}.releaseAccelerationDeltaMps2`, -2, 2),
        startDistanceDeltaM: boundedNumber(value.startDistanceDeltaM, `${label}.startDistanceDeltaM`, -250, 250),
        exitDistanceDeltaM: boundedNumber(value.exitDistanceDeltaM, `${label}.exitDistanceDeltaM`, -50, 50),
        stopOffsetM: boundedNumber(value.stopOffsetM, `${label}.stopOffsetM`, 0, 25),
        stopHeadingDeltaDeg: boundedNumber(value.stopHeadingDeltaDeg, `${label}.stopHeadingDeltaDeg`, 0, 180),
        startLaneOffsetDeltaM: boundedNumber(value.startLaneOffsetDeltaM, `${label}.startLaneOffsetDeltaM`, -16, 16),
        startHeadingDeltaDeg: boundedNumber(value.startHeadingDeltaDeg, `${label}.startHeadingDeltaDeg`, 0, 180),
        exitLaneOffsetDeltaM: boundedNumber(value.exitLaneOffsetDeltaM, `${label}.exitLaneOffsetDeltaM`, -16, 16),
        exitHeadingDeltaDeg: boundedNumber(value.exitHeadingDeltaDeg, `${label}.exitHeadingDeltaDeg`, 0, 180),
        timeDeltaMinutes: boundedNumber(value.timeDeltaMinutes, `${label}.timeDeltaMinutes`, 0, 720, true),
        weatherChanged: requiredBoolean(value.weatherChanged, `${label}.weatherChanged`),
        vehicleModelChanged: requiredBoolean(value.vehicleModelChanged, `${label}.vehicleModelChanged`),
        vehicleColorDeltaPct: boundedNumber(value.vehicleColorDeltaPct, `${label}.vehicleColorDeltaPct`, 0, 100),
    };
    validateInternalConsistency(profile, label);
    return profile;
}

export function cloneStopSignVariationProfile(profile: StopSignVariationProfile): StopSignVariationProfile {
    return {...profile, changedDimensions: [...profile.changedDimensions]};
}

function validateInternalConsistency(profile: StopSignVariationProfile, label: string) {
    const expected = changedDimensions(profile);
    if (profile.changeCount !== profile.changedDimensions.length) {
        throw new Error(`${label}.changeCount must match changedDimensions`);
    }
    if (expected.length !== profile.changedDimensions.length || expected.some((dimension, index) => dimension !== profile.changedDimensions[index])) {
        throw new Error(`${label}.changedDimensions contradicts the exact deltas or canonical order`);
    }
    if (profile.baseline && (profile.changeCount !== 0 || profile.combinationMagnitudePct !== 0)) {
        throw new Error(`${label} baseline must have zero changes and zero magnitude`);
    }
}

function changedDimensions(profile: StopSignVariationProfile): string[] {
    const changed = [
        nonZero(profile.targetSpeedDeltaMps),
        nonZero(profile.brakingDecelerationDeltaMps2),
        nonZero(profile.releaseAccelerationDeltaMps2),
        nonZero(profile.startDistanceDeltaM),
        nonZero(profile.exitDistanceDeltaM),
        nonZero(profile.stopOffsetM),
        nonZero(profile.stopHeadingDeltaDeg),
        nonZero(profile.startLaneOffsetDeltaM),
        nonZero(profile.startHeadingDeltaDeg),
        nonZero(profile.exitLaneOffsetDeltaM),
        nonZero(profile.exitHeadingDeltaDeg),
        profile.weatherChanged,
        profile.timeDeltaMinutes > 0,
        profile.vehicleModelChanged,
        nonZero(profile.vehicleColorDeltaPct),
    ];
    return canonicalDimensions.filter((_dimension, index) => changed[index]);
}

function parseDimensions(value: unknown, label: string): string[] {
    if (!Array.isArray(value) || value.some((dimension) => typeof dimension !== "string")) {
        throw new Error(`${label} must be an array of canonical dimension names`);
    }
    return [...value];
}

function boundedNumber(value: unknown, label: string, minimum: number, maximum: number, integer = false): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isInteger(value))) {
        throw new Error(`${label} must be ${integer ? "an integer " : ""}between ${minimum} and ${maximum}`);
    }
    return value;
}

function requiredBoolean(value: unknown, label: string): boolean {
    if (typeof value !== "boolean") {
        throw new Error(`${label} must be a boolean`);
    }
    return value;
}

function nonZero(value: number): boolean {
    return Math.abs(value) > 1e-6;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
