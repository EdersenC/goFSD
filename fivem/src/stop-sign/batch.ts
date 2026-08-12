import {StopSignCatalogPosition, StopSignJob, StopSignPose} from "./types";
import {poseAhead, poseBehind, relativeStopLinePose} from "./geometry";
import {parseStopSignVariationProfile} from "./variation-profile";

const derivedPoseToleranceM = 1e-4;

export function parseStopSignJobs(value: unknown): StopSignJob[] {
    if (!Array.isArray(value) || value.length === 0 || value.length > 5000) {
        throw new Error("stopSignJobs must contain between 1 and 5000 jobs");
    }
    const ids = new Set<string>();
    return value.map((raw, index) => parseJob(raw, index, ids));
}

function parseJob(raw: unknown, index: number, ids: Set<string>): StopSignJob {
    if (!isRecord(raw)) {
        throw new Error(`stopSignJobs[${index}] must be an object`);
    }
    const id = requiredString(raw.id, `stopSignJobs[${index}].id`);
    if (ids.has(id)) {
        throw new Error(`stopSignJobs contains duplicate id ${id}`);
    }
    ids.add(id);
    const attemptCount = boundedNumber(raw.attemptCount, `stopSignJobs[${index}].attemptCount`, 1, 50, true);
    const targetSpeedMps = boundedNumber(raw.targetSpeedMps, `stopSignJobs[${index}].targetSpeedMps`, 0.5, 15);
    const brakingDecelerationMps2 = boundedNumber(raw.brakingDecelerationMps2, `stopSignJobs[${index}].brakingDecelerationMps2`, 3.5, 4.2);
    const releaseAccelerationMps2 = boundedNumber(raw.releaseAccelerationMps2, `stopSignJobs[${index}].releaseAccelerationMps2`, 3, 5);
    const stopConfirmationMs = boundedNumber(raw.stopConfirmationMs, `stopSignJobs[${index}].stopConfirmationMs`, 100, 1000, true);
    const stopDistanceM = boundedNumber(raw.stopDistanceM, `stopSignJobs[${index}].stopDistanceM`, 0.5, 15);
    const egoCenterOffsetM = boundedNumber(raw.egoCenterOffsetM, `stopSignJobs[${index}].egoCenterOffsetM`, 0.5, 8);
    const startDistanceM = boundedNumber(raw.startDistanceM, `stopSignJobs[${index}].startDistanceM`, 5, 250);
    const exitDistanceM = boundedNumber(raw.exitDistanceM, `stopSignJobs[${index}].exitDistanceM`, 2, 50);
    const time = isRecord(raw.time) ? raw.time : {};
    const vehicle = isRecord(raw.vehicle) ? raw.vehicle : {};
    const color = isRecord(vehicle.color) ? {
        r: boundedNumber(vehicle.color.r, `stopSignJobs[${index}].vehicle.color.r`, 0, 255, true),
        g: boundedNumber(vehicle.color.g, `stopSignJobs[${index}].vehicle.color.g`, 0, 255, true),
        b: boundedNumber(vehicle.color.b, `stopSignJobs[${index}].vehicle.color.b`, 0, 255, true),
    } : undefined;

    const catalogId = optionalString(raw.catalogId);
    const job: StopSignJob = {
        id,
        entryId: requiredString(raw.entryId, `stopSignJobs[${index}].entryId`),
        variationId: requiredString(raw.variationId, `stopSignJobs[${index}].variationId`),
        catalogId,
        catalogPosition: catalogId
            ? parseCatalogPosition(raw.catalogPosition, `stopSignJobs[${index}].catalogPosition`)
            : undefined,
        signPose: parseStopSignPose(raw.signPose, `stopSignJobs[${index}].signPose`),
        stopLinePose: parseStopSignPose(raw.stopLinePose, `stopSignJobs[${index}].stopLinePose`),
        egoStopPose: parseStopSignPose(raw.egoStopPose, `stopSignJobs[${index}].egoStopPose`),
        startPose: parseStopSignPose(raw.startPose, `stopSignJobs[${index}].startPose`),
        exitPose: parseStopSignPose(raw.exitPose, `stopSignJobs[${index}].exitPose`),
        stopDistanceM,
        egoCenterOffsetM,
        startDistanceM,
        exitDistanceM,
        targetSpeedMps,
        brakingDecelerationMps2,
        releaseAccelerationMps2,
        stopConfirmationMs,
        attemptCount,
        weather: requiredString(raw.weather, `stopSignJobs[${index}].weather`),
        time: {
            hour: boundedNumber(time.hour, `stopSignJobs[${index}].time.hour`, 0, 23, true),
            minute: boundedNumber(time.minute, `stopSignJobs[${index}].time.minute`, 0, 59, true),
        },
        vehicle: {
            model: optionalString(vehicle.model),
            color,
        },
        seed: requiredString(raw.seed, `stopSignJobs[${index}].seed`),
        variationProfile: parseStopSignVariationProfile(raw.variationProfile, `stopSignJobs[${index}].variationProfile`),
    };
    rejectUncalibratedSignPose(job.signPose, index);
    validateDerivedGeometry(job, index);
    return job;
}

function rejectUncalibratedSignPose(pose: StopSignPose, index: number) {
    if (pose.x === 0 && pose.y === 0 && pose.z === 0) {
        throw new Error(`stopSignJobs[${index}].signPose is still the uncalibrated origin placeholder`);
    }
}

function validateDerivedGeometry(job: StopSignJob, index: number) {
    if (job.catalogId) {
        const expectedStopLine = poseAhead(job.egoStopPose, job.egoCenterOffsetM);
        const expectedSign = poseAhead(expectedStopLine, job.stopDistanceM);
        requireMatchingPose(job.stopLinePose, expectedStopLine, `stopSignJobs[${index}].stopLinePose`);
        requireMatchingPose(job.signPose, expectedSign, `stopSignJobs[${index}].signPose`);
        const startDistanceM = planarDistance(job.startPose, job.egoStopPose);
        const exitDistanceM = planarDistance(job.egoStopPose, job.exitPose);
        if (Math.abs(startDistanceM - job.startDistanceM) > derivedPoseToleranceM) {
            throw new Error(`stopSignJobs[${index}].startDistanceM contradicts the captured start pose`);
        }
        if (Math.abs(exitDistanceM - job.exitDistanceM) > derivedPoseToleranceM) {
            throw new Error(`stopSignJobs[${index}].exitDistanceM contradicts the captured end pose`);
        }
        const start = relativeStopLinePose(job.startPose, job.egoStopPose);
        const exit = relativeStopLinePose(job.exitPose, job.egoStopPose);
        if (start.longitudinalM > -5 || exit.longitudinalM < 2) {
            throw new Error(`stopSignJobs[${index}] captured route must travel Start -> Stop -> End`);
        }
        if (Math.abs(start.lateralM) > 8 || Math.abs(exit.lateralM) > 8) {
            throw new Error(`stopSignJobs[${index}] captured route must stay within 8m of one lane`);
        }
        return;
    }
    const expectedStopLine = poseBehind(job.signPose, job.stopDistanceM);
    const expectedEgoStop = poseBehind(expectedStopLine, job.egoCenterOffsetM);
    const expectedStart = poseBehind(expectedEgoStop, job.startDistanceM);
    const expectedExit = poseAhead(job.signPose, job.exitDistanceM);
    requireMatchingPose(job.stopLinePose, expectedStopLine, `stopSignJobs[${index}].stopLinePose`);
    requireMatchingPose(job.egoStopPose, expectedEgoStop, `stopSignJobs[${index}].egoStopPose`);
    requireMatchingPose(job.startPose, expectedStart, `stopSignJobs[${index}].startPose`);
    requireMatchingPose(job.exitPose, expectedExit, `stopSignJobs[${index}].exitPose`);
}

function parseCatalogPosition(raw: unknown, label: string): StopSignCatalogPosition {
    if (!isRecord(raw)) {
        throw new Error(`${label} must be an object`);
    }
    const position = {
        x: finiteNumber(raw.x, `${label}.x`),
        y: finiteNumber(raw.y, `${label}.y`),
        z: finiteNumber(raw.z, `${label}.z`),
    };
    if (Math.abs(position.x) > 10_000 || Math.abs(position.y) > 10_000 || position.z < -1_000 || position.z > 3_000) {
        throw new Error(`${label} is outside supported GTA world bounds`);
    }
    return position;
}

function planarDistance(first: StopSignPose, second: StopSignPose): number {
    return Math.hypot(first.x - second.x, first.y - second.y);
}

function requireMatchingPose(actual: StopSignPose, expected: StopSignPose, label: string) {
    const positionErrorM = Math.hypot(actual.x - expected.x, actual.y - expected.y, actual.z - expected.z);
    const headingErrorDeg = Math.abs(actual.heading - expected.heading);
    if (positionErrorM > derivedPoseToleranceM || headingErrorDeg > derivedPoseToleranceM) {
        throw new Error(`${label} contradicts the sign-relative distance contract`);
    }
}

export function parseStopSignPose(raw: unknown, label = "stop-sign pose"): StopSignPose {
    if (!isRecord(raw)) {
        throw new Error(`${label} must be an object`);
    }
    return {
        x: finiteNumber(raw.x, `${label}.x`),
        y: finiteNumber(raw.y, `${label}.y`),
        z: finiteNumber(raw.z, `${label}.z`),
        heading: boundedNumber(raw.heading, `${label}.heading`, 0, 360, false, true),
    };
}

function boundedNumber(
    value: unknown,
    label: string,
    minimum: number,
    maximum: number,
    integer = false,
    maximumExclusive = false,
): number {
    const parsed = finiteNumber(value, label);
    if (parsed < minimum || (maximumExclusive ? parsed >= maximum : parsed > maximum) || (integer && !Number.isInteger(parsed))) {
        const closing = maximumExclusive ? ")" : "]";
        throw new Error(`${label} must be ${integer ? "an integer " : ""}within [${minimum}, ${maximum}${closing}`);
    }
    return parsed;
}

function finiteNumber(value: unknown, label: string): number {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`${label} must be finite`);
    }
    return value;
}

function requiredString(value: unknown, label: string): string {
    const result = optionalString(value);
    if (!result) {
        throw new Error(`${label} is required`);
    }
    return result;
}

function optionalString(value: unknown): string | undefined {
    return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
