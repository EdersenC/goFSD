import {resolveParkingAttemptCount, resolveStraightParkingStartOffset} from "./curriculum";
import {ParkingPose} from "./types";

export const MAX_PARKING_BATCH_JOBS = 100;

export type ParkingBatchCommandPose = {
    x: number
    y: number
    z: number
    heading: number
};

export type ParkingBatchCommandJob = {
    id: string
    parkDest: ParkingBatchCommandPose
    startDest: ParkingBatchCommandPose
    collectionAmount: number
    seed: string
};

export type ValidatedParkingBatchJob = {
    id: string
    parkDest: ParkingPose
    startDest: ParkingPose
    collectionAmount: number
    seed: string
};

export type ParkingBatchOperations = {
    setParkingTarget: (pose: ParkingPose) => void
    setParkingStart: (pose: ParkingPose) => void
    startParkingRun: (collectionAmount: number, seed: string) => Promise<void>
    stopRequested: () => boolean
    onJobStart?: (job: ValidatedParkingBatchJob, jobIndex: number, jobCount: number) => void
    onJobComplete?: (job: ValidatedParkingBatchJob, jobIndex: number, jobCount: number) => void
};

export function validateParkingBatchJobs(value: unknown): ValidatedParkingBatchJob[] {
    if (!Array.isArray(value) || value.length < 1 || value.length > MAX_PARKING_BATCH_JOBS) {
        throw new Error(`parkingJobs must contain between 1 and ${MAX_PARKING_BATCH_JOBS} items`);
    }

    const ids = new Set<string>();
    return value.map((candidate, index) => {
        if (!candidate || typeof candidate !== "object") {
            throw new Error(`parkingJobs[${index}] must be an object`);
        }
        const source = candidate as Partial<ParkingBatchCommandJob>;
        const id = requireText(source.id, `parkingJobs[${index}].id`);
        if (ids.has(id)) {
            throw new Error(`parking job id "${id}" is duplicated`);
        }
        ids.add(id);
        const parkDest = parkingPoseFromCommand(source.parkDest, `parkingJobs[${index}].parkDest`);
        const startDest = parkingPoseFromCommand(source.startDest, `parkingJobs[${index}].startDest`);
        resolveStraightParkingStartOffset(startDest, parkDest);
        return {
            id,
            parkDest,
            startDest,
            collectionAmount: resolveParkingAttemptCount(source.collectionAmount),
            seed: requireText(source.seed, `parkingJobs[${index}].seed`),
        };
    });
}

// Runs one validated job at a time. The injected operations are the existing
// exact target, exact start, and parking-run entry points.
export async function executeParkingBatch(
    jobsValue: unknown,
    operations: ParkingBatchOperations
): Promise<number> {
    const jobs = validateParkingBatchJobs(jobsValue);
    let completedJobs = 0;
    for (const [jobOffset, job] of jobs.entries()) {
        if (operations.stopRequested()) {
            break;
        }
        const jobIndex = jobOffset + 1;
        operations.onJobStart?.(job, jobIndex, jobs.length);
        operations.setParkingTarget(job.parkDest);
        operations.setParkingStart(job.startDest);
        await operations.startParkingRun(job.collectionAmount, job.seed);
        if (operations.stopRequested()) {
            break;
        }
        completedJobs += 1;
        operations.onJobComplete?.(job, jobIndex, jobs.length);
    }
    return completedJobs;
}

function parkingPoseFromCommand(value: unknown, label: string): ParkingPose {
    if (!value || typeof value !== "object") {
        throw new Error(`${label} must be an object`);
    }
    const pose = value as Partial<ParkingBatchCommandPose>;
    const coords = [pose.x, pose.y, pose.z];
    if (!coords.every(Number.isFinite) || !Number.isFinite(pose.heading)) {
        throw new Error(`${label} must contain finite x, y, z, and heading values`);
    }
    if ((pose.heading as number) < 0 || (pose.heading as number) >= 360) {
        throw new Error(`${label}.heading must be in [0, 360)`);
    }
    return {
        coords: coords as [number, number, number],
        heading: pose.heading as number,
    };
}

function requireText(value: unknown, label: string): string {
    if (typeof value !== "string" || value.trim() === "") {
        throw new Error(`${label} is required`);
    }
    return value.trim();
}
