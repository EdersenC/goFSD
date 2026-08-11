import {brakingOnsetDistance} from "./expert";
import {StopSignClipStage, StopSignJob, StopSignPose} from "./types";

export type StopSignClipPlan = {
    stage: StopSignClipStage
    tripIndex: number
    fromPose: StopSignPose
    toPose: StopSignPose
    initialSpeedMps: number
};

export function planStopSignClips(job: StopSignJob, firstTripIndex: number): [StopSignClipPlan, StopSignClipPlan, StopSignClipPlan] {
    if (!Number.isSafeInteger(firstTripIndex) || firstTripIndex < 0) {
        throw new Error("Stop-sign clip plan requires a non-negative trip index");
    }
    const brakingStart = brakingStageStartPose(job);
    return [
        {
            stage: "approach",
            tripIndex: firstTripIndex,
            fromPose: {...job.startPose},
            toPose: brakingStart,
            initialSpeedMps: 0,
        },
        {
            stage: "brake_stop",
            tripIndex: firstTripIndex + 1,
            fromPose: brakingStart,
            toPose: {...job.egoStopPose},
            initialSpeedMps: job.targetSpeedMps,
        },
        {
            stage: "release",
            tripIndex: firstTripIndex + 2,
            fromPose: {...job.egoStopPose},
            toPose: {...job.exitPose},
            initialSpeedMps: 0,
        },
    ];
}

export function brakingStageStartPose(job: StopSignJob): StopSignPose {
    const totalDistanceM = Math.hypot(
        job.startPose.x - job.egoStopPose.x,
        job.startPose.y - job.egoStopPose.y,
        job.startPose.z - job.egoStopPose.z,
    );
    if (!Number.isFinite(totalDistanceM) || totalDistanceM <= 2) {
        throw new Error("Stop-sign stage split requires Start to be more than 2m before Stop");
    }
    const brakingDistanceM = Math.min(
        totalDistanceM - 1,
        Math.max(2, brakingOnsetDistance(job.targetSpeedMps) + 1),
    );
    const ratio = brakingDistanceM / totalDistanceM;
    return {
        x: job.egoStopPose.x + (job.startPose.x - job.egoStopPose.x) * ratio,
        y: job.egoStopPose.y + (job.startPose.y - job.egoStopPose.y) * ratio,
        z: job.egoStopPose.z + (job.startPose.z - job.egoStopPose.z) * ratio,
        heading: job.egoStopPose.heading,
    };
}
