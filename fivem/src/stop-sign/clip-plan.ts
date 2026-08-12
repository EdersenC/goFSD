import {StopSignBehaviorPhase, StopSignClipStage} from "./types";

export const STOP_SIGN_LOGICAL_CLIP_STAGES: readonly StopSignClipStage[] = [
    "approach",
    "brake_stop",
    "release",
];

/**
 * Assigns a logical training clip to an anchor frame without splitting the
 * underlying physical recording. History and future windows may cross these
 * boundaries so the temporal policy keeps the real transition context.
 */
export function logicalClipStageForPhase(phase: StopSignBehaviorPhase): StopSignClipStage {
    switch (phase) {
        case "accelerate":
        case "cruise_approach":
            return "approach";
        case "decelerate":
        case "stop_hold":
            return "brake_stop";
        case "release":
            return "release";
    }
}
