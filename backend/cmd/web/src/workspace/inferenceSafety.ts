import type {InferenceStatus} from "../types";

export function inferenceStateConfirmsHold(inference: InferenceStatus): boolean {
    if (inference.active) {
        return false;
    }
    const state = inference.state.trim().toLowerCase();
    return state === "idle" || state === "succeeded";
}
