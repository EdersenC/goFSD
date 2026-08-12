import type {ControlState} from "../types";

export type SafetyStartResult<T> =
    | {kind: "started", value: T}
    | {kind: "superseded-by-hold"};

export async function runSafetyStartWithHoldBarrier<T>(
    start: () => Promise<T>,
    safetyGeneration: number,
    currentSafetyGeneration: () => number,
    reassertStops: () => Promise<unknown>,
): Promise<SafetyStartResult<T>> {
    let settlement: {ok: true, value: T} | {ok: false, error: unknown};
    try {
        settlement = {ok: true, value: await start()};
    } catch (error) {
        settlement = {ok: false, error};
    }

    if (safetyGeneration !== currentSafetyGeneration()) {
        await reassertStops();
        return {kind: "superseded-by-hold"};
    }
    if (!settlement.ok) {
        throw settlement.error;
    }
    return {kind: "started", value: settlement.value};
}

export function requireCurrentSafetyEpoch(control: ControlState | undefined): number {
    const epoch = control?.safetyEpoch;
    if (typeof epoch !== "number" || !Number.isSafeInteger(epoch) || epoch < 1) {
        throw new Error("Refresh the live control state before starting vehicle motion.");
    }
    return epoch;
}

export function controlConsumerConfirmsHold(control: ControlState): boolean {
    return control.runtime.appliedSafetyEpoch === control.safetyEpoch
        && control.runtime.inFlightSafetyStarts === 0;
}

export function controlConsumerHoldBlocker(control: ControlState): string | undefined {
    if (control.runtime.appliedSafetyEpoch !== control.safetyEpoch) {
        return `FiveM has applied safety epoch ${control.runtime.appliedSafetyEpoch}, but epoch ${control.safetyEpoch} is required.`;
    }
    if (control.runtime.inFlightSafetyStarts !== 0) {
        return `${control.runtime.inFlightSafetyStarts} safety-sensitive start${control.runtime.inFlightSafetyStarts === 1 ? " is" : "s are"} still cleaning up in FiveM.`;
    }
    return undefined;
}
