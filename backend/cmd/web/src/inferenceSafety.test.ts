import type {ControlState, InferenceStatus} from "./types";
import {queueStopSignPlan, sendControlCommand} from "./api";
import {createStopSignPlan, stageStopSignCatalogLocation} from "./stop-sign-plan";
import {inferenceStateConfirmsHold} from "./workspace/inferenceSafety";
import {
    controlConsumerConfirmsHold,
    controlConsumerHoldBlocker,
    requireCurrentSafetyEpoch,
    runSafetyStartWithHoldBarrier,
} from "./workspace/safetyStartBarrier";

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

async function test(name: string, run: () => Promise<void> | void) {
    await run();
    console.log(`ok - ${name}`);
}

await test("a Hold superseding a rejected inference start reasserts stop without leaking AbortError", async () => {
    const abortError = new Error("The operation was aborted");
    abortError.name = "AbortError";
    let reassertions = 0;

    const result = await runSafetyStartWithHoldBarrier(
        async () => Promise.reject(abortError),
        4,
        () => 5,
        async () => {
            reassertions++;
        },
    );

    assert(result.kind === "superseded-by-hold", "the Hold generation must supersede the rejected start");
    assert(reassertions === 1, "the superseded start must reassert all safety stops exactly once");
});

await test("a rejected inference start still surfaces when no Hold supersedes it", async () => {
    const startError = new Error("model service unavailable");
    let caught: unknown;
    try {
        await runSafetyStartWithHoldBarrier(
            async () => Promise.reject(startError),
            7,
            () => 7,
            async () => undefined,
        );
    } catch (error) {
        caught = error;
    }
    assert(caught === startError, "a genuine start failure must retain its original error");
});

await test("a Hold superseding a resolved control start reasserts stop before completing", async () => {
    let reassertionFinished = false;
    const result = await runSafetyStartWithHoldBarrier(
        async () => ({status: "queued"}),
        11,
        () => 12,
        async () => {
            await Promise.resolve();
            reassertionFinished = true;
        },
    );

    assert(result.kind === "superseded-by-hold", "the late resolved start must be superseded");
    assert(reassertionFinished, "the barrier must await stop reassertion before releasing the operation");
});

await test("a fresh post-Hold start completes when its generation remains current", async () => {
    const result = await runSafetyStartWithHoldBarrier(
        async () => ({status: "queued", safetyEpoch: 19}),
        13,
        () => 13,
        async () => {
            throw new Error("a current start must not reassert Hold");
        },
    );

    assert(result.kind === "started" && result.value.safetyEpoch === 19, "a current start should preserve its response");
});

await test("fresh control state supplies an epoch while missing or unsafe values fail closed", () => {
    assert(requireCurrentSafetyEpoch({safetyEpoch: 17} as ControlState) === 17, "a current safe integer epoch should be returned");
    for (const safetyEpoch of [undefined, 0, -1, Number.MAX_SAFE_INTEGER + 1]) {
        let rejected = false;
        try {
            requireCurrentSafetyEpoch(safetyEpoch === undefined ? undefined : {safetyEpoch} as ControlState);
        } catch {
            rejected = true;
        }
        assert(rejected, `epoch ${String(safetyEpoch)} must fail closed`);
    }
});

await test("Hold waits for FiveM to apply the current epoch and finish every consumer start", () => {
    const control = {
        safetyEpoch: 32,
        runtime: {
            status: "idle",
            fivemConnected: true,
            appliedSafetyEpoch: 31,
            inFlightSafetyStarts: 0,
        },
        pendingCommands: [],
    } as ControlState;

    assert(!controlConsumerConfirmsHold(control), "a lower consumer epoch must block Hold settlement");
    assert(controlConsumerHoldBlocker(control)?.includes("epoch 32"), "the blocker must identify the required epoch");

    control.runtime.appliedSafetyEpoch = 32;
    control.runtime.inFlightSafetyStarts = 1;
    assert(!controlConsumerConfirmsHold(control), "an in-flight consumer start must block Hold settlement");
    assert(controlConsumerHoldBlocker(control)?.includes("still cleaning up"), "the blocker must explain pending cleanup");

    control.runtime.inFlightSafetyStarts = 0;
    assert(controlConsumerConfirmsHold(control), "matching epoch with no in-flight starts may settle");
});

await test("UI control and stop-sign batch requests carry their captured safety epoch", async () => {
    const originalFetch = globalThis.fetch;
    const bodies: Array<Record<string, unknown>> = [];
    const paths: string[] = [];
    const signals: Array<AbortSignal | null | undefined> = [];
    globalThis.fetch = (async (_input: string | URL | Request, init?: RequestInit) => {
        paths.push(String(_input));
        bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
        signals.push(init?.signal);
        return new Response(JSON.stringify({status: "queued", planId: "plan", jobCount: 0, jobs: [], command: {}}), {
            status: 200,
            headers: {"Content-Type": "application/json"},
        });
    }) as typeof fetch;
    try {
        const controller = new AbortController();
        await sendControlCommand("startEgo", {safetyEpoch: 23}, controller.signal);
        const plan = stageStopSignCatalogLocation(createStopSignPlan(), {id: "sign", x: 1, y: 2, z: 3}).plan;
        plan.id = "plan";
        await queueStopSignPlan(plan, 24, controller.signal);
        assert(signals.every((signal) => signal === controller.signal), "every UI start request must remain abortable by Hold");
    } finally {
        globalThis.fetch = originalFetch;
    }

    assert(bodies[0]?.type === "startEgo" && bodies[0]?.safetyEpoch === 23, "control starts must carry the captured epoch");
    assert(bodies[1]?.id === "plan" && bodies[1]?.safetyEpoch === 24, "stop-sign plan starts must carry the captured epoch");
    assert(paths[1] === "/control/stop-sign-batches", "stop-sign plans must use the canonical batch endpoint");
    assert(!("version" in bodies[1]!), "browser-only storage version must not leak into the strict backend contract");
    const entries = bodies[1]?.entries as Array<{autoVariations?: {count?: number}}>;
    assert(entries[0]?.autoVariations?.count === 50, "seeded auto-variation settings must reach the backend contract");
});

await test("Hold settlement accepts only inactive idle or succeeded inference", () => {
    assert(inferenceStateConfirmsHold(inferenceStatus("idle", false)), "inactive idle inference should confirm Hold");
    assert(inferenceStateConfirmsHold(inferenceStatus("succeeded", false)), "inactive succeeded inference should confirm Hold");

    for (const state of ["starting", "running", "stopping", "error", "unknown", ""]) {
        assert(!inferenceStateConfirmsHold(inferenceStatus(state, false)), `${state || "empty"} inference state must not confirm Hold`);
    }
    assert(!inferenceStateConfirmsHold(inferenceStatus("idle", true)), "active inference must not confirm Hold even if its state is idle");
});

console.log("inference Hold safety contract tests passed");

function inferenceStatus(state: string, active: boolean): InferenceStatus {
    return {
        state,
        active,
        actuatorReady: true,
        controllerReady: true,
        calibrationVerified: true,
        safetyReady: false,
        sourceFps: 30,
        inferenceHz: 10,
        windowSize: 5,
        frameStride: 2,
        dispatchStride: 3,
        frameWidth: 480,
        frameHeight: 480,
        framesSeen: 0,
        predictionsSent: 0,
        predictionErrors: 0,
    };
}
