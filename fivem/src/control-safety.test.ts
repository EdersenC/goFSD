import {
    canAcknowledgeControlSafetyEpochSync,
    ControlSafetyBarrier,
    parseControlSafetyEpochSync,
    runGuardedSafetyStart,
    StaleSafetyStartError,
} from "./control-safety";

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function assertThrowsStale(work: () => void, message: string) {
    try {
        work();
    } catch (error) {
        assert(error instanceof StaleSafetyStartError, `${message}: wrong error ${String(error)}`);
        return;
    }
    throw new Error(`${message}: expected a stale-start error`);
}

function testStopKeepsOlderStartInFlightUntilCleanupFinishes() {
    const barrier = new ControlSafetyBarrier();
    const start = barrier.beginStart(4);

    const stopping = barrier.applyEmergencyStop(5);
    assert(stopping.appliedSafetyEpoch === 5, "the emergency epoch must be applied immediately");
    assert(stopping.inFlightSafetyStarts === 1, "the interrupted start must remain visible while cleanup is pending");
    assert(stopping.hasStaleSafetyStarts, "the lower-epoch start must be marked stale");
    assert(start.isStale(), "the start lease must observe the newer emergency epoch");
    assertThrowsStale(start.throwIfStale, "a stale continuation must not resume");

    const settled = start.finish();
    assert(settled.inFlightSafetyStarts === 0, "cleanup completion must release the in-flight barrier");
    assert(!settled.hasStaleSafetyStarts, "no stale start may remain after cleanup");
}

function testOldDeliveryCannotStartAfterNewerStop() {
    const barrier = new ControlSafetyBarrier();
    barrier.applyEmergencyStop(12);
    assertThrowsStale(() => barrier.beginStart(11), "a delayed lower-epoch delivery must be rejected");

    const current = barrier.beginStart(12);
    assert(!current.isStale(), "a newly accepted current-epoch start remains allowed");
    current.finish();
}

function testFinishIsIdempotent() {
    const barrier = new ControlSafetyBarrier();
    const start = barrier.beginStart(2);
    start.finish();
    const snapshot = start.finish();
    assert(snapshot.inFlightSafetyStarts === 0, "finishing a lease twice must not underflow the barrier");
}

function testSafetyEpochSyncPayloadValidation() {
    const parsed = parseControlSafetyEpochSync({
        requestId: " sync-1 ",
        sessionId: " session-1 ",
        safetyEpoch: 9,
    });
    assert(parsed.requestId === "sync-1" && parsed.sessionId === "session-1" && parsed.safetyEpoch === 9, "valid sync identity must be normalized exactly");

    for (const invalid of [
        null,
        {},
        {requestId: "", sessionId: "session", safetyEpoch: 9},
        {requestId: "sync", sessionId: "", safetyEpoch: 9},
        {requestId: "sync", sessionId: "session", safetyEpoch: 0},
        {requestId: "sync", sessionId: "session", safetyEpoch: 1.5},
        {requestId: "sync", sessionId: "session", safetyEpoch: "9"},
    ]) {
        let rejected = false;
        try {
            parseControlSafetyEpochSync(invalid);
        } catch {
            rejected = true;
        }
        assert(rejected, `invalid safety sync payload must be rejected: ${JSON.stringify(invalid)}`);
    }
}

function testSafetyEpochSyncAcknowledgementRequiresSettledCleanup() {
    const sync = {requestId: "sync-1", sessionId: "session-1", safetyEpoch: 9};
    const settled = {
        appliedSafetyEpoch: 9,
        inFlightSafetyStarts: 0,
        hasStaleSafetyStarts: false,
    };

    assert(!canAcknowledgeControlSafetyEpochSync(sync, false, settled), "sync must wait for force cleanup to complete");
    assert(!canAcknowledgeControlSafetyEpochSync(sync, true, {...settled, appliedSafetyEpoch: 8}), "sync must wait for its epoch fence");
    assert(!canAcknowledgeControlSafetyEpochSync(sync, true, {...settled, inFlightSafetyStarts: 1}), "sync must wait for every older start lease to drain");
    assert(canAcknowledgeControlSafetyEpochSync(sync, true, settled), "settled cleanup at the synchronized epoch may be acknowledged");
    assert(canAcknowledgeControlSafetyEpochSync(sync, true, {...settled, appliedSafetyEpoch: 10}), "a newer applied epoch also satisfies the fence");
}

async function testDelayedStartCleansUpBeforeBarrierSettles() {
    const barrier = new ControlSafetyBarrier();
    let resume!: () => void;
    const delayed = new Promise<void>((resolve) => {
        resume = resolve;
    });
    let vehicleExists = false;
    let cleanupFinished = false;
    const snapshots: Array<{appliedSafetyEpoch: number, inFlightSafetyStarts: number}> = [];

    const start = runGuardedSafetyStart(
        barrier,
        20,
        async () => {
            await delayed;
            vehicleExists = true;
        },
        () => {
            vehicleExists = false;
            cleanupFinished = true;
        },
        ({appliedSafetyEpoch, inFlightSafetyStarts}) => snapshots.push({appliedSafetyEpoch, inFlightSafetyStarts}),
    );

    const stopping = barrier.applyEmergencyStop(21);
    assert(stopping.inFlightSafetyStarts === 1, "Hold must see the delayed start as in flight");
    resume();
    const result = await start;

    assert(result.kind === "canceled", "the delayed lower-epoch start must be canceled");
    assert(cleanupFinished && !vehicleExists, "a vehicle created by the delayed continuation must be cleaned up");
    assert(snapshots[snapshots.length - 1]?.inFlightSafetyStarts === 0, "the barrier may settle only after cleanup finishes");
}

async function testHigherEpochStartFencesEarlierLeaseWhileCleanupDrains() {
    const barrier = new ControlSafetyBarrier();
    const earlier = barrier.beginStart(30);
    const snapshots: Array<{appliedSafetyEpoch: number, inFlightSafetyStarts: number, hasStaleSafetyStarts: boolean}> = [];
    let newerWorkRan = false;

    const newer = await runGuardedSafetyStart(
        barrier,
        31,
        async () => {
            newerWorkRan = true;
        },
        () => undefined,
        (snapshot) => snapshots.push(snapshot),
    );

    assert(newer.kind === "canceled", "the newer start must wait for the earlier lease to clean up");
    assert(!newerWorkRan, "overlapping newer-epoch work must not begin");
    assert(earlier.isStale(), "advancing to the newer epoch must fence the earlier lease");
    const fenced = snapshots[snapshots.length - 1];
    assert(fenced?.appliedSafetyEpoch === 31, "the rejected newer start must still publish its epoch fence");
    assert(fenced?.inFlightSafetyStarts === 1 && fenced.hasStaleSafetyStarts, "the earlier cleanup lease must remain visible");
    earlier.finish();
}

async function testDelayedSceneInitializationChecksCancellation(kind: "startScene" | "runAllScenes") {
    const barrier = new ControlSafetyBarrier();
    let resumeModelLoad!: () => void;
    const modelLoad = new Promise<void>((resolve) => {
        resumeModelLoad = resolve;
    });
    let vehicleSpawned = false;
    let forceCleanupRan = false;

    const start = runGuardedSafetyStart(
        barrier,
        40,
        async (lease) => {
            await modelLoad;
            lease.throwIfStale();
            vehicleSpawned = true;
        },
        () => {
            vehicleSpawned = false;
            forceCleanupRan = true;
        },
    );

    barrier.applyEmergencyStop(41);
    resumeModelLoad();
    const result = await start;

    assert(result.kind === "canceled", `${kind} must be canceled after a stop during model loading`);
    assert(!vehicleSpawned, `${kind} must not spawn after its cancellation callback becomes true`);
    assert(forceCleanupRan, `${kind} must run fail-safe cleanup before releasing its barrier lease`);
    assert(barrier.snapshot().inFlightSafetyStarts === 0, `${kind} cleanup must finish before the barrier settles`);
}

async function main() {
    testStopKeepsOlderStartInFlightUntilCleanupFinishes();
    testOldDeliveryCannotStartAfterNewerStop();
    testFinishIsIdempotent();
    testSafetyEpochSyncPayloadValidation();
    testSafetyEpochSyncAcknowledgementRequiresSettledCleanup();
    await testDelayedStartCleansUpBeforeBarrierSettles();
    await testHigherEpochStartFencesEarlierLeaseWhileCleanupDrains();
    await testDelayedSceneInitializationChecksCancellation("startScene");
    await testDelayedSceneInitializationChecksCancellation("runAllScenes");
    console.log("control safety tests passed");
}

void main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
