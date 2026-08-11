import {
    ControlSessionFrontier,
    matchesControlSafetySyncAck,
} from "./control-session-frontier";

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function claim(frontier: ControlSessionFrontier, playerSource: number): number {
    const result = frontier.beginReconnect(playerSource);
    assert(result.kind === "accepted", `source ${playerSource} should claim the control session`);
    return result.generation;
}

async function testDeferredDispatchConfirmationCannotCrossReconnectFrontier() {
    const frontier = new ControlSessionFrontier();
    const initialGeneration = claim(frontier, 17);
    const initialConnect = frontier.beginConnect();
    assert(initialConnect?.generation === initialGeneration, "initial connect should claim the requested generation");
    assert(frontier.completeConnect(initialConnect).kind === "safety-sync-required", "initial connect should require safety sync");
    assert(frontier.beginPoll() === undefined, "polling must remain paused before safety sync acknowledgement");
    assert(frontier.completeSafetySync(initialGeneration, 17), "initial safety sync should make the session ready");

    const stalePoll = frontier.beginPoll();
    assert(stalePoll !== undefined, "ready session should produce a poll lease");
    assert(frontier.isPollCurrent(stalePoll), "poll response should be current before backend confirmation begins");
    let resolveConfirmation: ((commandId: string) => void) | undefined;
    const deferredConfirmation = new Promise<string>((resolve) => {
        resolveConfirmation = resolve;
    });
    const dispatched: string[] = [];
    const deferredDispatch = deferredConfirmation.then((commandId) => {
        frontier.dispatchConfirmedCommand(stalePoll, commandId, () => dispatched.push(commandId));
    });

    const resetGeneration = claim(frontier, 17);
    assert(frontier.beginPoll() === undefined, "polling must pause as soon as reconnect begins");
    const resetConnect = frontier.beginConnect();
    assert(resetConnect?.generation === resetGeneration, "reset should claim the latest generation");
    assert(frontier.completeConnect(resetConnect).kind === "safety-sync-required", "reset should require safety sync");
    assert(frontier.beginPoll() === undefined, "polling must stay paused while the reset epoch is unapplied");
    assert(frontier.completeSafetySync(resetGeneration, 17), "reset epoch acknowledgement should reopen polling");

    assert(resolveConfirmation !== undefined, "deferred confirmation resolver should be available");
    resolveConfirmation("epoch-old-command");
    await deferredDispatch;
    assert(dispatched.length === 0, "a confirmation begun before reset must never dispatch after reset");

    const freshPoll = frontier.beginPoll();
    assert(freshPoll?.lastSeenCommandId === "", "new session must begin with an empty command cursor");
    assert(
        frontier.dispatchConfirmedCommand(freshPoll, "epoch-new-command", () => dispatched.push("epoch-new-command")),
        "post-sync confirmation should dispatch",
    );
    assert(frontier.beginPoll()?.lastSeenCommandId === "epoch-new-command", "accepted poll should advance the new cursor");
}

function testReconnectDuringConnectRequiresAnotherReset() {
    const frontier = new ControlSessionFrontier();
    claim(frontier, 17);
    const firstConnect = frontier.beginConnect();
    assert(firstConnect !== undefined, "first connect should begin");

    const latestGeneration = claim(frontier, 17);
    assert(frontier.isCurrentGeneration(latestGeneration), "latest reconnect should replace the requested generation");
    assert(!frontier.isCurrentGeneration(firstConnect.generation), "in-flight connect should become superseded immediately");
    assert(frontier.beginConnect() === undefined, "connect requests must remain serialized");
    assert(frontier.completeConnect(firstConnect).kind === "stale", "superseded connect response must be discarded");

    const latestConnect = frontier.beginConnect();
    assert(latestConnect?.generation === latestGeneration, "a superseded reset must be followed by the latest reset");
    assert(frontier.completeConnect(latestConnect).kind === "safety-sync-required", "latest reset should require sync");
    assert(frontier.completeSafetySync(latestGeneration, 17), "latest sync should complete");
    assert(frontier.beginPoll()?.generation === latestGeneration, "only the latest generation may poll");
    assert(frontier.beginPoll()?.playerSource === 17, "same-owner registration must retain poll ownership");
}

function testSingleOwnerClaimAndRelease() {
    const frontier = new ControlSessionFrontier();
    const sourceA = 31;
    const sourceB = 47;
    const generationA = claim(frontier, sourceA);
    const connectA = frontier.beginConnect();
    assert(connectA?.playerSource === sourceA, "A registration must bind its connect lease to A");
    const preReadyB = frontier.beginReconnect(sourceB);
    assert(preReadyB.kind === "rejected" && preReadyB.ownerPlayerSource === sourceA, "B must be rejected while A is not ready");
    assert(!frontier.noteOwnerHeartbeat(sourceB), "B heartbeat must not refresh or retarget A's session");
    assert(!frontier.releaseOwner(sourceB), "dropping B must not release A's ownership");
    assert(frontier.completeConnect(connectA).kind === "safety-sync-required", "A connect should require sync");
    assert(frontier.completeSafetySync(generationA, sourceA), "only A acknowledgement should complete A's sync");
    assert(frontier.beginPoll()?.playerSource === sourceA, "dispatch must remain bound to A after B heartbeat");

    const stalePollA = frontier.beginPoll();
    assert(stalePollA !== undefined, "A should hold a poll lease before supersession");
    const rejectedB = frontier.beginReconnect(sourceB);
    assert(rejectedB.kind === "rejected" && rejectedB.ownerPlayerSource === sourceA, "B must be rejected while A owns the session");
    assert(frontier.acceptDispatchConfirmation(stalePollA, "a-command"), "rejected B registration must not invalidate A's confirmation");
    assert(frontier.releaseOwner(sourceA), "A drop should release ownership");
    assert(!frontier.acceptDispatchConfirmation(stalePollA, "stale-a-command"), "A drop must invalidate A's in-flight confirmation");
    const generationB = claim(frontier, sourceB);
    const connectB = frontier.beginConnect();
    assert(connectB?.generation === generationB && connectB.playerSource === sourceB, "B registration must bind the next reset to B");
    assert(frontier.completeConnect(connectB).kind === "safety-sync-required", "B connect should require sync");
    assert(!frontier.completeSafetySync(generationB, sourceA), "A must not acknowledge B's generation");
    assert(frontier.completeSafetySync(generationB, sourceB), "B acknowledgement should make B ready");
    assert(frontier.beginPoll()?.playerSource === sourceB, "post-supersession dispatch must target B");
}

function testOwnerDropInvalidatesPendingSync() {
    const frontier = new ControlSessionFrontier();
    const generationA = claim(frontier, 51);
    const connectA = frontier.beginConnect();
    assert(connectA !== undefined, "A connect should begin");
    assert(frontier.completeConnect(connectA).kind === "safety-sync-required", "A should reach pending sync");

    assert(frontier.releaseOwner(51), "A drop should release a pending owner");
    assert(!frontier.completeSafetySync(generationA, 51), "A acknowledgement after its drop must be invalid");
    assert(frontier.beginPoll() === undefined, "released owner must not poll");

    const generationB = claim(frontier, 52);
    const connectB = frontier.beginConnect();
    assert(connectB?.playerSource === 52, "B should claim after A drops");
    assert(frontier.completeConnect(connectB).kind === "safety-sync-required", "B reset should require sync");
    assert(frontier.completeSafetySync(generationB, 52), "B sync should complete");
    assert(frontier.beginPoll()?.playerSource === 52, "B should own polling after sync");
}

function testOwnerFreshnessPausesAndResumesPolling() {
    let now = 100;
    const frontier = new ControlSessionFrontier({
        ownerFreshnessTimeoutMs: 12,
        now: () => now,
    });
    const generation = claim(frontier, 61);
    const connect = frontier.beginConnect();
    assert(connect !== undefined, "owner connect should begin");
    assert(frontier.completeConnect(connect).kind === "safety-sync-required", "owner connect should require sync");
    assert(frontier.completeSafetySync(generation, 61), "owner sync should complete");
    const livePoll = frontier.beginPoll();
    assert(livePoll?.playerSource === 61, "fresh owner should poll");

    now += 13;
    assert(frontier.beginPoll() === undefined, "stale owner heartbeat must pause new polls");
    assert(!frontier.acceptDispatchConfirmation(livePoll, "late-command"), "a confirmation crossing the freshness timeout must be discarded");
    assert(!frontier.noteOwnerHeartbeat(62), "another source must not refresh the owner");
    assert(frontier.beginPoll() === undefined, "wrong-source heartbeat must leave polling paused");
    assert(frontier.noteOwnerHeartbeat(61), "owner heartbeat should refresh liveness");
    assert(frontier.beginPoll()?.playerSource === 61, "fresh owner heartbeat should resume polling");
}

function testFailedConfirmationDoesNotAdvanceCursor() {
    const frontier = new ControlSessionFrontier();
    const generation = claim(frontier, 71);
    const connect = frontier.beginConnect();
    assert(connect !== undefined, "connect should begin");
    assert(frontier.completeConnect(connect).kind === "safety-sync-required", "connect should require sync");
    assert(frontier.completeSafetySync(generation, 71), "sync should complete");

    const lease = frontier.beginPoll();
    assert(lease !== undefined, "ready session should poll");
    // Transport and payload validation failures return without accepting a dispatch decision.
    assert(frontier.beginPoll()?.lastSeenCommandId === "", "failed confirmation must leave the cursor unchanged");
    assert(frontier.acceptDispatchConfirmation(lease, "canceled-start"), "clean backend rejection should be accepted");
    assert(frontier.beginPoll()?.lastSeenCommandId === "canceled-start", "clean rejection must skip the canceled start");
}

function testBackendRestartForcesFreshSafetySync() {
    const frontier = new ControlSessionFrontier();
    const initialGeneration = claim(frontier, 72);
    const initialConnect = frontier.beginConnect();
    assert(initialConnect !== undefined, "initial backend connect should begin");
    assert(frontier.completeConnect(initialConnect).kind === "safety-sync-required", "initial connect should require sync");
    assert(frontier.completeSafetySync(initialGeneration, 72), "initial safety sync should complete");
    assert(frontier.beginPoll() !== undefined, "initial session should poll");

    assert(frontier.requestBackendReconnect(), "active owner should retain ownership across a backend restart");
    assert(!frontier.requestBackendReconnect(), "duplicate restart observations must not starve the reconnect");
    assert(frontier.beginPoll() === undefined, "backend restart must pause command polling immediately");
    const reconnect = frontier.beginConnect();
    assert(reconnect !== undefined && reconnect.generation !== initialGeneration, "backend restart should use a new generation");
    assert(frontier.completeConnect(reconnect).kind === "safety-sync-required", "backend restart should require a fresh safety sync");
    assert(frontier.completeSafetySync(reconnect.generation, 72), "fresh safety sync should restore polling");
    assert(frontier.beginPoll()?.lastSeenCommandId === "", "fresh backend session must reset the command cursor");
}

function testDispatchFailureDoesNotAdvanceCursor() {
    const frontier = new ControlSessionFrontier();
    const generation = claim(frontier, 81);
    const connect = frontier.beginConnect();
    assert(connect !== undefined, "connect should begin");
    assert(frontier.completeConnect(connect).kind === "safety-sync-required", "connect should require sync");
    assert(frontier.completeSafetySync(generation, 81), "sync should complete");

    const lease = frontier.beginPoll();
    assert(lease !== undefined, "ready session should poll");
    let dispatchAttempted = false;
    let dispatchFailed = false;
    try {
        frontier.dispatchConfirmedCommand(lease, "emergency-stop", () => {
            dispatchAttempted = true;
            throw new Error("emit failed");
        });
    } catch {
        dispatchFailed = true;
    }
    assert(dispatchAttempted && dispatchFailed, "dispatch error should reach the caller");
    assert(frontier.beginPoll()?.lastSeenCommandId === "", "failed emit must not advance past a confirmed emergency");

    const dispatched: string[] = [];
    assert(frontier.dispatchConfirmedCommand(lease, "emergency-stop", () => dispatched.push("emergency-stop")), "retry should dispatch");
    assert(dispatched.length === 1, "successful retry should emit exactly once");
    assert(frontier.beginPoll()?.lastSeenCommandId === "emergency-stop", "successful emit should advance the cursor");
}

function testSafetySyncAckRequiresExactIdentityAndSource() {
    const expected = {
        requestId: "sync-7",
        sessionId: "session-7",
        safetyEpoch: 19,
        playerSource: 42,
    };
    const ack = {
        requestId: expected.requestId,
        sessionId: expected.sessionId,
        safetyEpoch: expected.safetyEpoch,
    };
    assert(matchesControlSafetySyncAck(expected, 42, ack), "matching sync acknowledgement should be accepted");
    assert(!matchesControlSafetySyncAck(expected, 41, ack), "another player source must not acknowledge the sync");
    assert(!matchesControlSafetySyncAck(expected, 42, {...ack, requestId: "sync-old"}), "stale request id must be rejected");
    assert(!matchesControlSafetySyncAck(expected, 42, {...ack, sessionId: "session-old"}), "stale session id must be rejected");
    assert(!matchesControlSafetySyncAck(expected, 42, {...ack, safetyEpoch: 18}), "stale safety epoch must be rejected");
}

async function main() {
    await testDeferredDispatchConfirmationCannotCrossReconnectFrontier();
    testReconnectDuringConnectRequiresAnotherReset();
    testSingleOwnerClaimAndRelease();
    testOwnerDropInvalidatesPendingSync();
    testOwnerFreshnessPausesAndResumesPolling();
    testFailedConfirmationDoesNotAdvanceCursor();
    testBackendRestartForcesFreshSafetySync();
    testDispatchFailureDoesNotAdvanceCursor();
    testSafetySyncAckRequiresExactIdentityAndSource();
    console.log("control session frontier tests passed");
}

void main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
