import {
    ControlDispatchTimeoutError,
    ControlPollSingleFlight,
    dispatchConfirmationAdvancesCursor,
    parseControlDispatchConfirmation,
    runWithAbortTimeout,
} from "./control-dispatch";
import {ControlSessionFrontier} from "./control-session-frontier";

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function assertThrows(run: () => void, message: string) {
    let threw = false;
    try {
        run();
    } catch {
        threw = true;
    }
    assert(threw, message);
}

function testConfirmedResponseReturnsCanonicalCommand() {
    const confirmation = parseControlDispatchConfirmation({
        commandId: "cmd-1",
        confirmed: true,
        command: {
            id: "cmd-1",
            type: "startEgo",
            safetyEpoch: 7,
            seed: "canonical",
        },
    }, "cmd-1");

    assert(confirmation.confirmed, "valid confirmation should be accepted");
    assert(confirmation.command.seed === "canonical", "parser should preserve the canonical backend command");
    assert(dispatchConfirmationAdvancesCursor(confirmation), "confirmed command should advance the cursor");
}

function testStopSignCommandsAreCanonical() {
    for (const type of ["setStopSignTarget", "clearStopSignTarget", "setStopSignCatalogWaypoint", "probeStopSignTarget", "teleportStopSignStart", "startStopSignBatch"] as const) {
        const confirmation = parseControlDispatchConfirmation({
            commandId: `cmd-${type}`,
            confirmed: true,
            command: {id: `cmd-${type}`, type, safetyEpoch: 8},
        }, `cmd-${type}`);
        assert(confirmation.confirmed && confirmation.command.type === type, `${type} must be accepted`);
    }
}

function testCleanRejectionsAreExplicit() {
    for (const reason of ["missing", "canceled", "staleSafetyEpoch"] as const) {
        const confirmation = parseControlDispatchConfirmation({
            commandId: "cmd-2",
            confirmed: false,
            reason,
        }, "cmd-2");
        assert(!confirmation.confirmed && confirmation.reason === reason, `${reason} rejection should be accepted`);
        assert(
            dispatchConfirmationAdvancesCursor(confirmation) === (reason !== "missing"),
            `${reason} rejection returned the wrong cursor decision`,
        );
    }
}

function testMalformedResponsesFailClosed() {
    const malformed = [
        null,
        {},
        {commandId: "cmd-other", confirmed: false, reason: "canceled"},
        {commandId: "cmd-3", confirmed: "false", reason: "canceled"},
        {commandId: "cmd-3", confirmed: false, reason: "unknown"},
        {commandId: "cmd-3", confirmed: false, reason: "canceled", command: {}},
        {commandId: "cmd-3", confirmed: true},
        {commandId: "cmd-3", confirmed: true, reason: "canceled", command: {id: "cmd-3", type: "startEgo", safetyEpoch: 1}},
        {commandId: "cmd-3", confirmed: true, command: {id: "cmd-other", type: "startEgo", safetyEpoch: 1}},
        {commandId: "cmd-3", confirmed: true, command: {id: "cmd-3", type: "", safetyEpoch: 1}},
        {commandId: "cmd-3", confirmed: true, command: {id: "cmd-3", type: "launch", safetyEpoch: 1}},
        {commandId: "cmd-3", confirmed: true, command: {id: "cmd-3", type: "startEgo", safetyEpoch: "1"}},
    ];
    for (const value of malformed) {
        assertThrows(
            () => parseControlDispatchConfirmation(value, "cmd-3"),
            `malformed confirmation must fail closed: ${JSON.stringify(value)}`,
        );
    }
}

async function testTimeoutReleasesPollAndPreservesCursorForHold() {
    const frontier = new ControlSessionFrontier();
    const claim = frontier.beginReconnect(91);
    assert(claim.kind === "accepted", "test control owner should be accepted");
    const connect = frontier.beginConnect();
    assert(connect !== undefined, "test control session should begin connecting");
    assert(frontier.completeConnect(connect).kind === "safety-sync-required", "test connect should require safety sync");
    assert(frontier.completeSafetySync(claim.generation, 91), "test control session should become ready");

    const singleFlight = new ControlPollSingleFlight();
    let timedRequestAborted = false;
    const timedOutPoll = singleFlight.run(async () => {
        const lease = frontier.beginPoll();
        assert(lease?.lastSeenCommandId === "", "initial poll should start at the empty cursor");
        await runWithAbortTimeout("control poll", 10, (signal) => new Promise<never>((_resolve, reject) => {
            signal.addEventListener("abort", () => {
                timedRequestAborted = true;
                reject(new Error("request aborted"));
            }, {once: true});
        }));
        frontier.acceptDispatchConfirmation(lease, "start-before-hold");
    });

    const overlappingPollRan = await singleFlight.run(async () => {
        throw new Error("overlapping poll must not run");
    });
    assert(!overlappingPollRan, "the singleton must reject an overlapping poll");

    let timeoutObserved = false;
    try {
        await timedOutPoll;
    } catch (error) {
        timeoutObserved = error instanceof ControlDispatchTimeoutError;
    }
    assert(timeoutObserved, "a half-open control request must fail with the bounded timeout");
    assert(timedRequestAborted, "timing out must abort the underlying HTTP request");
    assert(frontier.beginPoll()?.lastSeenCommandId === "", "a timeout must preserve the prior command cursor");

    const dispatched: string[] = [];
    const recoveryPollRan = await singleFlight.run(async () => {
        const lease = frontier.beginPoll();
        assert(lease?.lastSeenCommandId === "", "the recovery poll must retry from the prior cursor");
        const hold = "hold-after-timeout";
        assert(
            frontier.dispatchConfirmedCommand(lease, hold, () => dispatched.push(hold)),
            "the next poll must be able to dispatch the queued Hold",
        );
    });
    assert(recoveryPollRan, "the timeout must release the singleton for the next poll");
    assert(dispatched.length === 1 && dispatched[0] === "hold-after-timeout", "the recovery poll must emit Hold");
    assert(frontier.beginPoll()?.lastSeenCommandId === "hold-after-timeout", "successful Hold dispatch must advance the cursor");
}

async function testTimedOutConnectCanBeRetriedByHeartbeat() {
    const frontier = new ControlSessionFrontier();
    const claim = frontier.beginReconnect(92);
    assert(claim.kind === "accepted", "test reconnect owner should be accepted");
    const timedOutLease = frontier.beginConnect();
    assert(timedOutLease !== undefined, "the reconnect should acquire a connect lease");

    let timeoutObserved = false;
    try {
        await runWithAbortTimeout("control reconnect POST /control/connect", 10, () => new Promise<never>(() => undefined));
    } catch (error) {
        timeoutObserved = error instanceof ControlDispatchTimeoutError;
        frontier.failConnect(timedOutLease);
    }

    assert(timeoutObserved, "a half-open control connect must fail within its bounded timeout");
    const retryLease = frontier.beginConnect();
    assert(retryLease?.generation === claim.generation, "the next owner heartbeat must be able to retry the same generation");
    assert(frontier.completeConnect(retryLease).kind === "safety-sync-required", "the retry must resume at safety synchronization");
    assert(frontier.completeSafetySync(claim.generation, 92), "the retried connect must be able to restore polling");
    assert(frontier.beginPoll()?.playerSource === 92, "Hold polling must resume after the retried connect");
}

async function main() {
    testConfirmedResponseReturnsCanonicalCommand();
    testStopSignCommandsAreCanonical();
    testCleanRejectionsAreExplicit();
    testMalformedResponsesFailClosed();
    await testTimeoutReleasesPollAndPreservesCursorForHold();
    await testTimedOutConnectCanBeRetriedByHeartbeat();
    console.log("control dispatch tests passed");
}

void main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
