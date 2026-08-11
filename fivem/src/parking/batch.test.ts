import {executeParkingBatch, ParkingBatchCommandJob} from "./batch";
import {
    resolveParkingOperationCompletionStatus,
    resolveStopCommandStatus,
} from "./control-status";

const jobs: ParkingBatchCommandJob[] = [
    {
        id: "bay-1:base",
        parkDest: {x: 0, y: 0, z: 0, heading: 0},
        startDest: {x: 0, y: -11, z: 0, heading: 0},
        collectionAmount: 3,
        seed: "fresh:bay-1:base",
    },
    {
        id: "bay-2:base",
        parkDest: {x: 10, y: 10, z: 0, heading: 90},
        startDest: {x: 21, y: 10, z: 0, heading: 90},
        collectionAmount: 3,
        seed: "fresh:bay-2:base",
    },
];

async function main() {
    await testSequentialCallbacks();
    await testCooperativeStopDoesNotCountPartialJob();
    testParkingControlStopStatus();
    console.log("parking batch execution tests passed");
}

async function testSequentialCallbacks() {
    const events: string[] = [];
    let runActive = false;
    const completed = await executeParkingBatch(jobs, {
        setParkingTarget: () => events.push("target"),
        setParkingStart: () => events.push("start"),
        startParkingRun: async (_amount, seed) => {
            assert(!runActive, "parking jobs must never overlap");
            runActive = true;
            events.push(`run:${seed}`);
            await Promise.resolve();
            runActive = false;
        },
        stopRequested: () => false,
        onJobStart: (job, index, count) => events.push(`begin:${index}/${count}:${job.id}`),
        onJobComplete: (job, index, count) => events.push(`complete:${index}/${count}:${job.id}`),
    });
    assert(completed === 2, "both sequential jobs should complete");
    assert(events.join("|") === [
        "begin:1/2:bay-1:base", "target", "start", "run:fresh:bay-1:base", "complete:1/2:bay-1:base",
        "begin:2/2:bay-2:base", "target", "start", "run:fresh:bay-2:base", "complete:2/2:bay-2:base",
    ].join("|"), `unexpected temporal event order: ${events.join("|")}`);
    console.log("ok - parking batch executes exactly one ordered job at a time");
}

async function testCooperativeStopDoesNotCountPartialJob() {
    let stopped = false;
    let completedCallbackCount = 0;
    let runCount = 0;
    const completed = await executeParkingBatch(jobs, {
        setParkingTarget: () => undefined,
        setParkingStart: () => undefined,
        startParkingRun: async () => {
            runCount += 1;
            stopped = true;
        },
        stopRequested: () => stopped,
        onJobComplete: () => {
            completedCallbackCount += 1;
        },
    });
    assert(runCount === 1, "stop must prevent later jobs from starting");
    assert(completed === 0, "a stopped partial job must not count as complete");
    assert(completedCallbackCount === 0, "a stopped partial job must not report completion");
    console.log("ok - cooperative stop excludes the partial job and blocks later jobs");
}

function testParkingControlStopStatus() {
    assert(
        resolveStopCommandStatus(true, false) === "idle",
        "a synchronous ego stop with no parking work can report idle"
    );
    assert(
        resolveStopCommandStatus(true, true) === "stopping",
        "an active parking operation must remain stopping even when the immediate scene stop is synchronous"
    );
    assert(
        resolveStopCommandStatus(false, false) === "stopping",
        "an asynchronous scene stop must not report idle"
    );
    assert(
        resolveParkingOperationCompletionStatus(true, false) === "stopping",
        "parking completion must remain stopping until the final vehicle hold succeeds"
    );
    assert(
        resolveParkingOperationCompletionStatus(true, true) === "idle",
        "a stopped parking operation can report idle after its final vehicle hold"
    );
    assert(
        resolveParkingOperationCompletionStatus(false, false) === "runningScene",
        "a normally completed parking run returns to manual ego control"
    );
    console.log("ok - parking stop status waits for asynchronous completion and final hold");
}

function assert(condition: boolean, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

void main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
