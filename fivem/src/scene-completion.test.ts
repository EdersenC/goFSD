import {completeSuccessfulScene} from "./scene-completion";
import {requireManagedEgoDisposed} from "./managed-ego-disposal";

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function testSuccessfulCompletionSynchronouslyHoldsAndDisposesManagedEgo() {
    const reasons: string[] = [];

    completeSuccessfulScene(" inner-city-driving:default ", {
        forceSafeStopAndDisposeCurrentEgo: (reason) => reasons.push(reason),
    });

    assert(reasons.length === 1, "successful completion must dispose exactly one managed ego");
    assert(
        reasons[0] === 'successful scene "inner-city-driving:default" completed',
        "successful completion must identify the completed scene",
    );
}

function testCleanupFailurePreventsSuccessfulCompletion() {
    const cleanupFailure = new Error("managed ego could not be disposed");
    let returned = false;

    try {
        completeSuccessfulScene("inner-city-driving:default", {
            forceSafeStopAndDisposeCurrentEgo: () => {
                throw cleanupFailure;
            },
        });
        returned = true;
    } catch (error) {
        assert(error === cleanupFailure, "completion must preserve the fail-safe cleanup error");
    }

    assert(!returned, "a scene must not complete when its managed ego remains live");
}

function testEmptySceneNameViolatesCompletionInvariant() {
    let cleanupCalled = false;
    let rejected = false;

    try {
        completeSuccessfulScene("  ", {
            forceSafeStopAndDisposeCurrentEgo: () => {
                cleanupCalled = true;
            },
        });
    } catch (error) {
        rejected = error instanceof Error && error.message.includes("scene name is empty");
    }

    assert(rejected, "successful completion must reject an empty scene name");
    assert(!cleanupCalled, "an invalid completion must not mutate managed ego state");
}

function testFailedDisposalRemainsTrackedAcrossRepeatedHolds() {
    const managedEgo = {vehicle: {id: 731}};
    let trackedEgo: typeof managedEgo | null = managedEgo;
    let vehicleExists = true;
    let failedAttempts = 0;

    const attemptVerifiedDisposal = () => {
        const candidate = trackedEgo;
        trackedEgo = null;
        try {
            requireManagedEgoDisposed(
                candidate,
                (vehicle) => vehicle === managedEgo.vehicle.id && vehicleExists,
                (failedEgo) => {
                    trackedEgo = failedEgo;
                },
            );
        } catch (error) {
            failedAttempts += 1;
            assert(
                error instanceof Error && error.message.includes(String(managedEgo.vehicle.id)),
                "verified deletion failure must identify the retained vehicle",
            );
        }
    };

    attemptVerifiedDisposal();
    assert(trackedEgo === managedEgo, "the first failed Hold must restore the managed ego handle");
    attemptVerifiedDisposal();
    assert(failedAttempts === 2, "a repeated Hold must retry and fail closed on the same live vehicle");
    assert(trackedEgo === managedEgo, "a repeated failed Hold must still retain the managed ego handle");

    vehicleExists = false;
    attemptVerifiedDisposal();
    assert(trackedEgo === null, "the managed ego may clear only after entity deletion is verified");
}

function main() {
    testSuccessfulCompletionSynchronouslyHoldsAndDisposesManagedEgo();
    testCleanupFailurePreventsSuccessfulCompletion();
    testEmptySceneNameViolatesCompletionInvariant();
    testFailedDisposalRemainsTrackedAcrossRepeatedHolds();
    console.log("scene completion tests passed");
}

main();
