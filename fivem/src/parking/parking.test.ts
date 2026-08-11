import {
    buildForwardBayCurriculum,
    buildForwardBayEvaluationPlan,
    DEFAULT_PARKING_EVALUATION_SEED,
    resolveParkingAttemptCount,
    resolveStraightParkingStartOffset,
} from "./curriculum";
import {
    gtaForwardVector,
    gtaHeadingFromVector,
    gtaRightVector,
    headingDeltaDegrees,
    isVehicleFootprintInsideBay,
    poseFromLocalOffset,
    relativePose,
} from "./geometry";
import {
    MAX_STRAIGHT_APPROACH_HEADING_ERROR_DEG,
    MAX_STRAIGHT_APPROACH_LATERAL_ERROR_M,
    MAX_STRAIGHT_TRACKING_HEADING_ERROR_DEG,
    MAX_STRAIGHT_TRACKING_LATERAL_ERROR_M,
    ParkingEvaluationOutcomeTracker,
    ParkingHazardLatch,
    ParkingObservation,
    ParkingOutcomeTracker,
    terminalParkingPhase,
} from "./outcome";
import {parkingMarkerCorners, PARKING_TARGET_MARKER_ID} from "./marker";
import {buildParkingFixturePlans} from "./fixtures";
import {
    planStraightStopExpert,
    STRAIGHT_STOP_ACCELERATION_MPS2,
    STRAIGHT_STOP_BRAKING_MPS2,
    STRAIGHT_STOP_EXPERT_INTERVAL_MS,
    STRAIGHT_STOP_HOLD_DISTANCE_M,
    STRAIGHT_STOP_HOLD_SPEED_MPS,
    STRAIGHT_STOP_MAX_DT_SECONDS,
    STRAIGHT_STOP_MAX_SPEED_MPS,
} from "./expert";
import {ParkingGoal, ParkingPose, VehicleFootprint} from "./types";
import {
    CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS,
    createNoEgoControlTelemetry,
} from "../controlTelemetry";
import {
    PARKING_DISABLED_CONTROL_IDS,
    PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS,
    PARKING_EVALUATION_SPAWN_SETTLE_MS,
    PARKING_POST_FLASH_CLEAN_FRAME_BUFFER_MS,
} from "./controls";
import {isActuatorStateNeutral, isInferenceCleanupComplete} from "../expertNeutral";
import {RECONNECT_CAPTURE_STOP_REQUEST} from "../captureLifecycle";
import {validateParkingBatchJobs} from "./batch";
import {
    PARKING_ISOLATION_RADIUS_M,
    PARKING_ISOLATION_ZONE_REFRESH_DISTANCE_M,
    parkingIsolationBounds,
    shouldRefreshParkingIsolationZone,
} from "./isolation";
import {
    normalizeVehicleSteering,
    resolveSteeringFullLockRadians,
    SULTAN_STEERING_FULL_LOCK_RADIANS,
} from "../vehicleSteering";

const epsilon = 1e-9;

test("parking isolation creates a 1000m native suppression box around its moving center", () => {
    const bounds = parkingIsolationBounds([120, -45, 8]);
    assert(PARKING_ISOLATION_RADIUS_M === 1000, "parking isolation radius must remain 1000m");
    assert(bounds.min[0] === -880, "minimum x should follow the isolation center");
    assert(bounds.min[1] === -1045, "minimum y should follow the isolation center");
    assert(bounds.min[2] === -992, "minimum z should follow the isolation center");
    assert(bounds.max[0] === 1120, "maximum x should follow the isolation center");
    assert(bounds.max[1] === 955, "maximum y should follow the isolation center");
    assert(bounds.max[2] === 1008, "maximum z should follow the isolation center");
});

test("persistent parking isolation refreshes its native zone only after meaningful movement", () => {
    assert(
        shouldRefreshParkingIsolationZone(null, [0, 0, 0]),
        "an uninitialized native zone must be created"
    );
    assert(
        !shouldRefreshParkingIsolationZone([0, 0, 0], [99, 0, 0]),
        "small movement must retain the existing native zone"
    );
    assert(
        shouldRefreshParkingIsolationZone(
            [0, 0, 0],
            [PARKING_ISOLATION_ZONE_REFRESH_DISTANCE_M, 0, 0]
        ),
        "the native zone must follow the player at the refresh threshold"
    );
});

test("client reconnect aborts instead of processing an unfinished capture", () => {
    assert(RECONNECT_CAPTURE_STOP_REQUEST.abortOnly === true, "reconnect cleanup must be abort-only");
});

test("parking batch jobs validate exact poses before execution", () => {
    const jobs = validateParkingBatchJobs([
        {
            id: "bay-a:base",
            parkDest: {x: 10, y: 20, z: 3, heading: 0},
            startDest: {x: 10, y: 9, z: 3, heading: 0},
            collectionAmount: 8,
            seed: "fresh-1:bay-a:base",
        },
    ]);
    assert(jobs.length === 1, "expected one validated job");
    assert(jobs[0]!.parkDest.coords[1] === 20, "expected exact target pose conversion");
    assert(jobs[0]!.startDest.coords[1] === 9, "expected exact start pose conversion");
    assert(jobs[0]!.collectionAmount === 8, "expected collection amount to survive validation");
});

test("parking batch validation fails before accepting an unsafe start", () => {
    assertThrows(
        () => validateParkingBatchJobs([
            {
                id: "bay-a:unsafe",
                parkDest: {x: 10, y: 20, z: 3, heading: 0},
                startDest: {x: 11, y: 16, z: 3, heading: 0},
                collectionAmount: 8,
                seed: "fresh-1:bay-a:unsafe",
            },
        ]),
        "9.5-15.0m behind",
    );
});

function test(name: string, body: () => void) {
    try {
        body();
        console.log(`ok - ${name}`);
    } catch (error) {
        console.error(`not ok - ${name}`);
        throw error;
    }
}

function assert(condition: boolean, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function assertApproximately(actual: number, expected: number, message: string) {
    assert(
        Math.abs(actual - expected) <= epsilon,
        `${message}: expected ${expected}, got ${actual}`
    );
}

function assertThrows(body: () => void, expectedMessage: string) {
    try {
        body();
    } catch (error) {
        assert(error instanceof Error, "expected an Error instance");
        assert(error.message.includes(expectedMessage), `unexpected error: ${error.message}`);
        return;
    }
    throw new Error("expected function to throw");
}

const target: ParkingPose = {coords: [120, -45, 8], heading: 90};
const footprint: VehicleFootprint = {halfWidthM: 0.9, halfLengthM: 2.2};

function buildGoal(): ParkingGoal {
    return {
        task: "parking",
        maneuver: "forward-bay",
        visualCueId: PARKING_TARGET_MARKER_ID,
        target: {coords: [0, 0, 0], heading: 0},
        bay: {widthM: 3.4, lengthM: 6.5},
        tolerances: {
            maxCenterDistanceM: 0.65,
            maxHeadingErrorDeg: 5,
            maxSpeedMps: 0.15,
            settledDurationMs: 1000,
        },
        attemptIndex: 0,
        attemptCount: 1,
        seed: "parking-test",
        startPose: {coords: [0, -10, 0], heading: 0},
        startOffset: {longitudinalM: -10, lateralM: 0, headingDeg: 0},
    };
}

function safeObservation(
    value: Pick<ParkingObservation, "nowMs" | "pose" | "speedMps"> & Partial<ParkingObservation>
): ParkingObservation {
    return {
        collision: false,
        reversing: false,
        onGround: true,
        pitchDeg: 0,
        rollDeg: 0,
        stopRequested: false,
        ...value,
    };
}

test("GTA cardinal headings map to native XY axes", () => {
    const north = gtaForwardVector(0);
    const west = gtaForwardVector(90);
    const south = gtaForwardVector(180);
    const east = gtaForwardVector(270);
    assertApproximately(north[0], 0, "heading 0 x");
    assertApproximately(north[1], 1, "heading 0 y");
    assertApproximately(west[0], -1, "heading 90 x");
    assertApproximately(west[1], 0, "heading 90 y");
    assertApproximately(south[1], -1, "heading 180 y");
    assertApproximately(east[0], 1, "heading 270 x");
    assertApproximately(gtaRightVector(0)[0], 1, "heading 0 right x");
});

test("heading/vector conversions round trip", () => {
    for (const heading of [0, 30, 90, 179, 270, 359]) {
        const roundTripped = gtaHeadingFromVector(gtaForwardVector(heading));
        assertApproximately(headingDeltaDegrees(roundTripped, heading), 0, `heading ${heading}`);
    }
});

test("negative longitudinal offsets are behind every cardinal target heading", () => {
    const origin: ParkingPose = {coords: [0, 0, 0], heading: 0};
    const expectedStarts: Array<[number, number, number]> = [
        [0, -10, 0],
        [10, 0, 0],
        [0, 10, 0],
        [-10, 0, 0],
    ];
    for (const [index, heading] of [0, 90, 180, 270].entries()) {
        const start = poseFromLocalOffset({...origin, heading}, -10, 0);
        const expected = expectedStarts[index];
        assert(expected !== undefined, `missing expected cardinal start ${index}`);
        assertApproximately(start.coords[0], expected[0], `heading ${heading} behind x`);
        assertApproximately(start.coords[1], expected[1], `heading ${heading} behind y`);
    }
});

test("local offsets and relative poses are inverse transformations", () => {
    for (const heading of [0, 90, 217]) {
        const pose = poseFromLocalOffset({...target, heading}, -11.25, 1.75, -7);
        const relative = relativePose(pose, {...target, heading});
        assertApproximately(relative.longitudinalM, -11.25, `heading ${heading} longitudinal`);
        assertApproximately(relative.lateralM, 1.75, `heading ${heading} lateral`);
        assertApproximately(relative.headingDeg, 7, `heading ${heading} heading delta`);
    }
});

test("vehicle footprint containment honors target rotation and bay edges", () => {
    const bay = {widthM: 4, lengthM: 6};
    assert(isVehicleFootprintInsideBay(target, target, bay, footprint), "centered vehicle should fit");
    const inside = poseFromLocalOffset(target, 0.79, 0.09);
    assert(isVehicleFootprintInsideBay(inside, target, bay, footprint), "near-edge vehicle should fit");
    const outsideLength = poseFromLocalOffset(target, 0.81, 0);
    assert(!isVehicleFootprintInsideBay(outsideLength, target, bay, footprint), "length edge should fail");
    const outsideWidth = poseFromLocalOffset(target, 0, 1.11);
    assert(!isVehicleFootprintInsideBay(outsideWidth, target, bay, footprint), "width edge should fail");
});

test("parking fixtures occupy the two adjacent bays deterministically", () => {
    const parkingTarget = {
        pose: target,
        bay: {widthM: 3.4, lengthM: 6.5},
        tolerances: buildGoal().tolerances,
    };
    const fixtures = buildParkingFixturePlans(parkingTarget);
    assert(fixtures.length === 2, "expected one fixture on each side of the target bay");

    const left = relativePose(fixtures[0]!.pose, target);
    const right = relativePose(fixtures[1]!.pose, target);
    assertApproximately(left.longitudinalM, 0, "left fixture longitudinal position");
    assertApproximately(left.lateralM, -3.4, "left fixture lateral position");
    assertApproximately(right.longitudinalM, 0, "right fixture longitudinal position");
    assertApproximately(right.lateralM, 3.4, "right fixture lateral position");
    assertApproximately(left.headingDeg, 0, "left fixture heading");
    assertApproximately(right.headingDeg, 0, "right fixture heading");
});

test("straight-stop expert accelerates at a bounded 20 Hz rate", () => {
    assert(STRAIGHT_STOP_EXPERT_INTERVAL_MS === 50, "expert cadence must match parking telemetry");
    const output = planStraightStopExpert({
        remainingDistanceM: 10,
        measuredSpeedMps: 0,
        previousDesiredSpeedMps: 0,
        dtSeconds: STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    assertApproximately(
        output.desiredSpeedMps,
        STRAIGHT_STOP_ACCELERATION_MPS2 * (STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000),
        "first expert step"
    );
    assert(!output.stopRequested, "far-away acceleration must not request a stop");

    const capped = planStraightStopExpert({
        remainingDistanceM: 100,
        measuredSpeedMps: STRAIGHT_STOP_MAX_SPEED_MPS,
        previousDesiredSpeedMps: STRAIGHT_STOP_MAX_SPEED_MPS,
        dtSeconds: STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    assertApproximately(capped.desiredSpeedMps, STRAIGHT_STOP_MAX_SPEED_MPS, "maximum speed cap");
});

test("straight-stop expert steers back toward the saved goal pose", () => {
    const baseInput = {
        remainingDistanceM: 10,
        measuredSpeedMps: 1,
        previousDesiredSpeedMps: 1,
        dtSeconds: STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000,
    };
    const rightOfPath = planStraightStopExpert({
        ...baseInput,
        lateralErrorM: 0.24,
        headingErrorDeg: 2,
    });
    assert(
        rightOfPath.desiredWheelSteerNormalized > 0,
        "a vehicle right of the goal path and heading left of target must steer left"
    );

    const leftOfPath = planStraightStopExpert({
        ...baseInput,
        lateralErrorM: -0.24,
        headingErrorDeg: -2,
    });
    assert(
        leftOfPath.desiredWheelSteerNormalized < 0,
        "a vehicle left of the goal path and heading right of target must steer right"
    );

    const aligned = planStraightStopExpert({
        ...baseInput,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    assertApproximately(aligned.desiredWheelSteerNormalized, 0, "aligned path steering");
});

test("straight-stop expert follows the guarded braking-distance profile", () => {
    const output = planStraightStopExpert({
        remainingDistanceM: 1,
        measuredSpeedMps: 2,
        previousDesiredSpeedMps: 2,
        dtSeconds: STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    const expected = Math.sqrt(2 * STRAIGHT_STOP_BRAKING_MPS2 * 0.75);
    assertApproximately(output.desiredSpeedMps, expected, "guarded braking speed");
    assert(!output.stopRequested, "braking before the hold zone must remain active");
});

test("straight-stop expert requests a zero-speed hold only near the target", () => {
    const holding = planStraightStopExpert({
        remainingDistanceM: STRAIGHT_STOP_HOLD_DISTANCE_M,
        measuredSpeedMps: STRAIGHT_STOP_HOLD_SPEED_MPS,
        previousDesiredSpeedMps: STRAIGHT_STOP_HOLD_SPEED_MPS,
        dtSeconds: STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    assert(holding.stopRequested, "slow vehicle in the hold zone must stop");
    assertApproximately(holding.desiredSpeedMps, 0, "hold speed");

    const stillMoving = planStraightStopExpert({
        remainingDistanceM: STRAIGHT_STOP_HOLD_DISTANCE_M,
        measuredSpeedMps: STRAIGHT_STOP_HOLD_SPEED_MPS + 0.001,
        previousDesiredSpeedMps: STRAIGHT_STOP_HOLD_SPEED_MPS,
        dtSeconds: STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    assert(!stillMoving.stopRequested, "hold must wait for measured speed to become safe");
});

test("straight-stop expert reaches the hold zone smoothly from every allowed start", () => {
    const dtSeconds = STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000;
    for (const startDistanceM of [9.5, 12, 15]) {
        let remainingDistanceM = startDistanceM;
        let measuredSpeedMps = 0;
        let previousDesiredSpeedMps = 0;
        let stopped = false;

        for (let step = 0; step < 500; step += 1) {
            const output = planStraightStopExpert({
                remainingDistanceM,
                measuredSpeedMps,
                previousDesiredSpeedMps,
                dtSeconds,
                lateralErrorM: 0,
                headingErrorDeg: 0,
            });
            assert(output.desiredSpeedMps >= 0, "expert speed must never reverse");
            assert(
                output.desiredSpeedMps <= STRAIGHT_STOP_MAX_SPEED_MPS,
                "expert speed must respect its maximum"
            );
            if (output.stopRequested) {
                assert(
                    remainingDistanceM <= STRAIGHT_STOP_HOLD_DISTANCE_M,
                    `start ${startDistanceM}m must finish inside the hold zone`
                );
                assertApproximately(output.desiredSpeedMps, 0, "terminal desired speed");
                stopped = true;
                break;
            }
            const speedIncreaseMps = output.desiredSpeedMps - previousDesiredSpeedMps;
            assert(
                speedIncreaseMps <= (STRAIGHT_STOP_ACCELERATION_MPS2 * dtSeconds) + epsilon,
                "expert acceleration must remain bounded"
            );
            remainingDistanceM = Math.max(0, remainingDistanceM - (output.desiredSpeedMps * dtSeconds));
            measuredSpeedMps = output.desiredSpeedMps;
            previousDesiredSpeedMps = output.desiredSpeedMps;
        }
        assert(stopped, `expert must stop from ${startDistanceM}m`);
    }
});

test("straight-stop expert rejects invalid state and controller intervals", () => {
    const valid = {
        remainingDistanceM: 10,
        measuredSpeedMps: 0,
        previousDesiredSpeedMps: 0,
        dtSeconds: STRAIGHT_STOP_EXPERT_INTERVAL_MS / 1000,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    };
    assertThrows(
        () => planStraightStopExpert({...valid, remainingDistanceM: Number.NaN}),
        "remainingDistanceM must be finite"
    );
    assertThrows(
        () => planStraightStopExpert({...valid, measuredSpeedMps: Number.POSITIVE_INFINITY}),
        "measuredSpeedMps must be finite"
    );
    assertThrows(
        () => planStraightStopExpert({...valid, previousDesiredSpeedMps: -0.01}),
        "previousDesiredSpeedMps must be finite"
    );
    assertThrows(
        () => planStraightStopExpert({...valid, lateralErrorM: Number.NaN}),
        "lateralErrorM must be finite"
    );
    assertThrows(
        () => planStraightStopExpert({...valid, headingErrorDeg: Number.POSITIVE_INFINITY}),
        "headingErrorDeg must be finite"
    );
    assertThrows(
        () => planStraightStopExpert({
            ...valid,
            previousDesiredSpeedMps: STRAIGHT_STOP_MAX_SPEED_MPS + 0.001,
        }),
        "previousDesiredSpeedMps must not exceed"
    );
    for (const dtSeconds of [0, -0.01, STRAIGHT_STOP_MAX_DT_SECONDS + 0.001, Number.NaN]) {
        assertThrows(
            () => planStraightStopExpert({...valid, dtSeconds}),
            "dtSeconds must be finite"
        );
    }
});

test("RGB target marker covers the calibrated parking bay", () => {
    const markerTarget: ParkingPose = {coords: [0, 0, 10], heading: 0};
    const corners = parkingMarkerCorners(markerTarget, {widthM: 4, lengthM: 6});
    const expected: Array<readonly [number, number, number]> = [
        [-2, -3, 10.04],
        [2, -3, 10.04],
        [2, 3, 10.04],
        [-2, 3, 10.04],
    ];
    assert(PARKING_TARGET_MARKER_ID === "orange-bay-highlight-v1", "marker identity must be stable");
    corners.forEach((corner, index) => {
        const expectedCorner = expected[index];
        assert(expectedCorner !== undefined, `missing expected marker corner ${index}`);
        assertApproximately(corner[0], expectedCorner[0], `marker corner ${index} x`);
        assertApproximately(corner[1], expectedCorner[1], `marker corner ${index} y`);
        assertApproximately(corner[2], expectedCorner[2], `marker corner ${index} z`);
    });
});

test("phase-one curriculum repeats the exact user-calibrated start", () => {
    const calibratedStart = poseFromLocalOffset(target, -12, 0, 0);
    const first = buildForwardBayCurriculum(target, calibratedStart, 50);
    const second = buildForwardBayCurriculum(target, calibratedStart, 50);
    assert(JSON.stringify(first) === JSON.stringify(second), "calibrated start should reproduce exactly");
    assert(first.length === 50, "curriculum count should match request");
    assert(first.every((plan, index) => plan.attemptIndex === index), "attempt indices should be zero based");
    for (const plan of first) {
        const relative = relativePose(plan.startPose, target);
        assertApproximately(plan.startOffset.longitudinalM, -12, "saved longitudinal start must not vary");
        assertApproximately(plan.startOffset.lateralM, 0, "phase-one start must be centered on the parking bay");
        assertApproximately(plan.startOffset.headingDeg, 0, "phase-one start must be aligned with the parking bay");
        assert(JSON.stringify(plan.startPose) === JSON.stringify(calibratedStart), "saved pose must be cloned exactly");
        assertApproximately(
            relative.longitudinalM,
            plan.startOffset.longitudinalM,
            `attempt ${plan.attemptIndex} longitudinal offset`
        );
        assertApproximately(relative.lateralM, 0, `attempt ${plan.attemptIndex} lateral offset`);
        assertApproximately(relative.headingDeg, 0, `attempt ${plan.attemptIndex} heading offset`);
    }
    assertThrows(() => resolveParkingAttemptCount(0), "1 through 50");
    assertThrows(() => resolveParkingAttemptCount(51), "1 through 50");
    assertThrows(() => resolveParkingAttemptCount(1.5), "1 through 50");
});

test("straight start validation rejects cheating geometry and preserves heading sign", () => {
    const valid = poseFromLocalOffset(target, -9.5, 0.34, 3);
    const offset = resolveStraightParkingStartOffset(valid, target);
    assertApproximately(offset.longitudinalM, -9.5, "valid start longitudinal offset");
    assertApproximately(offset.lateralM, 0.34, "valid start lateral offset");
    assertApproximately(offset.headingDeg, 3, "valid start heading offset");
    assertThrows(
        () => resolveStraightParkingStartOffset(poseFromLocalOffset(target, -9.49, 0, 0), target),
        "9.5-15.0m behind"
    );
    assertThrows(
        () => resolveStraightParkingStartOffset(poseFromLocalOffset(target, -15.01, 0, 0), target),
        "9.5-15.0m behind"
    );
    assertThrows(
        () => resolveStraightParkingStartOffset(poseFromLocalOffset(target, -12, 0.351, 0), target),
        "centered within 0.35m"
    );
    assertThrows(
        () => resolveStraightParkingStartOffset(poseFromLocalOffset(target, -12, 0, 3.01), target),
        "within 3 degrees"
    );
    assertThrows(
        () => resolveStraightParkingStartOffset({coords: [Number.NaN, 0, 0], heading: 0}, target),
        "finite coordinates"
    );
});

test("parking evaluation uses the exact calibrated start", () => {
    const calibratedStart = poseFromLocalOffset(target, -11, 0, 0);
    const first = buildForwardBayEvaluationPlan(target, calibratedStart, "evaluation-seed");
    const repeated = buildForwardBayEvaluationPlan(target, calibratedStart, "evaluation-seed");
    assert(JSON.stringify(first) === JSON.stringify(repeated), "evaluation plan must reproduce by seed");
    assert(first.plan.attemptIndex === 0, "evaluation must use the zero-based first attempt");
    assert(JSON.stringify(first.plan.startPose) === JSON.stringify(calibratedStart), "evaluation must reuse saved start");
    assertApproximately(first.plan.startOffset.lateralM, 0, "evaluation start must be centered on the parking bay");
    assertApproximately(first.plan.startOffset.headingDeg, 0, "evaluation start must be aligned with the parking bay");

    const defaulted = buildForwardBayEvaluationPlan(target, calibratedStart, "  ");
    assert(defaulted.seed === DEFAULT_PARKING_EVALUATION_SEED, "empty seed must use stable evaluation default");
});

test("parking succeeds only after continuously settled for the tolerance", () => {
    const tracker = new ParkingOutcomeTracker(buildGoal(), footprint, 100);
    const stablePose: ParkingPose = {coords: [0, 0, 0], heading: 0};
    const first = tracker.observe(safeObservation({
        nowMs: 100,
        pose: stablePose,
        speedMps: 0,
    }));
    assert(first.outcome === null && !first.state.parked, "first stable sample must not succeed");
    const early = tracker.observe(safeObservation({
        nowMs: 1099,
        pose: stablePose,
        speedMps: 0,
    }));
    assert(early.outcome === null && !early.state.parked, "999 ms must not succeed");
    const complete = tracker.observe(safeObservation({
        nowMs: 1100,
        pose: stablePose,
        speedMps: 0,
    }));
    assert(complete.outcome?.status === "succeeded", "1000 ms should succeed");
    assert(complete.outcome.success, "success outcome flag should be true");
});

test("parking evaluation waits for motion before timeout and settlement", () => {
    const tracker = new ParkingEvaluationOutcomeTracker(buildGoal(), footprint, 1000);
    const farPose: ParkingPose = {coords: [0, -10, 0], heading: 0};
    const waiting = tracker.observe(safeObservation({
        nowMs: 60_000,
        pose: farPose,
        speedMps: 0,
    }));
    assert(waiting.outcome === null && !tracker.isArmed(), "waiting at the start must not arm or timeout");

    const armed = tracker.observe(safeObservation({
        nowMs: 60_001,
        pose: farPose,
        speedMps: 0.05,
    }));
    assert(armed.outcome === null && tracker.isArmed(), "first meaningful motion must arm evaluation");

    const timedOut = tracker.observe(safeObservation({
        nowMs: 61_002,
        pose: farPose,
        speedMps: 0.05,
    }));
    assert(timedOut.outcome?.status === "timeout", "timeout must be relative to first motion");
});

test("collision, stop, invalid vehicle, and timeout are strict failures", () => {
    const goal = buildGoal();
    const farPose: ParkingPose = {coords: [0, -10, 0], heading: 0};
    const collision = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 1,
        collision: true,
    })).outcome;
    assert(collision?.status === "collision" && !collision.success, "collision must fail");

    const stopped = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 0,
        stopRequested: true,
    })).outcome;
    assert(stopped?.status === "stopped" && !stopped.success, "stop must fail");
    assert(terminalParkingPhase(stopped) === "failed", "stopped outcome must expose failed terminal phase");

    const invalid = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: null,
        speedMps: 0,
    })).outcome;
    assert(invalid?.status === "invalid_vehicle" && !invalid.success, "invalid vehicle must fail");

    const timeout = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 45_000,
        pose: farPose,
        speedMps: 0,
    })).outcome;
    assert(timeout?.status === "timeout" && !timeout.success, "timeout must fail");
    assert(terminalParkingPhase(timeout) === "failed", "timeout must expose failed terminal phase");
});

test("unsafe expert motion and pose are strict failures", () => {
    const goal = buildGoal();
    const farPose: ParkingPose = {coords: [0, -10, 0], heading: 0};
    const reversing = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 0.5,
        reversing: true,
    })).outcome;
    assert(reversing?.status === "reversing" && !reversing.success, "reverse motion must fail");

    const airborne = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 0,
        onGround: false,
    })).outcome;
    assert(airborne?.status === "off_ground" && !airborne.success, "off-ground motion must fail");

    const tilted = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 0,
        rollDeg: 5.01,
    })).outcome;
    assert(tilted?.status === "not_upright" && !tilted.success, "unsafe tilt must fail");

    const lateralDeviation = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: {coords: [MAX_STRAIGHT_TRACKING_LATERAL_ERROR_M + 0.01, -8, 0], heading: 0},
        speedMps: 1,
    })).outcome;
    assert(
        lateralDeviation?.status === "left_straight_corridor" && !lateralDeviation.success,
        "lateral deviation must fail the straight-only attempt"
    );

    const headingDeviation = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: {coords: [0, -8, 0], heading: MAX_STRAIGHT_TRACKING_HEADING_ERROR_DEG + 0.01},
        speedMps: 1,
    })).outcome;
    assert(
        headingDeviation?.status === "left_straight_corridor" && !headingDeviation.success,
        "heading deviation must fail the straight-only attempt"
    );
});

test("per-frame collision and reverse hazards remain latched", () => {
    const hazards = new ParkingHazardLatch(-0.1);
    hazards.observe(true, -0.2);
    hazards.observe(false, 1);
    assert(hazards.collisionDetected(), "one collision frame must remain latched");
    assert(hazards.reversingDetected(), "one reversing frame must remain latched");
});

test("expert input guards cover driving controls and require applied neutral state", () => {
    assert(
        CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS === 50,
        "expert capture and live-control history must use the declared 50 ms cadence"
    );
    assert(
        PARKING_POST_FLASH_CLEAN_FRAME_BUFFER_MS >= 300,
        "post-flash clean-frame buffer must cover the eight-frame training history"
    );
    assert(PARKING_EVALUATION_SPAWN_SETTLE_MS >= 1000, "evaluation spawn must settle before scoring");
    assert(
        PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS >= 300,
        "evaluation spawn must have a clean collision-history buffer before scoring"
    );
    for (const controlId of [59, 60, 63, 64, 71, 72, 75, 76]) {
        assert(
            (PARKING_DISABLED_CONTROL_IDS as readonly number[]).includes(controlId),
            `control ${controlId} must be disabled`
        );
    }
    assert(isActuatorStateNeutral({
        applied: {enabled: false, steer: 0, throttle: 0, brakePressureAvg: 0, handbrake: false},
    }), "disabled zero actuator should be neutral");
    assert(!isActuatorStateNeutral({
        applied: {enabled: true, steer: 0, throttle: 0, brakePressureAvg: 0, handbrake: false},
    }), "enabled actuator should not be accepted as neutral");
    assert(!isActuatorStateNeutral({
        applied: {enabled: false, steer: 0.1, throttle: 0, brakePressureAvg: 0, handbrake: false},
    }), "non-zero actuator should not be accepted as neutral");
    assert(
        !isInferenceCleanupComplete({state: "succeeded", active: true}),
        "terminal inference state must not bypass process cleanup"
    );
    assert(
        isInferenceCleanupComplete({state: "succeeded", active: false}),
        "inactive terminal inference state should be safe to hand off"
    );
    assert(
        isInferenceCleanupComplete({state: "idle", active: false}),
        "inactive idle inference should be safe to hand off"
    );
    assert(
        !isInferenceCleanupComplete({state: "running", active: false}),
        "a non-terminal inference state must not be treated as cleanup complete"
    );
});

test("physical wheel steering normalizes to the parking setpoint contract", () => {
    const fullLock = resolveSteeringFullLockRadians(42);
    const halfLeft = normalizeVehicleSteering(fullLock / 2, 42);
    assertApproximately(halfLeft.normalized, 0.5, "half-lock steering should normalize to 0.5");
    assertApproximately(halfLeft.wheelAngleRadians, fullLock / 2, "raw wheel angle should be retained");
    assertApproximately(halfLeft.fullLockRadians, fullLock, "handling full lock should be retained");

    const beyondRightLock = normalizeVehicleSteering(-2 * fullLock, 42);
    assert(beyondRightLock.normalized === -1, "normalized steering must clamp at full lock");
});

test("invalid handling steering lock uses the validated Sultan fallback", () => {
    for (const invalidLock of [Number.NaN, 0, 4.99, 90.01]) {
        assert(
            resolveSteeringFullLockRadians(invalidLock) === SULTAN_STEERING_FULL_LOCK_RADIANS,
            `handling lock ${String(invalidLock)} should use the Sultan fallback`
        );
    }
    const fallbackSample = normalizeVehicleSteering(SULTAN_STEERING_FULL_LOCK_RADIANS, Number.NaN);
    assert(fallbackSample.normalized === 1, "Sultan fallback full lock should normalize to 1");
});

test("no-ego heartbeat clears stale readiness and route state", () => {
    const telemetry = createNoEgoControlTelemetry(1234);
    assert(!telemetry.vehicleExists, "no-ego heartbeat must clear vehicle existence");
    assert(!telemetry.isInVehicle, "no-ego heartbeat must clear player readiness");
    assert(telemetry.currentSpeed === 0, "no-ego heartbeat must clear speed");
    assert(telemetry.vehicleModelHash === 0, "no-ego heartbeat must clear vehicle identity");
    assert(telemetry.routeDirectionUnknown === 1, "no-ego route should be explicitly unknown");
    assert(telemetry.routeForwardDelta === 0, "no-ego heartbeat must clear route forward delta");
    assert(telemetry.routeHeadingError === 0, "no-ego heartbeat must clear route heading error");
    assert(telemetry.routeDistance === 0, "no-ego heartbeat must clear route distance");
    assert(telemetry.gameTimeMs === 1234, "heartbeat should preserve sample time");
});

console.log("parking unit tests passed");
